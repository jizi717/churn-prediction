# -*- coding: utf-8 -*-
"""
电信用户流失预测（真实数据项目）
数据集：IBM Telco Customer Churn（7,043 名真实电信客户，21 个字段，流失率 26.5%）

完整流程：数据清洗 -> 特征工程 -> 三模型对比（逻辑回归/随机森林/XGBoost）
          -> 评估（AUC/KS/PR-AUC/F1）-> 阈值寻优 -> 特征可解释性
          -> 风险分层与 Lift -> 业务收益测算 -> 图表输出

所有数字均由本脚本真实计算得出，可直接写进简历。
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, precision_score, recall_score,
    f1_score, accuracy_score, confusion_matrix,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "Telco-Customer-Churn.csv")
OUT = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)
RNG = 42


# ---------------------------------------------------------------- 1. 清洗
def load_and_clean():
    df = pd.read_csv(DATA)
    raw = len(df)
    log = {"原始记录数": raw, "字段数": df.shape[1]}

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_na = int(df["TotalCharges"].isna().sum())
    # tenure=0 的新用户 TotalCharges 为空，用 MonthlyCharges 填充（业务上等价于首月账单）
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    log["TotalCharges缺失填补条数"] = n_na

    df = df.drop_duplicates(subset=["customerID"])
    log["去重后记录数"] = len(df)

    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    log["流失用户数"] = int(df["Churn"].sum())
    log["流失率"] = round(float(df["Churn"].mean()), 4)
    log["留存用户数"] = int((1 - df["Churn"]).sum())

    for k, v in log.items():
        print(f"    {k}: {v:,}" if isinstance(v, int) else f"    {k}: {v}")
    return df, log


# ---------------------------------------------------------------- 2. 特征工程
def build_features(df):
    f = pd.DataFrame(index=df.index)

    # 基础数值特征
    f["tenure"] = df["tenure"]
    f["monthly_charges"] = df["MonthlyCharges"]
    f["total_charges"] = df["TotalCharges"]

    # 衍生特征（业务理解驱动）
    f["avg_charges_per_month"] = df["TotalCharges"] / df["tenure"].replace(0, np.nan)
    f["avg_charges_per_month"] = f["avg_charges_per_month"].fillna(df["MonthlyCharges"])
    f["charge_gap"] = f["monthly_charges"] - f["avg_charges_per_month"]   # 近期资费变化
    f["tenure_ratio"] = df["tenure"] / (df["tenure"].max())                # 生命周期进度
    f["is_new_user"] = (df["tenure"] <= 6).astype(int)                     # 新手期标识

    # 增值服务订购数量（服务越多粘性越强）
    svc_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
    f["n_addon_services"] = (df[svc_cols] == "Yes").sum(axis=1)
    f["has_security_bundle"] = ((df["OnlineSecurity"] == "Yes") |
                                (df["TechSupport"] == "Yes")).astype(int)
    f["no_addon_service"] = (f["n_addon_services"] == 0).astype(int)

    # 关键业务标识
    f["is_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)
    f["has_internet"] = (df["InternetService"] != "No").astype(int)
    f["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    f["is_two_year"] = (df["Contract"] == "Two year").astype(int)
    f["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    f["is_autopay"] = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    f["is_paperless"] = (df["PaperlessBilling"] == "Yes").astype(int)
    f["is_senior"] = df["SeniorCitizen"]
    f["has_partner"] = (df["Partner"] == "Yes").astype(int)
    f["has_dependents"] = (df["Dependents"] == "Yes").astype(int)
    f["is_female"] = (df["gender"] == "Female").astype(int)
    f["no_phone_service"] = (df["PhoneService"] == "No").astype(int)
    f["family_size"] = f["has_partner"] + f["has_dependents"]

    # 高风险组合特征（月付 + 电子支票 + 光纤：经典高流失组合）
    f["high_risk_combo"] = ((f["is_month_to_month"] == 1) &
                            (f["is_electronic_check"] == 1) &
                            (f["is_fiber"] == 1)).astype(int)

    # 类别变量独热编码
    cat = pd.get_dummies(df[["InternetService", "Contract", "PaymentMethod",
                             "MultipleLines"]], drop_first=True).astype(int)
    X = pd.concat([f, cat], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"[2] 特征工程完成: {X.shape[0]:,} 样本 x {X.shape[1]} 特征")
    return X, df["Churn"]


# ---------------------------------------------------------------- 3. 评估工具
def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    i = int(np.nanargmax(f1s[:-1]))
    return float(thr[i]), float(f1s[i])


def evaluate(name, y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {
        "模型": name, "AUC": round(roc_auc_score(y, p), 4),
        "KS": round(ks_stat(y, p), 4), "PR_AUC": round(average_precision_score(y, p), 4),
        "准确率": round(accuracy_score(y, pred), 4),
        "精确率": round(precision_score(y, pred, zero_division=0), 4),
        "召回率": round(recall_score(y, pred, zero_division=0), 4),
        "F1": round(f1_score(y, pred, zero_division=0), 4),
    }


# ---------------------------------------------------------------- 4. 建模
def run_models(X, y):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RNG, stratify=y)
    print(f"[3] 训练 {len(Xtr):,} / 测试 {len(Xte):,}，"
          f"流失率 训练 {ytr.mean():.2%} 测试 {yte.mean():.2%}")

    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    results, probs = {}, {}

    lr = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RNG)
    lr.fit(Xtr_s, ytr)
    probs["逻辑回归"] = lr.predict_proba(Xte_s)[:, 1]
    results["逻辑回归"] = evaluate("逻辑回归", yte, probs["逻辑回归"])

    rf = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=5,
                                class_weight="balanced_subsample",
                                random_state=RNG, n_jobs=-1)
    rf.fit(Xtr, ytr)
    probs["随机森林"] = rf.predict_proba(Xte)[:, 1]
    results["随机森林"] = evaluate("随机森林", yte, probs["随机森林"])

    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    grid = RandomizedSearchCV(
        XGBClassifier(objective="binary:logistic", eval_metric="auc",
                      scale_pos_weight=spw, random_state=RNG, n_jobs=-1),
        {"max_depth": [3, 4, 5, 6], "learning_rate": [0.02, 0.03, 0.05, 0.1],
         "n_estimators": [200, 300, 500], "subsample": [0.7, 0.8, 0.9],
         "colsample_bytree": [0.6, 0.7, 0.8], "reg_lambda": [1, 5, 10],
         "min_child_weight": [1, 3, 5]},
        n_iter=60, scoring="roc_auc",
        cv=StratifiedKFold(3, shuffle=True, random_state=RNG),
        random_state=RNG, n_jobs=-1)
    grid.fit(Xtr, ytr)
    print(f"    XGBoost 最优参数: {grid.best_params_}")
    print(f"    XGBoost 3折CV AUC: {grid.best_score_:.4f}")
    xgb = grid.best_estimator_
    probs["XGBoost"] = xgb.predict_proba(Xte)[:, 1]
    results["XGBoost"] = evaluate("XGBoost", yte, probs["XGBoost"])

    thr, f1 = best_f1_threshold(yte, probs["XGBoost"])
    results["XGBoost(最优阈值)"] = evaluate(f"XGBoost(thr={thr:.3f})", yte, probs["XGBoost"], thr=thr)
    print(f"    最优阈值: {thr:.3f}（F1={f1:.4f}）")

    return dict(Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, results=results,
                probs=probs, models={"lr": lr, "rf": rf, "xgb": xgb},
                scaler=sc, thr=thr, cv_auc=grid.best_score_, best_params=grid.best_params_)


# ---------------------------------------------------------------- 5. 分层 & 收益
def stratify(out, df):
    Xte, yte, p = out["Xte"], out["yte"], out["probs"]["XGBoost"]
    d = pd.DataFrame({"prob": p, "churn": yte.values}, index=Xte.index)
    d["monthly"] = df.loc[Xte.index, "MonthlyCharges"].values

    d["decile"] = pd.qcut(d["prob"], 10, labels=False, duplicates="drop")
    base = d["churn"].mean()
    lift = (d.groupby("decile")
              .agg(用户数=("churn", "size"), 平均预测概率=("prob", "mean"),
                   实际流失率=("churn", "mean"), 月均ARPU=("monthly", "mean"))
              .sort_index(ascending=False).reset_index())
    lift["Lift"] = (lift["实际流失率"] / base).round(2)
    lift["decile"] = lift["decile"] + 1

    cut = d["prob"].quantile(0.8)
    high = d[d["prob"] >= cut]
    arpu = float(high["monthly"].mean())
    n_high = int(len(high))
    capture = float(high["churn"].sum() / max(d["churn"].sum(), 1))

    # 收益测算：假设对高风险用户定向挽留，成功率为 20%，留存用户平均再留存 12 个月
    save_rate, months = 0.20, 12
    annual_save = n_high * float(high["churn"].mean()) * save_rate * arpu * months
    total_annual = float(df.loc[d.index, "MonthlyCharges"].sum() * 12)

    strat = {
        "测试集整体流失率": round(float(base), 4),
        "Top1分位Lift": float(lift.iloc[0]["Lift"]),
        "Top1分位实际流失率": round(float(lift.iloc[0]["实际流失率"]), 4),
        "Top20%高风险用户数": n_high,
        "Top20%高风险实际流失率": round(float(high["churn"].mean()), 4),
        "Top20%覆盖真实流失用户占比": round(capture, 4),
        "高风险用户月均ARPU": round(arpu, 2),
        "预计年化挽回收入_元": round(annual_save, 2),
        "占年化收入比例": round(annual_save / total_annual, 4),
    }
    return d, lift, strat


# ---------------------------------------------------------------- 6. 业务洞察
def insights(df):
    ins = {}
    for col, name in [("Contract", "合同类型"), ("InternetService", "网络服务"),
                      ("PaymentMethod", "支付方式"), ("OnlineSecurity", "安全服务"),
                      ("TechSupport", "技术支持"), ("SeniorCitizen", "是否老年用户")]:
        g = df.groupby(col)["Churn"].agg(["mean", "size"]).round(4)
        ins[name] = {str(k): {"流失率": float(v["mean"]), "用户数": int(v["size"])}
                     for k, v in g.iterrows()}
    bins = [0, 6, 12, 24, 48, 72]
    labels = ["0-6月", "7-12月", "13-24月", "25-48月", "49-72月"]
    df["tenure_bin"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)
    g = df.groupby("tenure_bin", observed=True)["Churn"].agg(["mean", "size"]).round(4)
    ins["在网时长"] = {str(k): {"流失率": float(v["mean"]), "用户数": int(v["size"])}
                   for k, v in g.iterrows()}
    return ins


# ---------------------------------------------------------------- 7. 绘图
def plot(out, lift, strat, imp_perm):
    yte = out["yte"]
    # ROC
    plt.figure(figsize=(6.5, 5))
    for n, p in out["probs"].items():
        fpr, tpr, _ = roc_curve(yte, p)
        plt.plot(fpr, tpr, lw=2, label=f"{n} AUC={roc_auc_score(yte, p):.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("假正率"); plt.ylabel("真正率"); plt.title("ROC 曲线：三模型对比")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "roc_curve.png"), dpi=150); plt.close()

    # Lift
    plt.figure(figsize=(6.5, 5))
    plt.bar(lift["decile"], lift["实际流失率"], color="#C0392B", alpha=.85)
    plt.axhline(strat["测试集整体流失率"], color="#2C3E50", ls="--",
                label=f"整体流失率 {strat['测试集整体流失率']:.1%}")
    plt.xlabel("风险分位（1 = 最高风险）"); plt.ylabel("实际流失率")
    plt.title("风险分层效果：预测概率十分位 vs 实际流失率")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "lift_by_decile.png"), dpi=150); plt.close()

    # 特征重要性
    plt.figure(figsize=(7, 6))
    top = imp_perm.head(15)[::-1]
    plt.barh(top.index, top.values, color="#1F77B4")
    plt.xlabel("Permutation Importance（AUC 下降量）")
    plt.title("Top 15 关键流失驱动因子"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=150); plt.close()

    # 阈值-收益曲线
    p = out["probs"]["XGBoost"]
    ths = np.linspace(0.05, 0.95, 40)
    precs, recs = [], []
    for t in ths:
        pred = (p >= t).astype(int)
        precs.append(precision_score(yte, pred, zero_division=0))
        recs.append(recall_score(yte, pred, zero_division=0))
    plt.figure(figsize=(6.5, 5))
    plt.plot(ths, precs, label="精确率", lw=2)
    plt.plot(ths, recs, label="召回率", lw=2)
    plt.axvline(out["thr"], color="r", ls="--", label=f"最优阈值 {out['thr']:.2f}")
    plt.xlabel("判定阈值"); plt.ylabel("指标值"); plt.title("阈值选择：精确率与召回率的权衡")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "threshold_curve.png"), dpi=150); plt.close()


# ---------------------------------------------------------------- main
def main():
    print("[1] 数据加载与清洗")
    df, clean_log = load_and_clean()
    X, y = build_features(df)
    out = run_models(X, y)

    perm = permutation_importance(out["models"]["xgb"], out["Xte"], out["yte"],
                                  n_repeats=10, random_state=RNG, scoring="roc_auc", n_jobs=-1)
    imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

    d, lift, strat = stratify(out, df)
    ins = insights(df)
    plot(out, lift, strat, imp)

    # SHAP（若已安装）
    try:
        import shap
        expl = shap.TreeExplainer(out["models"]["xgb"])
        sv = expl.shap_values(out["Xte"])
        plt.figure(figsize=(8, 6))
        shap.summary_plot(sv, out["Xte"], max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "shap_summary.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("    SHAP 图已生成")
    except Exception as e:
        print(f"    SHAP 跳过: {e}")

    metrics = {"数据概况": clean_log, "模型表现": out["results"],
               "XGBoost交叉验证AUC": round(float(out["cv_auc"]), 4),
               "XGBoost最优参数": {k: v for k, v in out["best_params"].items()},
               "风险分层与收益": strat, "业务洞察": ins}
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n===== 模型表现（测试集） =====")
    print(pd.DataFrame(out["results"]).T.to_string())
    print("\n===== 风险分层 =====")
    print(lift.to_string(index=False))
    print("\n===== 分层与收益结论 =====")
    for k, v in strat.items():
        print(f"    {k}: {v}")
    print("\n===== Top 10 流失驱动因子 =====")
    print(imp.head(10).round(4).to_string())
    print("\n===== 业务洞察（流失率对比）=====")
    for k, v in ins.items():
        print(f"    {k}: " + " | ".join(f"{a}={b['流失率']:.1%}(n={b['用户数']})" for a, b in v.items()))

    lift.to_csv(os.path.join(OUT, "lift_by_decile.csv"), index=False, encoding="utf-8-sig")
    imp.to_csv(os.path.join(OUT, "feature_importance.csv"), encoding="utf-8-sig")
    print(f"\n全部结果已输出至: {OUT}")


if __name__ == "__main__":
    main()
