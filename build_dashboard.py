# -*- coding: utf-8 -*-
"""
把流失预测项目的真实结果渲染成一个交互式可视化看板（HTML + ECharts）
输出：results/dashboard.html
"""
import os
import json
import base64
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "results")

m = json.load(open(os.path.join(OUT, "metrics.json"), encoding="utf-8"))
lift = pd.read_csv(os.path.join(OUT, "lift_by_decile.csv"))
imp = pd.read_csv(os.path.join(OUT, "feature_importance.csv"), index_col=0).iloc[:, 0]
ins = m["业务洞察"]
strat = m["风险分层与收益"]
res = m["模型表现"]


def img_b64(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return ""
    with open(p, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def ins_data(key):
    d = ins[key]
    return list(d.keys()), [round(v["流失率"] * 100, 2) for v in d.values()]


cards = [
    ("客户总数", f"{m['数据概况']['原始记录数']:,}", "人"),
    ("整体流失率", f"{m['数据概况']['流失率'] * 100:.2f}", "%"),
    ("最优模型 AUC", f"{max(v['AUC'] for v in res.values()):.4f}", ""),
    ("KS 值", f"{max(v['KS'] for v in res.values()):.4f}", ""),
    ("Top1 分位 Lift", f"{strat['Top1分位Lift']}", "倍"),
    ("Top20% 捕获流失用户", f"{strat['Top20%覆盖真实流失用户占比'] * 100:.1f}", "%"),
]

lift_js = json.dumps({
    "x": [int(i) for i in lift["decile"]],
    "churn": [round(v * 100, 2) for v in lift["实际流失率"]],
    "arpu": [round(v, 1) for v in lift["月均ARPU"]],
}, ensure_ascii=False)

imp_top = imp.head(12)[::-1]
imp_js = json.dumps({
    "y": [str(i) for i in imp_top.index],
    "v": [round(float(x), 4) for x in imp_top.values],
}, ensure_ascii=False)

name_map = {"is_month_to_month": "月付合同", "tenure": "在网时长", "is_two_year": "两年合约",
            "has_security_bundle": "订购安全服务", "has_internet": "开通网络",
            "total_charges": "累计消费", "tenure_ratio": "生命周期进度",
            "avg_charges_per_month": "月均实际消费", "monthly_charges": "月费",
            "is_fiber": "光纤用户", "is_electronic_check": "电子支票支付",
            "n_addon_services": "增值服务数", "charge_gap": "资费变化",
            "is_new_user": "新客标识", "no_addon_service": "无增值服务"}
imp_js = json.dumps({
    "y": [name_map.get(str(i), str(i)) for i in imp_top.index],
    "v": [round(float(x), 4) for x in imp_top.values],
}, ensure_ascii=False)

c1k, c1v = ins_data("合同类型")
c2k, c2v = ins_data("支付方式")
c3k, c3v = ins_data("在网时长")
c4k, c4v = ins_data("网络服务")

model_rows = "".join(
    f"<tr><td>{v['模型']}</td><td>{v['AUC']}</td><td>{v['KS']}</td><td>{v['PR_AUC']}</td>"
    f"<td>{v['精确率']}</td><td>{v['召回率']}</td><td>{v['F1']}</td></tr>"
    for v in res.values())

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>电信用户流失预测 · 分析看板</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:"Microsoft YaHei",-apple-system,sans-serif;background:#f5f7fa;color:#1f2937;padding:24px}}
.header{{max-width:1280px;margin:0 auto 20px}}
h1{{font-size:24px;font-weight:600;margin-bottom:6px}}
.sub{{color:#6b7280;font-size:13px}}
.cards{{max-width:1280px;margin:0 auto 20px;display:grid;grid-template-columns:repeat(6,1fr);gap:14px}}
.card{{background:#fff;border-radius:10px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.card .label{{font-size:12px;color:#6b7280;margin-bottom:8px}}
.card .value{{font-size:26px;font-weight:700;color:#1d4ed8}}
.card .unit{{font-size:13px;color:#9ca3af;margin-left:2px}}
.grid{{max-width:1280px;margin:0 auto;display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.panel{{background:#fff;border-radius:10px;padding:18px;box-shadow:0 1px 3px rgba(0,0,0,.08);margin-bottom:16px}}
.panel h2{{font-size:15px;font-weight:600;margin-bottom:12px;color:#111827}}
.chart{{width:100%;height:320px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:9px 10px;text-align:left;border-bottom:1px solid #eef0f3}}
th{{background:#f9fafb;font-weight:600;color:#374151}}
.full{{grid-column:1 / -1}}
img{{width:100%;border-radius:8px;margin-top:8px}}
.foot{{max-width:1280px;margin:16px auto;color:#9ca3af;font-size:12px;text-align:center}}
</style>
</head>
<body>
<div class="header">
  <h1>电信用户流失预测分析看板</h1>
  <div class="sub">数据源：IBM Telco Customer Churn（{m['数据概况']['原始记录数']:,} 名真实客户）　|
  模型：Logistic Regression / Random Forest / XGBoost　|　所有指标由脚本真实计算</div>
</div>

<div class="cards">
{''.join(f'<div class="card"><div class="label">{l}</div><div class="value">{v}<span class="unit">{u}</span></div></div>' for l, v, u in cards)}
</div>

<div class="grid">
  <div class="panel"><h2>风险分层效果（十分位 Lift）</h2><div id="c1" class="chart"></div></div>
  <div class="panel"><h2>关键流失驱动因子（Permutation Importance）</h2><div id="c2" class="chart"></div></div>

  <div class="panel"><h2>合同类型 vs 流失率</h2><div id="c3" class="chart"></div></div>
  <div class="panel"><h2>支付方式 vs 流失率</h2><div id="c4" class="chart"></div></div>

  <div class="panel"><h2>在网时长 vs 流失率</h2><div id="c5" class="chart"></div></div>
  <div class="panel"><h2>网络服务类型 vs 流失率</h2><div id="c6" class="chart"></div></div>

  <div class="panel full"><h2>模型表现对比（测试集）</h2>
    <table><thead><tr><th>模型</th><th>AUC</th><th>KS</th><th>PR-AUC</th><th>精确率</th><th>召回率</th><th>F1</th></tr></thead>
    <tbody>{model_rows}</tbody></table>
  </div>

  <div class="panel"><h2>ROC 曲线</h2><img src="{img_b64('roc_curve.png')}"></div>
  <div class="panel"><h2>SHAP 可解释性摘要</h2><img src="{img_b64('shap_summary.png')}"></div>

  <div class="panel full"><h2>业务结论</h2>
    <table><tbody>
      <tr><td>高风险层（预测概率 Top 20%）实际流失率</td><td><b>{strat['Top20%高风险实际流失率']*100:.1f}%</b>（整体 {strat['测试集整体流失率']*100:.1f}%）</td></tr>
      <tr><td>Top 20% 用户覆盖的真实流失用户比例</td><td><b>{strat['Top20%覆盖真实流失用户占比']*100:.1f}%</b></td></tr>
      <tr><td>最高风险分位（Top 10%）Lift</td><td><b>{strat['Top1分位Lift']} 倍</b>，该层实际流失率 {strat['Top1分位实际流失率']*100:.1f}%</td></tr>
      <tr><td>高风险用户月均 ARPU</td><td><b>¥{strat['高风险用户月均ARPU']}</b></td></tr>
      <tr><td>预计年化挽回收入（按 20% 挽留成功率测算）</td><td><b>¥{strat['预计年化挽回收入_元']:,.0f}</b></td></tr>
    </tbody></table>
  </div>
</div>

<div class="foot">由 churn_telco.py + build_dashboard.py 自动生成 · 可直接放入作品集</div>

<script>
const L = {lift_js};
echarts.init(document.getElementById('c1')).setOption({{
  tooltip:{{trigger:'axis'}},
  xAxis:{{type:'category',data:L.x,name:'风险分位'}},
  yAxis:{{type:'value',name:'实际流失率(%)'}},
  series:[{{type:'bar',data:L.churn,itemStyle:{{color:'#c0392b'}},name:'实际流失率'}},
          {{type:'line',data:L.arpu,yAxisIndex:0,name:'月均ARPU',itemStyle:{{color:'#1d4ed8'}}}}]
}});
const I = {imp_js};
echarts.init(document.getElementById('c2')).setOption({{
  tooltip:{{trigger:'axis'}},
  grid:{{left:110,right:30}},
  xAxis:{{type:'value',name:'AUC 下降量'}},
  yAxis:{{type:'category',data:I.y}},
  series:[{{type:'bar',data:I.v,itemStyle:{{color:'#1f77b4'}}}}]
}});
function bar(id,keys,vals,color){{
  echarts.init(document.getElementById(id)).setOption({{
    tooltip:{{trigger:'axis',formatter:p=>p[0].name+': '+p[0].value+'%'}},
    grid:{{left:60,right:20,top:20}},
    xAxis:{{type:'category',data:keys,axisLabel:{{interval:0,rotate:15}}}},
    yAxis:{{type:'value',name:'流失率(%)'}},
    series:[{{type:'bar',data:vals,itemStyle:{{color:color}},
      label:{{show:true,position:'top',formatter:'{{c}}%'}}}}]
  }});
}}
bar('c3',{json.dumps(c1k, ensure_ascii=False)},{c1v},'#c0392b');
bar('c4',{json.dumps(c2k, ensure_ascii=False)},{c2v},'#e67e22');
bar('c5',{json.dumps(c3k, ensure_ascii=False)},{c3v},'#16a085');
bar('c6',{json.dumps(c4k, ensure_ascii=False)},{c4v},'#8e44ad');
</script>
</body>
</html>
"""

path = os.path.join(OUT, "dashboard.html")
with open(path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"看板已生成: {path}")
