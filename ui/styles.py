"""Visual theme kept separate from application behavior."""

CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --cyan:#42e8c4; --deep:#06161a; --panel:#0c2429; --muted:#89a0a2; --red:#ff706a; --violet:#ad8aff; --amber:#f1ba56; }
.stApp { background: #06161a; color: #f2f8f5; font-family:'Manrope',sans-serif; }
[data-testid="stHeader"] { background:rgba(6,22,26,.82); border-bottom:1px solid rgba(125,190,181,.14); }
[data-testid="stSidebar"] { background:#081d22; border-right:1px solid rgba(125,190,181,.14); }
[data-testid="stSidebar"] * { color:#dbe8e5; }
.block-container { max-width:1180px; padding-top:2.2rem; padding-bottom:5rem; }
h1,h2,h3 { font-family:'Manrope',sans-serif!important; letter-spacing:-.035em!important; }
p, label, .stMarkdown { font-family:'Manrope',sans-serif; }
.sf-brand { font-weight:800; letter-spacing:.18em; font-size:.82rem; margin-bottom:1.4rem; }
.sf-brand span { color:var(--cyan); }
.eyebrow { color:var(--cyan); font:500 .68rem 'DM Mono',monospace; letter-spacing:.16em; text-transform:uppercase; }
.hero { padding:2.2rem 0 1.5rem; }
.hero h1 { font-size:clamp(3rem,6vw,5.8rem); line-height:.98; margin:.8rem 0 1.4rem; max-width:1000px; }
.hero h1 span { color:var(--cyan); }
.hero p { color:#9eb1b2; font-size:1.05rem; line-height:1.75; max-width:760px; }
.status { display:inline-flex; gap:.55rem; align-items:center; border:1px solid rgba(125,190,181,.18); border-radius:99px; padding:.5rem .8rem; font:500 .65rem 'DM Mono',monospace; }
.status i { width:.45rem; height:.45rem; background:var(--cyan); box-shadow:0 0 12px var(--cyan); border-radius:50%; }
.impact { border:1px solid rgba(125,190,181,.16); background:linear-gradient(145deg,rgba(12,36,41,.95),rgba(7,25,29,.95)); padding:1.35rem; min-height:190px; }
.impact b { display:block; color:var(--cyan); font:500 .68rem 'DM Mono',monospace; letter-spacing:.12em; margin-bottom:2rem; }
.impact h3 { font-size:1.05rem; margin-bottom:.4rem; }
.impact p { color:#8ea2a3; font-size:.8rem; line-height:1.6; }
.objective { border-left:2px solid var(--cyan); background:rgba(66,232,196,.045); padding:1.4rem 1.6rem; margin:1.5rem 0 2.5rem; }
.objective strong { color:var(--cyan); }
.section-label { color:#6f8587; font:500 .58rem 'DM Mono',monospace; letter-spacing:.14em; margin-bottom:.35rem; }
.metric-card { border-top:1px solid rgba(125,190,181,.18); padding:1rem .2rem; min-height:90px; }
.metric-card b { color:var(--cyan); font-size:1.65rem; }
.metric-card span { display:block; color:#718789; font:500 .63rem 'DM Mono',monospace; text-transform:uppercase; margin-top:.25rem; }
.hybrid-summary { color:#9eb1b2; max-width:900px; line-height:1.7; margin-bottom:1rem; }
.hybrid-summary strong { color:#eff8f4; }
.section-heading { display:flex; align-items:flex-start; gap:1rem; border-top:1px solid rgba(125,190,181,.14); margin:3.2rem 0 1.2rem; padding-top:1.3rem; }
.section-heading>span { color:var(--cyan); font:500 .61rem 'DM Mono',monospace; border:1px solid rgba(66,232,196,.24); padding:.3rem .42rem; margin-top:.12rem; }
.section-heading h3 { font-size:1.45rem; margin:0 0 .25rem; }
.section-heading p { color:#728789; font-size:.74rem; margin:0; }
.model-stack { display:grid; grid-template-columns:1fr 1fr; gap:.8rem; margin-bottom:2.3rem; }
.model-card { border:1px solid rgba(125,190,181,.16); background:rgba(12,36,41,.72); padding:1.3rem; position:relative; overflow:hidden; min-height:255px; }
.model-card:before { content:""; position:absolute; left:0; top:0; bottom:0; width:2px; background:var(--cyan); }
.stage-two-card:before { background:var(--violet); }
.model-card>span, .evaluation-card>span, .support-card>span { color:var(--cyan); font:500 .58rem 'DM Mono',monospace; letter-spacing:.11em; }
.stage-two-card>span { color:var(--violet); }
.model-card h4, .evaluation-card h4 { margin:.65rem 0 1rem; font-size:1rem; }
.model-row { display:flex; justify-content:space-between; gap:1rem; border-top:1px solid rgba(125,190,181,.11); padding:.65rem 0; }
.model-row b { color:#dfeae7; font-size:.76rem; }
.model-row small { color:#778d8f; font-size:.68rem; text-align:right; }
.model-card p { color:#819597; font-size:.72rem; line-height:1.55; margin:.65rem 0 0; }
.decision-flow { margin:1.1rem 0 1.3rem; border:1px solid rgba(125,190,181,.16); background:linear-gradient(145deg,rgba(12,36,41,.92),rgba(6,22,26,.96)); padding:1.25rem; }
.flow-track { display:grid; grid-template-columns:minmax(0,1fr) 42px minmax(0,1.25fr) 42px minmax(0,1.25fr); align-items:stretch; }
.flow-node { position:relative; min-height:190px; padding:1.2rem; border:1px solid rgba(125,190,181,.17); background:rgba(5,21,25,.72); }
.flow-node:before { content:""; position:absolute; top:-1px; left:-1px; right:-1px; height:2px; background:#61787a; }
.stage-one-node:before { background:var(--cyan); box-shadow:0 0 15px rgba(66,232,196,.3); }
.stage-two-node:before { background:var(--violet); box-shadow:0 0 15px rgba(173,138,255,.3); }
.flow-number { float:right; color:#496164; font:500 .66rem 'DM Mono',monospace; }
.flow-kicker { color:#7e9496; font:500 .6rem 'DM Mono',monospace; letter-spacing:.12em; }
.stage-one-node .flow-kicker { color:var(--cyan); }
.stage-two-node .flow-kicker { color:var(--violet); }
.flow-node h4 { clear:both; color:#eff8f4; font-size:1.02rem; margin:1.15rem 0 .45rem; }
.flow-node p { color:#879b9d; font-size:.76rem; line-height:1.55; margin:0; }
.flow-signals { display:flex; flex-wrap:wrap; gap:.35rem; margin-top:1rem; }
.flow-signals span { color:#a9bcba; border:1px solid rgba(125,190,181,.16); border-radius:99px; padding:.24rem .48rem; font:500 .55rem 'DM Mono',monospace; }
.flow-arrow { display:flex; align-items:center; justify-content:center; color:#557174; }
.flow-arrow span { width:100%; text-align:center; font-size:1.3rem; }
.flow-routing { display:grid; grid-template-columns:1fr 1.8fr; gap:.8rem; margin-top:.8rem; }
.route-path { position:relative; padding:1rem 1.1rem; border:1px solid rgba(125,190,181,.14); background:rgba(5,21,25,.62); }
.route-source { display:block; color:#718789; font:500 .56rem 'DM Mono',monospace; letter-spacing:.08em; margin-bottom:.75rem; }
.route-line { display:block; width:2.2rem; height:2px; background:currentColor; margin-bottom:.75rem; }
.route-path strong { font-size:.78rem; }
.route-path small { display:block; color:#758b8d; font-size:.66rem; margin-top:.55rem; line-height:1.5; }
.normal-path { color:var(--cyan); }
.investigate-path { color:var(--violet); }
.outcome-group { display:flex; flex-wrap:wrap; gap:.5rem; }
.outcome-group strong { color:#e7eeee; border:1px solid rgba(173,138,255,.28); padding:.38rem .55rem; background:rgba(173,138,255,.055); }
.test-context { display:grid; grid-template-columns:repeat(4,1fr); gap:.55rem; margin:1rem 0 .8rem; }
.test-context>div { border:1px solid rgba(125,190,181,.14); background:rgba(5,21,25,.56); padding:1rem; }
.test-context b { display:block; color:#eff8f4; font-size:1.25rem; }
.test-context span { color:#718789; font:500 .58rem 'DM Mono',monospace; text-transform:uppercase; }
.evaluation-card { min-height:145px; border:1px solid rgba(125,190,181,.14); background:#0b2227; padding:1.2rem; margin-bottom:1rem; }
.evaluation-card p { color:#84999b; font-size:.76rem; line-height:1.6; }
.evaluation-card p b { color:var(--cyan); }
.family-panel { border:1px solid rgba(125,190,181,.14); background:#0b2227; padding:1.35rem; margin-top:.35rem; }
.family-panel-head { display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; border-bottom:1px solid rgba(125,190,181,.12); padding-bottom:1rem; margin-bottom:.25rem; }
.family-panel-head span { color:var(--cyan); font:500 .58rem 'DM Mono',monospace; letter-spacing:.11em; }
.family-panel-head h4 { margin:.35rem 0 0; font-size:1rem; }
.family-panel-head>small { color:#617779; font:400 .56rem 'DM Mono',monospace; white-space:nowrap; }
.family-row { display:grid; grid-template-columns:145px minmax(120px,1fr) 72px; gap:1rem; align-items:center; padding:.85rem 0; border-bottom:1px solid rgba(125,190,181,.08); }
.family-name b { display:block; color:#e2ece9; font-size:.78rem; }
.family-name small { display:block; color:#6f8587; font:400 .58rem 'DM Mono',monospace; margin-top:.2rem; }
.family-meter { height:7px; background:rgba(125,190,181,.1); overflow:hidden; }
.family-meter i { display:block; height:100%; background:linear-gradient(90deg,#279f8a,var(--cyan)); }
.family-row strong { color:#dce8e5; font:500 .7rem 'DM Mono',monospace; text-align:right; }
.low-support .family-meter { background:rgba(241,186,86,.12); }
.low-support strong { color:var(--amber); }
.family-note { color:#6f8587; font-size:.65rem; line-height:1.55; margin:.9rem 0 0; }
@media (max-width:850px) {
  .model-stack, .test-context { grid-template-columns:1fr; }
  .flow-track { grid-template-columns:1fr; }
  .flow-arrow { min-height:38px; transform:rotate(90deg); }
  .flow-node { min-height:0; }
  .flow-routing { grid-template-columns:1fr; }
  .family-row { grid-template-columns:110px minmax(80px,1fr) 64px; gap:.65rem; }
}
@media (max-width:560px) {
  .block-container { padding-left:1rem; padding-right:1rem; }
  .hero h1 { font-size:2.65rem; }
  .family-panel-head>small { display:none; }
  .family-row { grid-template-columns:1fr 58px; }
  .family-meter { grid-column:1 / -1; grid-row:2; }
}
.decision { border:1px solid; padding:1.4rem 1.5rem; margin:.5rem 0 1rem; }
.decision small { font:500 .62rem 'DM Mono',monospace; letter-spacing:.14em; }
.decision h2 { margin:.35rem 0 .2rem; font-size:2rem; }
.decision.normal { color:var(--cyan); background:rgba(66,232,196,.055); }
.decision.attack { color:var(--red); background:rgba(255,112,106,.055); }
.decision.unknown { color:var(--violet); background:rgba(173,138,255,.055); }
.decision.review { color:var(--amber); background:rgba(241,186,86,.055); }
.route { background:#0c2429; padding:1rem; border-left:2px solid var(--cyan); margin:.8rem 0; }
.route small { color:#708789; font:500 .6rem 'DM Mono',monospace; }
.route b { display:block; color:var(--cyan); font:500 .75rem 'DM Mono',monospace; margin:.35rem 0; }
.notice { color:#8ea1a2; font-size:.78rem; line-height:1.6; border:1px solid rgba(125,190,181,.15); padding:1rem; }
.footer { color:#5f7779; font:400 .63rem 'DM Mono',monospace; border-top:1px solid rgba(125,190,181,.14); margin-top:4rem; padding-top:1rem; }
div[data-testid="stMetric"] { background:#0b2227; border:1px solid rgba(125,190,181,.14); padding:1rem; }
div[data-testid="stMetric"] label { color:#809597!important; }
div[data-testid="stMetricValue"] { color:#eff8f4; }
.stButton>button, .stDownloadButton>button { background:var(--cyan); color:#05201b; border:0; border-radius:3px; font-weight:800; min-height:2.8rem; }
.stButton>button:hover, .stDownloadButton>button:hover { background:#8ffce5; color:#05201b; border:0; }
div[data-baseweb="select"]>div, .stTextInput input, .stNumberInput input { background:#0b2227; border-color:rgba(125,190,181,.18); }
[data-testid="stFileUploader"] { border:1px dashed rgba(125,190,181,.3); padding:1rem; background:rgba(12,36,41,.4); }
.stTabs [data-baseweb="tab-list"] { gap:1.3rem; border-bottom:1px solid rgba(125,190,181,.16); }
.stTabs [data-baseweb="tab"] { font-family:'DM Mono',monospace; font-size:.72rem; }
.stTabs [aria-selected="true"] { color:var(--cyan)!important; }
</style>
"""
