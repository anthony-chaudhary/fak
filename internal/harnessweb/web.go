package harnessweb

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/goalregistry"
	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

const defaultFakURL = "http://127.0.0.1:8080"

const (
	maxLocalWorkIDs     = 8
	maxLocalWorkIDBytes = 32
)

const page = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>fak local harness</title><style>
:root{color-scheme:dark;--ink:#edf5ef;--muted:#9dafaa;--panel:#101c17;--panel2:#0a1510;--line:#294038;--accent:#8ee8bd;--accent2:#8eb5ff;--warn:#f6c453;--danger:#fb7185;--bg:#07100c}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 16% 0,#173326 0,var(--bg) 38%);color:var(--ink);font:15px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}a{color:inherit}button,input{font:inherit}main{width:min(1180px,94vw);margin:24px auto 64px}.eyebrow{margin:0;color:var(--accent);font-weight:800;letter-spacing:.16em;text-transform:uppercase;font-size:.78rem}.topbar{display:flex;justify-content:space-between;gap:18px;align-items:center;margin-bottom:18px}.brand{display:flex;align-items:center;gap:12px}.mark{display:grid;place-items:center;width:34px;height:34px;border:1px solid var(--accent);border-radius:9px;color:var(--accent);font-weight:900}.mode{display:flex;gap:8px;align-items:center;color:var(--muted);font-size:.86rem}.dot{width:8px;height:8px;border-radius:50%;background:var(--accent)}.shell{border:1px solid var(--line);border-radius:18px;background:color-mix(in srgb,var(--panel) 94%,transparent);box-shadow:0 24px 80px #0008;overflow:hidden}.top{padding:26px 30px 22px;border-bottom:1px solid var(--line)}h1{font:650 clamp(2rem,5vw,3.6rem)/1 ui-sans-serif,system-ui;margin:.12em 0 .25em}.sub{margin:0;color:var(--muted);max-width:72ch}.actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:20px}.button,.skin{display:inline-flex;align-items:center;justify-content:center;min-height:38px;border:1px solid var(--line);border-radius:9px;background:transparent;color:var(--ink);padding:8px 12px;font-weight:750;text-decoration:none;cursor:pointer}.button.primary{background:var(--accent);border-color:var(--accent);color:#062116}.button[aria-disabled="true"]{color:var(--muted);cursor:default}.nav{display:flex;gap:5px;overflow:auto;padding:10px 22px;border-bottom:1px solid var(--line);background:#09130f}.nav a{white-space:nowrap;text-decoration:none;color:var(--muted);padding:7px 9px;border-radius:7px}.nav a:hover,.nav a:focus-visible{background:var(--panel);color:var(--ink);outline:none}.content{padding:24px 30px 30px}.section{scroll-margin-top:16px;margin-bottom:32px}.section:last-child{margin-bottom:0}.sectionhead{display:flex;align-items:end;justify-content:space-between;gap:16px;margin-bottom:13px}.sectionhead h2{font:650 1.2rem/1.2 ui-sans-serif,system-ui;margin:0}.sectionhead p{margin:0;color:var(--muted);font-size:.84rem}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.stat{border:1px solid var(--line);border-radius:12px;background:var(--panel2);padding:16px}.stat .value{display:block;font:700 1.8rem/1 ui-sans-serif,system-ui;margin-bottom:7px}.stat .label{display:block;color:var(--ink);font-weight:750}.stat .detail{display:block;color:var(--muted);font-size:.78rem;margin-top:3px}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:12px}.panel{border:1px solid var(--line);border-radius:12px;background:var(--panel2);padding:16px;min-width:0}.panel h3{font:650 1rem/1.2 ui-sans-serif,system-ui;margin:0 0 12px}.rows{display:grid;gap:8px}.row{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:12px;padding:10px 0;border-top:1px solid var(--line)}.row:first-child{border-top:0;padding-top:0}.row:last-child{padding-bottom:0}.row strong{display:block;overflow-wrap:anywhere}.meta{color:var(--muted);font-size:.78rem}.tag{align-self:start;border:1px solid var(--line);border-radius:99px;padding:3px 7px;color:var(--muted);font-size:.72rem}.dashboards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.dashboard{display:block;min-height:116px;border:1px solid var(--line);border-radius:12px;background:var(--panel2);padding:14px;text-decoration:none}.dashboard[href]:hover,.dashboard[href]:focus-visible{border-color:var(--accent2);transform:translateY(-1px);outline:none}.dashboard.disabled{opacity:.58}.dashboard strong{display:block;margin-bottom:5px}.dashboard span{display:block;color:var(--muted);font-size:.8rem}.path{margin-top:12px!important;color:var(--accent2)!important;overflow-wrap:anywhere}.composer{display:flex;gap:10px}.composer input{flex:1;min-width:0;background:#06100b;color:var(--ink);border:1px solid var(--line);border-radius:10px;padding:14px}.composer button,.approval button{background:var(--accent);color:#062116;border:0;border-radius:10px;padding:0 20px;font-weight:800;cursor:pointer}.examples{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 0}.examples button{background:transparent;border:1px solid var(--line);color:var(--muted);border-radius:99px;padding:6px 10px;cursor:pointer}.events{display:grid;gap:9px;margin-top:18px}.event{border-left:3px solid var(--line);background:#07110c;padding:11px 13px}.event[data-kind^="message"]{border-color:var(--accent)}.event[data-kind^="tool"],.event[data-kind^="approval"]{border-color:var(--warn)}.event[data-kind="error"]{border-color:var(--danger)}.kind{color:var(--muted);font-size:.74rem;text-transform:uppercase}.approval{display:flex;gap:8px;margin-top:10px}.approval button:last-child{background:transparent;border:1px solid var(--danger);color:var(--danger)}.empty{margin:0;color:var(--muted);border:1px dashed var(--line);padding:18px;border-radius:10px}code{color:var(--accent)}body[data-skin="minimal"]{--accent:#93c5fd;--panel:#111827;--panel2:#08101d;--line:#334155;--bg:#020617}@media(max-width:900px){.stats,.dashboards{grid-template-columns:repeat(2,minmax(0,1fr))}.twocol{grid-template-columns:1fr}}@media(max-width:600px){main{width:min(100% - 20px,1180px);margin-top:14px}.top,.content{padding-left:18px;padding-right:18px}.topbar{align-items:flex-start}.mode{max-width:45%;text-align:right}.stats,.dashboards{grid-template-columns:1fr}.composer{display:grid}.composer button{padding:13px}.sectionhead{align-items:start;flex-direction:column}}
.startup-grid{display:grid;grid-template-columns:minmax(220px,.7fr) minmax(0,1.3fr);gap:12px}.startup-summary{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.startup-summary .stat{padding:13px}.startup-summary .value{font-size:1.35rem}.startup-stack{display:grid;gap:12px}.startup-message{border-left:3px solid var(--accent2);padding-left:10px}.startup-message strong,.startup-message .meta{display:block}.startup-message[data-level="warn"],.startup-message[data-level="warning"]{border-color:var(--warn)}.startup-message[data-level="error"],.startup-message[data-level="failed"]{border-color:var(--danger)}@media(max-width:900px){.startup-grid{grid-template-columns:1fr}}@media(max-width:600px){.startup-summary{grid-template-columns:1fr}}
.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}.session-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}.session-card{min-width:0;border:1px solid var(--line);border-radius:16px;padding:14px;background:var(--panel);outline:none}.session-card:focus{box-shadow:0 0 0 3px var(--accent-soft)}.session-title{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.provider,.session-state{font-size:12px;border:1px solid var(--line);border-radius:999px;padding:3px 7px}.session-state{font-weight:700}.state-awaiting-approval,.state-awaiting-input{color:var(--warn)}.state-failed,.state-disconnected{color:var(--bad)}.state-working{color:var(--good)}.session-card dl{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:12px 0}.session-card dl div{min-width:0}.session-card dt{font-size:11px;color:var(--muted);text-transform:uppercase}.session-card dd{margin:2px 0;overflow-wrap:anywhere}.pending{padding:8px;border-left:3px solid var(--warn);background:var(--warn-soft)}.session-approval{margin:10px 0;padding:10px;border:1px solid var(--warn);border-radius:10px;background:rgba(246,196,83,0.06)}.session-approval-controls{display:flex;gap:8px;margin-top:10px}.button-approval-accept{background:var(--accent);color:#062116;border:1px solid var(--accent);border-radius:8px;padding:6px 14px;font-weight:700;cursor:pointer}.button-approval-decline{background:transparent;color:var(--danger);border:1px solid var(--danger);border-radius:8px;padding:6px 14px;font-weight:700;cursor:pointer}.approval-details{display:grid;gap:6px;margin:8px 0;font-size:12px}.approval-details dt{color:var(--muted);font-size:10px;text-transform:uppercase}.approval-details dd{margin:0}.session-controls{display:flex;gap:6px;flex-wrap:wrap}.session-controls button[disabled]{opacity:.55;cursor:not-allowed}@media(max-width:560px){.session-grid{grid-template-columns:minmax(0,1fr)}.session-title{align-items:flex-start}.session-state{margin-left:0}.session-card dl{grid-template-columns:minmax(0,1fr)}.session-controls{display:grid;grid-template-columns:repeat(2,minmax(0,1fr))}.session-controls button{width:100%;min-height:44px}}
</style></head><body><main><div class="topbar"><div class="brand"><span class="mark">f</span><p class="eyebrow">fak operator home</p></div><p class="mode"><span class="dot"></span><span id="status">loading local state…</span></p></div><section class="shell"><header class="top"><h1>Harness overview</h1><p class="sub">Agents, goals, and live operating surfaces in one place. Start a run after you know what is active.</p><div class="actions"><a id="gateway-link" class="button primary" aria-disabled="true">Web gateway</a><button id="refresh" class="button" type="button">Refresh status</button><button id="skin" class="skin" type="button">Theme</button></div></header><nav class="nav" aria-label="Harness pages"><a href="#overview">Overview</a><a href="#gateway-startup">Gateway startup</a><a href="#agents">Agent stats</a><a href="#goals">Goals</a><a href="#dashboards">Live dashboards</a><a href="#sessions">Sessions</a><a href="#run">Run agent</a></nav><div class="content">
<section id="overview" class="section"><div class="sectionhead"><h2>Overview</h2><p id="overview-note">reading current state</p></div><div id="stats" class="stats" aria-live="polite"><p class="empty">Loading operational totals…</p></div><div id="local-work" class="rows" aria-label="Active local work"></div></section>
<section id="gateway-startup" class="section"><div class="sectionhead"><h2>Gateway startup</h2><p id="startup-note">reading structured /debug/vars.startup</p></div><div id="startup-content" class="startup-grid" aria-live="polite"><p class="empty">Loading startup state...</p></div></section>
<section class="section"><div class="twocol"><div id="agents" class="panel"><h3>Agent stats</h3><div id="agent-rows" class="rows"><p class="empty">Loading agents…</p></div></div><div id="goals" class="panel"><h3>Goals</h3><div id="goal-rows" class="rows"><p class="empty">Loading goals…</p></div></div></div></section>
<section id="dashboards" class="section"><div class="sectionhead"><h2>Live dashboards</h2><p id="dashboard-note">Gateway dashboards connect automatically when fak serve is running.</p></div><div id="dashboard-links" class="dashboards"></div></section>
<section id="sessions" class="section"><div class="sectionhead"><h2>Codex sessions</h2><p>Authoritative state and capability-gated controls. Use arrow keys between cards.</p></div><div id="session-cards" class="session-grid" role="list" aria-label="Authoritative Codex sessions" aria-live="polite" aria-busy="true"><p class="empty">Loading sessions�</p></div><p id="session-feedback" class="muted" role="status"></p></section>
<section id="run" class="section"><div class="sectionhead"><h2>Run agent</h2><p>Live through the configured gateway; deterministic offline otherwise.</p></div><form id="prompt" class="composer"><input id="text" aria-label="Message" value="show the native harness works"><button>Run</button></form><nav class="examples" aria-label="Proof scenarios"><button data-example="normal">Tool run</button><button data-example="approval">Approval run</button><button data-example="failure">Failure run</button></nav><section id="events" class="events" aria-live="polite"><p class="empty">Choose a goal or start a run. Semantic events appear here.</p></section></section>
</div></section></main><script>
const query=new URLSearchParams(location.search);let run=query.get("run")||"",after=0,skin=query.get("skin")||"forest";document.body.dataset.skin=skin;const list=document.querySelector("#events"),status=document.querySelector("#status"),text=document.querySelector("#text");
const esc=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
function render(e){const p=e.payload||{},d=document.createElement("article");d.className="event";d.dataset.kind=e.type;const detail=p.text||p.summary||p.name||p.message||p.status||e.type;d.innerHTML='<div class="kind">'+esc(e.type)+' - '+e.sequence+'</div><div>'+esc(detail)+'</div>';if(e.type==="approval.requested"){const meta=document.createElement("dl");meta.className="approval-meta";[["Kind",p.kind],["Workspace",p.workspace],["Scope",p.scope],["Risk / authority",p.risk],["Consequence",p.consequence],["fak capability floor",p.fak_capability_floor],["Codex sandbox / approval policy",p.codex_sandbox_policy],["Reason",p.policy_reason]].forEach(([k,v])=>{if(v)meta.innerHTML+='<dt>'+esc(k)+'</dt><dd>'+esc(v)+'</dd>'});d.append(meta);if(p.status==="pending"&&p.fak_capability_floor!=="deny"){const a=document.createElement("div");a.className="approval";a.innerHTML='<button data-decision="approve">Approve once</button><button data-decision="deny">Deny</button>';a.addEventListener("click",async x=>{const decision=x.target.dataset.decision;if(!decision)return;await fetch("/api/approvals",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({run_id:run,approval_id:p.approval_id,decision})});await pull()});d.append(a)}}list.append(d);after=e.sequence}
function empty(message){const p=document.createElement("p");p.className="empty";p.textContent=message;return p}
function row(title,meta,tag){const d=document.createElement("div");d.className="row";const copy=document.createElement("div"),strong=document.createElement("strong"),small=document.createElement("span"),badge=document.createElement("span");strong.textContent=title;small.className="meta";small.textContent=meta;badge.className="tag";badge.textContent=tag;copy.append(strong,small);d.append(copy,badge);return d}
function stat(value,label,detail){const d=document.createElement("div");d.className="stat";d.innerHTML='<span class="value">'+esc(value)+'</span><span class="label">'+esc(label)+'</span><span class="detail">'+esc(detail)+'</span>';return d}
const seconds=n=>(Number(n)||0).toFixed(3)+" s";
const bytes=n=>{n=Number(n)||0;if(n>=1073741824)return(n/1073741824).toFixed(2)+" GiB";if(n>=1048576)return(n/1048576).toFixed(1)+" MiB";if(n>=1024)return(n/1024).toFixed(1)+" KiB";return n+" B"};
function startupPanel(title){const d=document.createElement("div"),h=document.createElement("h3");d.className="panel";h.textContent=title;d.append(h);return d}
function renderStartup(gateway){const target=document.querySelector("#startup-content"),note=document.querySelector("#startup-note"),startup=gateway.startup;target.replaceChildren();if(!gateway.reachable){note.textContent="gateway unavailable";target.append(empty("Connect fak serve to read structured startup state."));return}if(!startup){note.textContent="legacy gateway fallback";target.append(empty("Structured startup state is unavailable on this older gateway. Open Diagnostics for its legacy startup report."));return}const statusPanel=startupPanel("Readiness"),summary=document.createElement("div");summary.className="startup-summary";summary.append(stat(startup.status||"unknown","Startup status",startup.ready_at?"ready at "+startup.ready_at:(startup.started_at?"started at "+startup.started_at:"timestamp unavailable")),stat(seconds(startup.time_to_ready_seconds),"Time to ready",seconds(startup.unaccounted_seconds)+" unaccounted"));statusPanel.append(summary);target.append(statusPanel);const startupState=startup.status||"unknown";note.textContent=startupState==="ready"?"ready in "+seconds(startup.time_to_ready_seconds):startupState==="starting"?"starting; readiness pending":startupState+"; inspect typed startup messages";const stack=document.createElement("div");stack.className="startup-stack";const phases=startupPanel("Startup phases"),phaseRows=document.createElement("div");phaseRows.className="rows";for(const phase of startup.phases||[])phaseRows.append(row(phase.name||"unnamed phase",seconds(phase.seconds)+" | "+(phase.provenance||"unknown provenance"),phase.stage||"gateway-boot"));if(!phaseRows.children.length)phaseRows.append(empty("No phase timings reported."));phases.append(phaseRows);stack.append(phases);if(startup.model_load){const model=startup.model_load,modelPanel=startupPanel("Model load"),modelRows=document.createElement("div");modelRows.className="rows";modelRows.append(row((model.source||"unknown source")+" | "+(model.mode||"unknown mode"),seconds(model.total_seconds)+" | "+bytes(model.bytes)+" | "+(model.tensors||0)+" tensors",model.bottleneck||"no bottleneck"));for(const path of model.load_paths||[])modelRows.append(row((path.quant_type||"unknown quant")+" | "+(path.class||"unknown class"),(path.resident_tensors||0)+" resident / "+(path.dequant_tensors||0)+" dequant tensors | "+bytes(path.resident_bytes||0)+" resident",path.dequant_bytes?bytes(path.dequant_bytes)+" dequant":"resident"));modelPanel.append(modelRows);stack.append(modelPanel)}const messages=startupPanel("Startup messages"),messageRows=document.createElement("div");messageRows.className="rows";for(const message of startup.messages||[]){const d=document.createElement("div"),kind=document.createElement("strong"),copy=document.createElement("span");d.className="startup-message";d.dataset.level=message.level||"info";kind.textContent=(message.source||"gateway")+" / "+(message.kind||"note")+" / "+(message.level||"info");copy.className="meta";copy.textContent=message.text||"(empty message)";d.append(kind,copy);messageRows.append(d)}if(!messageRows.children.length)messageRows.append(empty("No retained startup messages."));messages.append(messageRows);stack.append(messages);target.append(stack)}function renderOverview(data){const agents=data.agents||{},goals=data.goals||{},work=data.work||{},gateway=data.gateway||{},workspace=data.workspace||{};status.textContent=data.mode==="live"?"live - gateway connected":"offline - gateway unavailable";document.querySelector("#overview-note").textContent=gateway.reachable?"live gateway and local registries":"local registries · gateway unavailable; start fak serve and this page reconnects automatically";const stats=document.querySelector("#stats");stats.replaceChildren(stat((agents.live_sessions||[]).length,"Live agents",gateway.reachable?"running through the gateway":"gateway not connected"),stat(agents.live_runs||0,"Live stored runs","gateway-backed operational runs"),stat(agents.offline_demo_runs||0,"Offline proof runs","deterministic demo evidence; not live activity"),stat(goals.active||0,"Active goals",(goals.blocked||0)+" blocked · "+(goals.paused||0)+" paused"),stat(work.active||0,"Active local work",(work.released||0)+" released"),stat(gateway.fleet_sessions||0,"Fleet sessions",(gateway.fleet_machines||0)+" machines visible"));const agentRows=document.querySelector("#agent-rows");agentRows.replaceChildren();for(const a of agents.live_sessions||[])agentRows.append(row(a.trace_id||"unnamed agent",(a.last_tool?"last tool "+a.last_tool+" · ":"")+(a.turns_left||0)+" turns left",a.run||"live"));for(const a of agents.recent_runs||[])agentRows.append(row(a.run_id,(a.source||"legacy-unknown")+" | "+a.events+" events | last "+(a.last_event||"unknown"),a.status));if(!agentRows.children.length)agentRows.append(empty("No agent sessions or stored runs yet."));const goalRows=document.querySelector("#goal-rows");goalRows.replaceChildren();if(!goals.readable)goalRows.append(empty("Goal registry unavailable."));else for(const g of goals.items||[])goalRows.append(row(g.title,g.goal_id,g.lifecycle));if(!goalRows.children.length)goalRows.append(empty("No registered goals. Create one with fak goal create."));const gatewayLink=document.querySelector("#gateway-link");if(gateway.url){gatewayLink.href=gateway.url;gatewayLink.target="_blank";gatewayLink.rel="noreferrer";gatewayLink.removeAttribute("aria-disabled")}else{gatewayLink.removeAttribute("href");gatewayLink.removeAttribute("target");gatewayLink.setAttribute("aria-disabled","true")}document.querySelector("#dashboard-note").textContent=gateway.reachable?"live from "+gateway.url:"Start fak serve; dashboards connect automatically to "+(gateway.target||"the configured gateway")+".";const links=document.querySelector("#dashboard-links");links.replaceChildren();for(const item of data.dashboards||[]){const d=document.createElement(item.url?"a":"div");d.className="dashboard"+(item.url?"":" disabled");if(item.url){d.href=item.url;d.target="_blank";d.rel="noreferrer"}const strong=document.createElement("strong"),desc=document.createElement("span"),path=document.createElement("span");strong.textContent=item.label;desc.textContent=item.description;path.className="path";path.textContent=item.path;d.append(strong,desc,path);links.append(d)}if(workspace.armed)document.querySelector("#overview-note").textContent+=" · workspace "+workspace.identity;renderStartup(gateway)}
function showLocalWork(data){const local=data.local_work||{},intents=local.issue_intents||{},leases=local.dos_leases||{},worktrees=local.worker_worktrees||{},target=document.querySelector("#local-work");target.replaceChildren();for(const issue of intents.ids||[])target.append(row(issue,"active local intent","intent"));for(const lane of leases.ids||[])target.append(row(lane,"active DOS lane lease","lease"));for(const item of worktrees.items||[]){const identity=item.issue_number?"#"+item.issue_number:(item.lane||item.session_id||"unknown association"),detail=item.state==="landed_witnessed"?"worker worktree: landed with independent commit witness":item.state==="cleanup_ready"?"worker worktree: cleanup ready, not complete":"worker worktree: "+String(item.state||"association_unknown").replaceAll("_"," ");target.append(row(identity,detail,item.complete?"complete":"evidence"))}if(!target.children.length)target.append(empty("No active local issue intents, DOS lane leases, or worker worktrees."));const states=worktrees.states||{},stats=document.querySelector("#stats");stats.append(stat(intents.active||0,"Local issue intents",intents.readable?"active repository claims":"intent state unavailable"),stat(leases.active||0,"DOS lane leases",leases.readable?"active local lanes":"DOS state unavailable"),stat(worktrees.total||0,"Worker worktrees",(states.landed_witnessed||0)+" landed witnessed; "+(states.cleanup_ready||0)+" cleanup ready (not complete)"))}
async function refreshOverview(){try{const r=await fetch("/api/status",{cache:"no-store"});if(!r.ok)throw new Error(r.status);const data=await r.json();renderOverview(data);showLocalWork(data)}catch(e){status.textContent="status unavailable"}}
async function pull(){const r=await fetch('/api/events?run_id='+encodeURIComponent(run)+'&after='+after);for(const e of await r.json())render(e);status.textContent='connected · cursor '+after}
async function start(message){list.replaceChildren();after=0;const r=await fetch("/api/runs",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({message})});run=(await r.json()).run_id;history.replaceState(null,"","?run="+encodeURIComponent(run));await pull();await refreshOverview()}
document.querySelector("#prompt").addEventListener("submit",async e=>{e.preventDefault();await start(text.value)});document.querySelector(".examples").addEventListener("click",async e=>{const x=e.target.dataset.example;if(!x)return;text.value=x==="approval"?"approval: inspect workspace":x==="failure"?"failure: demonstrate typed error":"show the native harness works";await start(text.value)});document.querySelector("#skin").addEventListener("click",()=>{skin=skin==="forest"?"minimal":"forest";document.body.dataset.skin=skin});const scenario=new URLSearchParams(location.search).get("scenario");if(scenario){text.value=scenario==="approval"?"approval: inspect workspace":scenario==="failure"?"failure: demonstrate typed error":"show the native harness works";start(text.value)}
async function loadSessions(focusID=""){sessionCards.setAttribute("aria-busy","true");try{const r=await fetch("/api/sessions"+(query.get("no_color")==="1"?"?no_color=1":""));const d=await r.json();if(!r.ok)throw new Error(d.error||"session load failed");sessionCards.innerHTML=d.html;bindSessionCards();if(focusID){const card=[...sessionCards.querySelectorAll(".session-card")].find(item=>item.dataset.sessionId===focusID);if(card){sessionCards.querySelectorAll(".session-card").forEach(item=>item.tabIndex=-1);card.tabIndex=0;card.focus()}}}catch(e){sessionCards.innerHTML='<p class="empty">Session state unavailable.</p>';sessionFeedback.textContent=e.message}finally{sessionCards.setAttribute("aria-busy","false")}}
function bindSessionCards(){const cards=[...sessionCards.querySelectorAll(".session-card")];const moveFocus=index=>{cards.forEach((card,i)=>card.tabIndex=i===index?0:-1);cards[index]?.focus()};cards.forEach((card,index)=>card.addEventListener("keydown",e=>{let next;if(["ArrowDown","ArrowRight"].includes(e.key))next=(index+1)%cards.length;else if(["ArrowUp","ArrowLeft"].includes(e.key))next=(index-1+cards.length)%cards.length;else if(e.key==="Home")next=0;else if(e.key==="End")next=cards.length-1;else return;e.preventDefault();moveFocus(next)}));sessionCards.querySelectorAll("[data-session-action]").forEach(button=>button.addEventListener("click",async()=>{const id=button.dataset.sessionId,action=button.dataset.sessionAction;button.disabled=true;sessionFeedback.textContent=action+" requested for "+id;try{const r=await fetch("/api/sessions/"+encodeURIComponent(id)+"/controls/"+action,{method:"POST"});const d=await r.json();if(!r.ok)throw new Error(d.error||"control failed");sessionFeedback.textContent=action+" sent to "+id;await loadSessions(id)}catch(e){sessionFeedback.textContent=e.message;button.disabled=false}}));sessionCards.querySelectorAll("[data-approval-action]").forEach(button=>button.addEventListener("click",async e=>{e.preventDefault();const id=button.dataset.sessionId,action=button.dataset.approvalAction,approvalId=button.dataset.approvalId||"";button.disabled=true;sessionFeedback.textContent="Approval "+action+" requested for "+id;try{const r=await fetch("/api/sessions/"+encodeURIComponent(id)+"/approval",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({resolution:action,approval_id:approvalId,reason:"web operator "+action})});const d=await r.json();if(!r.ok)throw new Error(d.error||"approval resolution failed");sessionFeedback.textContent="Approval "+action+" sent to "+id;await loadSessions(id)}catch(err){sessionFeedback.textContent=err.message;button.disabled=false}}));sessionCards.querySelectorAll(".approval-form").forEach(form=>form.addEventListener("submit",async e=>{e.preventDefault();const submitter=e.submitter;const action=submitter?.dataset?.approvalAction||submitter?.value||"accept";const id=form.dataset.sessionId;const approvalId=form.dataset.approvalId||form.querySelector('input[name="approval_id"]')?.value||"";sessionFeedback.textContent="Approval "+action+" requested for "+id;try{const r=await fetch("/api/sessions/"+encodeURIComponent(id)+"/approval",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({resolution:action,approval_id:approvalId,reason:"web operator "+action})});const d=await r.json();if(!r.ok)throw new Error(d.error||"approval resolution failed");sessionFeedback.textContent="Approval "+action+" sent to "+id;await loadSessions(id)}catch(err){sessionFeedback.textContent=err.message}}))}
function subscribeSessionEvents(){if(!window.EventSource)return;try{const es=new EventSource("/api/sessions/events");es.addEventListener("session_update",e=>{if(e.data){try{const p=JSON.parse(e.data);if(p.html){sessionCards.innerHTML=p.html;bindSessionCards();return}}catch(_){}}loadSessions()});es.addEventListener("approval_requested",e=>{if(e.data){try{const p=JSON.parse(e.data);sessionFeedback.textContent="Approval requested for "+(p.session_id||"session")}catch(_){}}loadSessions()});es.addEventListener("approval_resolved",e=>{if(e.data){try{const p=JSON.parse(e.data);sessionFeedback.textContent="Approval "+(p.resolution||"resolved")+" for "+(p.session_id||"session")}catch(_){}}loadSessions()});es.onmessage=()=>{loadSessions()}}catch(_){}}
subscribeSessionEvents();document.querySelector("#refresh").addEventListener("click",()=>{refreshOverview();loadSessions()});refreshOverview();loadSessions();if(run)pull();setInterval(refreshOverview,5000);setInterval(loadSessions,5000);
</script></body></html>`

type runState struct {
	events   []harnesskit.Envelope
	approval string
	resolved bool
}

type store struct {
	mu      sync.RWMutex
	runs    map[string]*runState
	next    uint64
	persist string
}

func newStore() *store { return &store{runs: make(map[string]*runState)} }

func newPersistentStore(path string) (*store, error) {
	s := &store{runs: make(map[string]*runState), persist: path}
	body, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return s, nil
		}
		return nil, err
	}
	var disk struct {
		Runs map[string][]harnesskit.Envelope `json:"runs"`
	}
	if err := json.Unmarshal(body, &disk); err != nil {
		return nil, fmt.Errorf("decode session store: %w", err)
	}
	for id, events := range disk.Runs {
		s.runs[id] = &runState{events: events}
		if strings.HasPrefix(id, "local-") {
			var n uint64
			_, _ = fmt.Sscanf(id, "local-%d", &n)
			if n > s.next {
				s.next = n
			}
		}
	}
	return s, nil
}

func (s *store) saveLocked() error {
	if s.persist == "" {
		return nil
	}
	disk := struct {
		Runs map[string][]harnesskit.Envelope `json:"runs"`
	}{Runs: make(map[string][]harnesskit.Envelope, len(s.runs))}
	for id, state := range s.runs {
		disk.Runs[id] = state.events
	}
	body, err := json.MarshalIndent(disk, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(s.persist), 0o755); err != nil {
		return err
	}
	tmp := s.persist + ".tmp"
	if err := os.WriteFile(tmp, body, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, s.persist)
}

func (s *store) nextRunID(prefix string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.next++
	return fmt.Sprintf("%s-%d", prefix, s.next)
}

func (s *store) create(message string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.next++
	runID := fmt.Sprintf("local-%d", s.next)
	state := &runState{}
	state.events = append(state.events, event(runID, 1, harnesskit.EventRunStarted, harnesskit.RunPayload{Status: "running"}))
	switch {
	case strings.HasPrefix(strings.ToLower(message), "approval:"):
		state.approval = "approval-1"
		state.events = append(state.events,
			event(runID, 2, harnesskit.EventMessageCompleted, harnesskit.MessagePayload{MessageID: "message-2", Role: "assistant", Text: "Workspace inspection needs approval."}),
			event(runID, 3, harnesskit.EventApprovalRequested, harnesskit.ApprovalPayload{ApprovalID: state.approval, Prompt: "Allow read-only workspace inspection?", Status: "pending", Scope: "workspace.read"}),
		)
	case strings.HasPrefix(strings.ToLower(message), "failure:"):
		state.events = append(state.events,
			event(runID, 2, harnesskit.EventError, harnesskit.ErrorPayload{Code: "OFFLINE_DEMO_FAILURE", Message: "Typed offline failure; no action was taken.", Retryable: true}),
			event(runID, 3, harnesskit.EventRunCompleted, harnesskit.RunPayload{Status: "failed", Reason: "offline witness"}),
		)
	default:
		state.events = append(state.events,
			event(runID, 2, harnesskit.EventMessageStarted, harnesskit.MessagePayload{MessageID: "message-2", Role: "assistant"}),
			event(runID, 3, harnesskit.EventMessageDelta, harnesskit.MessagePayload{MessageID: "message-2", Text: "offline reply: "}),
			event(runID, 4, harnesskit.EventMessageCompleted, harnesskit.MessagePayload{MessageID: "message-2", Role: "assistant", Text: "offline reply: " + message}),
			event(runID, 5, harnesskit.EventToolStarted, harnesskit.ToolPayload{CallID: "selfcheck-tool", Name: "record_selfcheck", Status: "running"}),
			event(runID, 6, harnesskit.EventToolCompleted, harnesskit.ToolPayload{CallID: "selfcheck-tool", Name: "record_selfcheck", Status: "completed", Summary: "record_selfcheck: ok"}),
			event(runID, 7, harnesskit.EventArtifactPublished, harnesskit.ArtifactPayload{ArtifactID: "receipt-1", MediaType: "text/plain", URI: "memory://selfcheck-receipt", Name: "Offline receipt"}),
			event(runID, 8, harnesskit.EventRunCompleted, harnesskit.RunPayload{Status: "completed"}),
		)
	}
	s.runs[runID] = state
	_ = s.saveLocked()
	return runID
}

func event(runID string, seq uint64, kind harnesskit.EventType, payload any) harnesskit.Envelope {
	body, _ := json.Marshal(payload)
	return harnesskit.Envelope{Version: harnesskit.ProtocolVersion, RunID: runID, Sequence: seq, EventID: fmt.Sprintf("%s-%d", runID, seq), Type: kind, Sensitivity: harnesskit.SensitivityPublic, Payload: body}
}

func (s *store) after(runID string, after uint64) []harnesskit.Envelope {
	s.mu.RLock()
	defer s.mu.RUnlock()
	state := s.runs[runID]
	if state == nil {
		return []harnesskit.Envelope{}
	}
	out := make([]harnesskit.Envelope, 0, len(state.events))
	for _, e := range state.events {
		if e.Sequence > after {
			out = append(out, e)
		}
	}
	return out
}

func (s *store) replace(runID string, events []harnesskit.Envelope) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.runs[runID] = &runState{events: append([]harnesskit.Envelope(nil), events...)}
	return s.saveLocked()
}

type runSummary struct {
	RunID     string `json:"run_id"`
	Source    string `json:"source"`
	Status    string `json:"status"`
	Events    int    `json:"events"`
	LastEvent string `json:"last_event,omitempty"`
}

type liveSessionSummary struct {
	TraceID   string `json:"trace_id"`
	Run       string `json:"run"`
	TurnsLeft int    `json:"turns_left"`
	LastTool  string `json:"last_tool,omitempty"`
}

type statusPayload struct {
	Mode       string            `json:"mode"`
	Workspace  workspaceStatus   `json:"workspace"`
	Gateway    gatewayOverview   `json:"gateway"`
	Agents     agentOverview     `json:"agents"`
	Goals      goalOverview      `json:"goals"`
	LocalWork  localWorkOverview `json:"local_work"`
	Dashboards []dashboardLink   `json:"dashboards"`
}

type agentOverview struct {
	TotalRuns        int                  `json:"total_runs"`
	LiveRuns         int                  `json:"live_runs"`
	OfflineDemoRuns  int                  `json:"offline_demo_runs"`
	Running          int                  `json:"running"`
	Completed        int                  `json:"completed"`
	Failed           int                  `json:"failed"`
	AwaitingApproval int                  `json:"awaiting_approval"`
	LiveSessions     []liveSessionSummary `json:"live_sessions"`
	RecentRuns       []runSummary         `json:"recent_runs"`
}

func (s *store) overview(live []liveSessionSummary) agentOverview {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := agentOverview{TotalRuns: len(s.runs), LiveSessions: live, RecentRuns: make([]runSummary, 0, len(s.runs))}
	for id, state := range s.runs {
		summary := summarizeRun(id, state.events)
		switch summary.Source {
		case "live":
			out.LiveRuns++
		case "offline-demo":
			out.OfflineDemoRuns++
		}
		switch summary.Status {
		case "completed":
			out.Completed++
		case "failed":
			out.Failed++
		case "awaiting approval":
			out.AwaitingApproval++
		default:
			out.Running++
		}
		out.RecentRuns = append(out.RecentRuns, summary)
	}
	sort.Slice(out.RecentRuns, func(i, j int) bool {
		iRun, jRun := runSerial(out.RecentRuns[i].RunID), runSerial(out.RecentRuns[j].RunID)
		if iRun != jRun {
			return iRun > jRun
		}
		return out.RecentRuns[i].RunID > out.RecentRuns[j].RunID
	})
	if len(out.RecentRuns) > 8 {
		out.RecentRuns = out.RecentRuns[:8]
	}
	return out
}

func runSource(id string) string {
	switch {
	case strings.HasPrefix(id, "live-"):
		return "live"
	case strings.HasPrefix(id, "local-"):
		return "offline-demo"
	default:
		return "legacy-unknown"
	}
}

func summarizeRun(id string, events []harnesskit.Envelope) runSummary {
	summary := runSummary{RunID: id, Source: runSource(id), Status: "running", Events: len(events)}
	var requested, resolved uint64
	for _, e := range events {
		summary.LastEvent = string(e.Type)
		switch e.Type {
		case harnesskit.EventApprovalRequested:
			requested = e.Sequence
		case harnesskit.EventApprovalResolved:
			resolved = e.Sequence
		case harnesskit.EventRunCompleted:
			var payload harnesskit.RunPayload
			if e.DecodePayload(&payload) == nil && strings.TrimSpace(payload.Status) != "" {
				summary.Status = strings.TrimSpace(payload.Status)
			}
		}
	}
	if requested > resolved {
		summary.Status = "awaiting approval"
	}
	return summary
}

func runSerial(id string) uint64 {
	idx := strings.LastIndexByte(id, '-')
	if idx < 0 || idx == len(id)-1 {
		return 0
	}
	n, _ := strconv.ParseUint(id[idx+1:], 10, 64)
	return n
}

type goalLister interface {
	List() ([]goalregistry.Goal, error)
}

type goalSummary struct {
	GoalID    string                 `json:"goal_id"`
	Title     string                 `json:"title"`
	Lifecycle goalregistry.Lifecycle `json:"lifecycle"`
	UpdatedAt time.Time              `json:"updated_at"`
}

type goalOverview struct {
	Readable bool          `json:"readable"`
	Total    int           `json:"total"`
	Active   int           `json:"active"`
	Blocked  int           `json:"blocked"`
	Paused   int           `json:"paused"`
	Achieved int           `json:"achieved"`
	Items    []goalSummary `json:"items"`
}

func readGoalOverview(source goalLister) goalOverview {
	if source == nil {
		return goalOverview{Items: []goalSummary{}}
	}
	goals, err := source.List()
	if err != nil {
		return goalOverview{Items: []goalSummary{}}
	}
	out := goalOverview{Readable: true, Total: len(goals), Items: make([]goalSummary, 0, len(goals))}
	for _, goal := range goals {
		switch goal.Lifecycle {
		case goalregistry.Active:
			out.Active++
		case goalregistry.Blocked:
			out.Blocked++
		case goalregistry.Paused:
			out.Paused++
		case goalregistry.Achieved:
			out.Achieved++
		}
		out.Items = append(out.Items, goalSummary{GoalID: goal.GoalID, Title: goal.Title, Lifecycle: goal.Lifecycle, UpdatedAt: goal.UpdatedAt})
	}
	sort.Slice(out.Items, func(i, j int) bool { return out.Items[i].UpdatedAt.After(out.Items[j].UpdatedAt) })
	return out
}

type dashboardLink struct {
	Label       string `json:"label"`
	Description string `json:"description"`
	Path        string `json:"path"`
	URL         string `json:"url,omitempty"`
}

type gatewayOverview struct {
	Configured    bool                    `json:"configured"`
	Reachable     bool                    `json:"reachable"`
	URL           string                  `json:"url,omitempty"`
	Target        string                  `json:"target,omitempty"`
	FleetMachines int                     `json:"fleet_machines"`
	FleetSessions int                     `json:"fleet_sessions"`
	Startup       *gatewayStartupOverview `json:"startup,omitempty"`
}

type gatewayStartupOverview struct {
	Status             string                   `json:"status"`
	StartedAt          string                   `json:"started_at,omitempty"`
	ReadyAt            string                   `json:"ready_at,omitempty"`
	TimeToReadySeconds float64                  `json:"time_to_ready_seconds"`
	UnaccountedSeconds float64                  `json:"unaccounted_seconds"`
	Phases             []gatewayStartupPhase    `json:"phases,omitempty"`
	Messages           []gatewayStartupMessage  `json:"messages,omitempty"`
	ModelLoad          *gatewayStartupModelLoad `json:"model_load,omitempty"`
}

type gatewayStartupPhase struct {
	Name       string  `json:"name"`
	Seconds    float64 `json:"seconds"`
	Provenance string  `json:"provenance"`
	Stage      string  `json:"stage"`
}

type gatewayStartupMessage struct {
	Source string `json:"source"`
	Kind   string `json:"kind"`
	Level  string `json:"level"`
	Text   string `json:"text"`
}

type gatewayStartupModelLoad struct {
	Source       string                   `json:"source"`
	Mode         string                   `json:"mode"`
	TotalSeconds float64                  `json:"total_seconds"`
	Bytes        int64                    `json:"bytes"`
	Tensors      int                      `json:"tensors"`
	Bottleneck   string                   `json:"bottleneck"`
	LoadPaths    []gatewayStartupLoadPath `json:"load_paths,omitempty"`
}

type gatewayStartupLoadPath struct {
	QuantType       string `json:"quant_type"`
	Class           string `json:"class"`
	ResidentTensors int    `json:"resident_tensors"`
	ResidentBytes   int64  `json:"resident_bytes"`
	DequantTensors  int    `json:"dequant_tensors"`
	DequantBytes    int64  `json:"dequant_bytes"`
}

func dashboardLinks(base string) []dashboardLink {
	rows := []dashboardLink{
		{Label: "Web gateway", Description: "Gateway identity and API map.", Path: "/"},
		{Label: "Health", Description: "Readiness, engine state, and serving diagnostics.", Path: "/healthz"},
		{Label: "Agent sessions", Description: "Every live session and remaining drive budget.", Path: "/v1/fak/sessions"},
		{Label: "Background loops", Description: "Supervised loops and their current progress.", Path: "/v1/fak/loops"},
		{Label: "Fleet status", Description: "Cross-session running, blocked, and pressure totals.", Path: "/v1/fak/fleet"},
		{Label: "Task manager", Description: "Read-only process task accounting when enabled.", Path: "/v1/fak/tasks"},
		{Label: "Metrics", Description: "Prometheus counters for cache, latency, and requests.", Path: "/metrics"},
		{Label: "Diagnostics", Description: "Bounded runtime, agent, fleet, and resource state.", Path: "/debug/vars"},
	}
	for i := range rows {
		if base != "" {
			rows[i].URL = strings.TrimRight(base, "/") + rows[i].Path
		}
	}
	return rows
}

func publicGatewayURL(raw string) string {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return ""
	}
	u.User = nil
	u.RawQuery = ""
	u.Fragment = ""
	u.Path = strings.TrimRight(u.Path, "/")
	return strings.TrimRight(u.String(), "/")
}

type liveAdapter struct {
	baseURL         string
	offlineFallback bool
	client          *http.Client
	workspace       workspaceStatus
	identity        string
}

func (a *liveAdapter) reachable(ctx context.Context) bool {
	ctx, cancel := context.WithTimeout(ctx, 500*time.Millisecond)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(a.baseURL, "/")+"/healthz", nil)
	if err != nil {
		return false
	}
	client := a.client
	if client == nil {
		client = &http.Client{Timeout: 500 * time.Millisecond}
	}
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode >= 200 && resp.StatusCode < 300
}

type workspaceStatus struct {
	Armed    bool     `json:"armed"`
	Tools    []string `json:"tools,omitempty"`
	Identity string   `json:"identity,omitempty"`
}

func (a *liveAdapter) overview(ctx context.Context) (gatewayOverview, []liveSessionSummary) {
	base := publicGatewayURL(a.baseURL)
	out := gatewayOverview{Configured: true, Target: base}
	if base == "" {
		return out, []liveSessionSummary{}
	}
	probeCtx, cancel := context.WithTimeout(ctx, 1500*time.Millisecond)
	defer cancel()
	req, err := http.NewRequestWithContext(probeCtx, http.MethodGet, strings.TrimRight(a.baseURL, "/")+"/debug/vars", nil)
	if err != nil {
		return out, []liveSessionSummary{}
	}
	client := a.client
	if client == nil {
		client = &http.Client{Timeout: 1500 * time.Millisecond}
	}
	resp, err := client.Do(req)
	if err != nil {
		return out, []liveSessionSummary{}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return out, []liveSessionSummary{}
	}
	var body struct {
		Sessions []liveSessionSummary    `json:"sessions"`
		Startup  *gatewayStartupOverview `json:"startup"`
		Fleet    *struct {
			Machines int `json:"machines"`
			Sessions int `json:"sessions"`
		} `json:"fleet"`
	}
	if json.NewDecoder(io.LimitReader(resp.Body, 2<<20)).Decode(&body) != nil {
		return out, []liveSessionSummary{}
	}
	out.Reachable = true
	out.URL = base
	out.Startup = body.Startup
	if body.Fleet != nil {
		out.FleetMachines = body.Fleet.Machines
		out.FleetSessions = body.Fleet.Sessions
	}
	return out, body.Sessions
}

func (a *liveAdapter) probeWorkspace(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(a.baseURL, "/")+"/healthz", nil)
	if err != nil {
		return err
	}
	resp, err := a.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("fak health returned %s", resp.Status)
	}
	var body struct {
		NativeCodeWorkspace *workspaceStatus `json:"native_code_workspace"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&body); err != nil {
		return err
	}
	if body.NativeCodeWorkspace != nil {
		a.workspace = *body.NativeCodeWorkspace
		a.workspace.Identity = a.identity
	}
	return nil
}

func (a *liveAdapter) run(ctx context.Context, runID, message string) ([]harnesskit.Envelope, error) {
	requestBody, _ := json.Marshal(map[string]any{
		"model": "fak-native", "max_tokens": 512, "stream": true,
		"messages": []map[string]string{{"role": "user", "content": message}},
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(a.baseURL, "/")+"/v1/messages", bytes.NewReader(requestBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Trace-Id", runID)
	resp, err := a.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("fak returned %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}

	events := []harnesskit.Envelope{
		event(runID, 1, harnesskit.EventRunStarted, harnesskit.RunPayload{Status: "running"}),
		event(runID, 2, harnesskit.EventMessageStarted, harnesskit.MessagePayload{MessageID: "assistant", Role: "assistant"}),
	}
	seq := uint64(2)
	var text strings.Builder
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 4096), 1<<20)
	var eventName string
	var data strings.Builder
	emit := func() error {
		if data.Len() == 0 {
			eventName = ""
			return nil
		}
		var obj map[string]any
		if err := json.Unmarshal([]byte(data.String()), &obj); err != nil {
			return fmt.Errorf("decode fak SSE: %w", err)
		}
		kind := eventName
		if kind == "" {
			kind, _ = obj["type"].(string)
		}
		switch kind {
		case "content_block_delta":
			delta, _ := obj["delta"].(map[string]any)
			chunk, _ := delta["text"].(string)
			if chunk != "" {
				text.WriteString(chunk)
				seq++
				events = append(events, event(runID, seq, harnesskit.EventMessageDelta, harnesskit.MessagePayload{MessageID: "assistant", Text: chunk}))
			}
		case "tool_started":
			seq++
			events = append(events, event(runID, seq, harnesskit.EventToolStarted, harnesskit.ToolPayload{CallID: stringField(obj, "call_id"), Name: stringField(obj, "tool"), Status: "running"}))
		case "result_admitted":
			seq++
			tool := stringField(obj, "tool")
			summary := stringField(obj, "summary")
			if summary == "" {
				summary = "result admitted by fak"
			}
			events = append(events, event(runID, seq, harnesskit.EventToolCompleted, harnesskit.ToolPayload{CallID: stringField(obj, "call_id"), Name: tool, Status: "completed", Summary: summary}))
			if tool == "Edit" || tool == "Write" || (tool == "Bash" && strings.Contains(summary, "git diff")) {
				seq++
				events = append(events, event(runID, seq, harnesskit.EventArtifactPublished, harnesskit.ArtifactPayload{ArtifactID: fmt.Sprintf("artifact-%d", seq), MediaType: "text/x-diff", URI: "fak-native://" + runID + "/" + stringField(obj, "call_id"), Name: "Workspace patch"}))
			}
		case "call_adjudicated":
			if strings.EqualFold(stringField(obj, "verdict"), "DENY") {
				seq++
				events = append(events, event(runID, seq, harnesskit.EventError, harnesskit.ErrorPayload{Code: stringField(obj, "reason"), Message: "fak denied " + stringField(obj, "tool"), Retryable: false}))
			}
		}
		eventName = ""
		data.Reset()
		return nil
	}
	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case line == "":
			if err := emit(); err != nil {
				return nil, err
			}
		case strings.HasPrefix(line, "event:"):
			eventName = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		case strings.HasPrefix(line, "data:"):
			if data.Len() > 0 {
				data.WriteByte('\n')
			}
			data.WriteString(strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if err := emit(); err != nil {
		return nil, err
	}
	seq++
	events = append(events, event(runID, seq, harnesskit.EventMessageCompleted, harnesskit.MessagePayload{MessageID: "assistant", Role: "assistant", Text: text.String()}))
	seq++
	events = append(events, event(runID, seq, harnesskit.EventRunCompleted, harnesskit.RunPayload{Status: "completed"}))
	return events, nil
}

func stringField(m map[string]any, key string) string { value, _ := m[key].(string); return value }

func (s *store) resolve(runID, approvalID, decision string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	state := s.runs[runID]
	if state == nil {
		return harnesskit.ErrInvalidProtocol
	}
	if approvalID == "" {
		approvalID = state.approval
	}
	if state.approval != approvalID || state.resolved || (decision != "approve" && decision != "deny") {
		return harnesskit.ErrInvalidProtocol
	}
	state.resolved = true
	seq := uint64(len(state.events) + 1)
	state.events = append(state.events, event(runID, seq, harnesskit.EventApprovalResolved, harnesskit.ApprovalPayload{ApprovalID: approvalID, Status: decision, Scope: "workspace.read"}))
	seq++
	text := "Approval denied; no tool ran."
	if decision == "approve" {
		text = "Approved read-only inspection completed."
		state.events = append(state.events, event(runID, seq, harnesskit.EventToolCompleted, harnesskit.ToolPayload{CallID: "approved-tool", Name: "inspect_workspace", Status: "completed", Summary: "read-only inspection: ok"}))
		seq++
	}
	state.events = append(state.events,
		event(runID, seq, harnesskit.EventMessageCompleted, harnesskit.MessagePayload{MessageID: fmt.Sprintf("message-%d", seq), Role: "assistant", Text: text}),
		event(runID, seq+1, harnesskit.EventRunCompleted, harnesskit.RunPayload{Status: "completed"}),
	)
	return s.saveLocked()
}

func (s *store) hasRun(runID string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.runs[runID] != nil
}

func handler(s *store) http.Handler { return handlerWithSources(s, nil, nil) }

func handlerWithLive(s *store, live *liveAdapter) http.Handler {
	return handlerWithSources(s, live, nil)
}

func handlerWithSources(s *store, live *liveAdapter, goals goalLister) http.Handler {
	return handlerWithAllSources(s, live, goals, nil, nil, "")
}

func handlerWithSessionSource(s *store, live *liveAdapter, goals goalLister, sessions SessionSource) http.Handler {
	return handlerWithAllSources(s, live, goals, sessions, nil, "")
}

func handlerWithAllSources(s *store, live *liveAdapter, goals goalLister, sessions SessionSource, local LocalWorkSource, root string) http.Handler {
	mux := http.NewServeMux()
	installSessionRoutesWithStore(mux, sessions, s)
	mux.HandleFunc("GET /", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
		_, _ = io.WriteString(w, page)
	})
	mux.HandleFunc("GET /api/status", func(w http.ResponseWriter, r *http.Request) {
		mode := "offline"
		workspace := workspaceStatus{}
		gateway := gatewayOverview{}
		liveSessions := []liveSessionSummary{}
		if live != nil {
			workspace = live.workspace
			gateway, liveSessions = live.overview(r.Context())
			if gateway.Reachable {
				mode = "live"
			}
		}
		w.Header().Set("Cache-Control", "no-store")
		agents := s.overview(liveSessions)
		writeJSON(w, statusPayload{
			Mode: mode, Workspace: workspace, Gateway: gateway,
			Agents: agents, Goals: readGoalOverview(goals),
			LocalWork:  readLocalWorkOverview(r.Context(), local, root, time.Now()),
			Dashboards: dashboardLinks(gateway.URL),
		})
	})
	mux.HandleFunc("POST /api/runs", func(w http.ResponseWriter, r *http.Request) {
		var input struct {
			Message string `json:"message"`
		}
		if err := json.NewDecoder(r.Body).Decode(&input); err != nil || strings.TrimSpace(input.Message) == "" {
			http.Error(w, "message is required", http.StatusBadRequest)
			return
		}
		message := strings.TrimSpace(input.Message)
		if live != nil && (!live.offlineFallback || live.reachable(r.Context())) && !strings.HasPrefix(strings.ToLower(message), "approval:") && !strings.HasPrefix(strings.ToLower(message), "failure:") {
			runID := s.nextRunID("live")
			events, err := live.run(r.Context(), runID, message)
			if err != nil {
				events = []harnesskit.Envelope{event(runID, 1, harnesskit.EventRunStarted, harnesskit.RunPayload{Status: "running"}), event(runID, 2, harnesskit.EventError, harnesskit.ErrorPayload{Code: "LIVE_FAK_ERROR", Message: err.Error(), Retryable: true}), event(runID, 3, harnesskit.EventRunCompleted, harnesskit.RunPayload{Status: "failed", Reason: "live fak error"})}
			}
			if err := s.replace(runID, events); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			writeJSON(w, map[string]string{"run_id": runID})
			return
		}
		writeJSON(w, map[string]string{"run_id": s.create(message)})
	})
	mux.HandleFunc("GET /api/events", func(w http.ResponseWriter, r *http.Request) {
		after, err := strconv.ParseUint(r.URL.Query().Get("after"), 10, 64)
		if err != nil && r.URL.Query().Get("after") != "" {
			http.Error(w, "invalid cursor", http.StatusBadRequest)
			return
		}
		writeJSON(w, s.after(r.URL.Query().Get("run_id"), after))
	})
	mux.HandleFunc("POST /api/approvals", func(w http.ResponseWriter, r *http.Request) {
		var input struct {
			RunID      string `json:"run_id"`
			ApprovalID string `json:"approval_id"`
			Decision   string `json:"decision"`
		}
		if json.NewDecoder(r.Body).Decode(&input) != nil || s.resolve(input.RunID, input.ApprovalID, input.Decision) != nil {
			http.Error(w, "invalid approval", http.StatusBadRequest)
			return
		}
		defaultSessionHub.broadcast("approval_resolved", []byte(fmt.Sprintf(`{"session_id":%q,"approval_id":%q,"resolution":%q}`, input.RunID, input.ApprovalID, input.Decision)))
		writeJSON(w, map[string]string{"status": "accepted"})
	})
	return mux
}

// SessionApprovalDirectResolver resolves approvals directly via sessionID, resolution ("accept" or "decline"), and reason.
type SessionApprovalDirectResolver interface {
	ResolveApproval(sessionID string, resolution string, reason string) error
}

// HandleSessionApproval exposes the HTTP handler for POST /api/sessions/{id}/approval.
func HandleSessionApproval(source any, s *store) http.HandlerFunc {
	return handleSessionApproval(source, s)
}

// handleSessionApproval returns an HTTP handler for POST /api/sessions/{id}/approval.
// It validates the request payload (accept or decline resolution) and dispatches
// the resolution to the underlying SessionSource or session coordinator, or the fallback store.
func handleSessionApproval(source any, s *store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id, err := url.PathUnescape(r.PathValue("id"))
		if err != nil || strings.TrimSpace(id) == "" {
			http.Error(w, "invalid session id", http.StatusBadRequest)
			return
		}

		var payload struct {
			Resolution string `json:"resolution"`
			Reason     string `json:"reason,omitempty"`
			ApprovalID string `json:"approval_id,omitempty"`
			Decision   string `json:"decision,omitempty"`
			Feedback   string `json:"feedback,omitempty"`
		}

		contentType := r.Header.Get("Content-Type")
		if strings.HasPrefix(contentType, "application/x-www-form-urlencoded") {
			if err := r.ParseForm(); err != nil {
				http.Error(w, "invalid form payload", http.StatusBadRequest)
				return
			}
			payload.Resolution = r.FormValue("resolution")
			payload.Reason = r.FormValue("reason")
			payload.ApprovalID = r.FormValue("approval_id")
			if payload.Resolution == "" {
				payload.Resolution = r.FormValue("decision")
			}
			if payload.Reason == "" {
				payload.Reason = r.FormValue("feedback")
			}
		} else {
			if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&payload); err != nil {
				http.Error(w, "invalid approval payload: invalid JSON", http.StatusBadRequest)
				return
			}
			if payload.Resolution == "" && payload.Decision != "" {
				payload.Resolution = payload.Decision
			}
			if payload.Reason == "" && payload.Feedback != "" {
				payload.Reason = payload.Feedback
			}
		}

		resolution := strings.ToLower(strings.TrimSpace(payload.Resolution))
		if resolution == "approve" {
			resolution = "accept"
		} else if resolution == "deny" || resolution == "reject" {
			resolution = "decline"
		}
		if resolution != "accept" && resolution != "decline" {
			http.Error(w, `invalid approval resolution: must be "accept" or "decline"`, http.StatusBadRequest)
			return
		}

		resolveViaStore := func(approvalID string) {
			decision := "approve"
			if resolution == "decline" {
				decision = "deny"
			}
			if approvalID == "" && s != nil {
				s.mu.RLock()
				if state := s.runs[id]; state != nil {
					approvalID = state.approval
				}
				s.mu.RUnlock()
			}
			if err := s.resolve(id, approvalID, decision); err != nil {
				writeSessionJSON(w, http.StatusConflict, map[string]string{"error": "approval resolution failed: " + err.Error()})
				return
			}
			defaultSessionHub.broadcastSession(id, "approval_resolved", []byte(fmt.Sprintf(`{"session_id":%q,"resolution":%q,"approval_id":%q}`, id, resolution, approvalID)))
			writeSessionJSON(w, http.StatusOK, map[string]any{
				"status":      "accepted",
				"session_id":  id,
				"approval_id": approvalID,
				"resolution":  resolution,
				"resolved":    true,
			})
		}

		if source != nil {
			var cards []SessionCard
			if lister, ok := source.(interface {
				Sessions(context.Context) ([]SessionCard, error)
			}); ok {
				var err error
				cards, err = lister.Sessions(r.Context())
				if err != nil {
					http.Error(w, err.Error(), http.StatusServiceUnavailable)
					return
				}
				cards, err = normalizeSessionCards(cards)
				if err != nil {
					writeSessionJSON(w, http.StatusBadGateway, map[string]string{"error": err.Error()})
					return
				}
			}
			var selected *SessionCard
			for i := range cards {
				if cards[i].ID == id {
					selected = &cards[i]
					break
				}
			}
			if len(cards) > 0 && selected == nil {
				if s != nil && s.hasRun(id) {
					resolveViaStore(payload.ApprovalID)
					return
				}
				http.Error(w, "logical session not found", http.StatusNotFound)
				return
			}

			if selected != nil {
				isAwaiting := selected.State == sessionAwaitingApproval || selected.PendingApproval != nil || strings.Contains(strings.ToLower(selected.PendingInteraction), "approval")
				if !isAwaiting {
					writeSessionJSON(w, http.StatusConflict, map[string]string{"error": "session is not awaiting approval"})
					return
				}
			}

			approvalID := payload.ApprovalID
			if approvalID == "" && selected != nil && selected.PendingApproval != nil {
				approvalID = selected.PendingApproval.ApprovalID
			}

			req := SessionApprovalRequest{
				SessionID:  id,
				ApprovalID: approvalID,
				Resolution: resolution,
				Reason:     payload.Reason,
			}

			var resolveErr error
			resolved := false
			if direct, ok := source.(SessionApprovalDirectResolver); ok {
				resolveErr = direct.ResolveApproval(id, resolution, payload.Reason)
				if resolveErr == nil {
					resolved = true
				}
			} else if resolver, ok := source.(SessionApprovalResolver); ok {
				resolveErr = resolver.ResolveApproval(r.Context(), req)
				if resolveErr == nil {
					resolved = true
				}
			} else if resolver, ok := source.(interface {
				ResolveApproval(context.Context, SessionApprovalRequest) error
			}); ok {
				resolveErr = resolver.ResolveApproval(r.Context(), req)
				if resolveErr == nil {
					resolved = true
				}
			} else if s != nil && s.hasRun(id) {
				resolveViaStore(approvalID)
				return
			} else {
				writeSessionJSON(w, http.StatusNotImplemented, map[string]string{"error": "session authority does not support approval resolution"})
				return
			}

			if resolveErr != nil {
				writeSessionJSON(w, http.StatusConflict, map[string]string{"error": resolveErr.Error()})
				return
			}

			if resolved {
				defaultSessionHub.broadcastSession(id, "approval_resolved", []byte(fmt.Sprintf(`{"session_id":%q,"resolution":%q,"approval_id":%q}`, id, resolution, approvalID)))
				writeSessionJSON(w, http.StatusOK, map[string]any{
					"status":      "accepted",
					"session_id":  id,
					"approval_id": approvalID,
					"resolution":  resolution,
					"resolved":    true,
				})
				return
			}

			writeSessionJSON(w, http.StatusNotImplemented, map[string]string{"error": "approval could not be resolved"})
			return
		}

		if s != nil && s.hasRun(id) {
			resolveViaStore(payload.ApprovalID)
			return
		}

		writeSessionJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "session authority is not connected"})
	}
}

// handleSessionSSE serves Server-Sent Events (SSE) for session cards and approval notifications.
// If scoped is true, it only streams events matching the path parameter {id} and global broadcasts.
func handleSessionSSE(scoped bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusBadRequest)
			return
		}
		sessionID := ""
		if scoped {
			var err error
			sessionID, err = url.PathUnescape(r.PathValue("id"))
			if err != nil || strings.TrimSpace(sessionID) == "" {
				http.Error(w, "invalid session id", http.StatusBadRequest)
				return
			}
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")

		ch := defaultSessionHub.subscribe(sessionID)
		defer defaultSessionHub.unsubscribe(ch)

		if sessionID != "" {
			fmt.Fprintf(w, "event: connected\ndata: {\"status\":\"connected\",\"session_id\":%q}\n\n", sessionID)
		} else {
			fmt.Fprintf(w, "event: connected\ndata: {\"status\":\"connected\"}\n\n")
		}
		flusher.Flush()

		ctx := r.Context()
		for {
			select {
			case <-ctx.Done():
				return
			case msg, ok := <-ch:
				if !ok {
					return
				}
				_, _ = w.Write(msg)
				flusher.Flush()
			}
		}
	}
}

// handleSessionEventHook receives external event streams or webhook envelopes for a session
// and broadcasts them into the live web SSE hub so cards refresh without polling.
func handleSessionEventHook(source SessionSource, s *store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id, err := url.PathUnescape(r.PathValue("id"))
		if err != nil || strings.TrimSpace(id) == "" {
			http.Error(w, "invalid session id", http.StatusBadRequest)
			return
		}
		body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
		if err != nil {
			http.Error(w, "failed to read event body", http.StatusBadRequest)
			return
		}
		if len(body) == 0 {
			http.Error(w, "empty event body", http.StatusBadRequest)
			return
		}

		app, parseErr := ParseSessionApproval(body)
		if parseErr == nil && app != nil {
			approvalPayload, _ := json.Marshal(map[string]any{
				"session_id":       id,
				"approval_id":      app.ApprovalID,
				"tool_name":        app.ToolName,
				"command":          app.Command,
				"arguments":        app.Arguments,
				"target_path":      app.TargetPath,
				"risk_explanation": app.RiskExplanation,
			})
			defaultSessionHub.broadcastSession(id, "approval_requested", approvalPayload)
			defaultSessionHub.broadcastSession(id, "session_update", approvalPayload)
		} else {
			var raw map[string]any
			_ = json.Unmarshal(body, &raw)
			eventType, _ := raw["type"].(string)
			if strings.Contains(eventType, "approval") || strings.Contains(string(body), "approval") {
				defaultSessionHub.broadcastSession(id, "approval_requested", body)
			} else {
				defaultSessionHub.broadcastSession(id, "session_update", body)
			}
		}

		writeSessionJSON(w, http.StatusAccepted, map[string]string{
			"status":     "admitted",
			"session_id": id,
		})
	}
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}

func selfcheck(out io.Writer) error {
	product, err := harnesskit.New("fak-local-ui", "0.2.0").WithProfile(harnesskit.Profile{ID: "offline-native-ui"}).Build()
	if err != nil || product.Spec().Profile.ID != "offline-native-ui" {
		return fmt.Errorf("public product contract: %w", err)
	}
	temp, err := os.MkdirTemp("", "fak-harness-web-selfcheck-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(temp)
	goals := goalregistry.Store{Path: filepath.Join(temp, "goals.json")}
	if _, err := goals.Create("Inspect the local harness", "", goalregistry.Provenance{Actor: "selfcheck", Authority: "test"}, nil); err != nil {
		return err
	}
	s := newStore()
	ts := httptest.NewServer(handlerWithSources(s, nil, goals))
	defer ts.Close()
	client := ts.Client()
	pageBody, err := get(client, ts.URL+"/")
	if err != nil {
		return err
	}
	for _, want := range []string{"Harness overview", "Web gateway", "Agent stats", "Goals", "Live dashboards", "aria-live=\"polite\"", "approval.requested", "data-skin", "e.payload"} {
		if !strings.Contains(pageBody, want) {
			return fmt.Errorf("captured render lacks %q", want)
		}
	}
	if strings.Contains(pageBody, "Local, bounded, yours.") {
		return fmt.Errorf("captured render still contains retired marketing headline")
	}
	normal, err := postRun(client, ts.URL, "prove local native UI")
	if err != nil {
		return err
	}
	events := s.after(normal, 0)
	if len(events) != 8 {
		return fmt.Errorf("normal events=%d want 8", len(events))
	}
	for _, e := range events {
		if err := e.Validate(); err != nil {
			return err
		}
	}
	resumed := s.after(normal, 6)
	if len(resumed) != 2 || resumed[0].Sequence != 7 {
		return fmt.Errorf("resume=%v", resumed)
	}
	approval, err := postRun(client, ts.URL, "approval: inspect workspace")
	if err != nil {
		return err
	}
	if err := s.resolve(approval, "approval-1", "approve"); err != nil {
		return err
	}
	if got := s.after(approval, 3); len(got) != 4 || got[0].Type != harnesskit.EventApprovalResolved {
		return fmt.Errorf("approval tail=%v", got)
	}
	failure, err := postRun(client, ts.URL, "failure: typed error")
	if err != nil {
		return err
	}
	if got := s.after(failure, 0); len(got) != 3 || got[1].Type != harnesskit.EventError {
		return fmt.Errorf("failure=%v", got)
	}
	statusBody, err := get(client, ts.URL+"/api/status")
	if err != nil {
		return err
	}
	var status statusPayload
	if err := json.Unmarshal([]byte(statusBody), &status); err != nil {
		return err
	}
	if status.Agents.TotalRuns != 3 || status.Goals.Active != 1 || status.LocalWork.IssueIntents.Active != 0 || status.LocalWork.DOSLeases.Active != 0 || len(status.Dashboards) != 8 {
		return fmt.Errorf("operator overview runs=%d active_goals=%d active_intents=%d active_leases=%d dashboards=%d", status.Agents.TotalRuns, status.Goals.Active, status.LocalWork.IssueIntents.Active, status.LocalWork.DOSLeases.Active, len(status.Dashboards))
	}
	h := sha256.Sum256([]byte(pageBody))
	fmt.Fprintf(out, "HARNESS_WEB_SELFCHECK ok protocol=%s normal=%d resumed=%d approval=4 failure=3 skins=2 runs=3 goals=1 dashboards=8 html_sha256=%s\n", harnesskit.ProtocolVersion, len(events), len(resumed), hex.EncodeToString(h[:]))
	return nil
}

func postRun(client *http.Client, baseURL, message string) (string, error) {
	body, _ := json.Marshal(map[string]string{"message": message})
	response, err := client.Post(baseURL+"/api/runs", "application/json", strings.NewReader(string(body)))
	if err != nil {
		return "", err
	}
	defer response.Body.Close()
	var created map[string]string
	if err := json.NewDecoder(response.Body).Decode(&created); err != nil {
		return "", err
	}
	return created["run_id"], nil
}

func get(client *http.Client, url string) (string, error) {
	response, err := client.Get(url)
	if err != nil {
		return "", err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	return string(body), err
}

func workspaceIdentity(root string) string {
	sum := sha256.Sum256([]byte(filepath.Clean(root)))
	return "ws-" + hex.EncodeToString(sum[:6])
}

func Run(ctx context.Context, stdout, stderr io.Writer, args []string) int {
	return RunWithLocalWork(ctx, stdout, stderr, args, nil)
}

func RunWithLocalWork(ctx context.Context, stdout, stderr io.Writer, args []string, local LocalWorkSource) int {
	fs := flag.NewFlagSet("harnesswebdemo", flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", "127.0.0.1:8787", "loopback listen address")
	check := fs.Bool("selfcheck", false, "run captured render and protocol witness")
	statePath := fs.String("state", "", "session state file (default: user config directory)")
	workspace := fs.String("workspace", "", "explicit workspace bound to the fak native gateway")
	fakURL := fs.String("fak-url", defaultFakURL, "stock fak base URL (defaults to the local fak serve gateway)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *check {
		if err := selfcheck(stdout); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		return 0
	}
	host, _, err := net.SplitHostPort(*addr)
	if err != nil || (host != "127.0.0.1" && host != "localhost" && host != "::1") {
		fmt.Fprintln(stderr, "-addr must use a loopback host")
		return 2
	}
	listener, err := net.Listen("tcp", *addr)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if *statePath == "" {
		configDir, configErr := os.UserConfigDir()
		if configErr != nil {
			fmt.Fprintln(stderr, configErr)
			return 1
		}
		*statePath = filepath.Join(configDir, "fak", "harnesswebdemo", "sessions.json")
	}
	s, err := newPersistentStore(*statePath)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	var live *liveAdapter
	if strings.TrimSpace(*workspace) != "" && strings.TrimSpace(*fakURL) == "" {
		fmt.Fprintln(stderr, "-workspace requires -fak-url")
		return 2
	}
	if strings.TrimSpace(*fakURL) != "" {
		identity := ""
		if strings.TrimSpace(*workspace) != "" {
			resolved, resolveErr := filepath.EvalSymlinks(*workspace)
			if resolveErr != nil {
				fmt.Fprintf(stderr, "harnesswebdemo: resolve workspace: %v\n", resolveErr)
				return 2
			}
			resolved, resolveErr = filepath.Abs(resolved)
			if resolveErr != nil {
				fmt.Fprintf(stderr, "harnesswebdemo: resolve workspace: %v\n", resolveErr)
				return 2
			}
			identity = workspaceIdentity(resolved)
		}
		live = &liveAdapter{baseURL: *fakURL, offlineFallback: *fakURL == defaultFakURL, client: &http.Client{Timeout: 10 * time.Minute}, identity: identity}
		probeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		err := live.probeWorkspace(probeCtx)
		cancel()
		if err != nil {
			fmt.Fprintf(stderr, "harnesswebdemo: fak capability probe: %v\n", err)
		}
		if identity != "" && !live.workspace.Armed {
			fmt.Fprintln(stderr, "harnesswebdemo: -workspace requires an armed native code workspace at -fak-url")
			return 2
		}
	}
	server := &http.Server{
		Handler:           handlerWithAllSources(s, live, goalregistry.Store{Path: goalregistry.DefaultPath()}, nil, local, localWorkRoot(*workspace)),
		ReadHeaderTimeout: 5 * time.Second,
	}
	fmt.Fprintf(stdout, "fak native harness UI: http://%s\n", listener.Addr())
	go func() { <-ctx.Done(); _ = server.Shutdown(context.Background()) }()
	if err := server.Serve(listener); err != nil && err != http.ErrServerClosed {
		fmt.Fprintln(stderr, err)
		return 1
	}
	return 0
}

func localWorkRoot(workspace string) string {
	if root := strings.TrimSpace(workspace); root != "" {
		return root
	}
	root, err := os.Getwd()
	if err != nil {
		return ""
	}
	return root
}
