package gateway

import (
	"fmt"
	"html/template"
	"net/http"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

// homePageTemplate is self-contained so the gateway origin stays useful without
// an asset server or frontend build pipeline.
var homePageTemplate = template.Must(template.New("gateway-home").Parse(`<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>fak gateway</title>
<style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#141b2d;--ink:#edf2ff;--muted:#a9b4cc;--line:#2a3652;--accent:#76e6c5;--accent2:#8eb5ff}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#18233c 0,#0b1020 44%);color:var(--ink);font:16px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}main{width:min(1040px,calc(100% - 32px));margin:0 auto;padding:56px 0 72px}header{margin-bottom:32px}.eyebrow{color:var(--accent);font-weight:700;letter-spacing:.12em;text-transform:uppercase;font-size:.78rem}h1{margin:.25rem 0 .4rem;font-size:clamp(2.25rem,7vw,4.75rem);line-height:1}.lede{max-width:720px;color:var(--muted);font-size:1.1rem}.facts{display:flex;flex-wrap:wrap;gap:8px;margin:22px 0 0;padding:0;list-style:none}.facts li{border:1px solid var(--line);border-radius:999px;padding:6px 11px;color:var(--muted);background:#10172a}.facts strong{color:var(--ink)}.live{margin:0 0 22px;padding:18px;border:1px solid var(--line);border-radius:14px;background:#10172a}.live-head{display:flex;align-items:center;justify-content:space-between;gap:12px}.live h2{margin:0;font-size:1.1rem}.live-state{color:var(--accent);font-weight:700}.live-state[data-state="unavailable"]{color:#ffb4a8}.live-values{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:14px}.live-value{padding:10px;border-radius:10px;background:#0b1020}.live-value span{display:block;color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.06em}.live-value strong{font-size:1.05rem}.live-note{margin:10px 0 0;color:var(--muted);font-size:.86rem}.section-title{margin:28px 0 10px}.section-note{margin:0 0 14px;color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}a.card{display:block;min-height:168px;padding:20px;border:1px solid var(--line);border-radius:14px;color:inherit;text-decoration:none;background:linear-gradient(145deg,#172139,#111829);transition:transform .12s ease,border-color .12s ease}a.card:hover,a.card:focus-visible{transform:translateY(-2px);border-color:var(--accent2);outline:none}.card h2{margin:0 0 8px;font-size:1.12rem}.card p{margin:0;color:var(--muted)}.path{display:block;margin-top:18px;color:var(--accent2);font:600 .85rem/1.3 ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}footer{margin-top:28px;color:var(--muted);font-size:.9rem}code{color:var(--accent)}
</style></head><body><main>
<header><div class="eyebrow">local agent kernel</div><h1>fak gateway</h1>
<p class="lede">This is the live gateway behind the URL shown in the TUI. Inspect its state, discover its APIs, or open the agent interface below.</p>
<ul class="facts"><li>engine <strong>{{.Engine}}</strong></li><li>model <strong>{{.Model}}</strong></li><li>provider <strong>{{.Provider}}</strong></li><li>vDSO <strong>{{if .VDSO}}on{{else}}off{{end}}</strong></li></ul></header>
<section class="live" aria-labelledby="live-title"><div class="live-head"><h2 id="live-title">Live gateway</h2><span id="live-state" class="live-state" data-state="loading" role="status">connecting</span></div><div class="live-values"><div class="live-value"><span>Readiness</span><strong id="live-ready">checking</strong></div><div class="live-value"><span>Requests</span><strong id="live-requests">—</strong></div><div class="live-value"><span>Cache hits</span><strong id="live-cache-hits">—</strong></div><div class="live-value"><span>In flight</span><strong id="live-inflight">—</strong></div></div><p id="live-note" class="live-note">Updates from this gateway every 5 seconds; no dashboard setup or extra process required.</p></section>
<h2 class="section-title">Rich dashboards</h2><p class="section-note">Choose a named view below; on-demand · 9 dashboards.</p>
<section class="grid" aria-label="Gateway surfaces">
<a class="card" href="/healthz"><h2>Health</h2><p>Live readiness, engine state, and serving diagnostics.</p><span class="path">GET /healthz</span></a>
<a class="card" href="/v1/models"><h2>Models</h2><p>OpenAI-compatible model discovery for this gateway.</p><span class="path">GET /v1/models - auth may apply</span></a>
<a class="card" href="/a2a/v1/agent-card"><h2>Agent</h2><p>Discover the A2A agent, its skills, and its task endpoint.</p><span class="path">GET /a2a/v1/agent-card - local discovery</span></a>
<a class="card" href="https://github.com/anthony-chaudhary/fak/blob/main/docs/fak/openapi.yaml"><h2>API map</h2><p>Open the gateway's machine-readable OpenAPI reference.</p><span class="path">docs/fak/openapi.yaml</span></a>
{{range .RichDashboards}}<a class="card rich-dashboard" data-dashboard-uid="{{.UID}}" href="/?dashboard=rich&amp;uid={{.UID}}"><h2>{{.Title}}</h2><p>{{.Description}}</p><span class="path">on-demand / {{.Category}}</span></a>{{end}}
<a class="card" href="/metrics"><h2>Metrics</h2><p>Prometheus counters for requests, cache behavior, and latency.</p><span class="path">GET /metrics - auth may apply</span></a>
<a class="card" href="/debug/vars"><h2>Diagnostics</h2><p>Runtime counters and bounded operational state.</p><span class="path">GET /debug/vars - auth may apply</span></a>
</section><footer>API clients can continue using <code>/v1/responses</code>, <code>/v1/chat/completions</code>, or <code>/mcp</code>. The homepage contains no credentials or upstream URL.</footer>
</main><script>
(()=>{const byId=id=>document.getElementById(id),state=byId("live-state"),note=byId("live-note");
const metric=(text,name)=>{let total=0,found=false;for(const line of text.split(/\r?\n/)){const match=line.match(new RegExp("^"+name+"(?:\\{[^}]*\\})?\\s+([0-9.eE+-]+)$"));if(match){total+=Number(match[1]);found=true}}return found?total:null};
const show=(id,value)=>{if(value!==null&&Number.isFinite(value))byId(id).textContent=value.toLocaleString()};
async function refresh(){try{const [health,metrics]=await Promise.all([fetch("/healthz",{headers:{Accept:"application/json"}}),fetch("/metrics",{headers:{Accept:"text/plain"}})]);if(!health.ok||!metrics.ok)throw new Error("health "+health.status+", metrics "+metrics.status);const healthBody=await health.json(),metricsBody=await metrics.text();byId("live-ready").textContent=(healthBody.ok===false||healthBody.ready===false)?"not ready":"ready";show("live-requests",metric(metricsBody,"fak_gateway_http_requests_total"));show("live-cache-hits",metric(metricsBody,"fak_gateway_inference_cached_prompt_hits_total"));show("live-inflight",metric(metricsBody,"fak_gateway_inflight_requests"));state.textContent="live";state.dataset.state="live";note.textContent="Updated "+new Date().toLocaleTimeString()+" / refreshes every 5 seconds."}catch(error){state.textContent="unavailable";state.dataset.state="unavailable";note.textContent="Live refresh failed; last good values are preserved. Open Health or Metrics below for details."}}
refresh();setInterval(refresh,5000)})();
</script></body></html>`))

type homePageData struct {
	Engine, Model, Provider string
	VDSO                    bool
	RichDashboards          []richDashboardLink
}

var dashboardUsageLedgerPath = filepath.Join(".fak", "nightrun", "gateway-usage.jsonl")

func recordDashboardAdoption(event string) {
	row, err := gatewayusageledger.DashboardEventRow(event, time.Now())
	if err != nil {
		return
	}
	_ = gatewayusageledger.Append(dashboardUsageLedgerPath, row)
}

func (s *Server) handleHome(w http.ResponseWriter, r *http.Request) {
	if s.richDashboards != nil && s.handleRichDashboard(w, r) {
		return
	}
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		w.Header().Set("Allow", "GET, HEAD")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; connect-src 'self'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	recordDashboardAdoption("lightweight_open")
	if r.Method == http.MethodHead {
		w.WriteHeader(http.StatusOK)
		return
	}
	data := homePageData{
		Engine: s.engineID, Model: s.model, Provider: s.provider, VDSO: s.k.VDSOEnabled(),
		RichDashboards: richDashboardLinks,
	}
	if err := homePageTemplate.Execute(w, data); err != nil {
		s.logf("gateway homepage render: %v", fmt.Errorf("execute template: %w", err))
	}
}
