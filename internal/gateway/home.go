package gateway

import (
	"fmt"
	"html/template"
	"net/http"
)

// homePageTemplate is self-contained so the gateway origin stays useful without
// an asset server or frontend build pipeline.
var homePageTemplate = template.Must(template.New("gateway-home").Parse(`<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>fak gateway</title>
<style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#141b2d;--ink:#edf2ff;--muted:#a9b4cc;--line:#2a3652;--accent:#76e6c5;--accent2:#8eb5ff}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#18233c 0,#0b1020 44%);color:var(--ink);font:16px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}main{width:min(1040px,calc(100% - 32px));margin:0 auto;padding:56px 0 72px}header{margin-bottom:32px}.eyebrow{color:var(--accent);font-weight:700;letter-spacing:.12em;text-transform:uppercase;font-size:.78rem}h1{margin:.25rem 0 .4rem;font-size:clamp(2.25rem,7vw,4.75rem);line-height:1}.lede{max-width:720px;color:var(--muted);font-size:1.1rem}.facts{display:flex;flex-wrap:wrap;gap:8px;margin:22px 0 0;padding:0;list-style:none}.facts li{border:1px solid var(--line);border-radius:999px;padding:6px 11px;color:var(--muted);background:#10172a}.facts strong{color:var(--ink)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}a.card{display:block;min-height:168px;padding:20px;border:1px solid var(--line);border-radius:14px;color:inherit;text-decoration:none;background:linear-gradient(145deg,#172139,#111829);transition:transform .12s ease,border-color .12s ease}a.card:hover,a.card:focus-visible{transform:translateY(-2px);border-color:var(--accent2);outline:none}.card h2{margin:0 0 8px;font-size:1.12rem}.card p{margin:0;color:var(--muted)}.path{display:block;margin-top:18px;color:var(--accent2);font:600 .85rem/1.3 ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}footer{margin-top:28px;color:var(--muted);font-size:.9rem}code{color:var(--accent)}
</style></head><body><main>
<header><div class="eyebrow">local agent kernel</div><h1>fak gateway</h1>
<p class="lede">This is the live gateway behind the URL shown in the TUI. Inspect its state, discover its APIs, or open the agent interface below.</p>
<ul class="facts"><li>engine <strong>{{.Engine}}</strong></li><li>model <strong>{{.Model}}</strong></li><li>provider <strong>{{.Provider}}</strong></li><li>vDSO <strong>{{if .VDSO}}on{{else}}off{{end}}</strong></li></ul></header>
<section class="grid" aria-label="Gateway surfaces">
<a class="card" href="/healthz"><h2>Health</h2><p>Live readiness, engine state, and serving diagnostics.</p><span class="path">GET /healthz</span></a>
<a class="card" href="/v1/models"><h2>Models</h2><p>OpenAI-compatible model discovery for this gateway.</p><span class="path">GET /v1/models - auth may apply</span></a>
<a class="card" href="/a2a/v1/agent-card"><h2>Agent</h2><p>Discover the A2A agent, its skills, and its task endpoint.</p><span class="path">GET /a2a/v1/agent-card - local discovery</span></a>
<a class="card" href="https://github.com/anthony-chaudhary/fak/blob/main/docs/fak/openapi.yaml"><h2>API map</h2><p>Open the gateway's machine-readable OpenAPI reference.</p><span class="path">docs/fak/openapi.yaml</span></a>
<a class="card" href="/metrics"><h2>Metrics</h2><p>Prometheus counters for requests, cache behavior, and latency.</p><span class="path">GET /metrics - auth may apply</span></a>
<a class="card" href="/debug/vars"><h2>Diagnostics</h2><p>Runtime counters and bounded operational state.</p><span class="path">GET /debug/vars - auth may apply</span></a>
</section><footer>API clients can continue using <code>/v1/responses</code>, <code>/v1/chat/completions</code>, or <code>/mcp</code>. The homepage contains no credentials or upstream URL.</footer>
</main></body></html>`))

type homePageData struct {
	Engine, Model, Provider string
	VDSO                    bool
}

func (s *Server) handleHome(w http.ResponseWriter, r *http.Request) {
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
	w.Header().Set("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	if r.Method == http.MethodHead {
		w.WriteHeader(http.StatusOK)
		return
	}
	data := homePageData{Engine: s.engineID, Model: s.model, Provider: s.provider, VDSO: s.k.VDSOEnabled()}
	if err := homePageTemplate.Execute(w, data); err != nil {
		s.logf("gateway homepage render: %v", fmt.Errorf("execute template: %w", err))
	}
}
