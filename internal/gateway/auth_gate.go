package gateway

import (
	"html/template"
	"net"
	"net/http"
	"strings"
)

var authGateTemplate = template.Must(template.New("gateway-auth-gate").Parse(`<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>fak gateway · authentication required</title>
<style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#141b2d;--ink:#edf2ff;--muted:#a9b4cc;--line:#2a3652;--accent:#76e6c5;--accent2:#8eb5ff;--danger:#ff6b6b}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(circle at top,#18233c 0,#0b1020 44%);color:var(--ink);font:16px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{width:min(720px,calc(100% - 32px));margin:0 auto;padding:56px 0 72px}
header{margin-bottom:28px}
.eyebrow{color:var(--accent);font-weight:700;letter-spacing:.12em;text-transform:uppercase;font-size:.78rem}
h1{margin:.25rem 0 .4rem;font-size:clamp(2rem,5vw,3rem);line-height:1.1}
.lede{color:var(--muted);font-size:1.05rem}
.box{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:24px;margin-bottom:20px}
.box h2{margin:0 0 12px;font-size:1.1rem;color:var(--ink)}
.box p{margin:0 0 12px;color:var(--muted);font-size:.92rem}
.alert{background:#2d1418;border:1px solid var(--danger);border-radius:10px;padding:12px 16px;margin-bottom:20px;color:#ffb4b4;font-size:.92rem}
form{display:flex;flex-direction:column;gap:12px}
label{font-size:.9rem;font-weight:600;color:var(--muted)}
.input-row{display:flex;gap:10px}
input[type="password"],input[type="text"]{flex:1;background:#0b1020;border:1px solid var(--line);border-radius:10px;padding:10px 14px;color:var(--ink);font:14px/1.4 ui-monospace,monospace}
input:focus{outline:none;border-color:var(--accent2)}
button{background:var(--accent);color:#0b1020;border:none;border-radius:10px;padding:10px 20px;font-weight:600;cursor:pointer;font-size:.95rem;transition:opacity .15s ease}
button:hover{opacity:.9}
pre{background:#0b1020;border:1px solid var(--line);border-radius:10px;padding:12px;overflow-x:auto;font:13px/1.5 ui-monospace,monospace;color:var(--accent2);margin:8px 0}
code{color:var(--accent);font-family:ui-monospace,monospace}
.links{display:flex;gap:16px;margin-top:12px}
.links a{color:var(--accent2);text-decoration:none;font-size:.9rem}
.links a:hover{text-decoration:underline}
footer{margin-top:32px;color:var(--muted);font-size:.85rem}
</style></head><body><main>
<header>
<div class="eyebrow">local agent kernel</div>
<h1>Authentication Required</h1>
<p class="lede">This FAK gateway is bound for remote access and requires a bearer token. Enter your gateway key to unlock the dashboard, or connect using an API client.</p>
</header>
{{if .InvalidKey}}
<div class="alert">Invalid gateway key. Please check the key on your appliance and try again.</div>
{{end}}
<div class="box">
<h2>Unlock Gateway Dashboard</h2>
<p>Paste the gateway bearer key to view live status, metrics, and models in this browser:</p>
<form method="GET" action="/">
<label for="key-input">Gateway Key (<code>FAK_GATEWAY_KEY</code>)</label>
<div class="input-row">
<input type="password" id="key-input" name="key" placeholder="Paste 64-character hex key..." autofocus autocomplete="off" required />
<button type="submit">Unlock Dashboard</button>
</div>
</form>
</div>
<div class="box">
<h2>Where is my key?</h2>
<p>On a <strong>Strix Halo appliance</strong>, the key is generated at install time and stored in:</p>
<pre>/etc/fak/gateway.env</pre>
<p>Retrieve it from your terminal or workstation:</p>
<pre># Via SSH:
ssh fak@strix1 "cat /etc/fak/gateway.env"

# Via fak-strix CLI:
go run ./cmd/fak-strix key</pre>
</div>
<div class="box">
<h2>API Client Configuration</h2>
<p>Set environment variables for OpenAI- or Anthropic-compatible clients:</p>
<pre>export OPENAI_BASE_URL="http://{{.Host}}:8080/v1"
export OPENAI_API_KEY="&lt;FAK_GATEWAY_KEY&gt;"</pre>
<div class="links">
<a href="/healthz">Test Liveness (/healthz - unauthenticated)</a>
<a href="https://github.com/anthony-chaudhary/fak">Documentation</a>
</div>
</div>
<footer>FAK Autonomous Serving Engine · Constant-time bearer verification active.</footer>
</main></body></html>`))

type authGateData struct {
	Host       string
	InvalidKey bool
}

func isBrowserHTMLRequest(r *http.Request) bool {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		return false
	}
	accept := r.Header.Get("Accept")
	return strings.Contains(accept, "text/html")
}

func writeAuthError(w http.ResponseWriter, r *http.Request, credPresented bool) {
	w.Header().Set("WWW-Authenticate", `Bearer realm="fak-gateway"`)
	if isBrowserHTMLRequest(r) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		w.Header().Set("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'self'")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.WriteHeader(http.StatusUnauthorized)
		if r.Method == http.MethodHead {
			return
		}
		host := r.Host
		if h, _, err := net.SplitHostPort(host); err == nil && h != "" {
			host = h
		}
		if host == "" {
			host = "appliance-ip"
		}
		data := authGateData{
			Host:       host,
			InvalidKey: credPresented,
		}
		_ = authGateTemplate.Execute(w, data)
		return
	}
	if !credPresented {
		writeErrCode(w, http.StatusUnauthorized, "missing_credentials", "missing or invalid credentials: no bearer token provided; send 'Authorization: Bearer <key>', 'X-Api-Key: <key>', or '?key=<key>'")
	} else {
		writeErrCode(w, http.StatusUnauthorized, "invalid_credentials", "missing or invalid credentials: provided gateway token is incorrect")
	}
}
