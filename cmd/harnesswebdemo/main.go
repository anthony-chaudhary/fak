package main

import (
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
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

const page = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>fak local harness</title><style>
:root{color-scheme:dark;--ink:#eef6ee;--muted:#9fb1a3;--panel:#122019;--line:#294034;--accent:#a7f3d0;--warn:#fbbf24;--danger:#fb7185;--bg:#09110d}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 18% 0,#173326 0,var(--bg) 42%);color:var(--ink);font:16px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}main{width:min(1040px,94vw);margin:5vh auto}.eyebrow{color:var(--accent);letter-spacing:.18em;text-transform:uppercase}.shell{border:1px solid var(--line);border-radius:18px;background:color-mix(in srgb,var(--panel) 92%,transparent);box-shadow:0 24px 80px #0008;overflow:hidden}.top{padding:26px 30px;border-bottom:1px solid var(--line);display:grid;grid-template-columns:1fr auto;gap:20px;align-items:end}h1{font:600 clamp(2rem,5vw,4.2rem)/1 ui-sans-serif,system-ui;margin:.15em 0}.sub{color:var(--muted);max-width:66ch}.run{padding:26px 30px}.composer{display:flex;gap:10px}.composer input{flex:1;background:#07100b;color:var(--ink);border:1px solid var(--line);border-radius:10px;padding:14px}.composer button,.approval button,.skin{background:var(--accent);color:#062116;border:0;border-radius:10px;padding:0 20px;font-weight:800;cursor:pointer}.skin{padding:9px 12px}.examples{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}.examples button{background:transparent;border:1px solid var(--line);color:var(--muted);border-radius:99px;padding:6px 10px;cursor:pointer}.events{display:grid;gap:10px;margin-top:24px}.event{border-left:3px solid var(--line);background:#0a1510;padding:12px 14px}.event[data-kind^="message"]{border-color:var(--accent)}.event[data-kind^="tool"],.event[data-kind^="approval"]{border-color:var(--warn)}.event[data-kind="error"]{border-color:var(--danger)}.kind{color:var(--muted);font-size:.78rem;text-transform:uppercase}.status{display:flex;gap:8px;align-items:center;color:var(--muted)}.dot{width:8px;height:8px;border-radius:50%;background:var(--accent)}.approval{display:flex;gap:8px;margin-top:10px}.approval button:last-child{background:transparent;border:1px solid var(--danger);color:var(--danger)}.empty{color:var(--muted);border:1px dashed var(--line);padding:24px;border-radius:12px}body[data-skin="minimal"]{--accent:#93c5fd;--panel:#111827;--line:#334155;--bg:#020617}body[data-skin="minimal"] .eyebrow{letter-spacing:.06em}body[data-skin="minimal"] h1{font-size:2.4rem}@media(max-width:720px){.top{grid-template-columns:1fr}.composer{display:grid}.composer button{padding:13px}}
</style></head><body><main><p class="eyebrow">fak / native harness</p><section class="shell"><header class="top"><div><h1>Local, bounded, yours.</h1><p class="sub">A separately built product UI over semantic harness events. No terminal scraping. Offline by default.</p><p class="status"><span class="dot"></span><span id="status">ready on loopback</span></p></div><button id="skin" class="skin" type="button">Switch skin</button></header><div class="run"><form id="prompt" class="composer"><input id="text" aria-label="Message" value="show the native harness works"><button>Run</button></form><nav class="examples" aria-label="Example runs"><button data-example="normal">Tool run</button><button data-example="approval">Approval run</button><button data-example="failure">Failure run</button></nav><section id="events" class="events" aria-live="polite"><p class="empty">Run an offline scenario. Events stay semantic and replayable.</p></section></div></section></main><script>
let run="",after=0,skin=new URLSearchParams(location.search).get("skin")||"forest";document.body.dataset.skin=skin;const list=document.querySelector("#events"),status=document.querySelector("#status"),text=document.querySelector("#text");
const esc=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
function render(e){const p=e.payload||{},d=document.createElement("article");d.className="event";d.dataset.kind=e.type;const detail=p.text||p.summary||p.name||p.message||p.status||e.type;d.innerHTML='<div class="kind">'+esc(e.type)+' · '+e.sequence+'</div><div>'+esc(detail)+'</div>';if(e.type==="approval.requested"){const a=document.createElement("div");a.className="approval";a.innerHTML='<button data-decision="approve">Approve once</button><button data-decision="deny">Deny</button>';a.addEventListener("click",async x=>{const decision=x.target.dataset.decision;if(!decision)return;await fetch("/api/approvals",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({run_id:run,approval_id:p.approval_id,decision})});await pull()});d.append(a)}list.append(d);after=e.sequence}
async function pull(){const r=await fetch('/api/events?run_id='+encodeURIComponent(run)+'&after='+after);for(const e of await r.json())render(e);status.textContent='connected · cursor '+after}
async function start(message){list.replaceChildren();after=0;const r=await fetch("/api/runs",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({message})});run=(await r.json()).run_id;await pull()}
document.querySelector("#prompt").addEventListener("submit",async e=>{e.preventDefault();await start(text.value)});document.querySelector(".examples").addEventListener("click",async e=>{const x=e.target.dataset.example;if(!x)return;text.value=x==="approval"?"approval: inspect workspace":x==="failure"?"failure: demonstrate typed error":"show the native harness works";await start(text.value)});document.querySelector("#skin").addEventListener("click",()=>{skin=skin==="forest"?"minimal":"forest";document.body.dataset.skin=skin});const scenario=new URLSearchParams(location.search).get("scenario");if(scenario){text.value=scenario==="approval"?"approval: inspect workspace":scenario==="failure"?"failure: demonstrate typed error":"show the native harness works";start(text.value)}
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

func (s *store) resolve(runID, approvalID, decision string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	state := s.runs[runID]
	if state == nil || state.approval != approvalID || state.resolved || (decision != "approve" && decision != "deny") {
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

func handler(s *store) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
		_, _ = io.WriteString(w, page)
	})
	mux.HandleFunc("POST /api/runs", func(w http.ResponseWriter, r *http.Request) {
		var input struct {
			Message string `json:"message"`
		}
		if err := json.NewDecoder(r.Body).Decode(&input); err != nil || strings.TrimSpace(input.Message) == "" {
			http.Error(w, "message is required", http.StatusBadRequest)
			return
		}
		writeJSON(w, map[string]string{"run_id": s.create(strings.TrimSpace(input.Message))})
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
		writeJSON(w, map[string]string{"status": "accepted"})
	})
	return mux
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
	s := newStore()
	ts := httptest.NewServer(handler(s))
	defer ts.Close()
	client := ts.Client()
	pageBody, err := get(client, ts.URL+"/")
	if err != nil {
		return err
	}
	for _, want := range []string{"Local, bounded, yours.", "aria-live=\"polite\"", "approval.requested", "data-skin", "e.payload"} {
		if !strings.Contains(pageBody, want) {
			return fmt.Errorf("captured render lacks %q", want)
		}
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
	h := sha256.Sum256([]byte(pageBody))
	fmt.Fprintf(out, "HARNESS_WEB_SELFCHECK ok protocol=%s normal=%d resumed=%d approval=4 failure=3 skins=2 html_sha256=%s\n", harnesskit.ProtocolVersion, len(events), len(resumed), hex.EncodeToString(h[:]))
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

func run(ctx context.Context, stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("harnesswebdemo", flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", "127.0.0.1:8787", "loopback listen address")
	check := fs.Bool("selfcheck", false, "run captured render and protocol witness")
	statePath := fs.String("state", "", "session state file (default: user config directory)")
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
	server := &http.Server{Handler: handler(s), ReadHeaderTimeout: 5 * time.Second}
	fmt.Fprintf(stdout, "fak native harness UI: http://%s\n", listener.Addr())
	go func() { <-ctx.Done(); _ = server.Shutdown(context.Background()) }()
	if err := server.Serve(listener); err != nil && err != http.ErrServerClosed {
		fmt.Fprintln(stderr, err)
		return 1
	}
	return 0
}

func main() { os.Exit(run(context.Background(), os.Stdout, os.Stderr, os.Args[1:])) }
