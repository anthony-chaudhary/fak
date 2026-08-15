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
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

const page = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>fak local harness</title><style>
:root{color-scheme:dark;--ink:#eef6ee;--muted:#9fb1a3;--panel:#122019;--line:#294034;--accent:#a7f3d0;--bg:#09110d}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% 0,#173326 0,var(--bg) 42%);color:var(--ink);font:16px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}main{width:min(900px,92vw);margin:7vh auto}.eyebrow{color:var(--accent);letter-spacing:.18em;text-transform:uppercase}.shell{border:1px solid var(--line);border-radius:18px;background:color-mix(in srgb,var(--panel) 92%,transparent);box-shadow:0 24px 80px #0008;overflow:hidden}.top{padding:28px 32px;border-bottom:1px solid var(--line)}h1{font:600 clamp(2rem,5vw,4.2rem)/1 ui-sans-serif,system-ui;margin:.2em 0}.sub{color:var(--muted);max-width:66ch}.run{padding:26px 32px}form{display:flex;gap:10px}input{flex:1;background:#07100b;color:var(--ink);border:1px solid var(--line);border-radius:10px;padding:14px}button{background:var(--accent);color:#062116;border:0;border-radius:10px;padding:0 20px;font-weight:800}.events{display:grid;gap:10px;margin-top:24px}.event{border-left:3px solid var(--line);background:#0a1510;padding:12px 14px}.event[data-kind^="message"]{border-color:var(--accent)}.event[data-kind^="tool"]{border-color:#fbbf24}.kind{color:var(--muted);font-size:.78rem;text-transform:uppercase}.status{display:flex;gap:8px;align-items:center;color:var(--muted)}.dot{width:8px;height:8px;border-radius:50%;background:var(--accent)}
</style></head><body><main><p class="eyebrow">fak / native harness</p><section class="shell"><header class="top"><h1>Local, bounded, yours.</h1><p class="sub">A separately built product UI over semantic harness events. No terminal scraping. Offline by default.</p><p class="status"><span class="dot"></span><span id="status">ready on loopback</span></p></header><div class="run"><form id="prompt"><input id="text" aria-label="Message" value="show the native harness works"><button>Run</button></form><section id="events" class="events" aria-live="polite"></section></div></section></main><script>
let run="",after=0;const list=document.querySelector("#events"),status=document.querySelector("#status");
function render(e){const d=document.createElement("article");d.className="event";d.dataset.kind=e.type;const detail=e.payload?.text||e.payload?.summary||e.payload?.name||e.payload?.message||e.type;d.innerHTML='<div class="kind">'+e.type+'</div><div>'+detail+'</div>';list.append(d);after=e.sequence}
async function pull(){const r=await fetch('/api/events?run_id='+encodeURIComponent(run)+'&after='+after);for(const e of await r.json())render(e);status.textContent='connected · cursor '+after}
document.querySelector("#prompt").addEventListener("submit",async e=>{e.preventDefault();list.replaceChildren();after=0;const r=await fetch("/api/runs",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({message:document.querySelector("#text").value})});run=(await r.json()).run_id;await pull()});
</script></body></html>`

type store struct {
	mu   sync.RWMutex
	runs map[string][]harnesskit.Envelope
	next uint64
}

func newStore() *store { return &store{runs: make(map[string][]harnesskit.Envelope)} }

func (s *store) create(message string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.next++
	runID := fmt.Sprintf("local-%d", s.next)
	reply := "offline reply: " + message
	s.runs[runID] = []harnesskit.Envelope{
		event(runID, 1, harnesskit.EventRunStarted, ""),
		messageEvent(runID, 2, "assistant", reply),
		toolEvent(runID, 3, harnesskit.EventToolStarted, "record_selfcheck", ""),
		toolEvent(runID, 4, harnesskit.EventToolCompleted, "record_selfcheck", "ok"),
		event(runID, 5, harnesskit.EventRunCompleted, ""),
	}
	return runID
}

func event(runID string, seq uint64, kind harnesskit.EventType, detail string) harnesskit.Envelope {
	return harnesskit.Envelope{Version: harnesskit.ProtocolVersion, RunID: runID, Sequence: seq, EventID: fmt.Sprintf("%s-%d", runID, seq), Type: kind, Sensitivity: harnesskit.SensitivityPublic}
}
func messageEvent(runID string, seq uint64, role, text string) harnesskit.Envelope {
	e := event(runID, seq, harnesskit.EventMessageCompleted, "")
	e.Payload, _ = json.Marshal(harnesskit.MessagePayload{MessageID: fmt.Sprintf("message-%d", seq), Role: role, Text: text})
	return e
}
func toolEvent(runID string, seq uint64, kind harnesskit.EventType, name, output string) harnesskit.Envelope {
	e := event(runID, seq, kind, "")
	e.Payload, _ = json.Marshal(harnesskit.ToolPayload{CallID: "selfcheck-tool", Name: name, Status: output, Summary: output})
	return e
}

func (s *store) after(runID string, after uint64) []harnesskit.Envelope {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var out []harnesskit.Envelope
	for _, e := range s.runs[runID] {
		if e.Sequence > after {
			out = append(out, e)
		}
	}
	return out
}

func handler(s *store) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
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
	return mux
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}

func selfcheck(out io.Writer) error {
	product, err := harnesskit.New("fak-local-ui", "0.1.0").WithProfile(harnesskit.Profile{ID: "offline-native-ui"}).Build()
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
	for _, want := range []string{"Local, bounded, yours.", "aria-live=\"polite\"", "e.payload?.text", "e.payload?.summary"} {
		if !strings.Contains(pageBody, want) {
			return fmt.Errorf("captured render lacks %q", want)
		}
	}
	response, err := client.Post(ts.URL+"/api/runs", "application/json", strings.NewReader(`{"message":"prove local native UI"}`))
	if err != nil {
		return err
	}
	defer response.Body.Close()
	var created map[string]string
	if err := json.NewDecoder(response.Body).Decode(&created); err != nil {
		return err
	}
	first, err := get(client, ts.URL+"/api/events?run_id="+created["run_id"]+"&after=0")
	if err != nil {
		return err
	}
	var events []harnesskit.Envelope
	if err := json.Unmarshal([]byte(first), &events); err != nil {
		return err
	}
	if len(events) != 5 {
		return fmt.Errorf("events=%d want 5", len(events))
	}
	for _, e := range events {
		if err := e.Validate(); err != nil {
			return err
		}
	}
	tail, err := get(client, ts.URL+"/api/events?run_id="+created["run_id"]+"&after=3")
	if err != nil {
		return err
	}
	var resumed []harnesskit.Envelope
	if err := json.Unmarshal([]byte(tail), &resumed); err != nil {
		return err
	}
	if len(resumed) != 2 || resumed[0].Sequence != 4 {
		return fmt.Errorf("resume=%v", resumed)
	}
	h := sha256.Sum256([]byte(pageBody))
	fmt.Fprintf(out, "HARNESS_WEB_SELFCHECK ok protocol=%s events=%d resumed=%d html_sha256=%s\n", harnesskit.ProtocolVersion, len(events), len(resumed), hex.EncodeToString(h[:]))
	return nil
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
	server := &http.Server{Handler: handler(newStore()), ReadHeaderTimeout: 5 * time.Second}
	fmt.Fprintf(stdout, "fak native harness UI: http://%s\n", listener.Addr())
	go func() { <-ctx.Done(); _ = server.Shutdown(context.Background()) }()
	if err := server.Serve(listener); err != nil && err != http.ErrServerClosed {
		fmt.Fprintln(stderr, err)
		return 1
	}
	return 0
}

func main() { os.Exit(run(context.Background(), os.Stdout, os.Stderr, os.Args[1:])) }
