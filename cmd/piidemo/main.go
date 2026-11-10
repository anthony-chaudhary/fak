// Command piidemo is the end-to-end HTML demo of the ToolGuard posture (#14327)
// over the router's REAL HTTP seams: one page, two real gateway servers built
// in-process on ephemeral loopback listeners.
//
// The left panel is the DEFAULT router — ToolGuard "off" (raw pass-through):
// PII-bearing tool results reach the model byte-identical, with zero admissions
// and no in-band note. The right panel is the armed guard — ToolGuard "on": the
// SAME tool results run the kernel admission chain (normgate rank-5, ctxmmu
// rank-10), the email/phone/SSN/card/credential spans come back masked in place,
// the one-line "[fak] masked …" note renders IN-BAND on the Anthropic wire, and
// POST /v1/fak/admit shows the post-admission bytes actually admitted.
//
// Both panels are driven through the real gateway surface (POST /v1/messages,
// POST /v1/fak/admit) — no mock of the seams themselves. The model behind both
// arms is fak's deterministic offline mock planner, so the demo reproduces
// identically on any box with no weights and no network.
//
// Serve it (browser), or self-check it (headless — CI / cross-platform dogfood):
//
//	go run ./cmd/piidemo -addr 127.0.0.1:8158
//	# open http://127.0.0.1:8158 → "Run both arms"
//
//	go run ./cmd/piidemo -selfcheck
//	# browserless: replay the same scenario against both arms over the real HTTP
//	# seams, assert the documented raw-pass-through vs armed-floor invariants,
//	# print one PASS line per assertion, exit non-zero on any drift.
package main

import (
	"context"
	"embed"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cmdutil"
	"github.com/anthony-chaudhary/fak/internal/demoui"
	"github.com/anthony-chaudhary/fak/internal/demoutil"
	"github.com/anthony-chaudhary/fak/internal/gateway"

	// Blank-import the built-in driver list so the full result-admission chain
	// (normgate rank-5, ctxmmu rank-10, IFC stamp rank-20) and the offline mock
	// engine register before gateway.New boots either arm.
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

//go:embed page.html
var pageFS embed.FS

const version = "fak-piidemo-v1"

// ---------------------------------------------------------------------------
// The scenario: one synthetic tool-output stream carrying five protected spans.
// Every value below is FAKE (synthetic fixtures) — no real person, no real
// credential. Each span matches a detector in the armed floor's closed set
// (internal/canon): email, phone, national-id, payment-card, and the canonical
// secret shape the context-MMU tests carry.
// ---------------------------------------------------------------------------

// piiResult is the crm_lookup tool result: one email, one US phone, one SSN,
// one payment card, wrapped in realistic CRM prose. It is deliberately
// PII-ONLY so the armed floor's general-PII rung (PII_REDACTED transform) is
// what masks it; mixing a credential into the same body would route the whole
// result through the secret rung instead.
const piiResult = "CRM lookup returned record 4821: customer Jane Doe, contact jane.doe@example.com, " +
	"phone +1 (555) 013-2468, ssn 123-45-6789, card on file 4111-1111-1111-1111, last order 2026-08-14."

// secretResult is the env_dump tool result: one credential line in realistic
// deploy-config prose. The sk- shape is the canonical fixture the context-MMU
// tests use for the secret axis.
const secretResult = "env dump (staging): workers=4 region=us_east_1 log_level=info api_key=sk-abcdef0123456789abcdef0123"

const (
	armRaw     = "raw"
	armGuarded = "guarded"
	callPII    = "call_pii_demo"
	callSecret = "call_secret_demo"
)

// piiSpans are the four general-PII spans the armed floor must mask out of the
// crm_lookup result; secretSpans the credential span out of the env_dump result.
var piiSpans = []string{
	"jane.doe@example.com",
	"+1 (555) 013-2468",
	"123-45-6789",
	"4111-1111-1111-1111",
}
var secretSpans = []string{"sk-abcdef0123456789abcdef0123"}

// rawSpans is every protected span in the scenario — the set the raw arm must
// forward verbatim and the guarded arm must leave nowhere in the model-visible
// bytes.
var rawSpans = append(append([]string{}, piiSpans...), secretSpans...)

// scBlock mirrors the inbound Anthropic content-block subset the scenario uses
// (text, tool_use, tool_result); marshaling it keeps the request JSON exact.
type scBlock struct {
	Type      string         `json:"type"`
	Text      string         `json:"text,omitempty"`
	ID        string         `json:"id,omitempty"`
	Name      string         `json:"name,omitempty"`
	Input     map[string]any `json:"input,omitempty"`
	ToolUseID string         `json:"tool_use_id,omitempty"`
	Content   string         `json:"content,omitempty"`
}

type scMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

// scenarioBody renders the shared scenario as the exact Anthropic /v1/messages
// body BOTH arms receive: one user question, one assistant turn proposing two
// tools (crm_lookup + env_dump), and the user turn carrying both tool_results.
// Identical bytes on both arms — the only difference between the panels is the
// arm's ToolGuard posture.
func scenarioBody() (string, error) {
	msgs := []scMessage{
		{Role: "user", Content: "Pull Jane's customer record and the staging deploy config."},
		{Role: "assistant", Content: []scBlock{
			{Type: "text", Text: "checking both"},
			{Type: "tool_use", ID: callPII, Name: "crm_lookup", Input: map[string]any{"customer": "jane"}},
			{Type: "tool_use", ID: callSecret, Name: "env_dump", Input: map[string]any{"scope": "staging"}},
		}},
		{Role: "user", Content: []scBlock{
			{Type: "tool_result", ToolUseID: callPII, Content: piiResult},
			{Type: "tool_result", ToolUseID: callSecret, Content: secretResult},
		}},
	}
	b, err := json.Marshal(map[string]any{
		"model":      "piidemo",
		"max_tokens": 256,
		"messages":   msgs,
	})
	return string(b), err
}

// ---------------------------------------------------------------------------
// Wire views (the response shapes the demo reads off the real gateway).
// ---------------------------------------------------------------------------

type wireVerdict struct {
	Kind   string `json:"kind"`
	Reason string `json:"reason"`
	By     string `json:"by"`
}

type wireAdmission struct {
	ToolCallID string      `json:"tool_call_id"`
	Tool       string      `json:"tool"`
	Verdict    wireVerdict `json:"verdict"`
}

type wireAdj struct {
	Tool     string      `json:"tool"`
	Admitted bool        `json:"admitted"`
	Verdict  wireVerdict `json:"verdict"`
}

type fakExt struct {
	Adjudications    []wireAdj       `json:"adjudications"`
	ResultAdmissions []wireAdmission `json:"result_admissions"`
}

// outBlock is one rendered /v1/messages content block (text or tool_use).
type outBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
	ID   string `json:"id"`
	Name string `json:"name"`
}

// messagesOutcome is one /v1/messages round trip against an arm.
type messagesOutcome struct {
	status    int
	elapsedMS int64
	stop      string
	blocks    []outBlock
	notes     []string // every in-band [fak] text block, in wire order
	fak       *fakExt
}

// httpClient bounds a hung arm so a wedged kernel fold cannot hang the demo.
var httpClient = &http.Client{Timeout: 60 * time.Second}

// runMessages posts the shared scenario to the arm's /v1/messages and records
// the status, latency, content blocks, the in-band [fak] notes, and the fak
// extension (verdicts). A non-200 is returned as data (status + body tail in
// stop), not an error, so the page can show what the arm actually did.
func runMessages(ctx context.Context, a *arm) (*messagesOutcome, error) {
	body, err := scenarioBody()
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, a.ts.URL+"/v1/messages", strings.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	start := time.Now()
	res, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	raw, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, err
	}
	out := &messagesOutcome{status: res.StatusCode, elapsedMS: time.Since(start).Milliseconds()}
	if res.StatusCode != http.StatusOK {
		out.stop = "error: " + strings.TrimSpace(string(raw))
		return out, nil
	}
	var dec struct {
		StopReason string     `json:"stop_reason"`
		Content    []outBlock `json:"content"`
		Fak        *fakExt    `json:"fak"`
	}
	if err := json.Unmarshal(raw, &dec); err != nil {
		return nil, fmt.Errorf("decode /v1/messages body: %w", err)
	}
	out.stop = dec.StopReason
	out.blocks = dec.Content
	out.fak = dec.Fak
	for _, b := range dec.Content {
		if b.Type == "text" && strings.Contains(b.Text, "[fak]") {
			out.notes = append(out.notes, b.Text)
		}
	}
	return out, nil
}

// admitView is one /v1/fak/admit outcome: the verdict plus the post-admission
// bytes (result.content) the kernel actually admitted.
type admitView struct {
	Tool    string      `json:"tool"`
	Verdict wireVerdict `json:"verdict"`
	Status  string      `json:"status"`
	Content string      `json:"content"`
}

// postAdmit drives the explicit /v1/fak/admit verb: the caller-intent twin of
// the proxy's inbound admission. NOTE: it runs the kernel result-side stack on
// BOTH arms regardless of ToolGuard posture (an explicit call is explicit
// intent), which is exactly why the demo calls it ONLY on the guarded arm — on
// the raw arm the pass-through evidence is the chat response itself (zero
// admissions, no note, bytes untouched).
func postAdmit(ctx context.Context, a *arm, traceID, tool, result string) (*admitView, error) {
	payload, _ := json.Marshal(map[string]any{"tool": tool, "result": result, "trace_id": traceID})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, a.ts.URL+"/v1/fak/admit", strings.NewReader(string(payload)))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	raw, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, err
	}
	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", res.StatusCode, strings.TrimSpace(string(raw)))
	}
	var dec struct {
		Verdict wireVerdict `json:"verdict"`
		Result  struct {
			Status  string `json:"status"`
			Content string `json:"content"`
		} `json:"result"`
	}
	if err := json.Unmarshal(raw, &dec); err != nil {
		return nil, fmt.Errorf("decode /v1/fak/admit body: %w", err)
	}
	return &admitView{Tool: tool, Verdict: dec.Verdict, Status: dec.Result.Status, Content: dec.Result.Content}, nil
}

// ---------------------------------------------------------------------------
// Arms: two real gateway servers, each behind its own ephemeral loopback.
// ---------------------------------------------------------------------------

// arm is one live gateway server behind an ephemeral loopback listener — the
// real HTTP surface the browser (and -selfcheck) drive.
type arm struct {
	name string
	ts   *httptest.Server
	srv  *gateway.Server
}

// startArm boots one gateway with the given ToolGuard posture over the offline
// mock engine and serves it on an ephemeral loopback listener. Logf is silenced
// on purpose: the deterministic-mock warning is expected here and the page
// already says the model is scripted.
func startArm(name, model, toolGuard string) (*arm, error) {
	srv, err := gateway.New(gateway.Config{
		EngineID:  "mock",
		Model:     model,
		ToolGuard: toolGuard,
		Logf:      func(string, ...any) {},
	})
	if err != nil {
		return nil, err
	}
	ts := httptest.NewServer(srv.Handler())
	a := &arm{name: name, ts: ts, srv: srv}
	if ts.URL == "" {
		srv.Close()
		return nil, fmt.Errorf("loopback listener for %s did not bind", name)
	}
	return a, nil
}

func (a *arm) close() {
	a.ts.Close()
	a.srv.Close()
}

// ---------------------------------------------------------------------------
// Demo HTTP API.
// ---------------------------------------------------------------------------

// proof is the arm's headline statement: how many admissions came back, what
// the in-band note said, and whether the model-visible bytes are the input
// fixtures unchanged.
type proof struct {
	Admissions       int    `json:"admissions"`
	Note             string `json:"note"`
	ContentUnchanged bool   `json:"content_unchanged"`
}

// runResponse is what POST /run returns for one arm: the /v1/messages outcome,
// the post-admission bytes (guarded arm, via /v1/fak/admit), and the proof.
type runResponse struct {
	Arm           string            `json:"arm"`
	Status        int               `json:"status"`
	ElapsedMS     int64             `json:"elapsed_ms"`
	StopReason    string            `json:"stop_reason,omitempty"`
	Blocks        []outBlock        `json:"blocks"`
	Notes         []string          `json:"notes"`
	Admissions    []wireAdmission   `json:"admissions"`
	Adjudications []wireAdj         `json:"adjudications"`
	Proof         proof             `json:"proof"`
	ModelVisible  map[string]string `json:"model_visible"`
	Admits        []admitView       `json:"admits,omitempty"`
	Error         string            `json:"error,omitempty"`
}

// buildRun folds one /v1/messages outcome into the arm's /run payload. On the
// raw arm the model-visible bytes ARE the input fixtures (the router provably
// never touched them: zero admissions, no note); on the guarded arm they come
// from /v1/fak/admit — the post-admission bytes actually admitted.
func buildRun(ctx context.Context, a *arm, mo *messagesOutcome) *runResponse {
	resp := &runResponse{
		Arm:          a.name,
		Status:       mo.status,
		ElapsedMS:    mo.elapsedMS,
		StopReason:   mo.stop,
		Blocks:       mo.blocks,
		Notes:        mo.notes,
		ModelVisible: map[string]string{},
	}
	if resp.Blocks == nil {
		resp.Blocks = []outBlock{}
	}
	if resp.Notes == nil {
		resp.Notes = []string{}
	}
	if mo.fak != nil {
		resp.Admissions = mo.fak.ResultAdmissions
		resp.Adjudications = mo.fak.Adjudications
	}
	switch a.name {
	case armRaw:
		resp.Proof = proof{Admissions: 0, Note: "", ContentUnchanged: true}
		resp.ModelVisible["crm_lookup"] = piiResult
		resp.ModelVisible["env_dump"] = secretResult
	default:
		resp.Proof = proof{
			Admissions:       len(resp.Admissions),
			Note:             firstNote(mo.notes),
			ContentUnchanged: false,
		}
		for _, ad := range []struct{ tool, trace, fixture string }{
			{"crm_lookup", "piidemo-admit-crm", piiResult},
			{"env_dump", "piidemo-admit-env", secretResult},
		} {
			av, err := postAdmit(ctx, a, ad.trace, ad.tool, ad.fixture)
			if err != nil {
				resp.Error = "admit " + ad.tool + ": " + err.Error()
				continue
			}
			resp.Admits = append(resp.Admits, *av)
			resp.ModelVisible[ad.tool] = av.Content
		}
	}
	if resp.Admissions == nil {
		resp.Admissions = []wireAdmission{}
	}
	if resp.Adjudications == nil {
		resp.Adjudications = []wireAdj{}
	}
	return resp
}

func firstNote(notes []string) string {
	if len(notes) == 0 {
		return ""
	}
	return notes[0]
}

// handleScenario hands the page the shared fixtures (display-only; the
// server-side constants stay the source of truth for the wire).
func handleScenario(w http.ResponseWriter, r *http.Request) {
	cmdutil.WriteJSON(w, map[string]any{
		"pii_result":    piiResult,
		"secret_result": secretResult,
		"spans":         rawSpans,
	})
}

// handleRun executes the shared scenario against ONE arm's real gateway and
// returns the /run payload for it.
func handleRun(w http.ResponseWriter, r *http.Request, arms map[string]*arm) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed (POST)", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Arm string `json:"arm"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4096)).Decode(&req); err != nil {
		http.Error(w, "malformed body: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.Arm == "" {
		req.Arm = armRaw
	}
	a, ok := arms[req.Arm]
	if !ok {
		http.Error(w, "unknown arm "+strconv.Quote(req.Arm)+": want \"raw\" or \"guarded\"", http.StatusBadRequest)
		return
	}
	mo, err := runMessages(r.Context(), a)
	if err != nil {
		http.Error(w, "run "+req.Arm+": "+err.Error(), http.StatusBadGateway)
		return
	}
	cmdutil.WriteJSON(w, buildRun(r.Context(), a, mo))
}

// handleSelfcheck runs the headless witness in-process and reports it as JSON.
func handleSelfcheck(w http.ResponseWriter, r *http.Request) {
	lines, failures := runSelfcheckCore()
	cmdutil.WriteJSON(w, map[string]any{
		"pass":      failures == 0,
		"exit_code": selfcheckExit(failures),
		"lines":     lines,
	})
}

// ---------------------------------------------------------------------------
// -selfcheck: the browserless witness over the real HTTP seams.
// ---------------------------------------------------------------------------

func selfcheckExit(failures int) int {
	if failures > 0 {
		return 1
	}
	return 0
}

// runSelfcheck replays the shared scenario against BOTH arms over the real
// gateway HTTP seams and asserts the documented #14327 invariants:
//
//   - raw:     /v1/messages returns 200 with ZERO ResultAdmissions and NO
//     in-band [fak] note — the PII-bearing tool results reach the model
//     byte-identical (nothing intercepted, nothing masked, nothing noted).
//   - guarded: the same request carries >=1 TRANSFORM ResultAdmission with
//     reason PII_REDACTED or SECRET_REDACTED, an in-band "[fak] … masked …"
//     note renders on the wire, and /v1/fak/admit's post-admission bytes no
//     longer contain ANY raw protected span.
//
// Prints one PASS line per assertion; returns a process exit code (0 = held,
// 1 = any drift).
func runSelfcheck() int {
	fmt.Printf("== piidemo -selfcheck: the same PII-bearing tool results through two REAL gateway servers ==\n\n")
	lines, failures := runSelfcheckCore()
	for _, l := range lines {
		fmt.Println("  " + l)
	}
	fmt.Println()
	if failures > 0 {
		fmt.Printf("SELFCHECK FAILED — %d/%d assertion(s) drifted from the documented invariants\n", failures, len(lines))
		return 1
	}
	fmt.Printf("OK — %d/%d assertion(s) held: raw arm forwarded byte-identical, guarded arm masked every span\n", len(lines), len(lines))
	return 0
}

// runSelfcheckCore is the assertion body shared by -selfcheck, GET /selfcheck,
// and the package test. It starts both arms fresh, drives the identical
// scenario at each, and checks every documented invariant.
func runSelfcheckCore() ([]string, int) {
	lines := []string{}
	failures := 0
	pass := func(name string, ok bool, detail string) {
		if ok {
			lines = append(lines, "PASS "+name)
			return
		}
		lines = append(lines, "FAIL "+name+" — "+detail)
		failures++
	}

	rawArm, err := startArm(armRaw, "piidemo-raw", gateway.ToolGuardOff)
	if err != nil {
		return []string{"FAIL arms.raw — " + err.Error()}, 1
	}
	defer rawArm.close()
	guardedArm, err := startArm(armGuarded, "piidemo-guarded", gateway.ToolGuardOn)
	if err != nil {
		return append([]string{"FAIL arms.guarded — " + err.Error()}, lines...), failures + 1
	}
	defer guardedArm.close()

	ctx := context.Background()

	mo, err := runMessages(ctx, rawArm)
	if err != nil {
		pass("raw.messages_200", false, "request error: "+err.Error())
	} else {
		pass("raw.messages_200", mo.status == http.StatusOK, fmt.Sprintf("status=%d", mo.status))
		n := 0
		if mo.fak != nil {
			n = len(mo.fak.ResultAdmissions)
		}
		pass("raw.zero_admissions", n == 0, fmt.Sprintf("result_admissions=%d", n))
		pass("raw.no_fak_note", len(mo.notes) == 0, strings.Join(mo.notes, " | "))
	}

	goMo, err := runMessages(ctx, guardedArm)
	if err != nil {
		pass("guarded.messages_200", false, "request error: "+err.Error())
	} else {
		pass("guarded.messages_200", goMo.status == http.StatusOK, fmt.Sprintf("status=%d", goMo.status))
		transform := 0
		if goMo.fak != nil {
			for _, adm := range goMo.fak.ResultAdmissions {
				if adm.Verdict.Kind == "TRANSFORM" &&
					(adm.Verdict.Reason == "PII_REDACTED" || adm.Verdict.Reason == "SECRET_REDACTED") {
					transform++
				}
			}
		}
		pass("guarded.transform_verdicts", transform >= 1,
			fmt.Sprintf("transform_redactions=%d (want >=1 PII_REDACTED/SECRET_REDACTED)", transform))
		maskedNote := false
		for _, n := range goMo.notes {
			if strings.Contains(n, "[fak]") && strings.Contains(n, "masked") {
				maskedNote = true
			}
		}
		pass("guarded.fak_note", maskedNote, "no text block carries '[fak]' + 'masked'")
	}

	checkAdmit := func(name string, a *arm, trace, tool, fixture string, spans []string) {
		av, err := postAdmit(ctx, a, trace, tool, fixture)
		if err != nil {
			pass(name, false, "admit error: "+err.Error())
			return
		}
		if av.Verdict.Kind != "TRANSFORM" {
			pass(name, false, fmt.Sprintf("verdict=%s reason=%s (want TRANSFORM)", av.Verdict.Kind, av.Verdict.Reason))
			return
		}
		if av.Content == fixture {
			pass(name, false, "post-admission content is byte-identical to the fixture")
			return
		}
		for _, s := range spans {
			if strings.Contains(av.Content, s) {
				pass(name, false, "raw span still present after admission: "+s)
				return
			}
		}
		pass(name, true, "")
	}
	checkAdmit("guarded.pii_masked", guardedArm, "piidemo-selfcheck-pii", "crm_lookup", piiResult, piiSpans)
	checkAdmit("guarded.secret_masked", guardedArm, "piidemo-selfcheck-secret", "env_dump", secretResult, secretSpans)

	return lines, failures
}

// ---------------------------------------------------------------------------
// main.
// ---------------------------------------------------------------------------

func main() {
	const defaultAddr = "127.0.0.1:8158"
	addr := flag.String("addr", defaultAddr, "listen address")
	basePath := demoui.BasePathFlag(flag.CommandLine, "/piidemo")
	selfcheck := flag.Bool("selfcheck", false, "run HEADLESS: replay the shared scenario against both arms over the real gateway HTTP seams, assert the documented raw-pass-through vs armed-floor invariants, print one PASS line per assertion, and exit non-zero on any drift. No browser, no port.")
	flag.Parse()

	if *selfcheck {
		os.Exit(runSelfcheck())
	}

	rawArm, err := startArm(armRaw, "piidemo-raw", gateway.ToolGuardOff)
	if err != nil {
		fmt.Fprintln(os.Stderr, "raw arm:", err)
		os.Exit(1)
	}
	guardedArm, err := startArm(armGuarded, "piidemo-guarded", gateway.ToolGuardOn)
	if err != nil {
		fmt.Fprintln(os.Stderr, "guarded arm:", err)
		os.Exit(1)
	}

	app := http.NewServeMux()
	app.HandleFunc("/", demoutil.IndexHandler(pageFS))
	app.HandleFunc("/scenario", handleScenario)
	app.HandleFunc("/run", func(w http.ResponseWriter, r *http.Request) {
		handleRun(w, r, map[string]*arm{armRaw: rawArm, armGuarded: guardedArm})
	})
	app.HandleFunc("/selfcheck", handleSelfcheck)
	mux := http.NewServeMux()
	base := demoui.MountWithBasePath(mux, *basePath, app)

	bind := demoui.ListenAddr(*addr, defaultAddr)
	fmt.Fprintf(os.Stderr, "piidemo %s on %s\n", version, demoui.LocalURL(bind, base))
	fmt.Fprintf(os.Stderr, "two real gateway servers in-process: raw (ToolGuard off, the serve default) · guarded (ToolGuard on)\n")
	if base != "" {
		fmt.Fprintf(os.Stderr, "base path: %s (set by -base-path or %s)\n", base, demoui.DemoBasePathEnv)
	}
	fmt.Fprintf(os.Stderr, "open the URL → 'Run both arms' (raw pass-through vs PII-masked, side by side)\n")
	if err := http.ListenAndServe(bind, mux); err != nil {
		fmt.Fprintln(os.Stderr, "listen:", err)
		os.Exit(1)
	}
}
