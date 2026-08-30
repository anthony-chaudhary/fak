package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchpost"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/perfrsiscore"
)

// loopCardSlack is a minimal httptest Slack Web API for the `fak loop run`
// card path: chat.postMessage + chat.update + conversations.history.
type loopCardSlack struct {
	mu     sync.Mutex
	nextTS int
	msgs   []loopCardMsg
}

func TestLoopRunAutomaticallyScoresPerformanceRSIOnceAtCompletion(t *testing.T) {
	oldNewCommand := loopNewCommand
	defer func() { loopNewCommand = oldNewCommand }()
	var childEnv []string
	loopNewCommand = func(argv, env []string, stdout, stderr io.Writer) loopCommand {
		childEnv = env
		return &fakeLoopCommand{pid: 9777}
	}
	t.Setenv(perfrsiscore.LoopTurnInputEnv, filepath.Join("..", "..", "internal", "perfrsiscore", "testdata", "complete.json"))
	usageLedger := filepath.Join(t.TempDir(), "performance-rsi-usage.jsonl")
	t.Setenv(perfrsiscore.UsageLedgerEnv, usageLedger)

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{
		"run",
		"--ledger", filepath.Join(t.TempDir(), "loops.jsonl"),
		"--loop", "dispatch/issues",
		"--run", "issue-9777-scored",
		"--no-guard",
		"--",
		"worker",
	})
	if code != 0 {
		t.Fatalf("runLoop code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if childEnv != nil {
		t.Fatalf("explicit %s behavior injected a child environment: %v", perfrsiscore.LoopTurnInputEnv, childEnv)
	}
	const marker = "fak loop run: performance-rsi loop-turn "
	if got := strings.Count(stderr.String(), marker); got != 1 {
		t.Fatalf("performance RSI invocation count=%d, want exactly 1:\n%s", got, stderr.String())
	}
	for _, want := range []string{`"schema":"fak-performance-rsi-loop-turn/1"`, `"status":"scored"`, `"reason":"SCORE_COMPLETE"`} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("loop-turn receipt missing %q:\n%s", want, stderr.String())
		}
	}
	fold, err := perfrsiscore.FoldUsage(usageLedger)
	if err != nil {
		t.Fatalf("fold performance RSI usage: %v", err)
	}
	if len(fold.Weeks) != 1 || fold.Weeks[0].Invocations != 1 || fold.Weeks[0].Scored != 1 || fold.Weeks[0].InvocationOutcomes.Success != 1 {
		t.Fatalf("performance RSI usage fold = %+v", fold)
	}
}

func TestLoopRunScoresMatchingRunScopedPerformanceRSIOutput(t *testing.T) {
	t.Setenv(perfrsiscore.LoopTurnInputEnv, "")
	t.Setenv("FAK_LOOP_RUN_HELPER", "performance-rsi")

	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{
		"run",
		"--ledger", ledger,
		"--loop", "dispatch/issues",
		"--run", "issue-10156-matching",
		"--no-guard",
		"--",
		os.Args[0], "-test.run=^TestLoopRunHelper$",
	})
	if code != 0 {
		t.Fatalf("runLoop code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}

	receipt := loopPerformanceRSIReceipt(t, stderr.String())
	if receipt.Status != perfrsiscore.LoopTurnScored || receipt.Reason != "SCORE_COMPLETE" || receipt.Snapshot != "issue-10156-matching" {
		t.Fatalf("matching output receipt = %+v", receipt)
	}
	if receipt.InvocationOutcomes.Success != 1 || receipt.InvocationOutcomes.Refusal != 0 || receipt.InvocationOutcomes.Error != 0 || receipt.InvocationOutcomes.Total() != 1 {
		t.Fatalf("matching output invocation outcomes = %+v", receipt.InvocationOutcomes)
	}
	if receipt.Input == "" || filepath.Dir(receipt.Input) != filepath.Dir(ledger) {
		t.Fatalf("run-scoped output %q is not beside ledger %q", receipt.Input, ledger)
	}
	events, err := loopmgr.Load(ledger)
	if err != nil {
		t.Fatalf("load loop ledger: %v", err)
	}
	if got := gotKinds(events); got != "fire,admit,start,end" {
		t.Fatalf("automatic score changed loop events: got %s", got)
	}
}

func TestLoopRunRejectsCrossRunPerformanceRSIOutput(t *testing.T) {
	t.Setenv(perfrsiscore.LoopTurnInputEnv, "")
	t.Setenv("FAK_LOOP_RUN_HELPER", "performance-rsi")
	t.Setenv("FAK_LOOP_TEST_RSI_SNAPSHOT", "another-run")

	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{
		"run",
		"--ledger", ledger,
		"--loop", "dispatch/issues",
		"--run", "issue-10156-current",
		"--no-guard",
		"--",
		os.Args[0], "-test.run=^TestLoopRunHelper$",
	})
	if code != 0 {
		t.Fatalf("cross-run output changed dispatch code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	receipt := loopPerformanceRSIReceipt(t, stderr.String())
	if receipt.Status != perfrsiscore.LoopTurnUnavailable || receipt.Reason != "SCORE_INPUT_UNAVAILABLE" {
		t.Fatalf("cross-run output receipt = %+v", receipt)
	}
	if receipt.InvocationOutcomes.Success != 0 || receipt.InvocationOutcomes.Refusal != 1 || receipt.InvocationOutcomes.Error != 0 || receipt.InvocationOutcomes.Total() != 1 {
		t.Fatalf("cross-run invocation outcomes = %+v", receipt.InvocationOutcomes)
	}
	if !strings.Contains(receipt.UnavailableDiagnostic, `snapshot "another-run" does not match loop run "issue-10156-current"`) {
		t.Fatalf("cross-run diagnostic = %q", receipt.UnavailableDiagnostic)
	}
	events, err := loopmgr.Load(ledger)
	if err != nil {
		t.Fatalf("load loop ledger: %v", err)
	}
	if got := gotKinds(events); got != "fire,admit,start,end" {
		t.Fatalf("cross-run score changed loop events: got %s", got)
	}
}

func TestLoopRunPreservesDispatchWhenPerformanceRSIInputUnavailable(t *testing.T) {
	oldNewCommand := loopNewCommand
	defer func() { loopNewCommand = oldNewCommand }()
	loopNewCommand = func(argv, env []string, stdout, stderr io.Writer) loopCommand {
		return &fakeLoopCommand{pid: 9777}
	}
	t.Setenv(perfrsiscore.LoopTurnInputEnv, "")
	usageLedger := filepath.Join(t.TempDir(), "performance-rsi-usage.jsonl")
	t.Setenv(perfrsiscore.UsageLedgerEnv, usageLedger)

	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{
		"run",
		"--ledger", ledger,
		"--loop", "dispatch/issues",
		"--run", "issue-9777-unavailable",
		"--json",
		"--no-guard",
		"--",
		"worker",
	})
	var report struct {
		ExitCode int `json:"exit_code"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unavailable score input corrupted loop JSON: %v\nstdout=%s\nstderr=%s", err, stdout.String(), stderr.String())
	}
	if code != 0 || report.ExitCode != 0 {
		t.Fatalf("unavailable score input changed dispatch result: code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	events, err := loopmgr.Load(ledger)
	if err != nil {
		t.Fatalf("load loop ledger: %v", err)
	}
	if got := gotKinds(events); got != "fire,admit,start,end" {
		t.Fatalf("unavailable score input changed loop events: got %s", got)
	}
	const marker = "fak loop run: performance-rsi loop-turn "
	if got := strings.Count(stderr.String(), marker); got != 1 {
		t.Fatalf("performance RSI invocation count=%d, want exactly 1:\n%s", got, stderr.String())
	}
	for _, want := range []string{`"status":"unavailable"`, `"reason":"SCORE_INPUT_UNAVAILABLE"`, `"unavailable_diagnostic":"FAK_PERFORMANCE_RSI_OUTPUT was not produced for this run"`, `"invocation_outcomes":{"success":0,"refusal":1,"error":0}`} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("explicit unavailable receipt missing %q:\n%s", want, stderr.String())
		}
	}
	receipt := loopPerformanceRSIReceipt(t, stderr.String())
	if receipt.InvocationOutcomes.Total() != 1 || receipt.InvocationOutcomes.Refusal != 1 {
		t.Fatalf("missing output invocation outcomes = %+v", receipt.InvocationOutcomes)
	}
	fold, err := perfrsiscore.FoldUsage(usageLedger)
	if err != nil {
		t.Fatalf("fold unavailable performance RSI usage: %v", err)
	}
	if len(fold.Weeks) != 1 || fold.Weeks[0].Invocations != 1 || fold.Weeks[0].Unavailable != 1 || fold.Weeks[0].InvocationOutcomes.Refusal != 1 {
		t.Fatalf("unavailable performance RSI usage fold = %+v", fold)
	}
}

func TestScoreAutomaticLoopPerformanceRSIRejectsUnsafeOutputShapes(t *testing.T) {
	tests := []struct {
		name  string
		setup func(t *testing.T, path string)
	}{
		{
			name: "zero-size regular file",
			setup: func(t *testing.T, path string) {
				t.Helper()
				if err := os.WriteFile(path, nil, 0o600); err != nil {
					t.Fatalf("write empty output: %v", err)
				}
			},
		},
		{
			name: "nonregular directory",
			setup: func(t *testing.T, path string) {
				t.Helper()
				if err := os.Mkdir(path, 0o700); err != nil {
					t.Fatalf("create output directory: %v", err)
				}
			},
		},
		{
			name: "oversized regular file",
			setup: func(t *testing.T, path string) {
				t.Helper()
				f, err := os.Create(path)
				if err != nil {
					t.Fatalf("create oversized output: %v", err)
				}
				if err := f.Truncate(loopPerformanceRSIMaxBytes + 1); err != nil {
					_ = f.Close()
					t.Fatalf("truncate oversized output: %v", err)
				}
				if err := f.Close(); err != nil {
					t.Fatalf("close oversized output: %v", err)
				}
			},
		},
		{name: "missing file", setup: func(t *testing.T, path string) {}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "performance-rsi.json")
			tt.setup(t, path)
			receipt := scoreAutomaticLoopPerformanceRSI(path, "run-current", nil)
			if receipt.Status != perfrsiscore.LoopTurnUnavailable || receipt.Reason != "SCORE_INPUT_UNAVAILABLE" {
				t.Fatalf("receipt status/reason = %q/%q, want unavailable/SCORE_INPUT_UNAVAILABLE", receipt.Status, receipt.Reason)
			}
			counts := receipt.InvocationOutcomes
			if counts.Success != 0 || counts.Refusal != 1 || counts.Error != 0 || counts.Total() != 1 {
				t.Fatalf("invocation outcomes = %+v, want exactly one refusal", counts)
			}
		})
	}
}

func writeLoopPerformanceRSIOutput(target, snapshot string) error {
	f, err := os.Open(filepath.Join("..", "..", "internal", "perfrsiscore", "testdata", "complete.json"))
	if err != nil {
		return err
	}
	evidence, err := perfrsiscore.Decode(f)
	closeErr := f.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	evidence.Snapshot = snapshot
	body, err := json.Marshal(evidence)
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(target), ".performance-rsi-child-*.json")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(body); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, target)
}

func loopPerformanceRSIReceipt(t *testing.T, stderr string) perfrsiscore.LoopTurnReceipt {
	t.Helper()
	const marker = "fak loop run: performance-rsi loop-turn "
	if got := strings.Count(stderr, marker); got != 1 {
		t.Fatalf("performance RSI invocation count=%d, want exactly 1:\n%s", got, stderr)
	}
	encoded := strings.SplitN(strings.SplitN(stderr, marker, 2)[1], "\n", 2)[0]
	var receipt perfrsiscore.LoopTurnReceipt
	if err := json.Unmarshal([]byte(encoded), &receipt); err != nil {
		t.Fatalf("decode performance RSI receipt: %v\n%s", err, encoded)
	}
	return receipt
}

type loopCardMsg struct {
	Channel  string
	TS       string
	ThreadTS string
	Text     string
}

func (f *loopCardSlack) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/chat.postMessage", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Channel  string `json:"channel"`
			Text     string `json:"text"`
			ThreadTS string `json:"thread_ts"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		f.mu.Lock()
		f.nextTS++
		ts := fmt.Sprintf("%d.0", f.nextTS)
		f.msgs = append(f.msgs, loopCardMsg{Channel: body.Channel, TS: ts, ThreadTS: body.ThreadTS, Text: body.Text})
		f.mu.Unlock()
		fmt.Fprintf(w, `{"ok":true,"ts":%q}`, ts)
	})
	mux.HandleFunc("/chat.update", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Channel string `json:"channel"`
			TS      string `json:"ts"`
			Text    string `json:"text"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		f.mu.Lock()
		for i := range f.msgs {
			if f.msgs[i].TS == body.TS && f.msgs[i].Channel == body.Channel {
				f.msgs[i].Text = body.Text
			}
		}
		f.mu.Unlock()
		fmt.Fprint(w, `{"ok":true}`)
	})
	mux.HandleFunc("/conversations.history", func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		msgs := make([]map[string]any, 0, len(f.msgs))
		for _, m := range f.msgs {
			msgs = append(msgs, map[string]any{"type": "message", "ts": m.TS, "text": m.Text, "thread_ts": m.ThreadTS})
		}
		f.mu.Unlock()
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "messages": msgs})
	})
	return mux
}

func (f *loopCardSlack) topLevel(channel string) []loopCardMsg {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []loopCardMsg
	for _, m := range f.msgs {
		if m.Channel == channel && m.ThreadTS == "" {
			out = append(out, m)
		}
	}
	return out
}

func (f *loopCardSlack) replies(channel, ts string) []loopCardMsg {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []loopCardMsg
	for _, m := range f.msgs {
		if m.Channel == channel && m.ThreadTS == ts {
			out = append(out, m)
		}
	}
	return out
}

// TestLoopRunCardPostsOnceAndFinalizesInPlace drives the REAL `fak loop run`
// Slack path (openDispatchRunCard -> postDispatchResult) against an httptest
// fake: the dispatch channel gets ONE card at run start, and the run's end
// EDITS that same message into the witness fold — with the full result body
// threaded under it — instead of appending a terminal message (#2263).
func TestLoopRunCardPostsOnceAndFinalizesInPlace(t *testing.T) {
	slack := &loopCardSlack{}
	srv := httptest.NewServer(slack.handler())
	defer srv.Close()
	oldBase := dispatchAPIBase
	dispatchAPIBase = srv.URL + "/"
	defer func() { dispatchAPIBase = oldBase }()
	t.Setenv("FAK_SLACK_OUTBOX_DIR", filepath.Join(t.TempDir(), "outbox"))

	var stderr bytes.Buffer
	res := dispatchpost.Result{LoopID: "nightly", RunID: "r-7", Command: "job.ps1"}
	// An explicit destination is notification intent even without --notify-slack.
	card := openDispatchRunCardIfRequested(&stderr, false, "C7", "test-token", res)
	if card == nil {
		t.Fatalf("card did not arm: %s", stderr.String())
	}
	top := slack.topLevel("C7")
	if len(top) != 1 || !strings.Contains(top[0].Text, "running `job.ps1`") {
		t.Fatalf("run start must post one card: %+v (%s)", top, stderr.String())
	}

	// Run ends: equal HEADs, exit 0 — an honest check-result, nothing landed.
	res.ExitCode = 0
	res.HeadBefore, res.HeadAfter = "abc1234", "abc1234"
	postDispatchResult(&stderr, false, "C7", "test-token", card, res)

	top = slack.topLevel("C7")
	if len(top) != 1 {
		t.Fatalf("channel must stay one line per run, got %d: %+v (%s)", len(top), top, stderr.String())
	}
	if !strings.Contains(top[0].Text, "dispatch nightly · run `r-7` — NOT_SHIPPED") ||
		!strings.Contains(top[0].Text, "verify=none · exit=0") {
		t.Fatalf("final edit must carry the witness fold: %q", top[0].Text)
	}
	reps := slack.replies("C7", top[0].TS)
	if len(reps) != 1 || !strings.Contains(reps[0].Text, "dispatch result nightly") {
		t.Fatalf("result body must ride in the thread: %+v", reps)
	}
	if !strings.Contains(stderr.String(), "dispatch run card finalized") {
		t.Fatalf("card path did not report success: %s", stderr.String())
	}
}

// TestLoopRunCardAmbientChannelDoesNotOptIn is the transport witness for #4677:
// a machine-wide operator channel may exist for roll-ups, but an ordinary loop that did
// not request per-run notification must produce ZERO top-level or threaded Slack traffic.
func TestLoopRunCardAmbientChannelDoesNotOptIn(t *testing.T) {
	slack := &loopCardSlack{}
	srv := httptest.NewServer(slack.handler())
	defer srv.Close()
	oldBase := dispatchAPIBase
	dispatchAPIBase = srv.URL + "/"
	defer func() { dispatchAPIBase = oldBase }()
	t.Setenv("FAK_DISPATCH_CHANNEL", "C-operator")
	t.Setenv("FAK_DISPATCH_TOKEN", "test-token")
	t.Setenv("FAK_SLACK_OUTBOX_DIR", filepath.Join(t.TempDir(), "outbox"))

	var stderr bytes.Buffer
	res := dispatchpost.Result{LoopID: "scheduler-test", RunID: "r-ambient", Command: "test.ps1"}
	card := openDispatchRunCardIfRequested(&stderr, false, "", "", res)
	if card != nil {
		t.Fatal("ambient FAK_DISPATCH_CHANNEL armed a per-run card without explicit opt-in")
	}
	postDispatchResult(&stderr, false, "", "", card, res)
	if got := len(slack.topLevel("C-operator")); got != 0 {
		t.Fatalf("ambient loop emitted %d top-level operator messages, want 0", got)
	}
	if stderr.Len() != 0 {
		t.Fatalf("silent no-op wrote stderr: %s", stderr.String())
	}
}

// TestLoopRunCardUnarmedWithoutChannel keeps the unconfigured-box behavior:
// no dispatch channel means no card and no error.
func TestLoopRunCardUnarmedWithoutChannel(t *testing.T) {
	t.Setenv("FAK_DISPATCH_CHANNEL", "")
	t.Setenv("FAK_SLACK_OUTBOX_DIR", filepath.Join(t.TempDir(), "outbox"))
	// Channel resolution falls back to a .env.slack.local walked up from the
	// cwd; run from an empty dir so a configured dev box stays hermetic.
	t.Chdir(t.TempDir())
	var stderr bytes.Buffer
	if card := openDispatchRunCard(&stderr, "", "", dispatchpost.Result{LoopID: "l", RunID: "r"}); card != nil {
		t.Fatalf("card must stay unarmed without a channel (stderr: %s)", stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("unarmed card must be silent: %s", stderr.String())
	}
}
