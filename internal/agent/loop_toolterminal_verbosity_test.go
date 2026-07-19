package agent

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/toolproc"
	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

const wakeTrace = "owned-session"

func doneProc(call string) toolproc.Proc {
	return toolproc.Proc{
		CallID: call, Tool: "shell", Session: wakeTrace,
		State: toolproc.StateDone, ExitStatus: "ok", EndMS: 9, RuntimeMS: 8,
		Pulses: 3, Liveness: toolproc.LivenessLive,
		Findings: []toolproc.Finding{{Reason: "TOOL_RAN_LONG", Advice: toolproc.AdviceProbe, Detail: "ran long"}},
	}
}

func killedProc(call string) toolproc.Proc {
	return toolproc.Proc{
		CallID: call, Tool: "shell", Session: wakeTrace,
		State: toolproc.StateKilled, KillReason: "TOOL_DEADLINE_EXCEEDED", EndMS: 9, RuntimeMS: 8,
		Findings: []toolproc.Finding{{Reason: "TOOL_DEADLINE_EXCEEDED", Advice: toolproc.AdviceKill, Detail: "overdue"}},
	}
}

func statuses(q *ToolTerminalWakeQueue) []string {
	var out []string
	for _, r := range q.Journal() {
		out = append(out, r.Status)
	}
	return out
}

// TestToolTerminalVerbosityGatesTheWake is the #2932 acceptance witness: the
// configured verbosity decides whether a completed background process wakes a
// turn at all, and a suppressed wake stays inspectable in the journal.
func TestToolTerminalVerbosityGatesTheWake(t *testing.T) {
	cases := []struct {
		name      string
		verbosity ToolTerminalVerbosity
		proc      toolproc.Proc
		wantWake  bool
	}{
		{"off-suppresses-success", ToolTerminalVerbosityOff, doneProc("c1"), false},
		{"off-suppresses-failure", ToolTerminalVerbosityOff, killedProc("c2"), false},
		{"error-suppresses-success", ToolTerminalVerbosityError, doneProc("c3"), false},
		{"error-admits-killed", ToolTerminalVerbosityError, killedProc("c4"), true},
		{"result-admits-success", ToolTerminalVerbosityResult, doneProc("c5"), true},
		{"all-admits-success", ToolTerminalVerbosityAll, doneProc("c6"), true},
		{"zero-value-admits-success", "", doneProc("c7"), true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			q := NewToolTerminalWakeQueue(wakeTrace)
			TerminalWakeSink(q, tc.verbosity)(tc.proc)

			got := statuses(q)
			if !tc.wantWake {
				if len(got) != 1 || got[0] != "SUPPRESSED" {
					t.Fatalf("verbosity %q: journal=%v want one SUPPRESSED", tc.verbosity, got)
				}
				select {
				case <-q.signal:
					t.Fatalf("verbosity %q raised a wake signal for a suppressed verdict", tc.verbosity)
				default:
				}
				return
			}
			if len(got) != 1 || got[0] != "ENQUEUED" {
				t.Fatalf("verbosity %q: journal=%v want one ENQUEUED", tc.verbosity, got)
			}
			select {
			case <-q.signal:
			default:
				t.Fatalf("verbosity %q admitted a verdict but raised no wake signal", tc.verbosity)
			}
		})
	}
}

// TestToolTerminalVerbosityProjectsDetail pins the DETAIL axis: `result` carries
// the outcome only, while `all` and `error` carry the diagnostic findings a
// caller woken for a failure needs in order to act.
func TestToolTerminalVerbosityProjectsDetail(t *testing.T) {
	q := NewToolTerminalWakeQueue(wakeTrace)
	TerminalWakeSink(q, ToolTerminalVerbosityResult)(doneProc("c1"))
	got := q.Journal()[0].Wake.Verdict
	if got.CallID != "c1" || got.State != toolproc.StateDone || got.ExitStatus != "ok" {
		t.Fatalf("result projection lost the outcome: %+v", got)
	}
	if len(got.Findings) != 0 || got.Pulses != 0 || got.Liveness != "" {
		t.Fatalf("result projection kept the diagnostic surface: %+v", got)
	}

	full := NewToolTerminalWakeQueue(wakeTrace)
	TerminalWakeSink(full, ToolTerminalVerbosityError)(killedProc("c2"))
	kept := full.Journal()[0].Wake.Verdict
	if len(kept.Findings) != 1 || kept.KillReason != "TOOL_DEADLINE_EXCEEDED" {
		t.Fatalf("error verbosity dropped the diagnosis it exists to deliver: %+v", kept)
	}

	all := NewToolTerminalWakeQueue(wakeTrace)
	TerminalWakeSink(all, ToolTerminalVerbosityAll)(doneProc("c3"))
	if !reflect.DeepEqual(all.Journal()[0].Wake.Verdict, doneProc("c3")) {
		t.Fatalf("all verbosity is not the verbatim historical verdict")
	}
}

// TestTerminalWakeSinkWiresIntoSupervisor drives the real seam: an `error`-level
// sink registered on a live toolprocgate.Supervisor lets a genuine background
// process exit through Tick and confirms the successful completion is suppressed
// while a killed one wakes the turn. This is the primitive #2932 asks for —
// fire-and-forget, woken on exit, at the configured verbosity.
func TestTerminalWakeSinkWiresIntoSupervisor(t *testing.T) {
	q := NewToolTerminalWakeQueue(wakeTrace)
	sup := toolprocgate.NewSupervisor(toolproc.Config{})
	sup.SetTerminalSink(TerminalWakeSink(q, ToolTerminalVerbosityError))

	if err := sup.Spawn("ok-job", "shell", wakeTrace, 0, 0, 1, nil); err != nil {
		t.Fatal(err)
	}
	if err := sup.Exit("ok-job", 2, "ok"); err != nil {
		t.Fatal(err)
	}
	if _, err := sup.Tick(2); err != nil {
		t.Fatal(err)
	}
	if got := statuses(q); len(got) != 1 || got[0] != "SUPPRESSED" {
		t.Fatalf("clean exit under `error` verbosity: journal=%v want one SUPPRESSED", got)
	}

	// A job that blows its deadline is killed by the fold and MUST wake the turn.
	if err := sup.Spawn("doomed", "shell", wakeTrace, 5, 0, 1, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := sup.Tick(1_000); err != nil {
		t.Fatal(err)
	}
	got := statuses(q)
	if len(got) != 2 || got[1] != "ENQUEUED" {
		t.Fatalf("killed job under `error` verbosity: journal=%v want a trailing ENQUEUED", got)
	}
	woke := q.Journal()[1].Wake
	if woke.TraceID != "doomed" || woke.Verdict.State != toolproc.StateKilled {
		t.Fatalf("wrong verdict woke the turn: %+v", woke)
	}
}

func TestParseToolTerminalVerbosity(t *testing.T) {
	for in, want := range map[string]ToolTerminalVerbosity{
		"":       ToolTerminalVerbosityAll,
		"all":    ToolTerminalVerbosityAll,
		"result": ToolTerminalVerbosityResult,
		"error":  ToolTerminalVerbosityError,
		"off":    ToolTerminalVerbosityOff,
	} {
		got, ok := ParseToolTerminalVerbosity(in)
		if !ok || got != want {
			t.Fatalf("Parse(%q)=(%q,%v) want (%q,true)", in, got, ok, want)
		}
	}
	// A typo is refused, never silently defaulted: defaulting a misconfigured
	// "erorr" to `all` would make a session unexpectedly chatty with no signal.
	if got, ok := ParseToolTerminalVerbosity("erorr"); ok {
		t.Fatalf("Parse(\"erorr\")=(%q,true) want refused", got)
	}
}

// TestToolTerminalWakeCarriesNoRawProcessOutput pins the adjudication invariant
// #2932 names: the completion injected into a resumed turn is the kernel's
// FOLDED verdict, never the background process's own bytes. toolproc.Proc has no
// stdout/stderr/output channel at all, so the guarantee is structural — this test
// fails the moment someone widens Proc with a raw-output field, which is exactly
// when the quarantine argument would stop holding.
func TestToolTerminalWakeCarriesNoRawProcessOutput(t *testing.T) {
	rawish := map[string]bool{
		"stdout": true, "stderr": true, "output": true, "body": true,
		"content": true, "text": true, "log": true, "logs": true, "result": true,
	}
	rt := reflect.TypeOf(toolproc.Proc{})
	for i := 0; i < rt.NumField(); i++ {
		tag := strings.Split(rt.Field(i).Tag.Get("json"), ",")[0]
		if rawish[tag] {
			t.Fatalf("toolproc.Proc gained raw-process-output field %q — the terminal wake "+
				"would now splice unadjudicated process bytes into the resumed turn", tag)
		}
	}

	// And the rendered payload the loop splices carries only the folded verdict.
	q := NewToolTerminalWakeQueue(wakeTrace)
	q.Enqueue(doneProc("c1"))
	payload, err := json.Marshal(q.Journal()[0].Wake)
	if err != nil {
		t.Fatal(err)
	}
	var round map[string]any
	if err := json.Unmarshal(payload, &round); err != nil {
		t.Fatal(err)
	}
	if round["kind"] != ToolTerminalWakeKind {
		t.Fatalf("spliced payload is not a typed wake: %s", payload)
	}
}
