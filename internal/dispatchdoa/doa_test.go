package dispatchdoa

import (
	"strings"
	"testing"
)

// doaLogFixture is a VERBATIM copy of a real outage record —
// .dispatch-runs/resolve-2419-20260728-213124.log, 1589 bytes, one of the 350 units
// burned between 2026-07-28 and 2026-08-03 (#5868). Every one of the 350 is
// byte-shaped like this: the dispatcher's pre-exec `# fak-spawn` header, then Go's
// flag package rejecting an argv the binary did not define, then the usage block.
// No guard launch banner, no turn, no commit. Its .witness recorded
// `claim: CLAIM_NO_COMMIT, reason: unknown` — the bucket that hid the outage.
const doaLogFixture = `# fak-spawn 20260728-213124 issue=2419 lane=gateway backend=claude argv0=fak.exe
flag provided but not defined: -compact-solvency-floor
usage: fak guard [flags] -- <agent command...>
  e.g. fak guard -- claude
       fak guard --provider openai -- codex
       fak guard --policy my-floor.json -- claude
       fak guard allow <tool>   # operator: always-allow a blocked tool, out-of-band (fak guard allow -h)

common flags:
  --policy         capability-floor manifest to enforce (default: the built-in floor; see --dump-policy)
  --provider       upstream wire: anthropic|openai|gemini|xai (default: auto-detected from the agent name)
  --api-key-env    opt IN to API billing using this env var (default: your subscription / passthrough)
  --probe          one-shot smoke mode: prove the guarded wire without requiring a task handoff
  --gguf           run a small model in-kernel, no API key or network (e.g. --gguf qwen2.5:7b)
  --local          auto-detect an already-running local model server (Ollama / LM Studio / llama.cpp)
  --log            write per-request + per-verdict structured logs to a file (or '-' for stderr)
  --audit          change where the decision journal is written ('off' disables it)
  --quiet          suppress the startup banner and the exit audit summary
  --dump-policy    print the built-in capability floor (an editable manifest) and exit
  --split          auto|on|off: a live fak-info pane beside the agent (auto-on inside a multiplexer)

81 flags in this build. 'fak guard -h -all' lists every one grouped; docs/fak/api-reference.md has the deep dive.
`

// healthyLogFixture is the HEAD of a real post-outage record —
// .dispatch-runs/resolve-1348-20260807-055116.log, 2026-08-07, from the clean
// 08-04-onward window. The discriminating line is the third one: the guard's
// `— kernel-adjudicated:` agent-launch banner. Its .witness ALSO recorded
// `claim: CLAIM_NO_COMMIT, reason: unknown` — same witness verdict as the outage
// runs, which is precisely why the witness reason could not have told the two apart
// and the detector must key off the log's own recorded launch evidence.
const healthyLogFixture = `# fak-spawn 20260807-055116 issue=1348 lane=gateway backend=claude argv0=fak.exe
fak guard: account "aug5-netra" is cooling down — rotating to "aug5TWO-netra" (resets 2026-08-07T06:16:04Z) (headroom=room)
fak guard: fleetspine: self-discovery on (group 239.255.70.65:4765 as "desktop-bb3fmhp")
fak guard 0.43.0 — kernel-adjudicated: claude -p --permission-mode bypassPermissions --model claude-opus-5 --effort xhigh
  build      : b225bb1ca20f
  gateway    : http://127.0.0.1:56430   (in-process; torn down when the command exits)
  upstream   : https://api.anthropic.com   (via the anthropic wire)
`

// TestOutageFixtureClassifiesDOA is the FIRES direction: the real 2026-07-28 record
// must be named, with the argv-drift cause, not left in the residual bucket.
func TestOutageFixtureClassifiesDOA(t *testing.T) {
	got := Classify(doaLogFixture, int64(len(doaLogFixture)))
	if !got.DOA || got.Cause != CauseFlagParse {
		t.Fatalf("Classify(real 20260728 outage log) = %+v, want {DOA:true Cause:%s}", got, CauseFlagParse)
	}
	if n := len(doaLogFixture); int64(n) > StubMaxBytes {
		t.Fatalf("fixture is %d bytes, over the %d-byte stub floor — the floor no longer covers the real corpus", n, StubMaxBytes)
	}
}

// TestHealthyFixtureNeverDOA is the SILENT direction, and the false-positive guard
// that matters most: a real post-outage run from the clean 08-04+ window, carrying
// the SAME `reason: unknown` witness as the outage runs, must not be labelled DOA.
func TestHealthyFixtureNeverDOA(t *testing.T) {
	// Full-size, as it lands on disk (the real record is 12 KiB).
	if got := Classify(healthyLogFixture, 12476); got.DOA {
		t.Fatalf("Classify(real healthy log) = %+v, want DOA:false", got)
	}
	// And still not DOA if it were somehow stub-sized: the launch banner is the
	// discriminator, so size alone can never manufacture a DOA verdict.
	if got := Classify(healthyLogFixture, int64(len(healthyLogFixture))); got.DOA {
		t.Fatalf("Classify(healthy head at stub size) = %+v, want DOA:false — the launch banner must veto", got)
	}
}

// TestFastLegitimateRunNotDOA pins the discriminator the issue calls out by name: a
// worker that started, did essentially nothing and exited immediately is NOT DOA. It
// reached launch, so it printed the banner; the agent simply chose to land nothing.
// Confusing this with a spawn death would blame the dispatcher for a worker's own
// verdict — the expensive false positive.
func TestFastLegitimateRunNotDOA(t *testing.T) {
	fast := "# fak-spawn 20260805-010101 issue=99 lane=docs backend=claude argv0=fak.exe\n" +
		"fak guard 0.43.0 — kernel-adjudicated: claude -p\n" +
		"nothing to do; exiting\n"
	if got := Classify(fast, int64(len(fast))); got.DOA {
		t.Fatalf("Classify(fast legitimate run) = %+v, want DOA:false", got)
	}
	// Same run with the banner suppressed but a turn trace present: still launched.
	turned := "# fak-spawn 20260805-010101 issue=99 lane=docs backend=claude argv0=fak.exe\n" +
		"fak-turn in=120 out=8 cost=0.001\n"
	if got := Classify(turned, int64(len(turned))); got.DOA {
		t.Fatalf("Classify(turn-trace run) = %+v, want DOA:false", got)
	}
}

// TestShapeNotKeyedToOneFlag proves the detector generalizes past
// -compact-solvency-floor: the next drift will be a different flag, a missing binary
// or a bad cwd, and an unrecognized stub is still reported as DOA rather than dropped.
func TestShapeNotKeyedToOneFlag(t *testing.T) {
	hdr := "# fak-spawn 20260901-000000 issue=1 lane=x backend=claude argv0=fak.exe\n"
	cases := []struct {
		name, body, want string
	}{
		{"a different undefined flag", "flag provided but not defined: -some-future-knob\nusage: fak guard\n", CauseFlagParse},
		{"cobra-style unknown flag", "Error: unknown flag: --turbo\n", CauseFlagParse},
		{"binary missing", "fork/exec C:\\bin\\fak.exe: executable file not found in %PATH%\n", CauseExecFailure},
		{"wrong architecture", "exec format error\n", CauseExecFailure},
		{"bad working directory", "chdir C:\\work\\gone: no such file or directory\n", CauseWorkingDir},
		{"unknown subcommand", "usage: fak guard [flags] -- <agent command...>\n", CauseUsageError},
		{"missing evidence", "", CauseMissingEvidence},
		{"shape holds, signature unknown", "some brand new startup fault nobody has seen\n", CauseUnrecognized},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			log := hdr + tc.body
			got := Classify(log, int64(len(log)))
			if !got.DOA || got.Cause != tc.want {
				t.Fatalf("Classify(%q) = %+v, want {DOA:true Cause:%s}", tc.body, got, tc.want)
			}
		})
	}
}

// TestFailOpenGates pins every way the classifier must decline to accuse.
func TestFailOpenGates(t *testing.T) {
	stub := "# fak-spawn 20260901-000000 issue=1 lane=x backend=claude argv0=fak.exe\nflag provided but not defined: -x\n"
	cases := []struct {
		name string
		head string
		size int64
	}{
		{"unstat-able log", stub, -1},
		{"over the stub floor", stub, StubMaxBytes + 1},
		{"no dispatcher spawn header", "flag provided but not defined: -x\n", 60},
		{"foreign artifact", "some other tool's output\n", 25},
		{"empty log", "", 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := Classify(tc.head, tc.size); got.DOA {
				t.Fatalf("Classify(%s) = %+v, want DOA:false (fail-open)", tc.name, got)
			}
		})
	}
}

// TestFoldAlarmsOnOutageDayOne replays the FIRST day of the outage — 2026-07-28,
// 30 DOA of 34 runs, re-derived from the retained corpus — and requires the alarm
// rung to fire. Day one is the whole point: the real outage ran six more days
// because nothing said this.
func TestFoldAlarmsOnOutageDayOne(t *testing.T) {
	var runs []Run
	for i := 0; i < 30; i++ {
		runs = append(runs, Run{Log: "resolve-1-20260728.log", Verdict: Verdict{DOA: true, Cause: CauseFlagParse}})
	}
	for i := 0; i < 4; i++ {
		runs = append(runs, Run{Log: "resolve-2-20260728.log"})
	}
	rep := Fold(runs)
	if rep.Runs != 34 || rep.DOA != 30 {
		t.Fatalf("Fold day-one = %d DOA of %d runs, want 30 of 34", rep.DOA, rep.Runs)
	}
	if rep.Status != StatusAlarm {
		t.Fatalf("Fold day-one status = %q, want %q (rate %.3f)", rep.Status, StatusAlarm, rep.Rate)
	}
	if rep.Rate < 0.88 || rep.Rate > 0.883 {
		t.Fatalf("Fold day-one rate = %.4f, want ~0.882", rep.Rate)
	}
	if got := strings.Join(rep.TopCauses(), " "); got != CauseFlagParse+"=30" {
		t.Fatalf("TopCauses = %q, want %q", got, CauseFlagParse+"=30")
	}
}

func TestClassifyOperationalFailureFamilies(t *testing.T) {
	tests := []struct {
		name string
		log  string
		want string
	}{
		{"auth invalid verdict", "worker preflight refused: AUTH_INVALID", CauseAuthInvalid},
		{"invalid refresh token", `401 Unauthorized: {"code":"invalid_refresh_token"}`, CauseAuthInvalid},
		{"guarded upstream credential", "upstream rejected the credential (HTTP 401)", CauseAuthInvalid},
		{"process start", "CreateProcess failed to start worker", CauseProcessStart},
		{"immediate exit", "harness crashed (NONZERO_EXIT, exit 1)", CauseImmediateExit},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			log := SpawnHeaderPrefix + "20260821 issue=1 lane=dispatch backend=codex argv0=fak.exe\n" + tc.log
			got := Classify(log, int64(len(log)))
			if !got.DOA || got.Cause != tc.want {
				t.Fatalf("Classify(%q) = %+v, want DOA cause %q", tc.log, got, tc.want)
			}
		})
	}
}

func TestFoldProvidesActionForEveryCause(t *testing.T) {
	runs := []Run{
		{Log: "auth.log", Verdict: Verdict{DOA: true, Cause: CauseAuthInvalid}},
		{Log: "novel.log", Verdict: Verdict{DOA: true, Cause: CauseUnrecognized, Signature: "sha256:0123456789abcdef"}},
	}
	rep := Fold(runs)
	for cause := range rep.Causes {
		if strings.TrimSpace(rep.NextActions[cause]) == "" {
			t.Fatalf("cause %q has no next action: %+v", cause, rep)
		}
	}
	if got := rep.Diagnostics["novel.log"]; got != "sha256:0123456789abcdef" {
		t.Fatalf("diagnostic = %q, want scrubbed signature", got)
	}
}

func TestUnknownSignatureIsStableAndScrubbed(t *testing.T) {
	const secret = "sk-secret-material"
	log := "# fak-spawn 20260821 issue=8406\nnovel startup failure token=" + secret + "\n"
	got := Classify(log, int64(len(log)))
	if !got.DOA || got.Cause != CauseUnrecognized || !strings.HasPrefix(got.Signature, "sha256:") {
		t.Fatalf("verdict = %+v, want fingerprinted unrecognized DOA", got)
	}
	if strings.Contains(got.Signature, secret) || got.Signature != Classify(log, int64(len(log))).Signature {
		t.Fatalf("signature must be stable and scrubbed: %q", got.Signature)
	}
}

// TestFoldClearOnHealthyWindow is the other direction: the 08-04-onward window held
// 287 runs and 0 DOA, so the fold must stay silent. A detector that cries on a
// healthy fleet is worse than none.
func TestFoldClearOnHealthyWindow(t *testing.T) {
	runs := make([]Run, 287)
	for i := range runs {
		runs[i] = Run{Log: "resolve-9-20260805.log"}
	}
	rep := Fold(runs)
	if rep.DOA != 0 || rep.Rate != 0 || rep.Status != StatusClear {
		t.Fatalf("Fold(healthy 287-run window) = %+v, want 0 DOA / clear", rep)
	}
	if len(rep.Sample) != 0 || len(rep.Causes) != 0 {
		t.Fatalf("Fold(healthy) named evidence %v/%v, want none", rep.Sample, rep.Causes)
	}
}

// TestFoldRungs pins the ladder, including the single-run guard: one DOA in a
// one-run window is 100% but must WARN, not ALARM, so a fluke cannot page.
func TestFoldRungs(t *testing.T) {
	doa := Run{Log: "a.log", Verdict: Verdict{DOA: true, Cause: CauseFlagParse}}
	ok := Run{Log: "b.log"}
	cases := []struct {
		name string
		runs []Run
		want string
	}{
		{"empty window", nil, StatusClear},
		{"all healthy", []Run{ok, ok, ok, ok}, StatusClear},
		{"a single DOA in a busy window", []Run{doa, ok, ok, ok, ok, ok}, StatusWarn},
		{"one of one", []Run{doa}, StatusWarn},
		{"two of two", []Run{doa, doa}, StatusWarn},
		{"three of three", []Run{doa, doa, doa}, StatusAlarm},
		{"three of six", []Run{doa, doa, doa, ok, ok, ok}, StatusAlarm},
		{"three of seven", []Run{doa, doa, doa, ok, ok, ok, ok}, StatusWarn},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := Fold(tc.runs).Status; got != tc.want {
				t.Fatalf("Fold(%s).Status = %q, want %q", tc.name, got, tc.want)
			}
		})
	}
}

// TestFoldSampleBounded keeps the operator line short and deterministic while the
// count stays exact.
func TestFoldSampleBounded(t *testing.T) {
	var runs []Run
	for i := 0; i < 50; i++ {
		runs = append(runs, Run{Log: string(rune('a'+i%26)) + ".log", Verdict: Verdict{DOA: true, Cause: CauseFlagParse}})
	}
	rep := Fold(runs)
	if rep.DOA != 50 {
		t.Fatalf("DOA = %d, want 50", rep.DOA)
	}
	if len(rep.Sample) != SampleMax {
		t.Fatalf("Sample len = %d, want %d", len(rep.Sample), SampleMax)
	}
	for i := 1; i < len(rep.Sample); i++ {
		if rep.Sample[i-1] > rep.Sample[i] {
			t.Fatalf("Sample not sorted: %v", rep.Sample)
		}
	}
}
