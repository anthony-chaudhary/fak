package rsiloop

// cadence_test.go — the #2877 witnesses. The named acceptance witness kills a REAL
// process mid-session (a re-exec'd helper that appends session rows and dies via
// os.Exit with no shutdown handshake) and proves a fresh process, rebuilt from the
// durable ledger alone, derives the same counters and fires the next review exactly
// on schedule — the crash that resets Hermes' in-memory counters to 0 (their #22357)
// loses nothing here. The remaining tests pin the confusion-risk fence (derive on
// every read, never hydrate-once) and the pure fold itself.

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionledger"
)

const (
	cadenceWitnessDirEnv   = "FAK_CADENCE_WITNESS_DIR"
	cadenceWitnessTraceEnv = "FAK_CADENCE_WITNESS_TRACE"
	// cadenceWitnessDeathCode is the helper's deliberate hard-death exit code, so the
	// parent can tell "died mid-session as scripted" from a build/run failure.
	cadenceWitnessDeathCode = 3
)

// TestCadenceKillAndRebuildFiresReviewOnSchedule is the #2877 acceptance witness:
// a separate process runs the first half of a session (4 turns, a fired review,
// 3 more turns, 2 skill iterations) and is killed mid-session; a fresh process
// rebuilds from the durable ledger alone, derives the exact counters, and the next
// review fires at EXACTLY the scheduled boundary — not early, not never.
func TestCadenceKillAndRebuildFiresReviewOnSchedule(t *testing.T) {
	dir := t.TempDir()
	const trace = "sess-2877"

	// Phase 1 — the doomed process: re-exec this test binary so the session's first
	// half runs in a genuinely separate process image that dies hard (os.Exit skips
	// every defer/handshake). Durability must come from the ledger alone.
	cmd := exec.Command(os.Args[0], "-test.run=TestCadenceWitnessHelperProcess$")
	cmd.Env = append(os.Environ(),
		cadenceWitnessDirEnv+"="+dir,
		cadenceWitnessTraceEnv+"="+trace,
	)
	out, err := cmd.CombinedOutput()
	var ee *exec.ExitError
	if !errors.As(err, &ee) || ee.ExitCode() != cadenceWitnessDeathCode {
		t.Fatalf("helper process: want deliberate death exit %d, got err=%v output=%s",
			cadenceWitnessDeathCode, err, out)
	}

	// Phase 2 — the rebuilt process: open the ledger from disk with zero carried-over
	// memory and derive. An in-memory counter (Hermes' design) reads 0 here and the
	// scheduled review never fires; the ledger fold reads the truth.
	l, err := sessionledger.Open(dir)
	if err != nil {
		t.Fatalf("rebuild from durable ledger: %v", err)
	}
	got, err := SessionCadence(l, trace)
	if err != nil {
		t.Fatalf("SessionCadence after rebuild: %v", err)
	}
	want := Cadence{Turns: 7, TurnsSinceReview: 3, SkillIters: 2, ReviewsFired: 1}
	if got != want {
		t.Fatalf("cadence after kill+rebuild = %+v, want %+v (a zeroed counter is the Hermes bug)", got, want)
	}

	// Phase 3 — the session continues in the rebuilt process and the next review
	// fires ON SCHEDULE: with the default every-10-turns cadence and 3 since-review
	// turns already durable, the review is due after exactly 7 more turns — never
	// before (the counters were not lost, and not inflated either).
	cfg := DefaultCadenceConfig()
	for i := 0; i < 7; i++ {
		c, err := SessionCadence(l, trace)
		if err != nil {
			t.Fatalf("SessionCadence mid-continuation: %v", err)
		}
		if c.ReviewDue(cfg) {
			t.Fatalf("review due EARLY at %+v (want due only at %d since-review turns)", c, cfg.ReviewEveryTurns)
		}
		if _, err := l.Append(trace, CadenceKindTurnComplete, nil); err != nil {
			t.Fatalf("append continuation turn: %v", err)
		}
	}
	c, err := SessionCadence(l, trace)
	if err != nil {
		t.Fatalf("SessionCadence at schedule boundary: %v", err)
	}
	if c.TurnsSinceReview != cfg.ReviewEveryTurns || !c.ReviewDue(cfg) {
		t.Fatalf("next review did not fire on schedule after kill+rebuild: %+v (want due at %d)", c, cfg.ReviewEveryTurns)
	}

	// The hash chain survived the kill intact — the ledger's own integrity witness.
	entries, err := l.Chain(trace)
	if err != nil {
		t.Fatalf("Chain after rebuild: %v", err)
	}
	if err := sessionledger.Verify(entries); err != nil {
		t.Fatalf("ledger chain broken across process death: %v", err)
	}
}

// TestCadenceWitnessHelperProcess is the doomed half of the kill-and-rebuild
// witness. It is a no-op under a normal test run; re-exec'd by the witness with the
// env set, it appends the session's first half and dies hard mid-session.
func TestCadenceWitnessHelperProcess(t *testing.T) {
	dir := os.Getenv(cadenceWitnessDirEnv)
	if dir == "" {
		t.Skip("helper process: only run re-exec'd by TestCadenceKillAndRebuildFiresReviewOnSchedule")
	}
	trace := os.Getenv(cadenceWitnessTraceEnv)
	die := func(err error) {
		if err != nil {
			fmt.Fprintln(os.Stderr, "cadence witness helper:", err)
			os.Exit(1)
		}
	}
	l, err := sessionledger.Open(dir)
	die(err)
	for i := 0; i < 4; i++ {
		_, err := l.Append(trace, CadenceKindTurnComplete, nil)
		die(err)
	}
	die(MarkReviewFired(l, trace))
	for i := 0; i < 3; i++ {
		_, err := l.Append(trace, CadenceKindTurnComplete, nil)
		die(err)
	}
	for i := 0; i < 2; i++ {
		die(MarkSkillIter(l, trace))
	}
	os.Exit(cadenceWitnessDeathCode) // die mid-session: no handshake, no defers, no flush
}

// TestCadenceDerivedOnEveryReadNotHydratedOnce pins #2877's confusion-risk fence:
// the counters must be a fold over the ledger at each read, not a copy hydrated
// once at startup. New rows appended AFTER the first read move the very next read,
// through the same ledger handle, with no reopen.
func TestCadenceDerivedOnEveryReadNotHydratedOnce(t *testing.T) {
	l, err := sessionledger.Open(t.TempDir())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	const trace = "sess-live"
	if _, err := l.Append(trace, CadenceKindTurnComplete, nil); err != nil {
		t.Fatalf("append: %v", err)
	}
	first, err := SessionCadence(l, trace)
	if err != nil {
		t.Fatalf("first read: %v", err)
	}
	if want := (Cadence{Turns: 1, TurnsSinceReview: 1}); first != want {
		t.Fatalf("first read = %+v, want %+v", first, want)
	}

	// A hydrate-once snapshot would still answer `first` after these appends.
	if _, err := l.Append(trace, CadenceKindTurnComplete, nil); err != nil {
		t.Fatalf("append: %v", err)
	}
	if err := MarkSkillIter(l, trace); err != nil {
		t.Fatalf("MarkSkillIter: %v", err)
	}
	second, err := SessionCadence(l, trace)
	if err != nil {
		t.Fatalf("second read: %v", err)
	}
	if want := (Cadence{Turns: 2, TurnsSinceReview: 2, SkillIters: 1}); second != want {
		t.Fatalf("second read = %+v, want %+v (a stale value means hydrate-once, the Hermes patch shape)", second, want)
	}

	// A fired review resets BOTH since-review counters; the whole-session totals keep counting.
	if err := MarkReviewFired(l, trace); err != nil {
		t.Fatalf("MarkReviewFired: %v", err)
	}
	third, err := SessionCadence(l, trace)
	if err != nil {
		t.Fatalf("third read: %v", err)
	}
	if want := (Cadence{Turns: 2, ReviewsFired: 1}); third != want {
		t.Fatalf("post-review read = %+v, want %+v", third, want)
	}
}

// TestDeriveCadenceFoldAndSchedule pins the pure fold and the two schedule axes:
// unknown row kinds are ignored, the skill-iters axis fires independently of the
// turns axis, a disabled (non-positive) axis never fires, and an unknown trace is
// a fresh session (zero cadence, no error).
func TestDeriveCadenceFoldAndSchedule(t *testing.T) {
	entries := []sessionledger.Entry{
		{Kind: CadenceKindTurnComplete},
		{Kind: "turn_begin"}, // a non-cadence kind sharing the trace is ignored
		{Kind: CadenceKindSkillIter},
		{Kind: CadenceKindReviewFired},
		{Kind: CadenceKindSkillIter},
		{Kind: CadenceKindSkillIter},
		{Kind: CadenceKindTurnComplete},
	}
	got := DeriveCadence(entries)
	want := Cadence{Turns: 2, TurnsSinceReview: 1, SkillIters: 2, ReviewsFired: 1}
	if got != want {
		t.Fatalf("DeriveCadence = %+v, want %+v", got, want)
	}

	if got.ReviewDue(CadenceConfig{ReviewEveryTurns: 10, ReviewEverySkillIters: 2}) != true {
		t.Fatal("skill-iters axis at cadence must fire the review")
	}
	if got.ReviewDue(CadenceConfig{ReviewEveryTurns: 2, ReviewEverySkillIters: 0}) != false {
		t.Fatal("turns axis below cadence with skill axis disabled must not fire")
	}
	if got.ReviewDue(CadenceConfig{}) {
		t.Fatal("a fully disabled schedule must never fire")
	}

	l, err := sessionledger.Open(t.TempDir())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	fresh, err := SessionCadence(l, "never-seen-trace")
	if err != nil || fresh != (Cadence{}) {
		t.Fatalf("unknown trace: got %+v, %v; want zero cadence, nil error", fresh, err)
	}
}
