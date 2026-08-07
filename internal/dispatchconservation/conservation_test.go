package dispatchconservation

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// now is the fixed reporting clock; every fixture stamp is relative to it.
var now = time.Date(2026, 7, 3, 12, 0, 0, 0, time.UTC)

func stamp(hoursAgo float64) string {
	return now.Add(-time.Duration(hoursAgo * float64(time.Hour))).Format("20060102-150405")
}

func iso(hoursAgo float64) string {
	return now.Add(-time.Duration(hoursAgo * float64(time.Hour))).Format("2006-01-02T15:04:05Z")
}

type workerOpts struct {
	kind              string
	lane              string
	backend           string
	body              string
	bodyExplicitEmpty bool
	witness           map[string]any
	pid               *int
}

func mkWorker(t *testing.T, runs string, issue int, hoursAgo float64, opts workerOpts) string {
	t.Helper()
	if opts.kind == "" {
		opts.kind = "resolve"
	}
	if opts.lane == "" {
		opts.lane = "tools"
	}
	if opts.backend == "" {
		opts.backend = "claude"
	}
	body := opts.body
	if opts.body == "" && !opts.bodyExplicitEmpty {
		body = "worker output\n"
	}
	name := opts.kind + "-" + strconv.Itoa(issue) + "-" + stamp(hoursAgo) + ".log"
	log := filepath.Join(runs, name)
	header := "# fak-spawn " + stamp(hoursAgo) + " issue=" + strconv.Itoa(issue) +
		" lane=" + opts.lane + " backend=" + opts.backend + " argv0=claude.exe\n"
	if err := os.WriteFile(log, []byte(header+body), 0o644); err != nil {
		t.Fatalf("write log: %v", err)
	}
	if opts.witness != nil {
		b, _ := json.Marshal(opts.witness)
		if err := os.WriteFile(sidecar(log, ".witness"), b, 0o644); err != nil {
			t.Fatalf("write witness: %v", err)
		}
	}
	if opts.pid != nil {
		if err := os.WriteFile(sidecar(log, ".pid"), []byte(strconv.Itoa(*opts.pid)), 0o644); err != nil {
			t.Fatalf("write pid: %v", err)
		}
	}
	return log
}

// bodyExplicitEmpty lets a fixture write a header-only (spawn-failed) log; the
// default empty body string means "use the standard worker output".
func (o workerOpts) withEmptyBody() workerOpts { o.bodyExplicitEmpty = true; return o }

func aliveSet(pids ...int) AliveProbe {
	m := map[int]bool{}
	for _, p := range pids {
		m[p] = true
	}
	return AliveProbe{Scanned: true, PIDs: m}
}

func report(runs string, alive AliveProbe, windowH float64) Report {
	since := now.Add(-time.Duration(windowH * float64(time.Hour)))
	units := CollectUnits(runs, since, alive)
	closes := WindowedCloses(filepath.Join(runs, "progress.jsonl"), since)
	holds := WindowedContractHolds(filepath.Join(runs, "contract-holds.jsonl"), since)
	return FoldConservation(units, closes, holds, windowH, iso(0))
}

func ptr(i int) *int { return &i }

func TestStampParsesAsUTC(t *testing.T) {
	ts, ok := ParseLogStampUTC("resolve-42-" + stamp(1.0) + ".log")
	if !ok {
		t.Fatal("expected a parseable stamp")
	}
	if got := ts.Sub(now.Add(-time.Hour)); got < -time.Second || got > time.Second {
		t.Errorf("stamp off by %v", got)
	}
	if _, ok := ParseLogStampUTC("resolve-42-garbage.log"); ok {
		t.Error("garbage stamp should not parse")
	}
	if _, ok := ParseLogStampUTC("unrelated.log"); ok {
		t.Error("non-worker log should not parse")
	}
}

func TestWindowKeysOnSpawnStampNotMtime(t *testing.T) {
	runs := t.TempDir()
	inside := mkWorker(t, runs, 1, 2.0, workerOpts{witness: map[string]any{"claim": "CLAIM_WITNESSED", "sha": strings.Repeat("a", 40)}})
	outside := mkWorker(t, runs, 2, 30.0, workerOpts{witness: map[string]any{"claim": "CLAIM_WITNESSED", "sha": strings.Repeat("b", 40)}})
	// A fresh mtime on an OLD unit must not pull it into the window.
	_ = os.Chtimes(outside, now, now)
	_ = os.Chtimes(inside, now, now)
	rep := report(runs, aliveSet(), 6.0)
	if rep.Units.ResolveTotal != 1 {
		t.Errorf("resolve_total = %d, want 1", rep.Units.ResolveTotal)
	}
	if rep.Units.ShippedWitnessed != 1 {
		t.Errorf("shipped_witnessed = %d, want 1", rep.Units.ShippedWitnessed)
	}
}

func TestEveryBucketAndIdentity(t *testing.T) {
	runs := t.TempDir()
	mkWorker(t, runs, 10, 1.0, workerOpts{witness: map[string]any{"claim": "CLAIM_WITNESSED", "sha": strings.Repeat("a", 40)}})
	mkWorker(t, runs, 11, 1.1, workerOpts{witness: map[string]any{"claim": "CLAIM_UNWITNESSED", "sha": strings.Repeat("b", 40), "verdict": "ABSTAIN"}})
	mkWorker(t, runs, 12, 1.2, workerOpts{witness: map[string]any{"claim": "CLAIM_NO_COMMIT", "sha": nil, "reason": "policy_block"}})
	mkWorker(t, runs, 13, 1.3, workerOpts{}.withEmptyBody()) // header-only: spawn failed
	mkWorker(t, runs, 14, 1.4, workerOpts{})                 // dead, real log, no witness: LEAK
	mkWorker(t, runs, 15, 1.5, workerOpts{pid: ptr(4242)})   // alive pid: live, not spent
	rep := report(runs, aliveSet(4242), 6.0)
	u := rep.Units
	if u.ResolveTotal != 6 || u.Live != 1 || u.Spent != 5 {
		t.Fatalf("totals: resolve=%d live=%d spent=%d, want 6/1/5", u.ResolveTotal, u.Live, u.Spent)
	}
	if u.ShippedWitnessed != 1 || u.CommittedUnwitnessed != 1 || u.NoCommit != 1 {
		t.Errorf("shipped=%d unwitnessed=%d no_commit=%d, want 1/1/1", u.ShippedWitnessed, u.CommittedUnwitnessed, u.NoCommit)
	}
	if u.SpawnFailed != 1 || u.LeakedUnswept != 1 {
		t.Errorf("spawn_failed=%d leaked=%d, want 1/1", u.SpawnFailed, u.LeakedUnswept)
	}
	if u.NoCommitReasons["policy_block"] != 1 || len(u.NoCommitReasons) != 1 {
		t.Errorf("no_commit_reasons = %v, want {policy_block:1}", u.NoCommitReasons)
	}
	if !rep.IdentityHolds {
		t.Error("identity should hold")
	}
	if rep.Verdict != "LEAKING" {
		t.Errorf("verdict = %s, want LEAKING", rep.Verdict)
	}
	if len(rep.LeakedUnits) != 1 || rep.LeakedUnits[0].Issue != 14 {
		t.Errorf("leaked_units = %+v, want issue 14", rep.LeakedUnits)
	}
}

func TestDeadPidWithNoWitnessIsLeak(t *testing.T) {
	runs := t.TempDir()
	mkWorker(t, runs, 20, 1.0, workerOpts{pid: ptr(999)}) // pid sidecar survives but pid is dead
	rep := report(runs, aliveSet(1), 6.0)
	if rep.Units.LeakedUnswept != 1 || rep.Units.Live != 0 {
		t.Errorf("leaked=%d live=%d, want 1/0", rep.Units.LeakedUnswept, rep.Units.Live)
	}
}

func TestUnscannableHostCountsPidUnitsLive(t *testing.T) {
	runs := t.TempDir()
	mkWorker(t, runs, 21, 1.0, workerOpts{pid: ptr(999)})
	rep := report(runs, AliveProbe{Scanned: false}, 6.0) // host scan unavailable
	if rep.Units.Live != 1 || rep.Units.LeakedUnswept != 0 {
		t.Errorf("live=%d leaked=%d, want 1/0", rep.Units.Live, rep.Units.LeakedUnswept)
	}
	if rep.Verdict != "CONSERVED" {
		t.Errorf("verdict = %s, want CONSERVED", rep.Verdict)
	}
}

func TestWitnessOutranksLivePid(t *testing.T) {
	runs := t.TempDir()
	mkWorker(t, runs, 22, 1.0, workerOpts{pid: ptr(4242),
		witness: map[string]any{"claim": "CLAIM_NO_COMMIT", "sha": nil, "reason": "auth_wall"}})
	rep := report(runs, aliveSet(4242), 6.0)
	if rep.Units.NoCommit != 1 || rep.Units.Live != 0 {
		t.Errorf("no_commit=%d live=%d, want 1/0", rep.Units.NoCommit, rep.Units.Live)
	}
}

func TestUnknownWitnessReasonFoldsToUnknown(t *testing.T) {
	runs := t.TempDir()
	mkWorker(t, runs, 23, 1.0, workerOpts{witness: map[string]any{"claim": "CLAIM_NO_COMMIT", "sha": nil, "reason": "not-a-real-bucket"}})
	rep := report(runs, aliveSet(), 6.0)
	if rep.Units.NoCommitReasons["unknown"] != 1 || len(rep.Units.NoCommitReasons) != 1 {
		t.Errorf("no_commit_reasons = %v, want {unknown:1}", rep.Units.NoCommitReasons)
	}
}

// TestModelSwitchableReasonsAccountedNotFolded pins that the Layer-2 model-switchable
// classes (usage_cap / model_unknown / rate_limit) are first-class no-commit buckets,
// not folded into "unknown" — so conservation accounting names where each unit went.
func TestModelSwitchableReasonsAccountedNotFolded(t *testing.T) {
	runs := t.TempDir()
	mkWorker(t, runs, 31, 1.0, workerOpts{witness: map[string]any{"claim": "CLAIM_NO_COMMIT", "sha": nil, "reason": "usage_cap"}})
	mkWorker(t, runs, 32, 1.0, workerOpts{witness: map[string]any{"claim": "CLAIM_NO_COMMIT", "sha": nil, "reason": "model_unknown"}})
	mkWorker(t, runs, 33, 1.0, workerOpts{witness: map[string]any{"claim": "CLAIM_NO_COMMIT", "sha": nil, "reason": "rate_limit"}})
	rep := report(runs, aliveSet(), 6.0)
	r := rep.Units.NoCommitReasons
	if r["usage_cap"] != 1 || r["model_unknown"] != 1 || r["rate_limit"] != 1 || r["unknown"] != 0 {
		t.Errorf("no_commit_reasons = %v, want each switchable class = 1 and no unknown", r)
	}
}

// TestPythonSweepReasonsAccountedNotFolded pins the classes that ONLY the Python
// witness sweep stamps (tools/issue_resolve_dispatch.py: NO_COMMIT_RESTART_EXHAUSTED,
// NO_COMMIT_PREVIEW_CONFIRM, NO_COMMIT_MISSING_LOG) as first-class buckets. That sweep
// writes the very .witness sidecars this package reads, but its vocabulary is wider
// than internal/dispatchtick's NoCommit* consts, and noCommitReasons had only copied
// the dispatchtick half — so these three hit the fold and were booked as "unknown".
//
// Regression witnessed on this repo's own .dispatch-runs, 78h window ending
// 2026-08-07T06:19:45Z: the sidecars held unknown=143 + restart_exhausted=26, and
// `fak dispatch-conservation --window-h 78` reported `unknown=169` with
// restart_exhausted missing from the breakdown — 15% of the window's no-commit units
// relabelled from a named, actionable guard terminal to the residual bucket.
func TestPythonSweepReasonsAccountedNotFolded(t *testing.T) {
	runs := t.TempDir()
	mkWorker(t, runs, 34, 1.0, workerOpts{witness: map[string]any{"claim": "CLAIM_NO_COMMIT", "sha": nil, "reason": "restart_exhausted"}})
	mkWorker(t, runs, 35, 1.0, workerOpts{witness: map[string]any{"claim": "CLAIM_NO_COMMIT", "sha": nil, "reason": "preview_confirm_feedback"}})
	mkWorker(t, runs, 36, 1.0, workerOpts{witness: map[string]any{"claim": "CLAIM_NO_COMMIT", "sha": nil, "reason": "missing_log_artifact"}})
	rep := report(runs, aliveSet(), 6.0)
	r := rep.Units.NoCommitReasons
	if r["restart_exhausted"] != 1 || r["preview_confirm_feedback"] != 1 || r["missing_log_artifact"] != 1 || r["unknown"] != 0 {
		t.Errorf("no_commit_reasons = %v, want each python-sweep class = 1 and no unknown", r)
	}
}

// guardEpilogueFixture is a worker log tail carrying the guard exit summary: two real
// section rules from cmd/fak/guard_format_layout.go guardSection(). The second one is
// deliberately the "cache window" section #5867 proposed keying on, so the
// zero-cache-turn fixture below can drop it while keeping a complete epilogue.
const guardEpilogueFixture = "" +
	"fak-turn trace=win-deadbeef ok prov=80.0k tok cache=healthy_cache\n" +
	"── guard · audit journal ──────────────────────────────\n" +
	"  appended                  38 decision(s) appended this session\n" +
	"── guard · cache window ───────────────────────────────\n" +
	"  recorded                  20 turn(s)\n"

// TestUnknownNoCommitSplitByGuardEpilogue is the #5867 fail-before/pass-after pin: the
// witness sweep's residual "unknown" — the fleet's LARGEST terminal disposition — must
// be split by the evidence the worker log tail already carries, instead of booking half
// the fleet under the one label that means "we could not tell".
//
// Measured on this repo's own .dispatch-runs over the clean current-fleet window
// (2026-08-04T00:00Z..08-07T06:15Z; the 07-28..08-03 -compact-solvency-floor spawn
// outage is excluded on purpose, a trailing-7d slice would measure that instead):
// 283 graded resolve units, of which unknown=149 (52.8% of runs, 55.2 of 102.3
// wall-to-witness seat-hours = 54.0%). `fak dispatch-conservation --window-h 78.6`
// printed a single opaque `unknown=149`. The same 149 split 114 / 30 / 2 on the tail
// markers pinned here — no new log retention needed, the sweep's own 4 KiB window
// already holds every marker (re-running the split at 24 KiB moves zero units).
func TestUnknownNoCommitSplitByGuardEpilogue(t *testing.T) {
	runs := t.TempDir()
	noCommit := map[string]any{"claim": "CLAIM_NO_COMMIT", "sha": nil, "reason": "unknown"}
	// A session that reached its guard epilogue and simply landed nothing.
	mkWorker(t, runs, 40, 1.0, workerOpts{body: guardEpilogueFixture, witness: noCommit})
	// A session killed MID-TURN: the log ends on an in-flight fak-turn row with no
	// guard summary at all. 30/30 of the real died-before-epilogue units look like this.
	mkWorker(t, runs, 41, 1.0, workerOpts{
		body:    "fak-turn trace=win-cafebabe ok prov=104.4k tok (89% of prompt)\n",
		witness: noCommit})
	// The guard never exec'd the agent. Checked FIRST: this run also prints a partial
	// epilogue, and "the child never ran" is the more specific, more fixable statement.
	mkWorker(t, runs, 42, 1.0, workerOpts{
		body: guardEpilogueFixture +
			"fak guard: could not run \"claude\": snapshot generated child config\n",
		witness: noCommit})

	r := report(runs, aliveSet(), 6.0).Units.NoCommitReasons
	if r[ReasonCleanExitNoCommit] != 1 || r[ReasonDiedBeforeEpilogue] != 1 ||
		r[ReasonGuardChildSpawnFailed] != 1 || r["unknown"] != 0 {
		t.Errorf("no_commit_reasons = %v, want one of each split class and NO residual unknown", r)
	}
}

// TestCleanExitKeysOnSectionRuleNotCacheWindow refutes the marker #5867 proposed. That
// issue keys "epilogue present" on the final `guard · cache window` section, but
// formatVCacheSnapshotPointer (cmd/fak/guard_child_supervision.go) returns "" when the
// session recorded zero cache turns — so a complete epilogue with no cached turn carries
// no cache-window section and would be mis-booked as a death.
//
// This is not hypothetical: over the full retained history (2382 graded resolve units)
// the cache-window marker finds 0/69 auth_wall runs, every one of which carries an
// epilogue the section rule finds (69/69). Inside the unknown bucket the same gate hides
// 7 runs. Keying on the section rule instead costs nothing and loses none of them.
func TestCleanExitKeysOnSectionRuleNotCacheWindow(t *testing.T) {
	runs := t.TempDir()
	zeroTurnEpilogue := "" +
		"fak-turn trace=win-deadbeef ok prov=80.0k tok cache=n/a\n" +
		"── guard · audit journal ──────────────────────────────\n" +
		"  appended                  12 decision(s) appended this session\n" +
		"  Track 2 OBSERVED-$ row not written: no provider-cache tokens\n"
	if strings.Contains(zeroTurnEpilogue, "cache window") {
		t.Fatal("fixture must NOT contain the cache-window section; that is the point")
	}
	mkWorker(t, runs, 43, 1.0, workerOpts{body: zeroTurnEpilogue,
		witness: map[string]any{"claim": "CLAIM_NO_COMMIT", "sha": nil, "reason": "unknown"}})
	r := report(runs, aliveSet(), 6.0).Units.NoCommitReasons
	if r[ReasonCleanExitNoCommit] != 1 || r[ReasonDiedBeforeEpilogue] != 0 {
		t.Errorf("no_commit_reasons = %v, want a zero-cache-turn epilogue read as a CLEAN EXIT", r)
	}
}

// bulkyEpilogue pads a fixture epilogue past a byte budget with real guard section
// rules, so a test can prove a marker survives being pushed AWAY from EOF by the
// summary the guard prints after it. The live epilogue is a median 7199 bytes over the
// 118 measured clean-exit units (112 of 118 exceed 4096), so this is not a synthetic
// worry: it is the actual geometry of every worker log the sweep grades.
func bulkyEpilogue(t *testing.T, atLeast int) string {
	t.Helper()
	var b strings.Builder
	b.WriteString("── guard · avoided-spend attribution ──────────────────\n")
	for b.Len() < atLeast {
		b.WriteString("  owner split               provider ~1.4M (100%) + fak ~0 (0%)\n")
	}
	b.WriteString(guardEpilogueFixture)
	return b.String()
}

// TestCleanExitSplitsProviderQuotaAndWallClockTerminals is the fail-before/pass-after pin
// for #5870. clean_exit_no_commit — 118 of 291 graded resolve units over the clean
// current-fleet window (2026-08-04..08-07), ~41% of fleet seat-time — was read as a
// THROUGHPUT problem: sessions that ran a full ~26 minutes and simply landed nothing.
// It is not. 105 of those 118 (89.0%) end on a TYPED guard terminal the ledger was not
// looking for:
//
//   - 67 (56.8%) hit a provider quota wall: a 429 whose Retry-After exceeds
//     goalpark.LongWaitFloor parks the goal with reason=LONG_RETRY_AFTER
//     (internal/goalpark.RecordLongRetry), and/or the account is dropped from the
//     servable pool by cmd/fak/accounts_cooldown.go. The turn stream shows
//     `FAILED reason=rate_limited ... kind=weekly_limit announced_wait=1h13m40s`.
//   - 38 (32.2%) hit the wall-clock envelope: dispatch always passes --max-duration
//     (1740s) and the guard drains the session with TIME_BUDGET_EXHAUSTED
//     (internal/session.ReasonTimeBudgetExhausted).
//   - 13 (11.0%) are the residual — a session that really did end quietly with nothing.
//
// The two named classes are DISJOINT in the measured window (a parked goal tears down
// immediately, so it never also reaches the 29m envelope), and neither is an agent
// behaviour the worker prompt can fix: one is purchased capacity, one is a configured
// envelope. Booking them as "the agent under-produced" pointed the fleet's largest
// remaining loss at the wrong lever.
func TestCleanExitSplitsProviderQuotaAndWallClockTerminals(t *testing.T) {
	runs := t.TempDir()
	noCommit := map[string]any{"claim": "CLAIM_NO_COMMIT", "sha": nil, "reason": "unknown"}
	// A 429 weekly-limit park. The terminal line sits ABOVE the epilogue, exactly as it
	// does on disk.
	mkWorker(t, runs, 50, 1.0, workerOpts{witness: noCommit, body: "" +
		"fak guard: PARKED goal=\"compute\" parked_until=1785815999 reason=LONG_RETRY_AFTER\n" +
		"fak-turn trace=win-611ca8 FAILED reason=rate_limited kind=weekly_limit announced_wait=1h13m40s\n" +
		bulkyEpilogue(t, 6000)})
	// An account dropped from the servable pool by a live usage cap, with no park line.
	// 2 of the 67 look only like this.
	mkWorker(t, runs, 51, 1.0, workerOpts{witness: noCommit, body: "" +
		"fak guard: account cooled by a live usage cap until 2026-08-04T01:35:27Z — it drops from the servable pool\n" +
		bulkyEpilogue(t, 6000)})
	// The wall-clock envelope.
	mkWorker(t, runs, 52, 1.0, workerOpts{witness: noCommit, body: "" +
		"fak guard: TIME_BUDGET_EXHAUSTED — wall-clock --max-duration envelope elapsed for compute-claude-54880\n" +
		bulkyEpilogue(t, 6000)})
	// The residual: a complete epilogue and no typed terminal at all. This one must KEEP
	// clean_exit_no_commit — the class has to keep meaning "we looked and found nothing",
	// or the split has just moved the blindness somewhere else.
	mkWorker(t, runs, 53, 1.0, workerOpts{witness: noCommit, body: bulkyEpilogue(t, 6000)})

	// The two new class names are asserted as LITERALS, not as the ReasonProviderQuotaWall
	// / ReasonWallClockExhausted consts the rest of this file would use. That is
	// deliberate on both counts: it keeps the assertion behavioral rather than tautological
	// (a const on both sides can be renamed in lockstep and still pass), and it let this
	// test compile and RED against the pre-#5870 implementation, where it reported
	// `map[clean_exit_no_commit:4]` — all four fixtures folded into the one bucket.
	r := report(runs, aliveSet(), 6.0).Units.NoCommitReasons
	if r["provider_quota_wall"] != 2 || r["wall_clock_exhausted"] != 1 ||
		r[ReasonCleanExitNoCommit] != 1 {
		t.Errorf("no_commit_reasons = %v, want provider_quota_wall=2 wall_clock_exhausted=1 clean_exit_no_commit=1", r)
	}
}

// TestProducerStampedCleanExitIsStillRefined is the forward-compatibility pin for the
// change landing NEXT DOOR. Every affected .witness on disk today reads
// reason="unknown", because tools/issue_resolve_dispatch.py has no signature for the
// epilogue split and falls through to it — so the #5870 refinement is reached via the
// unknown branch. That producer is in the middle of mirroring #5867's three classes
// into its own NO_COMMIT_* vocabulary, and once it lands, these units arrive already
// stamped "clean_exit_no_commit".
//
// Without this case the split would keep passing every other test in this file while
// firing on none of the runs it exists for. Pinning it here means the producer can land
// its half whenever it likes and the ledger keeps naming the terminal.
func TestProducerStampedCleanExitIsStillRefined(t *testing.T) {
	runs := t.TempDir()
	mkWorker(t, runs, 55, 1.0, workerOpts{
		witness: map[string]any{"claim": "CLAIM_NO_COMMIT", "sha": nil,
			"reason": ReasonCleanExitNoCommit},
		body: "fak guard: TIME_BUDGET_EXHAUSTED — wall-clock envelope elapsed\n" + bulkyEpilogue(t, 6000)})
	r := report(runs, aliveSet(), 6.0).Units.NoCommitReasons
	if r["wall_clock_exhausted"] != 1 || r[ReasonCleanExitNoCommit] != 0 {
		t.Errorf("no_commit_reasons = %v, want a PRODUCER-stamped clean exit refined to its typed terminal", r)
	}
}

// TestTypedTerminalSurvivesTheEpilogueThatBuriesIt is the mechanism pin, kept separate
// from the classification pin above because it is the whole reason those 105 units were
// invisible. The witness sweep classifies from a 4 KiB tail
// (tools/issue_resolve_dispatch.py _CAP_TAIL_BYTES), and the guard prints its exit
// summary AFTER the terminal that caused the exit. Measured over the 118 units: the
// TIME_BUDGET_EXHAUSTED marker is a median 7671 bytes from EOF and lands inside the
// 4 KiB window 0/38 times — but 38/38 inside 16 KiB. Same for the 429 park (4/65 vs
// 65/65) and the usage cap (0/34 vs 34/34).
//
// 16 KiB is not a new liberty: it is _RESTART_EXHAUSTED_TAIL_BYTES, the window the
// producer's OWN classify_restart_exhaustion already reads for the same reason ("the
// live epilogue can exceed the generic 4 KiB quota-banner tail"). So this package still
// never claims to see something the producer could not have seen.
func TestTypedTerminalSurvivesTheEpilogueThatBuriesIt(t *testing.T) {
	dir := t.TempDir()
	log := filepath.Join(dir, "resolve-54-20260804-000000.log")
	body := "fak guard: TIME_BUDGET_EXHAUSTED — wall-clock envelope elapsed\n" + bulkyEpilogue(t, 6000)
	if err := os.WriteFile(log, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := ReadTailBytes(log, unknownRefineTailBytes); strings.Contains(got, "TIME_BUDGET_EXHAUSTED") {
		t.Fatal("fixture must bury the terminal past the 4 KiB window; that is the point")
	}
	if got := refineUnknownNoCommit(log); got != "wall_clock_exhausted" {
		t.Errorf("refineUnknownNoCommit = %q, want the terminal read through the epilogue that buries it", got)
	}
}

// TestUnrecognizedReasonStaysUnknownNotRefined keeps the #5866 lesson intact. The split
// above runs ONLY on a reason a producer itself stamped "unknown". A reason string this
// package merely fails to recognize must still fold to a bare "unknown" — after this
// change that residual is a VOCABULARY-DRIFT alarm ("a sidecar writer stamped a class we
// have never heard of"), which is worth acting on, and silently dressing it up as
// clean_exit_no_commit would re-hide exactly what #5866 uncovered.
func TestUnrecognizedReasonStaysUnknownNotRefined(t *testing.T) {
	runs := t.TempDir()
	mkWorker(t, runs, 44, 1.0, workerOpts{body: guardEpilogueFixture,
		witness: map[string]any{"claim": "CLAIM_NO_COMMIT", "sha": nil, "reason": "a_future_producers_class"}})
	r := report(runs, aliveSet(), 6.0).Units.NoCommitReasons
	if r["unknown"] != 1 || r[ReasonCleanExitNoCommit] != 0 {
		t.Errorf("no_commit_reasons = %v, want an UNRECOGNIZED reason to stay a bare unknown", r)
	}
}

// TestUnknownRefinementFailsOpenOnMissingLog pins the read-only fail-open rule: this
// package never invents evidence. A graded unknown whose log artifact is gone has no
// tail to read, so it must land in died_before_epilogue (no exit evidence exists) and
// must not panic or claim a clean exit.
func TestUnknownRefinementFailsOpenOnMissingLog(t *testing.T) {
	if got := refineUnknownNoCommit(filepath.Join(t.TempDir(), "absent.log")); got != ReasonDiedBeforeEpilogue {
		t.Errorf("refineUnknownNoCommit(missing) = %q, want %q", got, ReasonDiedBeforeEpilogue)
	}
}

func TestRepairUnitsCountedSeparately(t *testing.T) {
	runs := t.TempDir()
	mkWorker(t, runs, 30, 1.0, workerOpts{kind: "repair", lane: "contract-repair"})
	rep := report(runs, aliveSet(), 6.0)
	if rep.Units.RepairTotal != 1 || rep.Units.ResolveTotal != 0 || rep.Units.LeakedUnswept != 0 {
		t.Errorf("repair=%d resolve=%d leaked=%d, want 1/0/0",
			rep.Units.RepairTotal, rep.Units.ResolveTotal, rep.Units.LeakedUnswept)
	}
}

func TestClosesAndHoldsWindowed(t *testing.T) {
	runs := t.TempDir()
	progress := []string{
		`{"schema":"fleet-issue-resolve-progress/1","utc":"` + iso(1.0) + `","closed_now":2,"open_now":700,"baseline_open":483}`,
		`{"schema":"fleet-issue-resolve-progress/1","utc":"` + iso(0.5) + `","closed_now":1,"open_now":698,"baseline_open":483}`,
		`{"schema":"fleet-issue-resolve-progress/1","utc":"` + iso(30.0) + `","closed_now":9}`,
		`not json`,
	}
	os.WriteFile(filepath.Join(runs, "progress.jsonl"), []byte(strings.Join(progress, "\n")+"\n"), 0o644)
	holds := []string{
		`{"utc":"` + iso(1.0) + `","ts":` + strconv.FormatInt(now.Add(-time.Hour).Unix(), 10) + `,"issue":100,"score":8,"reason":"x"}`,
		`{"utc":"` + iso(0.9) + `","ts":` + strconv.FormatInt(now.Add(-54*time.Minute).Unix(), 10) + `,"issue":100,"score":8,"reason":"x"}`,
		`{"utc":"` + iso(40.0) + `","ts":` + strconv.FormatInt(now.Add(-40*time.Hour).Unix(), 10) + `,"issue":200,"score":8,"reason":"x"}`,
	}
	os.WriteFile(filepath.Join(runs, "contract-holds.jsonl"), []byte(strings.Join(holds, "\n")+"\n"), 0o644)
	rep := report(runs, aliveSet(), 6.0)
	if rep.Yield.IssuesClosedInWindow != 3 {
		t.Errorf("issues_closed_in_window = %d, want 3", rep.Yield.IssuesClosedInWindow)
	}
	if rep.Yield.OpenNow == nil || *rep.Yield.OpenNow != 698 {
		t.Errorf("open_now = %v, want 698", rep.Yield.OpenNow)
	}
	if rep.ContractHolds.Rows != 2 || rep.ContractHolds.DistinctIssues != 1 {
		t.Errorf("contract_holds = %+v, want {2,1}", rep.ContractHolds)
	}
}

func TestChurnSurfacesIssuesBurningMultipleUnits(t *testing.T) {
	runs := t.TempDir()
	mkWorker(t, runs, 50, 2.0, workerOpts{witness: map[string]any{"claim": "CLAIM_NO_COMMIT", "sha": nil, "reason": "self_modify"}})
	mkWorker(t, runs, 50, 1.0, workerOpts{witness: map[string]any{"claim": "CLAIM_NO_COMMIT", "sha": nil, "reason": "self_modify"}})
	mkWorker(t, runs, 51, 1.0, workerOpts{witness: map[string]any{"claim": "CLAIM_WITNESSED", "sha": strings.Repeat("c", 40)}})
	rep := report(runs, aliveSet(), 6.0)
	if rep.Churn.IssuesWith2Plus != 1 {
		t.Fatalf("issues_with_2plus_units = %d, want 1", rep.Churn.IssuesWith2Plus)
	}
	if len(rep.Churn.Worst) != 1 || rep.Churn.Worst[0].Issue != 50 || rep.Churn.Worst[0].Units != 2 {
		t.Errorf("worst = %+v, want [{50 2}]", rep.Churn.Worst)
	}
}

func runMain(t *testing.T, runs string, alive AliveProbe, extra ...string) (int, Report) {
	t.Helper()
	argv := append([]string{"--runs-dir", runs, "--json"}, extra...)
	var buf bytes.Buffer
	rc := Run(argv, func() AliveProbe { return alive }, now, &buf, &buf)
	var rep Report
	if err := json.Unmarshal(buf.Bytes(), &rep); err != nil {
		t.Fatalf("decode report: %v\n%s", err, buf.String())
	}
	return rc, rep
}

func TestJSONReportAndFailOnLeakGate(t *testing.T) {
	runs := t.TempDir()
	mkWorker(t, runs, 60, 1.0, workerOpts{}) // one leaked unit
	rc, rep := runMain(t, runs, aliveSet())
	if rc != 0 {
		t.Errorf("default rc = %d, want 0 (report-only)", rc)
	}
	if rep.Schema != Schema || rep.Units.LeakedUnswept != 1 {
		t.Errorf("schema=%q leaked=%d, want %q/1", rep.Schema, rep.Units.LeakedUnswept, Schema)
	}
	if rc, _ := runMain(t, runs, aliveSet(), "--fail-on-leak", "0"); rc != 1 {
		t.Errorf("--fail-on-leak 0 rc = %d, want 1", rc)
	}
	if rc, _ := runMain(t, runs, aliveSet(), "--fail-on-leak", "1"); rc != 0 {
		t.Errorf("--fail-on-leak 1 rc = %d, want 0", rc)
	}
}

func TestMissingRunsDirDegradesToEmptyConserved(t *testing.T) {
	runs := t.TempDir()
	rc, rep := runMain(t, filepath.Join(runs, "nope"), aliveSet())
	if rc != 0 {
		t.Errorf("rc = %d, want 0", rc)
	}
	if rep.Units.ResolveTotal != 0 || rep.Verdict != "CONSERVED" {
		t.Errorf("resolve=%d verdict=%s, want 0/CONSERVED", rep.Units.ResolveTotal, rep.Verdict)
	}
}
