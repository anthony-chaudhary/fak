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
