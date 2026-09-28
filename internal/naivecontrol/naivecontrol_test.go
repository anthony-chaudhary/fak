package naivecontrol

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const (
	headSHA   = "1111111111111111111111111111111111111111"
	baseSHA   = "2222222222222222222222222222222222222222"
	landedSHA = "3333333333333333333333333333333333333333"
	strandSHA = "4444444444444444444444444444444444444444"
	fakeSHA   = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
)

type gitReply struct {
	out  string
	code int
	err  error
}

// scriptedGit is a GitRunner that answers only the probes a test names and fails
// the test on any other one, so no verdict can come from an unscripted default.
func scriptedGit(t *testing.T, script map[string]gitReply) GitRunner {
	t.Helper()
	return func(_ context.Context, _ string, args ...string) (string, int, error) {
		key := strings.Join(args, " ")
		r, ok := script[key]
		if !ok {
			t.Fatalf("unscripted git %q", key)
		}
		return r.out, r.code, r.err
	}
}

// landedScript is the probe chain for one commit that exists, is reachable from
// HEAD and was never reverted.
func landedScript(sha string) map[string]gitReply {
	return map[string]gitReply{
		"rev-parse --verify --quiet " + sha + "^{commit}": {out: sha + "\n"},
		"cat-file -e " + sha + "^{commit}":                {},
		"rev-parse --verify " + sha + "^{commit}":         {out: sha + "\n"},
		"merge-base --is-ancestor " + sha + " HEAD":       {},
		"log --format=%H%x00%B%x00 " + sha + "..HEAD":     {},
	}
}

func merge(ms ...map[string]gitReply) map[string]gitReply {
	out := map[string]gitReply{}
	for _, m := range ms {
		for k, v := range m {
			out[k] = v
		}
	}
	return out
}

var t0 = time.Date(2026, 9, 28, 12, 0, 0, 0, time.UTC)

// TestShippedCountsComeFromGitAncestry pins the admissibility rule. Two picks; the
// workers self-report three commits. Git lands one of them, a second exists but is
// not reachable from HEAD, and the third names no commit at all. Only the landed one
// counts, the fabricated claim contributes zero, and a pick is NOT_SHIPPED only
// because this package's own base..HEAD scan found nothing for it.
func TestShippedCountsComeFromGitAncestry(t *testing.T) {
	run := scriptedGit(t, merge(
		landedScript(landedSHA),
		map[string]gitReply{
			"rev-parse --verify --quiet HEAD^{commit}": {out: headSHA + "\n"},
			"log --format=%H%x1f%s " + baseSHA + "..HEAD": {
				out: landedSHA + "\x1ffix(gateway): stop the leak (#101)\n" +
					headSHA + "\x1fdocs: unrelated (#9999)\n",
			},
			"rev-parse --verify --quiet " + fakeSHA + "^{commit}":   {code: 1},
			"rev-parse --verify --quiet " + strandSHA + "^{commit}": {out: strandSHA + "\n"},
			"cat-file -e " + strandSHA + "^{commit}":                {},
			"rev-parse --verify " + strandSHA + "^{commit}":         {out: strandSHA + "\n"},
			"merge-base --is-ancestor " + strandSHA + " HEAD":       {code: 1},
		},
	))
	picks := []Pick{{Issue: 101, Base: baseSHA}, {Issue: 102, Base: baseSHA}}
	claims := []Claim{
		{Issue: 101, SHA: landedSHA},
		{Issue: 102, SHA: fakeSHA},
		{Issue: 102, SHA: strandSHA},
		{Issue: 555, SHA: landedSHA}, // not a pick: never verified, never counted
	}
	v := Verify(context.Background(), run, "", picks, claims)
	row := Build(RunInput{
		Arm: ArmNaive, RunID: "r1", RecordedAt: t0,
		Picks: picks, PicksKnown: true,
		SelfReported: claims, SelfReportRead: true,
		Verification: &v,
	})

	if got, ok := row.PicksShipped.Get(); !ok || got != 1 {
		t.Fatalf("picks_shipped = %s (%s), want 1", row.PicksShipped, row.PicksShipped.WhyMissing())
	}
	if got, ok := row.CommitsLanded.Get(); !ok || got != 1 {
		t.Fatalf("commits_landed = %s (%s), want 1 distinct landed sha", row.CommitsLanded, row.CommitsLanded.WhyMissing())
	}
	if got, _ := row.CommitsClaimed.Get(); got != 4 {
		t.Fatalf("commits_claimed = %d, want 4 (the self-report, kept visible)", got)
	}
	want := map[string]Verdict{
		fakeSHA:   VerdictNotLanded,
		strandSHA: VerdictNotLanded,
	}
	for _, c := range row.Commits {
		if c.Issue == 555 {
			if c.Verdict != VerdictOffPick {
				t.Fatalf("claim for unoffered #555 = %s, want OFF_PICK", c.Verdict)
			}
			continue
		}
		if w, ok := want[c.SHA]; ok && c.Verdict != w {
			t.Fatalf("claim %s = %s (%s), want %s", c.SHA[:8], c.Verdict, c.Detail, w)
		}
		if c.SHA == landedSHA && c.Verdict != VerdictLanded {
			t.Fatalf("landed claim = %s (%s)", c.Verdict, c.Detail)
		}
	}
	status := map[int]PickStatus{}
	for _, p := range row.Picks {
		status[p.Issue] = p.Status
	}
	if status[101] != PickShipped || status[102] != PickNotShipped {
		t.Fatalf("pick statuses = %v, want #101 SHIPPED, #102 NOT_SHIPPED", status)
	}
	if row.HeadSHA != headSHA || row.Stage != StageHarvested {
		t.Fatalf("row not pinned to HEAD: verified_against=%q stage=%s", row.HeadSHA, row.Stage)
	}
}

// TestSelfReportAloneCannotShip is the lying-driver case: every worker claims
// success, git can resolve none of the claims, and no base was recorded to scan.
// The result is zero landed commits and UNKNOWN shipped picks, never the claim.
func TestSelfReportAloneCannotShip(t *testing.T) {
	run := scriptedGit(t, map[string]gitReply{
		"rev-parse --verify --quiet HEAD^{commit}":            {out: headSHA + "\n"},
		"rev-parse --verify --quiet " + fakeSHA + "^{commit}": {code: 1},
	})
	picks := []Pick{{Issue: 7}}
	claims := []Claim{{Issue: 7, SHA: fakeSHA}, {Issue: 7, SHA: "HEAD"}}
	v := Verify(context.Background(), run, "", picks, claims)
	row := Build(RunInput{Arm: ArmNaive, RunID: "r", RecordedAt: t0, Picks: picks, PicksKnown: true,
		SelfReported: claims, SelfReportRead: true, Verification: &v})
	for _, c := range row.Commits {
		if c.Verdict != VerdictNotLanded {
			t.Fatalf("claim %q = %s, want NOT_LANDED (a word or a fabricated sha is not a commit)", c.SHA, c.Verdict)
		}
	}
	if row.PicksShipped.IsMeasured() {
		t.Fatalf("picks_shipped = %s, want UNKNOWN: with no base scan, an absent landed commit proves nothing", row.PicksShipped)
	}
	if !strings.Contains(row.PicksShipped.WhyMissing(), "no base sha") {
		t.Fatalf("reason %q does not name the missing base", row.PicksShipped.WhyMissing())
	}
}

// TestGitOutageIsUnknownNotZero: when git cannot run, nothing it would have said is
// recorded as 0. This is the DOA shape: an outage must not read like a quiet night.
func TestGitOutageIsUnknownNotZero(t *testing.T) {
	down := errors.New("exec: \"git\": executable file not found")
	run := func(context.Context, string, ...string) (string, int, error) { return "", -1, down }
	picks := []Pick{{Issue: 1, Base: baseSHA}}
	v := Verify(context.Background(), run, "", picks, []Claim{{Issue: 1, SHA: landedSHA}})
	row := Build(RunInput{Arm: ArmNaive, RunID: "r", RecordedAt: t0, Picks: picks, PicksKnown: true, Verification: &v})
	for name, m := range map[string]Metric[int64]{"picks_shipped": row.PicksShipped, "commits_landed": row.CommitsLanded} {
		if m.IsMeasured() || m.String() != Unknown {
			t.Fatalf("%s = %s under a git outage, want UNKNOWN", name, m)
		}
		if !strings.Contains(m.WhyMissing(), "git") {
			t.Fatalf("%s reason %q does not name git", name, m.WhyMissing())
		}
	}
}

// TestMissingMetricRendersUnknownNotZero pins the missing-measurement state on every
// surface: the value API, JSON round trip, a field absent from a decoded row, and
// the rendered comparison. An observed 0 still renders as 0.
func TestMissingMetricRendersUnknownNotZero(t *testing.T) {
	if got := Missing[int64]("x").String(); got != Unknown {
		t.Fatalf("Missing.String() = %q", got)
	}
	var zero Metric[float64]
	if got := zero.String(); got != Unknown {
		t.Fatalf("zero Metric renders %q, want UNKNOWN", got)
	}
	if got := Measured[int64](0).String(); got != "0" {
		t.Fatalf("an observed zero renders %q, want 0", got)
	}

	row := Build(RunInput{Arm: ArmNaive, RunID: "r", RecordedAt: t0, Picks: []Pick{{Issue: 1}}, PicksKnown: true,
		ControllerMS: Measured[int64](0)})
	b, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	var back Row
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if back.CostUSD.IsMeasured() || back.CostUSD.State != StateMissing {
		t.Fatalf("cost_usd after round trip = %+v, want MISSING_MEASUREMENT", back.CostUSD)
	}
	if v, ok := back.ControllerMS.Get(); !ok || v != 0 {
		t.Fatalf("an observed controller_ms 0 did not survive the round trip: %+v", back.ControllerMS)
	}

	// A row that omits a field entirely (an older recorder) must not surface 0.
	var sparse Row
	if err := json.Unmarshal([]byte(`{"schema":"fak-naive-control/1","arm":"naive","run_id":"old","stage":"PLANNED","ts_unix_nano":1}`), &sparse); err != nil {
		t.Fatal(err)
	}
	if err := Validate(sparse); err != nil {
		t.Fatalf("a sparse row is legal, got %v", err)
	}
	if sparse.WallMS.String() != Unknown || sparse.WallMS.WhyMissing() == "" {
		t.Fatalf("absent wall_ms = %q (%q), want UNKNOWN with a reason", sparse.WallMS, sparse.WallMS.WhyMissing())
	}

	var out bytes.Buffer
	if err := RenderComparison(&out, Compare([]Row{row}, LedgerHealth{Rows: 1})); err != nil {
		t.Fatal(err)
	}
	text := out.String()
	costLine := lineWith(t, text, "cost (USD)")
	if !strings.Contains(costLine, Unknown) || strings.Contains(costLine, "$0.00") {
		t.Fatalf("cost line %q, want UNKNOWN and no $0.00", costLine)
	}
	if !strings.Contains(text, "naive.cost_usd: 1 of 1 run(s) unmeasured") {
		t.Fatalf("render does not give the reason cost is UNKNOWN:\n%s", text)
	}
	if line := lineWith(t, text, "controller time"); !strings.Contains(line, "0s") {
		t.Fatalf("observed zero controller time line %q, want 0s", line)
	}
}

// TestCompareWithOneArmEmpty: a ledger holding only naive rows still prints both
// columns; the orchestrated column says it has no data, every one of its cells is
// UNKNOWN, and the verdict refuses to conclude and names the empty arm.
func TestCompareWithOneArmEmpty(t *testing.T) {
	rows := []Row{
		harvestedRow(ArmNaive, "n1", t0, 4, 3, 5),
		harvestedRow(ArmNaive, "n2", t0.Add(time.Hour), 4, 4, 4),
	}
	c := Compare(rows, LedgerHealth{Path: "ledger.jsonl", Rows: 2})
	if len(c.Arms) != 2 || c.Arms[0].Arm != ArmNaive || c.Arms[1].Arm != ArmOrchestrated {
		t.Fatalf("arms = %+v, want naive then orchestrated", c.Arms)
	}
	naive, orch := c.Arms[0], c.Arms[1]
	if v, _ := naive.PicksShipped.Total.Get(); v != 7 {
		t.Fatalf("naive shipped total = %d, want 7", v)
	}
	if r, _ := naive.ShipRate.Get(); r != 7.0/8.0 {
		t.Fatalf("naive ship rate = %v, want 0.875", r)
	}
	if orch.Runs != 0 || orch.PicksShipped.Total.IsMeasured() || orch.ShipRate.IsMeasured() {
		t.Fatalf("empty orchestrated arm reports measurements: %+v", orch)
	}
	if c.Conclusive || !strings.Contains(c.Verdict, "orchestrated ship rate UNKNOWN") {
		t.Fatalf("verdict = %q (conclusive=%v), want INCONCLUSIVE naming the orchestrated arm", c.Verdict, c.Conclusive)
	}

	var out bytes.Buffer
	if err := RenderComparison(&out, c); err != nil {
		t.Fatal(err)
	}
	text := out.String()
	if line := lineWith(t, text, "runs"); !strings.Contains(line, "2 (2 harvested)") || !strings.HasSuffix(strings.TrimSpace(line), "0 (no data)") {
		t.Fatalf("runs row = %q, want naive 2 (2 harvested) and orchestrated 0 (no data)", line)
	}
	for _, tc := range []struct{ metric, naive string }{
		{"picks offered", "8"},
		{"picks shipped (git)", "7"},
		{"ship rate", "87.5%"},
		{"commits landed (git)", "9"},
	} {
		cells := strings.Fields(strings.TrimPrefix(lineWith(t, text, tc.metric), tc.metric))
		if len(cells) != 2 || cells[0] != tc.naive || cells[1] != Unknown {
			t.Fatalf("%s cells = %q, want [%s UNKNOWN]", tc.metric, cells, tc.naive)
		}
	}
	if !strings.Contains(text, "orchestrated: no runs recorded, so every field is UNKNOWN") {
		t.Fatalf("render does not say the orchestrated arm has no data:\n%s", text)
	}

	// No data at all: both columns empty, still a well-formed, inconclusive screen.
	empty := Compare(nil, LedgerHealth{Path: "x", Missing: true})
	if empty.Conclusive || len(empty.Arms) != 2 || empty.Arms[0].Runs != 0 {
		t.Fatalf("empty comparison = %+v", empty)
	}
}

// TestCompareConclusiveWhenBothArmsMeasured: with both arms harvested and cost read,
// one line carries the conclusion.
func TestCompareConclusiveWhenBothArmsMeasured(t *testing.T) {
	n := harvestedRow(ArmNaive, "n", t0, 10, 9, 9)
	n.CostUSD = Measured(121.5)
	o := harvestedRow(ArmOrchestrated, "o", t0, 10, 1, 1)
	o.CostUSD = Measured(42.96)
	c := Compare([]Row{n, o}, LedgerHealth{Rows: 2})
	if !c.Conclusive {
		t.Fatalf("verdict inconclusive: %s", c.Verdict)
	}
	for _, want := range []string{"naive ships 90.0% of picks (9/10) at $13.50", "orchestrated ships 10.0% of picks (1/10) at $42.96"} {
		if !strings.Contains(c.Verdict, want) {
			t.Fatalf("verdict %q missing %q", c.Verdict, want)
		}
	}
}

// TestAggregateRefusesPartialSum: one unharvested run makes the arm's shipped total
// UNKNOWN; the partial sum over the measured runs is disclosed, not promoted.
func TestAggregateRefusesPartialSum(t *testing.T) {
	planned := Build(RunInput{Arm: ArmNaive, RunID: "p", RecordedAt: t0, Picks: []Pick{{Issue: 1}, {Issue: 2}}, PicksKnown: true})
	s := Summarize(ArmNaive, []Row{harvestedRow(ArmNaive, "h", t0, 4, 3, 3), planned})
	if s.PicksShipped.Total.IsMeasured() {
		t.Fatalf("shipped total measured over an unharvested run: %s", s.PicksShipped.Total)
	}
	if s.PicksShipped.PartialSum == nil || *s.PicksShipped.PartialSum != 3 || s.PicksShipped.Measured != 1 {
		t.Fatalf("partial = %+v, want 3 over 1 measured run", s.PicksShipped)
	}
	if v, _ := s.PicksOffered.Total.Get(); v != 6 {
		t.Fatalf("offered total = %d, want 6 (both runs read their picks)", v)
	}
	if !strings.Contains(s.PicksShipped.Total.WhyMissing(), "not harvested") {
		t.Fatalf("reason %q does not say the run was not harvested", s.PicksShipped.Total.WhyMissing())
	}

	// A window that starts after the unharvested run measures the arm again; the
	// run is excluded by when it was recorded, never by what it measured.
	recent := Since([]Row{planned, harvestedRow(ArmNaive, "h", t0.Add(time.Hour), 4, 3, 3)}, t0.Add(time.Minute))
	if v, ok := Summarize(ArmNaive, recent).PicksShipped.Total.Get(); !ok || v != 3 {
		t.Fatalf("windowed shipped total = %d (measured=%v), want 3", v, ok)
	}
}

// TestArmRoundTripsThroughLedger: both arms written to one ledger come back as
// themselves, a HARVESTED row supersedes its PLANNED row, and a row with an arm
// outside the closed vocabulary is refused on write and counted on read.
func TestArmRoundTripsThroughLedger(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "naive-control.jsonl")
	planned := Build(RunInput{Arm: ArmNaive, RunID: "run-1", RecordedAt: t0, Picks: []Pick{{Issue: 5}}, PicksKnown: true})
	for _, r := range []Row{
		planned,
		harvestedRow(ArmNaive, "run-1", t0.Add(time.Minute), 1, 1, 1),
		harvestedRow(ArmOrchestrated, "run-1", t0, 3, 1, 2),
	} {
		if err := Append(path, r); err != nil {
			t.Fatal(err)
		}
	}
	bad := planned
	bad.Arm = "Naive"
	if err := Append(path, bad); err == nil {
		t.Fatal("Append accepted an arm outside naive|orchestrated")
	}

	rows, health, err := ReadLedger(path)
	if err != nil || health.Rows != 3 || health.Rejected != 0 {
		t.Fatalf("read %d rows, health %+v, err %v", len(rows), health, err)
	}
	latest := Latest(rows)
	if len(latest) != 2 {
		t.Fatalf("latest = %d rows, want one per (arm, run_id)", len(latest))
	}
	got, ok := FindRun(rows, ArmNaive, "run-1")
	if !ok || got.Arm != ArmNaive || got.Stage != StageHarvested {
		t.Fatalf("naive run-1 = %+v, want the HARVESTED naive row", got)
	}
	if o, ok := FindRun(rows, ArmOrchestrated, "run-1"); !ok || o.Arm != ArmOrchestrated {
		t.Fatalf("orchestrated run-1 lost its arm: %+v", o)
	}

	_, h := ParseLedger(`{"schema":"fak-naive-control/1","arm":"human-driven","run_id":"x","stage":"PLANNED","ts_unix_nano":1}` + "\n" + "not json\n")
	if h.Rows != 0 || h.Rejected != 2 || len(h.Errors) != 2 {
		t.Fatalf("health = %+v, want both lines rejected and disclosed", h)
	}
}

// TestOrchestratedSidecarsUseTheSameRuler: a CLAIM_WITNESSED sidecar is only a
// claim; the pick counts shipped because git lands it, and a CLAIM_NO_COMMIT
// sidecar with no .basesha stays UNKNOWN rather than becoming a measured miss.
func TestOrchestratedSidecarsUseTheSameRuler(t *testing.T) {
	var ws []OrchestratedWorker
	for _, raw := range []string{
		`{"claim": "CLAIM_WITNESSED", "issue": 10296, "log": "resolve-10296-20260901-091239.log", "sha": "` + landedSHA + `", "verdict": "OK", "witness": "diff-witnessed"}`,
		`{"claim": "CLAIM_NO_COMMIT", "issue": 10000, "log": "resolve-10000-20260901-075204.log", "reason": "clean_exit_no_commit", "sha": null, "verdict": null, "witness": null}`,
	} {
		sc, err := ParseWitnessSidecar([]byte(raw))
		if err != nil {
			t.Fatal(err)
		}
		ws = append(ws, OrchestratedWorker{Sidecar: sc})
	}
	picks, claims := OrchestratedPicks(ws)
	if len(picks) != 2 || len(claims) != 1 {
		t.Fatalf("picks=%v claims=%v", picks, claims)
	}
	run := scriptedGit(t, merge(landedScript(landedSHA), map[string]gitReply{
		"rev-parse --verify --quiet HEAD^{commit}": {out: headSHA + "\n"},
	}))
	v := Verify(context.Background(), run, "", picks, claims)
	row := Build(RunInput{Arm: ArmOrchestrated, RunID: "o", RecordedAt: t0, Picks: picks, PicksKnown: true,
		SelfReported: claims, SelfReportRead: true, Verification: &v})
	if row.PicksShipped.IsMeasured() {
		t.Fatalf("picks_shipped = %s, want UNKNOWN (#10000 was never scanned)", row.PicksShipped)
	}
	if !strings.Contains(row.PicksShipped.WhyMissing(), "1 verified shipped") {
		t.Fatalf("reason %q does not disclose the verified lower bound", row.PicksShipped.WhyMissing())
	}
	if ts, ok := SpawnStamp("resolve-10296-20260901-091239-ab12.witness"); !ok || !ts.Equal(time.Date(2026, 9, 1, 9, 12, 39, 0, time.UTC)) {
		t.Fatalf("SpawnStamp = %v %v", ts, ok)
	}
}

// TestRealGitFabricatedCommitCountsZero runs the verifier against a real fixture
// repository, because the fake runner can only be as right as its script: real git
// must answer a fabricated sha with a witnessed "no" (NOT_LANDED), a commit off
// HEAD's history with NOT_LANDED, and the landed fix with LANDED.
func TestRealGitFabricatedCommitCountsZero(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	dir := t.TempDir()
	g := func(args ...string) string {
		t.Helper()
		out, code, err := ExecGit(context.Background(), dir, args...)
		if err != nil || code != 0 {
			t.Fatalf("git %v: code %d err %v out %q", args, code, err, out)
		}
		return strings.TrimSpace(out)
	}
	g("init", "-q", "-b", "main")
	g("config", "user.email", "fixture@example.invalid")
	g("config", "user.name", "fixture")
	g("config", "commit.gpgsign", "false")
	g("commit", "-q", "--allow-empty", "-m", "chore: base")
	base := g("rev-parse", "HEAD")
	g("switch", "-q", "-c", "side")
	g("commit", "-q", "--allow-empty", "-m", "fix: never merged (#8)")
	side := g("rev-parse", "HEAD")
	g("switch", "-q", "main")
	g("commit", "-q", "--allow-empty", "-m", "fix(x): the real fix (#7)")
	fix := g("rev-parse", "HEAD")

	picks := []Pick{{Issue: 7, Base: base}, {Issue: 8, Base: base}}
	claims := []Claim{{Issue: 7, SHA: fix[:10]}, {Issue: 8, SHA: fakeSHA}, {Issue: 8, SHA: side}}
	v := Verify(context.Background(), ExecGit, dir, picks, claims)
	row := Build(RunInput{Arm: ArmNaive, RunID: "real", RecordedAt: t0, Picks: picks, PicksKnown: true,
		SelfReported: claims, SelfReportRead: true, Verification: &v})

	if got, ok := row.PicksShipped.Get(); !ok || got != 1 {
		t.Fatalf("picks_shipped = %s (%s), want 1", row.PicksShipped, row.PicksShipped.WhyMissing())
	}
	if got, ok := row.CommitsLanded.Get(); !ok || got != 1 {
		t.Fatalf("commits_landed = %s (%s), want 1 (short and full sha of one commit)", row.CommitsLanded, row.CommitsLanded.WhyMissing())
	}
	for _, c := range row.Commits {
		switch c.SHA {
		case fakeSHA, side:
			if c.Verdict != VerdictNotLanded {
				t.Fatalf("claim %s = %s (%s), want NOT_LANDED", c.SHA[:8], c.Verdict, c.Detail)
			}
		}
	}
	if row.HeadSHA != fix {
		t.Fatalf("verified_against = %s, want HEAD %s", row.HeadSHA, fix)
	}
}

func harvestedRow(arm Arm, runID string, at time.Time, offered, shipped, landed int64) Row {
	return Row{
		Schema: Schema, Arm: arm, RunID: runID, Stage: StageHarvested, TSUnixNano: at.UnixNano(),
		HeadSHA:        headSHA,
		PicksOffered:   Measured(offered),
		PicksShipped:   Measured(shipped),
		CommitsClaimed: Missing[int64]("no self-report was read for this run"),
		CommitsLanded:  Measured(landed),
		WallMS:         Missing[int64]("wall time not observed"),
		ControllerMS:   Measured[int64](1200),
		CostUSD:        Missing[float64]("no cost source was read for this run"),
	}
}

func lineWith(t *testing.T, text, prefix string) string {
	t.Helper()
	for _, l := range strings.Split(text, "\n") {
		if strings.HasPrefix(l, prefix+" ") {
			return l
		}
	}
	t.Fatalf("no line starting %q in:\n%s", prefix, text)
	return ""
}
