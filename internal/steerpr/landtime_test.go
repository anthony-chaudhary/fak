package steerpr

// landtime_test.go — the #5026 witness. Three of these tests are the issue's
// acceptance gate and one is its stated worst outcome:
//
//   - TestTickFoldIdenticalWithAndWithoutLandTimeHook — the optimization is
//     UNOBSERVABLE in the result. This is the gate.
//   - TestBrokenLedgerNeverBlocksTheLanding — a broken/unwritable ledger costs a
//     row, never the commit. This is the hard fence.
//   - TestLandingSeamStaysWithinLatencyBudget — the GATE_LATENCY_REGRESSION
//     assertion over a hot, shared, latency-budgeted path.
//   - TestLandingCarriesNoBand — the hook must never become the oracle.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

// landAt is a fixed instant: a land-time row's only non-deterministic input is
// the clock, so the tests pin it and stay reproducible.
var landAt = time.Date(2026, 8, 7, 5, 30, 0, 0, time.UTC)

// These tests drive the REAL tick path (logRecord from steerpr_test.go, then
// ParseLog) rather than hand-built Commit values that could quietly disagree
// with what git actually produces.
//
// sampleLog is the range under test: two commits in one unit, one in another,
// and one UNSTAMPED commit (the orphan the fold surfaces as legibility debt).
func sampleLog() string {
	return logRecord("1111111111111111111111111111111111111111", "feat(hooks): add the land-time seam (#5026) (fak hooks)", "body\n", "internal/hooks/hooks.go") +
		logRecord("2222222222222222222222222222222222222222", "fix(hooks): treat a torn row as absent (#5026) (fak hooks)", "refs #5015\n", "internal/steerpr/landtime.go") +
		logRecord("3333333333333333333333333333333333333333", "feat(gateway): widen the ready window (#4242) (fak gateway)", "", "internal/gateway/g.go") +
		logRecord("4444444444444444444444444444444444444444", "chore: no ship-stamp here at all", "", "README.md")
}

// recordAll runs the land-time seam over every commit in a range, the way the
// post-commit hook does one commit at a time.
func recordAll(t *testing.T, path string, commits []Commit) {
	t.Helper()
	for _, c := range commits {
		l, err := AssignLanded(c.SHA, c.Subject, "", landAt)
		if err != nil {
			t.Fatalf("AssignLanded(%s): %v", c.SHA, err)
		}
		if _, err := RecordLanding(path, l); err != nil {
			t.Fatalf("RecordLanding(%s): %v", c.SHA, err)
		}
	}
}

// foldJSON renders a fold canonically so two folds can be compared as bytes.
func foldJSON(t *testing.T, units []Unit, unstamped []Commit) string {
	t.Helper()
	buf, err := json.Marshal(struct {
		Units     []Unit   `json:"units"`
		Unstamped []Commit `json:"unstamped"`
	}{units, unstamped})
	if err != nil {
		t.Fatalf("marshal fold: %v", err)
	}
	return string(buf)
}

// TestTickFoldIdenticalWithAndWithoutLandTimeHook is #5026's acceptance gate:
// the tick must produce the IDENTICAL fold whether or not the hook ran. It holds
// structurally — the fold reads git and never opens the landing ledger — and
// this test is what keeps it structural, because the day someone teaches the
// fold to read the cache, the cache stops being a cache and becomes a second
// oracle that can disagree with git.
func TestTickFoldIdenticalWithAndWithoutLandTimeHook(t *testing.T) {
	raw := sampleLog()

	// Tick A: no hook ever ran.
	unitsA, unstampedA := FoldUnits(ParseLog(raw))
	without := foldJSON(t, unitsA, unstampedA)

	// Tick B: the hook ran over every commit first, into a live ledger.
	path := LandingLedgerPath(t.TempDir())
	recordAll(t, path, ParseLog(raw))
	if got := len(LoadLandings(path)); got != 4 {
		t.Fatalf("ledger holds %d rows, want 4 — the hook did not actually run", got)
	}
	unitsB, unstampedB := FoldUnits(ParseLog(raw))
	with := foldJSON(t, unitsB, unstampedB)

	if with != without {
		t.Errorf("the land-time hook changed the tick's fold; it is a cache, not an oracle\nwithout: %s\nwith:    %s", without, with)
	}
}

// TestLandTimeAssignsTheSameUnitTheTickWould is the other half of "identical":
// the fold is unchanged AND the cache agrees with it. Membership is computed by
// one shared parser (parseCommit), so this asserts the property that makes the
// optimization correct rather than merely invisible.
func TestLandTimeAssignsTheSameUnitTheTickWould(t *testing.T) {
	raw := sampleLog()
	commits := ParseLog(raw)
	path := LandingLedgerPath(t.TempDir())
	recordAll(t, path, commits)

	units, unstamped := FoldUnits(commits)
	if drift := ReconcileLandings(LoadLandings(path), units, unstamped); len(drift) != 0 {
		t.Errorf("land-time cache drifted from the tick's fold: %+v", drift)
	}

	// And the orphan is recorded as an orphan, not dropped: the cache's partition
	// is as total as the fold's.
	rows := LoadLandings(path)
	var orphans int
	for _, r := range rows {
		if r.Orphan {
			orphans++
			if r.Unit != "" {
				t.Errorf("orphan row names unit %q", r.Unit)
			}
		}
	}
	if orphans != 1 {
		t.Errorf("recorded %d orphan rows, want 1 (the unstamped commit)", orphans)
	}
}

// TestReconcileReportsAMissedHookInTheTicksFavour proves the drift is stated ONE
// way. A hook that never ran is drift the tick has already repaired, and it must
// read as "the fold assigned it", never as the cache correcting the fold.
func TestReconcileReportsAMissedHookInTheTicksFavour(t *testing.T) {
	commits := ParseLog(sampleLog())
	units, unstamped := FoldUnits(commits)

	drift := ReconcileLandings(nil, units, unstamped) // the hook never ran at all
	if len(drift) != len(commits) {
		t.Fatalf("drift covers %d commits, want %d", len(drift), len(commits))
	}
	for _, d := range drift {
		if d.Got != "" {
			t.Errorf("%s: cache reported %q when no row exists", d.SHA, d.Got)
		}
		if !strings.Contains(d.Why, "the tick assigned it") {
			t.Errorf("%s: drift is not stated in the tick's favour: %q", d.SHA, d.Why)
		}
	}

	// A STALE row (the cache disagreeing outright) is reported with the fold's
	// answer as Want — never the other way around.
	stale := []Landing{{
		Schema: LandingSchema, At: landAt.Format(time.RFC3339), GroupedBy: GroupedByLeaf,
		SHA: "1111111111111111111111111111111111111111", Unit: "wrong-leaf", Subject: "s",
	}}
	got := ReconcileLandings(stale, units, unstamped)
	var found bool
	for _, d := range got {
		if d.SHA == "1111111111111111111111111111111111111111" {
			found = true
			if d.Want != "hooks" || d.Got != "wrong-leaf" {
				t.Errorf("drift = want %q got %q; the fold's answer must be Want", d.Want, d.Got)
			}
		}
	}
	if !found {
		t.Error("a stale cache row was not reported as drift")
	}
}

// TestBrokenLedgerNeverBlocksTheLanding is #5026's hard fence: a deliberately
// broken overlay ledger must not fail the commit. Every failure path returns an
// ordinary error the caller drops on the floor — nothing here panics, exits, or
// refuses, because a commit-time refusal would re-introduce the PR gate through
// the back door, which the issue names as the single worst outcome available.
func TestBrokenLedgerNeverBlocksTheLanding(t *testing.T) {
	root := t.TempDir()

	// A regular file standing where the ledger's PARENT directory must be:
	// MkdirAll cannot proceed. This is the portable "unwritable ledger".
	blocker := filepath.Join(root, "blocked")
	if err := os.WriteFile(blocker, []byte("not a directory"), 0o644); err != nil {
		t.Fatal(err)
	}
	// A directory standing where the ledger FILE must be: OpenFile cannot proceed.
	asDir := filepath.Join(root, "dir-ledger", ".fak", "steer-landings.jsonl")
	if err := os.MkdirAll(asDir, 0o755); err != nil {
		t.Fatal(err)
	}

	broken := map[string]string{
		"parent is a regular file": filepath.Join(blocker, ".fak", "steer-landings.jsonl"),
		"ledger path is a dir":     asDir,
	}
	for name, path := range broken {
		t.Run(name, func(t *testing.T) {
			l, err := AssignLanded("5555555555555555555555555555555555555555", "feat(hooks): x (fak hooks)", "", landAt)
			if err != nil {
				t.Fatalf("AssignLanded must not depend on the ledger: %v", err)
			}
			// The contract under test: this RETURNS, with an error, and the
			// caller is free to ignore it. A panic here would take the hook
			// process down mid-commit; that is the failure this test catches.
			appended, err := RecordLanding(path, l)
			if appended {
				t.Error("reported a write into a broken ledger")
			}
			if err == nil {
				t.Error("a broken ledger returned no error: the failure must be recordable")
			}
			// Reading back a broken ledger is an EMPTY ledger, never an error
			// the caller has to handle and never an invented landing.
			if rows := LoadLandings(path); len(rows) != 0 {
				t.Errorf("broken ledger yielded %d rows, want 0", len(rows))
			}
		})
	}
}

// TestTornRowsDoNotPoisonTheLedger — the other broken-ledger shape: a half-written
// line (a crash mid-append) or a foreign schema must cost only itself. Skipping
// the torn row keeps the surrounding cache usable, which is why a broken ledger
// degrades instead of failing.
func TestTornRowsDoNotPoisonTheLedger(t *testing.T) {
	path := LandingLedgerPath(t.TempDir())
	good, err := AssignLanded("6666666666666666666666666666666666666666", "feat(hooks): keep (fak hooks)", "", landAt)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := RecordLanding(path, good); err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	// A torn line, a foreign-schema line, and blank noise.
	if _, err := f.WriteString(`{"schema":"fak.steerpr.landin` + "\n" + `{"schema":"someone.else.v9","sha":"x"}` + "\n\n"); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	rows := LoadLandings(path)
	if len(rows) != 1 || rows[0].SHA != good.SHA {
		t.Fatalf("torn/foreign rows poisoned the ledger: got %+v", rows)
	}
	if unit, ok := UnitOfLanding(rows, good.SHA); !ok || unit != "hooks" {
		t.Errorf("UnitOfLanding = %q, %v; want \"hooks\", true", unit, ok)
	}
}

// TestLandingCarriesNoBand is the anti-oracle assertion. The band needs the
// witness rung, which may legitimately not exist yet at land time; a guessed
// band that later read CLEARED would launder unwitnessed work. So the row must
// carry no band-shaped field AT ALL — asserted over the struct's shape, not over
// one constructed value, so adding such a field later fails here rather than
// silently shipping.
func TestLandingCarriesNoBand(t *testing.T) {
	forbidden := []string{"band", "verdict", "attention", "cleared", "rung"}
	rt := reflect.TypeOf(Landing{})
	for i := 0; i < rt.NumField(); i++ {
		name := strings.ToLower(rt.Field(i).Name)
		tag := strings.ToLower(rt.Field(i).Tag.Get("json"))
		for _, bad := range forbidden {
			if strings.Contains(name, bad) || strings.Contains(tag, bad) {
				t.Errorf("Landing carries a %q-shaped field (%s): land time must not resolve the band", bad, rt.Field(i).Name)
			}
		}
	}
	// And the serialized row genuinely has no band key.
	l, err := AssignLanded("7777777777777777777777777777777777777777", "feat(hooks): x (fak hooks)", "", landAt)
	if err != nil {
		t.Fatal(err)
	}
	buf, err := json.Marshal(l)
	if err != nil {
		t.Fatal(err)
	}
	for _, bad := range forbidden {
		if strings.Contains(strings.ToLower(string(buf)), `"`+bad) {
			t.Errorf("serialized landing contains a %q key: %s", bad, buf)
		}
	}
}

// TestCheckLandingRefusesAnIncoherentRow pins the row-level invariants: the
// partition stays total and disjoint (assigned XOR orphan), the grouping basis
// is stated rather than implied, and a row that cannot name its commit or its
// time is not a cache entry.
func TestCheckLandingRefusesAnIncoherentRow(t *testing.T) {
	ok := Landing{Schema: LandingSchema, At: landAt.Format(time.RFC3339), SHA: "abc", Unit: "hooks", GroupedBy: GroupedByLeaf}
	if err := CheckLanding(ok); err != nil {
		t.Fatalf("a coherent row was refused: %v", err)
	}
	orphan := ok
	orphan.Unit, orphan.Orphan = "", true
	if err := CheckLanding(orphan); err != nil {
		t.Fatalf("a coherent orphan row was refused: %v", err)
	}

	bad := map[string]Landing{
		"foreign schema":   {Schema: "other", At: ok.At, SHA: "abc", Unit: "hooks", GroupedBy: GroupedByLeaf},
		"no commit":        {Schema: LandingSchema, At: ok.At, Unit: "hooks", GroupedBy: GroupedByLeaf},
		"unparseable time": {Schema: LandingSchema, At: "yesterday", SHA: "abc", Unit: "hooks", GroupedBy: GroupedByLeaf},
		"implied grouping": {Schema: LandingSchema, At: ok.At, SHA: "abc", Unit: "hooks"},
		"orphan with unit": {Schema: LandingSchema, At: ok.At, SHA: "abc", Unit: "hooks", GroupedBy: GroupedByLeaf, Orphan: true},
		"neither":          {Schema: LandingSchema, At: ok.At, SHA: "abc", GroupedBy: GroupedByLeaf},
	}
	for name, l := range bad {
		if err := CheckLanding(l); err == nil {
			t.Errorf("%s: accepted an incoherent row", name)
		}
	}
}

// TestAssignLandedRefusesAnUnnamedCommit — a row that cannot name its commit or
// carry the subject the unit is parsed from is noise, not a cache entry. An
// UNSTAMPED subject is NOT an error: it is an orphan, exactly as the fold treats it.
func TestAssignLandedRefusesAnUnnamedCommit(t *testing.T) {
	if _, err := AssignLanded("", "feat(hooks): x (fak hooks)", "", landAt); err == nil {
		t.Error("accepted a landing with no sha")
	}
	if _, err := AssignLanded("abc", "   ", "", landAt); err == nil {
		t.Error("accepted a landing with no subject")
	}
	l, err := AssignLanded("abc", "chore: unstamped", "", landAt)
	if err != nil {
		t.Fatalf("an unstamped subject must yield an orphan, not an error: %v", err)
	}
	if !l.Orphan || l.Unit != "" {
		t.Errorf("unstamped subject gave unit %q orphan=%v; want orphan", l.Unit, l.Orphan)
	}
}

// TestRecordLandingIsIdempotentOnADoubleFire — two triggers racing over one
// commit (the realistic duplicate) must resolve to one row, while a genuine
// re-landing after other commits is a new fact and is appended.
func TestRecordLandingIsIdempotentOnADoubleFire(t *testing.T) {
	path := LandingLedgerPath(t.TempDir())
	first, err := AssignLanded("8888888888888888888888888888888888888888", "feat(hooks): x (fak hooks)", "", landAt)
	if err != nil {
		t.Fatal(err)
	}
	if appended, err := RecordLanding(path, first); err != nil || !appended {
		t.Fatalf("first record: appended=%v err=%v", appended, err)
	}
	if appended, err := RecordLanding(path, first); err != nil || appended {
		t.Errorf("double-fire appended a duplicate row: appended=%v err=%v", appended, err)
	}
	if got := len(LoadLandings(path)); got != 1 {
		t.Fatalf("ledger holds %d rows after a double-fire, want 1", got)
	}
}

// TestLandingSeamStaysWithinLatencyBudget is the GATE_LATENCY_REGRESSION
// assertion. The commit-hook path is hot, shared, and latency-budgeted, so the
// added cost is measured rather than asserted to be small. The budget is far
// above the real cost on purpose: it fires when someone teaches this path to
// shell out to git, take a lock, or rewrite the whole ledger — not to grade the
// current implementation.
func TestLandingSeamStaysWithinLatencyBudget(t *testing.T) {
	path := LandingLedgerPath(t.TempDir())
	const n = 200

	start := time.Now()
	for i := 0; i < n; i++ {
		sha := strings.Repeat("a", 39) + string(rune('0'+i%10))
		// A distinct unit per iteration keeps the idempotence short-circuit from
		// turning most iterations into a no-op read.
		l, err := AssignLanded(sha, "feat(lane"+string(rune('a'+i%26))+"): x (fak lane"+string(rune('a'+i%26))+")", "", landAt)
		if err != nil {
			t.Fatalf("iteration %d: %v", i, err)
		}
		if _, err := RecordLanding(path, l); err != nil {
			t.Fatalf("iteration %d: %v", i, err)
		}
	}
	per := time.Since(start) / n

	t.Logf("land-time seam: %v per commit over %d commits (budget %v)", per, n, LandingLatencyBudget)
	if per > LandingLatencyBudget {
		t.Errorf("GATE_LATENCY_REGRESSION: %v per commit exceeds the %v budget", per, LandingLatencyBudget)
	}
}
