package cachevaluereport

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardvars"
)

func censusNow(t *testing.T) time.Time {
	t.Helper()
	return time.Date(2026, 7, 26, 12, 0, 0, 0, time.UTC)
}

func mustPct(t *testing.T, p *float64, want float64, what string) {
	t.Helper()
	if p == nil {
		t.Fatalf("%s: want %.1f, got nil", what, want)
	}
	if diff := *p - want; diff > 0.01 || diff < -0.01 {
		t.Fatalf("%s = %.4f, want %.1f", what, *p, want)
	}
}

// TestFoldCensusMostlyPassiveFleet is the issue's headline case (#3650): a synthetic fleet
// where most workers run PASSIVE must be VISIBLE as such, with both ratios reported.
func TestFoldCensusMostlyPassiveFleet(t *testing.T) {
	rows := []WorkerRow{
		RowFromVars("w-active-fired", &guardvars.ManagedCacheVars{Active: true, Upgraded: 12}),
		RowFromVars("w-active-silent", &guardvars.ManagedCacheVars{Active: true, Inert: true}),
		RowFromVars("w-passive-a", nil), // no managed_cache block => affirmative PASSIVE
		RowFromVars("w-passive-b", nil),
		RowFromVars("w-passive-c", nil),
		RowFromVars("w-passive-d", &guardvars.ManagedCacheVars{Active: false, Upgraded: 0}),
	}
	rep := FoldCensus(rows, censusNow(t))

	if rep.Schema != CensusSchema {
		t.Fatalf("schema = %q, want %q", rep.Schema, CensusSchema)
	}
	if !rep.OK {
		t.Fatalf("census is a report, not a gate: OK must stay true, got %v", rep.OK)
	}
	if rep.Verdict != VerdictMostlyPassive {
		t.Fatalf("verdict = %q, want %q", rep.Verdict, VerdictMostlyPassive)
	}
	if rep.Workers != 6 || rep.Reached != 6 || rep.Unreached != 0 {
		t.Fatalf("coverage = workers %d reached %d unreached %d, want 6/6/0", rep.Workers, rep.Reached, rep.Unreached)
	}
	if rep.Active != 2 || rep.Passive != 4 {
		t.Fatalf("active/passive = %d/%d, want 2/4", rep.Active, rep.Passive)
	}
	mustPct(t, rep.ActivePct, 100*2.0/6.0, "ActivePct")

	// Among the 2 ACTIVE workers (both on the Anthropic wire, which HAS the lever), 1 fired.
	if rep.ActiveWithLever != 2 || rep.UpgradeFired != 1 {
		t.Fatalf("upgrade-fired among active = %d/%d, want 1/2", rep.UpgradeFired, rep.ActiveWithLever)
	}
	mustPct(t, rep.UpgradeFiredPct, 50, "UpgradeFiredPct")

	if !strings.Contains(rep.Finding, "MOSTLY_PASSIVE") {
		t.Fatalf("finding must name the verdict: %q", rep.Finding)
	}
	if !strings.Contains(rep.NextAction, "PASSIVE") {
		t.Fatalf("next action must point at the passive fleet: %q", rep.NextAction)
	}
}

// TestFoldCensusAdoptedFleet pins the other side of the majority split.
func TestFoldCensusAdoptedFleet(t *testing.T) {
	rows := []WorkerRow{
		RowFromVars("w1", &guardvars.ManagedCacheVars{Active: true, Upgraded: 3}),
		RowFromVars("w2", &guardvars.ManagedCacheVars{Active: true, Upgraded: 1}),
		RowFromVars("w3", &guardvars.ManagedCacheVars{Active: true, Upgraded: 0}),
		RowFromVars("w4", nil),
	}
	rep := FoldCensus(rows, censusNow(t))
	if rep.Verdict != VerdictAdopted {
		t.Fatalf("verdict = %q, want %q", rep.Verdict, VerdictAdopted)
	}
	mustPct(t, rep.ActivePct, 75, "ActivePct")
	mustPct(t, rep.UpgradeFiredPct, 100*2.0/3.0, "UpgradeFiredPct")
}

// TestFoldCensusUnreachedWorkersExcludedFromRatios is the honesty fence: a worker the
// scrape could not reach carries NO posture claim, so it must not be folded in as PASSIVE
// (which would manufacture a low adoption number out of a network failure).
func TestFoldCensusUnreachedWorkersExcludedFromRatios(t *testing.T) {
	rows := []WorkerRow{
		RowFromVars("reachable-active", &guardvars.ManagedCacheVars{Active: true, Upgraded: 5}),
		UnreachedRow("gone-a"),
		UnreachedRow("gone-b"),
		UnreachedRow("gone-c"),
	}
	rep := FoldCensus(rows, censusNow(t))

	if rep.Workers != 4 || rep.Reached != 1 || rep.Unreached != 3 {
		t.Fatalf("coverage = workers %d reached %d unreached %d, want 4/1/3", rep.Workers, rep.Reached, rep.Unreached)
	}
	if rep.Passive != 0 {
		t.Fatalf("unreachable workers must not count as PASSIVE, got passive=%d", rep.Passive)
	}
	// The one worker actually observed was ACTIVE, so the ACTIVE share is 100% of REACHED.
	mustPct(t, rep.ActivePct, 100, "ActivePct")
	if rep.Verdict != VerdictAdopted {
		t.Fatalf("verdict = %q, want %q", rep.Verdict, VerdictAdopted)
	}
	if !strings.Contains(rep.Finding, "3 unreachable worker(s) excluded") {
		t.Fatalf("finding must disclose the excluded workers: %q", rep.Finding)
	}
}

// TestFoldCensusLeverlessWireExcludedFromUpgradeRatio pins the wire-awareness the producer
// already enforces: on the OpenAI Responses (codex) wire there IS no 1h-TTL upgrade lever,
// so an ACTIVE worker there can never fire one. Counting it in the upgrade denominator
// would report a fleet-wide upgrade failure that is really a wire without the lever.
func TestFoldCensusLeverlessWireExcludedFromUpgradeRatio(t *testing.T) {
	rows := []WorkerRow{
		RowFromVars("anthropic-fired", &guardvars.ManagedCacheVars{Active: true, Upgraded: 4}),
		RowFromVars("codex-1", &guardvars.ManagedCacheVars{Active: true, Wire: guardvars.WireOpenAIResponses}),
		RowFromVars("codex-2", &guardvars.ManagedCacheVars{Active: true, Wire: guardvars.WireOpenAIResponses}),
	}
	rep := FoldCensus(rows, censusNow(t))

	if rep.Active != 3 {
		t.Fatalf("active = %d, want 3 (the leverless wire is still ACTIVE posture)", rep.Active)
	}
	if rep.ActiveWithLever != 1 || rep.ActiveLeverless != 2 {
		t.Fatalf("lever split = %d with / %d without, want 1/2", rep.ActiveWithLever, rep.ActiveLeverless)
	}
	mustPct(t, rep.ActivePct, 100, "ActivePct")
	mustPct(t, rep.UpgradeFiredPct, 100, "UpgradeFiredPct")
	if !strings.Contains(rep.Finding, "no 1h-TTL lever excluded from the upgrade ratio") {
		t.Fatalf("finding must disclose the leverless exclusion: %q", rep.Finding)
	}
}

// TestFoldCensusActiveButNoneFired is the fleet-scale sibling of the per-session
// CONFIGURED_BUT_INERT finding: posture ADOPTED, zero upgrades anywhere.
func TestFoldCensusActiveButNoneFired(t *testing.T) {
	rows := []WorkerRow{
		RowFromVars("w1", &guardvars.ManagedCacheVars{Active: true, Inert: true}),
		RowFromVars("w2", &guardvars.ManagedCacheVars{Active: true, Inert: true}),
	}
	rep := FoldCensus(rows, censusNow(t))
	if rep.Verdict != VerdictAdopted {
		t.Fatalf("verdict = %q, want %q", rep.Verdict, VerdictAdopted)
	}
	mustPct(t, rep.UpgradeFiredPct, 0, "UpgradeFiredPct")
	if !strings.Contains(rep.NextAction, "not one worker fired an upgrade") {
		t.Fatalf("next action must name the unfired fleet: %q", rep.NextAction)
	}
}

// TestFoldCensusInsufficient pins the two honest no-evidence shapes: an empty fleet, and a
// fleet where every scrape failed. Neither may report a 0% adoption number.
func TestFoldCensusInsufficient(t *testing.T) {
	empty := FoldCensus(nil, censusNow(t))
	if empty.Verdict != VerdictInsufficient {
		t.Fatalf("empty fleet verdict = %q, want %q", empty.Verdict, VerdictInsufficient)
	}
	if empty.ActivePct != nil || empty.UpgradeFiredPct != nil {
		t.Fatalf("empty fleet must report nil ratios, got active=%v upgrade=%v", empty.ActivePct, empty.UpgradeFiredPct)
	}

	dark := FoldCensus([]WorkerRow{UnreachedRow("a"), UnreachedRow("b")}, censusNow(t))
	if dark.Verdict != VerdictInsufficient {
		t.Fatalf("all-unreachable verdict = %q, want %q", dark.Verdict, VerdictInsufficient)
	}
	if dark.ActivePct != nil {
		t.Fatalf("all-unreachable fleet must not report an ACTIVE share, got %v", *dark.ActivePct)
	}
	if !strings.Contains(dark.Finding, "none reachable") {
		t.Fatalf("finding must say the fleet was dark: %q", dark.Finding)
	}
}

// TestRowFromVarsAbsentBlockIsAffirmativePassive pins the producer contract this census
// leans on: internal/gateway.managedCacheVars omits the managed_cache block ONLY when the
// lever is off and nothing was observed, so an absent block is a PASSIVE witness — not an
// unknown, and not a scrape failure.
func TestRowFromVarsAbsentBlockIsAffirmativePassive(t *testing.T) {
	row := RowFromVars("w", nil)
	if !row.Reached {
		t.Fatalf("a worker that answered is Reached, got %v", row.Reached)
	}
	if row.Published {
		t.Fatalf("an absent block must not read as published, got %v", row.Published)
	}
	if got := row.State(); got != StatePassive {
		t.Fatalf("state = %q, want %q", got, StatePassive)
	}

	unreached := UnreachedRow("w")
	if got := unreached.State(); got != StateUnknown {
		t.Fatalf("unreached state = %q, want %q", got, StateUnknown)
	}
}

// TestFoldCensusDeterministicOrder pins the stable breakdown order (ACTIVE, PASSIVE, then
// UNKNOWN; label within a state) so the rendered census and its JSON never churn.
func TestFoldCensusDeterministicOrder(t *testing.T) {
	rows := []WorkerRow{
		UnreachedRow("zz-gone"),
		RowFromVars("b-passive", nil),
		RowFromVars("a-active", &guardvars.ManagedCacheVars{Active: true, Upgraded: 1}),
		RowFromVars("a-passive", nil),
	}
	rep := FoldCensus(rows, censusNow(t))
	want := []string{"a-active", "a-passive", "b-passive", "zz-gone"}
	if len(rep.Rows) != len(want) {
		t.Fatalf("rows = %d, want %d", len(rep.Rows), len(want))
	}
	for i, w := range want {
		if rep.Rows[i].Worker != w {
			t.Fatalf("row[%d] = %q, want %q", i, rep.Rows[i].Worker, w)
		}
	}
	// The fold is pure: same input, same output.
	again := FoldCensus(rows, censusNow(t))
	a, _ := json.Marshal(rep)
	b, _ := json.Marshal(again)
	if string(a) != string(b) {
		t.Fatalf("fold is not deterministic:\n%s\n%s", a, b)
	}
}

// TestRenderCensusNamesBothHeadlines is the census ARTIFACT witness: the rendered report
// must carry the fleet %ACTIVE and the %upgrade-fired-among-active the issue asks for.
func TestRenderCensusNamesBothHeadlines(t *testing.T) {
	rows := []WorkerRow{
		RowFromVars("w-active-fired", &guardvars.ManagedCacheVars{Active: true, Upgraded: 7}),
		RowFromVars("w-passive-a", nil),
		RowFromVars("w-passive-b", nil),
		UnreachedRow("w-gone"),
	}
	out := RenderCensus(FoldCensus(rows, censusNow(t)))

	for _, want := range []string{
		"fleet managed-cache posture census",
		VerdictMostlyPassive,
		"ACTIVE share 33%",
		"upgrade fired 1/1",
		"workers 4 (reached 3, unreachable 1)",
		"w-active-fired",
		StateUnknown,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("rendered census missing %q:\n%s", want, out)
		}
	}
}

// TestFoldCensusUnlabelledWorkerStaysAddressable keeps every row nameable in the artifact.
func TestFoldCensusUnlabelledWorkerStaysAddressable(t *testing.T) {
	rep := FoldCensus([]WorkerRow{RowFromVars("   ", &guardvars.ManagedCacheVars{Active: true})}, censusNow(t))
	if len(rep.Rows) != 1 || rep.Rows[0].Worker != "unknown" {
		t.Fatalf("unlabelled worker = %+v, want label %q", rep.Rows, "unknown")
	}
}
