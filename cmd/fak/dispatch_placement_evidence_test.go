package main

import (
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

const placementEvidenceRoster = `{
  "version": "fak-accounts/v1",
  "accounts": [
    {"id": "laptop", "kind": "local", "base_url": "http://127.0.0.1:11434/v1"},
    {"id": "cluster", "kind": "fleet", "base_url": "http://glm.infer.internal:8000/v1"},
    {"id": "frontier", "kind": "anthropic", "cred_env": "ANTHROPIC_API_KEY"}
  ],
  "bindings": [
    {"model": "qwen3.6-4b", "account": "laptop"},
    {"model": "glm-5.2", "account": "cluster"},
    {"model": "opus-5", "account": "frontier"}
  ],
  "default": "laptop"
}`

func writePlacementRoster(t *testing.T, dir string) string {
	t.Helper()
	p := filepath.Join(dir, "accounts.json")
	if err := os.WriteFile(p, []byte(placementEvidenceRoster), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// setDispatchPlacementEvidence declares the placement-evidence setting for the duration of one
// test and restores it afterwards — what a test does now that the setting lives on the tick's
// config surface instead of in the process environment (t.Setenv's job before #5416's reads
// were relocated off the environment).
func setDispatchPlacementEvidence(t *testing.T, on bool) {
	t.Helper()
	old := dispatchPlacementEvidence
	dispatchPlacementEvidence = on
	t.Cleanup(func() { dispatchPlacementEvidence = old })
}

// setDispatchAccountsRoster declares the attribution roster override the same way.
func setDispatchAccountsRoster(t *testing.T, path string) {
	t.Helper()
	old := dispatchAccountsRoster
	dispatchAccountsRoster = path
	t.Cleanup(func() { dispatchAccountsRoster = old })
}

func TestPlacementEvidenceStaysOffUntilAnOperatorTurnsItOn(t *testing.T) {
	// The whole seam writes sidecars, payload keys, and a journal file. Defaulting it on
	// would change every existing fleet's runs directory and payload shape without anyone
	// asking, so a tick that DECLARES nothing must read as off — and the declaration has to be
	// on the command surface, where --help names it and a caller sets it per invocation, not in
	// the ambient environment (internal/envconfiglint, CONFIG_NOT_ENV).
	silent, _, code := parseDispatchTickFlags(io.Discard, []string{"--workspace", t.TempDir()})
	if code != 0 {
		t.Fatalf("parse of a bare tick failed with code %d", code)
	}
	if silent.PlacementEvidence || silent.AccountsRoster != "" {
		t.Errorf("an undeclared tick opted in: evidence=%v roster=%q", silent.PlacementEvidence, silent.AccountsRoster)
	}
	declared, _, code := parseDispatchTickFlags(io.Discard, []string{
		"--workspace", t.TempDir(), "--placement-evidence", "--accounts-roster", "roster.json",
	})
	if code != 0 {
		t.Fatalf("parse of a declared tick failed with code %d", code)
	}
	if !declared.PlacementEvidence || declared.AccountsRoster != "roster.json" {
		t.Errorf("--placement-evidence/--accounts-roster did not reach the options: %+v", declared)
	}

	// And the seam the leaf helpers read is the one the tick publishes.
	setDispatchPlacementEvidence(t, false)
	if dispatchPlacementEvidenceEnabled() {
		t.Error("an undeclared setting enabled the seam")
	}
	setDispatchPlacementEvidence(t, true)
	if !dispatchPlacementEvidenceEnabled() {
		t.Error("a declared setting did not enable the seam")
	}
}

func TestARungIsAttributedOnlyForAModelTheRosterBinds(t *testing.T) {
	// The roster's default is the LOCAL account, so a resolver built on Roster.Resolve would
	// report every unbound id as running on this box — counting a vendor call as a saving.
	dir := t.TempDir()
	setDispatchAccountsRoster(t, writePlacementRoster(t, dir))
	resolve := dispatchZoneResolver(dir)
	if resolve == nil {
		t.Fatal("no resolver from a valid roster")
	}
	for model, want := range map[string]modelroute.PlacementZone{
		"qwen3.6-4b": modelroute.ZoneDevice,
		"glm-5.2":    modelroute.ZoneFleet,
		"opus-5":     modelroute.ZoneVendor,
	} {
		z, why := dispatchtick.AttributeZone(resolve, model)
		if z != want || why != dispatchtick.ZoneAttributed {
			t.Errorf("%q attributed %q/%q, want %q", model, z, why, want)
		}
	}
	for _, unbound := range []string{"claude-opus-5", "qwen3.6-4b-typo", "sonnet"} {
		z, why := dispatchtick.AttributeZone(resolve, unbound)
		if z != "" || why != dispatchtick.ZoneUnboundModel {
			t.Errorf("unbound %q attributed %q/%q — the default account's rung leaked into attribution", unbound, z, why)
		}
	}
}

func TestNoRosterAndABrokenRosterBothAttributeNothing(t *testing.T) {
	dir := t.TempDir()
	setDispatchAccountsRoster(t, "")
	if got := dispatchAccountsRosterPath(dir); got != "" {
		t.Errorf("roster path %q from an empty root", got)
	}
	if resolve := dispatchZoneResolver(dir); resolve != nil {
		t.Error("a resolver appeared with no roster on disk")
	}
	// A malformed roster must attribute nothing rather than half-attributing from a
	// partially parsed file — LoadRoster validates, and a failure here is a config work
	// item, not a rung.
	bad := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(bad, []byte(`{"version":"fak-accounts/v1","accounts":[{"id":"x","kind":"not-a-kind"}]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	setDispatchAccountsRoster(t, bad)
	if resolve := dispatchZoneResolver(dir); resolve != nil {
		t.Error("a malformed roster produced a resolver")
	}
	// And the payload then says WHY, rather than saying device.
	payload := map[string]any{}
	recordDispatchPlacementEvidence(dir, []string{"tier/T2-optimal", "tier/T2-required"}, "qwen3.6-4b", payload)
	if _, ok := payload[dispatchZoneKey]; ok {
		t.Errorf("a rung was recorded with no usable roster: %v", payload)
	}
	if got := dispatchMapString(payload, dispatchZoneUnknownKey); got != string(dispatchtick.ZoneNoRoster) {
		t.Errorf("zone reason = %q, want %q", got, dispatchtick.ZoneNoRoster)
	}
	// The class half is independent of the roster and must still be recorded.
	if got := dispatchMapString(payload, dispatchWorkClassKey); got != string(modelroute.ClassRoutine) {
		t.Errorf("class = %q, want routine", got)
	}
}

func TestTheTickRecordsAReasonForEachFactItCannotName(t *testing.T) {
	dir := t.TempDir()
	setDispatchAccountsRoster(t, writePlacementRoster(t, dir))
	// An untriaged issue on an unpinned (seat-default) slot: neither fact is knowable, and
	// both refusals must be distinct and named, because they are different work items —
	// triage the backlog vs. pin and bind the model.
	payload := map[string]any{}
	recordDispatchPlacementEvidence(dir, []string{"bug"}, "", payload)
	if _, ok := payload[dispatchWorkClassKey]; ok {
		t.Errorf("an untagged issue named a class: %v", payload)
	}
	if got := dispatchMapString(payload, dispatchWorkClassUnknownKey); got != string(dispatchtick.ClassNoTierLabel) {
		t.Errorf("class reason = %q", got)
	}
	if got := dispatchMapString(payload, dispatchZoneUnknownKey); got != string(dispatchtick.ZoneNoModelPin) {
		t.Errorf("zone reason = %q, want %q", got, dispatchtick.ZoneNoModelPin)
	}
	// A coordination slot's refusal must not be spelled the same as an untriaged one.
	pm := map[string]any{}
	recordDispatchPlacementEvidence(dir, []string{dispatchtick.PMLabel}, "glm-5.2", pm)
	if got := dispatchMapString(pm, dispatchWorkClassUnknownKey); got != string(dispatchtick.ClassCoordinationBucket) {
		t.Errorf("PM class reason = %q, want %q", got, dispatchtick.ClassCoordinationBucket)
	}
	if got := dispatchMapString(pm, dispatchZoneKey); got != string(modelroute.ZoneFleet) {
		t.Errorf("PM zone = %q, want fleet — a coordination slot still ran somewhere", got)
	}
}

func TestOnlyResolvedFactsBecomeSidecarsAndTheyRoundTrip(t *testing.T) {
	dir := t.TempDir()
	log := filepath.Join(dir, "resolve-issue-42.log")
	if err := os.WriteFile(log, []byte("worker output\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// A payload with both facts writes both files, byte-exact.
	writeDispatchPlacementSidecars(log, map[string]any{
		dispatchWorkClassKey: string(modelroute.ClassUltraHard),
		dispatchZoneKey:      string(modelroute.ZoneFleet),
	})
	for suffix, want := range map[string]string{
		dispatchtick.WorkClassSidecarSuffix: string(modelroute.ClassUltraHard),
		dispatchtick.ZoneSidecarSuffix:      string(modelroute.ZoneFleet),
	} {
		b, err := os.ReadFile(dispatchtick.SidecarPath(log, suffix))
		if err != nil {
			t.Fatalf("%s sidecar: %v", suffix, err)
		}
		if string(b) != want {
			t.Errorf("%s sidecar = %q, want %q", suffix, b, want)
		}
	}
	// A payload carrying only reasons writes NOTHING — an unconfigured or unnameable slot
	// leaves a runs directory byte-identical to before this seam.
	log2 := filepath.Join(dir, "resolve-issue-43.log")
	if err := os.WriteFile(log2, []byte("x\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	writeDispatchPlacementSidecars(log2, map[string]any{
		dispatchWorkClassUnknownKey: string(dispatchtick.ClassNoTierLabel),
		dispatchZoneUnknownKey:      string(dispatchtick.ZoneNoRoster),
	})
	for _, suffix := range []string{dispatchtick.WorkClassSidecarSuffix, dispatchtick.ZoneSidecarSuffix} {
		if _, err := os.Stat(dispatchtick.SidecarPath(log2, suffix)); err == nil {
			t.Errorf("a refusal wrote a %s sidecar", suffix)
		}
	}
	// An empty payload (the seam switched off entirely) is the same no-op.
	writeDispatchPlacementSidecars(log2, map[string]any{})
	if _, err := os.Stat(dispatchtick.SidecarPath(log2, dispatchtick.ZoneSidecarSuffix)); err == nil {
		t.Error("the disabled seam wrote a sidecar")
	}
}

// evidenceSlot writes a finished slot's log + class sidecar and returns its witness record.
func evidenceSlot(t *testing.T, runsDir string, issue int, model string, zone modelroute.PlacementZone, class modelroute.WorkClass) dispatchtick.WitnessRecord {
	t.Helper()
	name := "resolve-issue-" + strconv.Itoa(issue) + ".log"
	log := filepath.Join(runsDir, name)
	if err := os.WriteFile(log, []byte("worker output\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if class != "" {
		if err := os.WriteFile(dispatchtick.SidecarPath(log, dispatchtick.WorkClassSidecarSuffix), []byte(class), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dispatchtick.WitnessRecord{
		Issue:     issue,
		Log:       name,
		SHA:       "sha-" + strconv.Itoa(issue),
		Model:     model,
		Zone:      string(zone),
		Claim:     dispatchtick.ClaimWitnessed,
		TestClaim: dispatchtick.ClaimTestGreen,
	}
}

func TestASweepBecomesJournalRowsAGraderCanActuallyRead(t *testing.T) {
	// The end-to-end claim of this file: a live sweep produces a durable corpus that
	// modelroute reads back and grades. Before this wiring every pure piece was correct and
	// the journal on a live fleet was empty.
	runs := t.TempDir()
	records := []dispatchtick.WitnessRecord{
		evidenceSlot(t, runs, 1, "qwen3.6-4b", modelroute.ZoneDevice, modelroute.ClassRoutine),
		evidenceSlot(t, runs, 2, "qwen3.6-4b", modelroute.ZoneDevice, modelroute.ClassRoutine),
		evidenceSlot(t, runs, 3, "glm-5.2", modelroute.ZoneFleet, modelroute.ClassNormalImpl),
	}
	ev := appendDispatchTurnOutcomes(runs, records)
	if got, want := ev["produced"], 3; got != want {
		t.Fatalf("produced = %v, want %d (%v)", got, want, ev)
	}
	if got := ev["appended"]; got != 3 {
		t.Fatalf("appended = %v", got)
	}
	journal, ok := ev["journal"].(string)
	if !ok {
		t.Fatalf("no journal path in %v", ev)
	}
	f, err := os.Open(journal)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	outcomes, stats, err := modelroute.ReadTurnOutcomes(f)
	if err != nil || stats.Malformed != 0 || len(outcomes) != 3 {
		t.Fatalf("read back %d outcome(s), stats %+v, err %v", len(outcomes), stats, err)
	}
	for _, o := range outcomes {
		if o.At.IsZero() {
			t.Errorf("outcome %q undated — it could never satisfy a freshness window", o.ID)
		}
		if o.ID == "" {
			t.Errorf("outcome for %q carries no id, so a second sweep would double-count it", o.Model)
		}
		if o.Verify == modelroute.VerifyNone {
			t.Errorf("outcome %q recorded as self-reported; a witnessed slot must carry its provenance", o.ID)
		}
	}
	// And the grade the corpus supports is a real one.
	evidence, fold := modelroute.FoldTurnOutcomes(outcomes, modelroute.FoldOptions{})
	if fold.Counted != 3 || fold.Duplicates != 0 {
		t.Errorf("fold = %+v", fold)
	}
	if g := modelroute.GradeCapability("qwen3.6-4b", evidence["qwen3.6-4b"], modelroute.GradeFloor{}); g.Class != modelroute.ClassRoutine {
		t.Errorf("laptop model graded against %q, want the routine work it actually did (%s)", g.Class, g.Reason)
	}
	// The zone travelled with the row, so the headline is answerable from the journal alone.
	self := 0
	for _, o := range outcomes {
		if o.Zone.SelfHosted() {
			self++
		}
	}
	if self != 3 {
		t.Errorf("%d of 3 rows attributed to a self-hosted rung", self)
	}
	if head, _ := ev["zone_share"].(string); !strings.Contains(head, "self-hosted 100%") || !strings.Contains(head, "0 of 3 slot(s) unattributed") {
		t.Errorf("zone headline = %q", head)
	}
}

func TestASlotWithNoClassSidecarIsCountedRatherThanGuessed(t *testing.T) {
	runs := t.TempDir()
	records := []dispatchtick.WitnessRecord{
		evidenceSlot(t, runs, 1, "qwen3.6-4b", modelroute.ZoneDevice, modelroute.ClassRoutine),
		evidenceSlot(t, runs, 2, "qwen3.6-4b", modelroute.ZoneDevice, ""),            // untriaged issue
		evidenceSlot(t, runs, 3, "", modelroute.ZoneDevice, modelroute.ClassRoutine), // seat-default slot
	}
	// A hand-edited sidecar naming the class whose floor guards destructive work must not
	// reach the journal either.
	log4 := filepath.Join(runs, "resolve-issue-4.log")
	if err := os.WriteFile(log4, []byte("x\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dispatchtick.SidecarPath(log4, dispatchtick.WorkClassSidecarSuffix),
		[]byte(modelroute.ClassSecurityRelease), 0o644); err != nil {
		t.Fatal(err)
	}
	records = append(records, dispatchtick.WitnessRecord{
		Issue: 4, Log: "resolve-issue-4.log", SHA: "sha-4", Model: "qwen3.6-4b",
		Zone: string(modelroute.ZoneDevice), Claim: dispatchtick.ClaimWitnessed, TestClaim: dispatchtick.ClaimTestGreen,
	})

	ev := appendDispatchTurnOutcomes(runs, records)
	if ev["produced"] != 1 {
		t.Errorf("produced = %v, want 1 (%v)", ev["produced"], ev)
	}
	if ev["unclassified"] != 2 {
		t.Errorf("unclassified = %v, want 2 — the untriaged slot and the forged sidecar", ev["unclassified"])
	}
	if ev["unattributed"] != 1 {
		t.Errorf("unattributed = %v, want 1 — the unpinned slot names no model", ev["unattributed"])
	}
	b, err := os.ReadFile(filepath.Join(runs, dispatchTurnJournalName))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), string(modelroute.ClassSecurityRelease)) {
		t.Errorf("a hand-written sidecar minted the security-release class into the journal:\n%s", b)
	}
}

func TestASweepWithNothingToRecordWritesNoJournalAtAll(t *testing.T) {
	// An operator who enables the seam on a quiet fleet should not find a zero-byte journal
	// implying evidence exists, and a sweep with no finished slots must not create one.
	runs := t.TempDir()
	if ev := appendDispatchTurnOutcomes(runs, nil); ev != nil {
		t.Errorf("an empty sweep reported %v", ev)
	}
	records := []dispatchtick.WitnessRecord{evidenceSlot(t, runs, 1, "", "", "")}
	ev := appendDispatchTurnOutcomes(runs, records)
	if ev == nil || ev["produced"] != 0 {
		t.Fatalf("ev = %v", ev)
	}
	if _, err := os.Stat(filepath.Join(runs, dispatchTurnJournalName)); err == nil {
		t.Error("a sweep that produced no evidence created a journal file")
	}
	// The accounting still says WHY, so the operator can act on it.
	if ev["unattributed"] != 1 {
		t.Errorf("unattributed = %v", ev["unattributed"])
	}
	if head, _ := ev["zone_share"].(string); !strings.Contains(head, "1 of 1 slot(s) unattributed") {
		t.Errorf("zone headline = %q — an unmeasured fleet must not read as a measured one", head)
	}
}

func TestTheJournalAppendsAcrossSweepsRatherThanRewriting(t *testing.T) {
	// Two ticks, two sweeps. A rewrite would silently discard every earlier grade, and the
	// corpus would never grow past one tick's worth of slots.
	runs := t.TempDir()
	first := []dispatchtick.WitnessRecord{evidenceSlot(t, runs, 1, "qwen3.6-4b", modelroute.ZoneDevice, modelroute.ClassRoutine)}
	second := []dispatchtick.WitnessRecord{evidenceSlot(t, runs, 2, "qwen3.6-4b", modelroute.ZoneDevice, modelroute.ClassRoutine)}
	appendDispatchTurnOutcomes(runs, first)
	appendDispatchTurnOutcomes(runs, second)
	f, err := os.Open(filepath.Join(runs, dispatchTurnJournalName))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	outcomes, stats, err := modelroute.ReadTurnOutcomes(f)
	if err != nil || stats.Malformed != 0 {
		t.Fatalf("stats %+v err %v", stats, err)
	}
	if len(outcomes) != 2 {
		t.Fatalf("journal holds %d outcome(s) after two sweeps, want 2", len(outcomes))
	}
	// Distinct slots must carry distinct ids, or the read-side dedupe would collapse a real
	// second attempt into the first one and cap the corpus at one row per model+class.
	if outcomes[0].ID == outcomes[1].ID {
		t.Errorf("both sweeps wrote id %q", outcomes[0].ID)
	}
	_, fold := modelroute.FoldTurnOutcomes(outcomes, modelroute.FoldOptions{})
	if fold.Counted != 2 || fold.Duplicates != 0 {
		t.Errorf("fold = %+v", fold)
	}
}
