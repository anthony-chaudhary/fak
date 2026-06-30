package rsiloop

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

// improveRow is a terse constructor for an "improve" journal row in the meta tests.
func improveRow(decision string, kept, truthClean, suiteGreen bool) Row {
	return Row{Mode: "improve", Decision: decision, Kept: kept, TruthClean: truthClean, SuiteGreen: suiteGreen}
}

// TestMetaFold_ClusteredEscalationProposesBoundedAdjustment is acceptance bullet 1: a
// clustered-escalation fixture produces a BOUNDED keep-policy proposal, a quiet
// journal produces none, and the proposal can never exceed the configured ceiling.
func TestMetaFold_ClusteredEscalationProposesBoundedAdjustment(t *testing.T) {
	cfg := DefaultMetaConfig()
	cur := KeepPolicy{GainThreshold: 0.10, BreakerK: 3, Throttle: 4}

	// A journal where the breaker escalated repeatedly (the cluster).
	clustered := []Row{
		improveRow("REVERT", false, true, true),
		improveRow("ESCALATE", false, false, true),
		improveRow("REVERT", false, true, true),
		improveRow("ESCALATE", false, false, true),
		improveRow("KEEP", true, true, true),
	}
	p, ok := Fold(clustered, cur, cfg)
	if !ok {
		t.Fatal("clustered escalations produced no proposal, want one")
	}
	if p.Knob != KnobGainThreshold {
		t.Errorf("proposal knob = %s, want gain_threshold", p.Knob)
	}
	if !(p.After > p.Before) {
		t.Errorf("proposal did not tighten: before=%v after=%v", p.Before, p.After)
	}
	if p.After > cfg.GainCeiling {
		t.Errorf("proposal %v exceeds the ceiling %v — not bounded", p.After, cfg.GainCeiling)
	}
	if p.Before != cur.GainThreshold {
		t.Errorf("proposal Before=%v, want the current policy %v", p.Before, cur.GainThreshold)
	}
	if p.Escalations < cfg.MinEscalations {
		t.Errorf("proposal cites %d escalations, want >= %d", p.Escalations, cfg.MinEscalations)
	}
	// Fold must be pure: cur is unchanged.
	if cur.GainThreshold != 0.10 {
		t.Errorf("Fold mutated the input policy: %v", cur.GainThreshold)
	}

	// A quiet journal (one lone escalation, below the cluster threshold) proposes nothing.
	quiet := []Row{
		improveRow("KEEP", true, true, true),
		improveRow("ESCALATE", false, false, true),
		improveRow("KEEP", true, true, true),
	}
	if _, ok := Fold(quiet, cur, cfg); ok {
		t.Error("a single escalation tripped a proposal — clustering not enforced")
	}

	// Bounded: a policy already at the ceiling proposes nothing (no unbounded swing).
	atCeil := KeepPolicy{GainThreshold: cfg.GainCeiling}
	if _, ok := Fold(clustered, atCeil, cfg); ok {
		t.Error("a policy at the ceiling produced a proposal — bound not enforced")
	}
}

// TestMetaFold_LooseningTruthCleanIsReverted is acceptance bullet 2 — the load-bearing
// anti-goodhart pinned test: a proposal that raises the RAW keep count by admitting
// truth-DIRTY keeps does NOT raise the truth-clean keep-rate, so EvaluateProposal (the
// reused shipgate keep-bit) REVERTS it. The companion case proves the gate is not just
// "always revert": a genuine tightening that raises truth-clean keeps is KEPT.
func TestMetaFold_LooseningTruthCleanIsReverted(t *testing.T) {
	// Baseline policy: 10 cycles, 3 truth-clean keeps -> rate 0.3.
	before := []Row{
		improveRow("KEEP", true, true, true),
		improveRow("KEEP", true, true, true),
		improveRow("KEEP", true, true, true),
		improveRow("REVERT", false, true, true),
		improveRow("REVERT", false, true, true),
		improveRow("REVERT", false, true, true),
		improveRow("REVERT", false, true, true),
		improveRow("REVERT", false, true, true),
		improveRow("REVERT", false, false, true),
		improveRow("REVERT", false, false, true),
	}

	// LOOSENED policy: the gate now admits 4 MORE keeps, but they are truth-DIRTY
	// (TruthClean=false). Raw keeps rose 3 -> 7; truth-clean keeps are still 3.
	loosened := []Row{
		improveRow("KEEP", true, true, true),
		improveRow("KEEP", true, true, true),
		improveRow("KEEP", true, true, true),
		improveRow("KEEP", true, false, true), // slop-keep
		improveRow("KEEP", true, false, true), // slop-keep
		improveRow("KEEP", true, false, true), // slop-keep
		improveRow("KEEP", true, false, true), // slop-keep
		improveRow("REVERT", false, true, true),
		improveRow("REVERT", false, true, true),
		improveRow("REVERT", false, true, true),
	}

	if got := KeepRateTruthClean(before); got != 0.3 {
		t.Fatalf("before truth-clean keep-rate = %v, want 0.3", got)
	}
	if got := KeepRateTruthClean(loosened); got > KeepRateTruthClean(before) {
		t.Fatalf("loosening RAISED the truth-clean keep-rate (%v > %v) — the meta-objective is gameable", got, KeepRateTruthClean(before))
	}

	dec, w := EvaluateProposal(before, loosened)
	if dec != shipgate.REVERT {
		t.Errorf("loosening truth-clean was %s, want REVERT", dec)
	}
	if w.Kept() {
		t.Error("loosening truth-clean set the non-forgeable keep-bit — fence breached")
	}

	// Companion: a GENUINE tightening — 5 truth-clean keeps, no slop -> rate 0.5 > 0.3 — IS kept.
	tightened := []Row{
		improveRow("KEEP", true, true, true),
		improveRow("KEEP", true, true, true),
		improveRow("KEEP", true, true, true),
		improveRow("KEEP", true, true, true),
		improveRow("KEEP", true, true, true),
		improveRow("REVERT", false, true, true),
		improveRow("REVERT", false, true, true),
		improveRow("REVERT", false, true, true),
		improveRow("REVERT", false, true, true),
		improveRow("REVERT", false, true, true),
	}
	dec2, w2 := EvaluateProposal(before, tightened)
	if dec2 != shipgate.KEEP {
		t.Errorf("genuine truth-clean gain was %s, want KEEP", dec2)
	}
	if !w2.Kept() {
		t.Error("genuine truth-clean gain did not set the keep-bit")
	}
}

// TestMetaApply_ProposeOnlyByDefault_ApplyExplicitAndLogged is acceptance bullet 3:
// propose-only by default (a no-op that still logs), and an explicit allow=true apply
// that mutates the policy and logs the change. Apply never mutates the caller's policy.
func TestMetaApply_ProposeOnlyByDefault_ApplyExplicitAndLogged(t *testing.T) {
	cur := KeepPolicy{GainThreshold: 0.10, BreakerK: 3, Throttle: 4}
	p := Proposal{Knob: KnobGainThreshold, Before: 0.10, After: 0.15, Rationale: "test"}

	// Default: propose-only — no change, but a record/log exists.
	noop := Apply(cur, p, false)
	if noop.Applied {
		t.Error("Apply(allow=false) reported Applied=true — not propose-only by default")
	}
	if noop.Policy != cur {
		t.Errorf("Apply(allow=false) changed the policy to %+v, want unchanged %+v", noop.Policy, cur)
	}
	if noop.Log == "" {
		t.Error("propose-only apply left no log line")
	}

	// Explicit --apply: the change lands and is logged.
	applied := Apply(cur, p, true)
	if !applied.Applied {
		t.Error("Apply(allow=true) reported Applied=false")
	}
	if applied.Policy.GainThreshold != p.After {
		t.Errorf("applied policy gain = %v, want %v", applied.Policy.GainThreshold, p.After)
	}
	if applied.Log == "" {
		t.Error("applied retune left no log line")
	}
	// Apply must not mutate the caller's policy value.
	if cur.GainThreshold != 0.10 {
		t.Errorf("Apply mutated the caller's policy: %v", cur.GainThreshold)
	}
}

func TestMetaApplyRunner_KeepIsProposeOnlyByDefault(t *testing.T) {
	cur := KeepPolicy{GainThreshold: 0.10, BreakerK: 3, Throttle: 4}
	p := Proposal{Knob: KnobGainThreshold, Before: 0.10, After: 0.15, Rationale: "test"}
	before := []Row{
		improveRow("KEEP", true, true, true),
		improveRow("REVERT", false, true, true),
		improveRow("REVERT", false, true, true),
	}
	after := []Row{
		improveRow("KEEP", true, true, true),
		improveRow("KEEP", true, true, true),
		improveRow("REVERT", false, true, true),
	}
	var measured KeepPolicy
	rec, err := ApplyProposalWithWitness(cur, before, p, false, "worktree:/tmp/meta-rsi", func(pol KeepPolicy) ([]Row, error) {
		measured = pol
		return after, nil
	})
	if err != nil {
		t.Fatalf("ApplyProposalWithWitness: %v", err)
	}
	if measured.GainThreshold != p.After {
		t.Fatalf("witness measured policy gain=%v, want proposal after=%v", measured.GainThreshold, p.After)
	}
	if rec.Decision != shipgate.KEEP || !rec.Witness.Kept() {
		t.Fatalf("witnessed decision = %s kept=%v, want KEEP", rec.Decision, rec.Witness.Kept())
	}
	if rec.Applied {
		t.Fatal("allow=false applied a kept proposal; default must remain propose-only")
	}
	if rec.Policy != cur {
		t.Fatalf("propose-only policy = %+v, want unchanged %+v", rec.Policy, cur)
	}
	if rec.WitnessRef != "worktree:/tmp/meta-rsi" || !strings.Contains(rec.Log, "propose-only") {
		t.Fatalf("audit record missing witness/propose-only detail: %+v", rec)
	}
	if rec.Score == nil || rec.Score.Name != MetaMetricName || rec.Score.Grade != "kept" {
		t.Fatalf("meta-RSI apply record missing scorecard: %+v", rec.Score)
	}
	if got := scoreComponentValue(rec.Score, "rate_delta"); got <= 0 {
		t.Fatalf("meta-RSI score should expose positive truth-clean rate delta, got %.3f in %+v", got, rec.Score)
	}
}

func TestMetaApplyRunner_ExplicitApplyAfterWitnessKeep(t *testing.T) {
	cur := KeepPolicy{GainThreshold: 0.10, BreakerK: 3, Throttle: 4}
	p := Proposal{Knob: KnobGainThreshold, Before: 0.10, After: 0.15, Rationale: "test"}
	before := []Row{
		improveRow("KEEP", true, true, true),
		improveRow("REVERT", false, true, true),
	}
	after := []Row{
		improveRow("KEEP", true, true, true),
		improveRow("KEEP", true, true, true),
	}
	rec, err := ApplyProposalWithWitness(cur, before, p, true, "worktree:/tmp/meta-rsi", func(KeepPolicy) ([]Row, error) {
		return after, nil
	})
	if err != nil {
		t.Fatalf("ApplyProposalWithWitness: %v", err)
	}
	if rec.Decision != shipgate.KEEP || !rec.Applied {
		t.Fatalf("explicit apply decision/applied = %s/%v, want KEEP/true", rec.Decision, rec.Applied)
	}
	if rec.Policy.GainThreshold != p.After {
		t.Fatalf("applied policy gain=%v, want %v", rec.Policy.GainThreshold, p.After)
	}
	if !strings.Contains(rec.Log, "APPLIED") {
		t.Fatalf("apply log missing APPLIED: %q", rec.Log)
	}
}

func TestMetaApplyRunner_RevertsLooseningEvenWhenAllowed(t *testing.T) {
	cur := KeepPolicy{GainThreshold: 0.10, BreakerK: 3, Throttle: 4}
	p := Proposal{Knob: KnobGainThreshold, Before: 0.10, After: 0.15, Rationale: "test"}
	before := []Row{
		improveRow("KEEP", true, true, true),
		improveRow("KEEP", true, true, true),
		improveRow("KEEP", true, true, true),
		improveRow("REVERT", false, true, true),
		improveRow("REVERT", false, true, true),
		improveRow("REVERT", false, true, true),
		improveRow("REVERT", false, true, true),
		improveRow("REVERT", false, true, true),
		improveRow("REVERT", false, false, true),
		improveRow("REVERT", false, false, true),
	}
	loosened := []Row{
		improveRow("KEEP", true, true, true),
		improveRow("KEEP", true, true, true),
		improveRow("KEEP", true, true, true),
		improveRow("KEEP", true, false, true),
		improveRow("KEEP", true, false, true),
		improveRow("KEEP", true, false, true),
		improveRow("KEEP", true, false, true),
		improveRow("REVERT", false, true, true),
		improveRow("REVERT", false, true, true),
		improveRow("REVERT", false, true, true),
	}
	rec, err := ApplyProposalWithWitness(cur, before, p, true, "worktree:/tmp/meta-rsi", func(KeepPolicy) ([]Row, error) {
		return loosened, nil
	})
	if err != nil {
		t.Fatalf("ApplyProposalWithWitness: %v", err)
	}
	if rec.Decision != shipgate.REVERT || rec.Applied {
		t.Fatalf("truth-dirty loosening decision/applied = %s/%v, want REVERT/false", rec.Decision, rec.Applied)
	}
	if rec.Policy != cur {
		t.Fatalf("reverted policy = %+v, want unchanged %+v", rec.Policy, cur)
	}
	if !strings.Contains(rec.Log, "REVERT") {
		t.Fatalf("revert log missing REVERT: %q", rec.Log)
	}
}

func TestMetaApplyRunnerRequiresWitnessEnvironment(t *testing.T) {
	cur := KeepPolicy{GainThreshold: 0.10}
	p := Proposal{Knob: KnobGainThreshold, Before: 0.10, After: 0.15}
	if _, err := ApplyProposalWithWitness(cur, nil, p, true, "", func(KeepPolicy) ([]Row, error) { return nil, nil }); err == nil {
		t.Fatal("empty witness ref succeeded")
	}
	if _, err := ApplyProposalWithWitness(cur, nil, p, true, "worktree:/tmp/meta-rsi", nil); err == nil {
		t.Fatal("nil witness function succeeded")
	}
}

// TestReadJournal_RoundTripsAndSkipsTornLine proves the fold's input loader reads real
// journal rows and is corruption-tolerant: a torn final line does not lose the valid
// rows before it (the fail-open discipline the regression guard depends on).
func TestReadJournal_RoundTripsAndSkipsTornLine(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "rsi.jsonl")

	j, err := NewJournal(path)
	if err != nil {
		t.Fatalf("NewJournal: %v", err)
	}
	want := []Row{
		improveRow("ESCALATE", false, false, true),
		improveRow("KEEP", true, true, true),
	}
	for _, r := range want {
		if err := j.Append(r); err != nil {
			t.Fatalf("append: %v", err)
		}
	}
	if err := j.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	// Simulate a crash mid-write: append a torn (non-JSON) final line.
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if _, err := f.WriteString(`{"mode":"improve","decision":"KE`); err != nil {
		t.Fatalf("write torn: %v", err)
	}
	f.Close()

	got, err := ReadJournal(path)
	if err != nil {
		t.Fatalf("ReadJournal: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("ReadJournal returned %d rows, want %d (torn line should be skipped, valid rows kept)", len(got), len(want))
	}
	if got[0].Decision != "ESCALATE" || got[1].Decision != "KEEP" {
		t.Errorf("rows out of order or wrong: %+v", got)
	}

	// A missing file is (nil, nil): no history, no crash.
	rows, err := ReadJournal(filepath.Join(dir, "absent.jsonl"))
	if err != nil || rows != nil {
		t.Errorf("ReadJournal(missing) = (%v, %v), want (nil, nil)", rows, err)
	}
}
