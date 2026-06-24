package main

import (
	"bytes"
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob" // registers the CAS backend the MMU pages quarantined bytes through
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// TestPoisonedResultsAreQuarantined is the load-bearing assertion for issue #573: the
// REAL context-MMU (internal/ctxmmu — the gate fak_admit / fak_syscall fold every tool
// result through) holds a poisoned tool RESULT out of the model's context, and the trap
// bytes do NOT survive into the post-admit payload the model would see. The benign
// control must be ALLOWED, proving the floor is not a blanket block.
func TestPoisonedResultsAreQuarantined(t *testing.T) {
	ctx := context.Background()
	for _, sc := range resultScenarios {
		v, qid, sees := admitThroughFAK(ctx, ctxmmu.New(), sc.Tool, []byte(sc.Body))
		if sc.Benign {
			if v.Kind != abi.VerdictAllow {
				t.Errorf("%s: benign result want ALLOW, got %s (reason %s)",
					sc.Tool, kindName(v.Kind), abi.ReasonName(v.Reason))
			}
			continue
		}
		if v.Kind != abi.VerdictQuarantine {
			t.Fatalf("%s: poisoned result want QUARANTINE, got %s (reason %s)",
				sc.Tool, kindName(v.Kind), abi.ReasonName(v.Reason))
		}
		if qid == "" {
			t.Fatalf("%s: quarantined but no quarantine_id minted (not a recoverable hold)", sc.Tool)
		}
		if sc.Trap != "" && bytes.Contains(sees, []byte(sc.Trap)) {
			t.Errorf("%s: trap %q survived into model-visible context after quarantine", sc.Tool, sc.Trap)
		}
	}
}

// TestScreenBytesFlagsBothVectors pins the deterministic signals the A/B rests on:
// the injection-shaped body and the secret-shaped body are each flagged by the regex
// floor (ctxmmu.ScreenBytes), so the quarantine in the test above is structural, not
// luck. The benign body clears the floor.
func TestScreenBytesFlagsBothVectors(t *testing.T) {
	for _, sc := range resultScenarios {
		_, hit := ctxmmu.ScreenBytes([]byte(sc.Body))
		if sc.Benign && hit {
			t.Errorf("%s: benign body must clear ScreenBytes", sc.Tool)
		}
		if !sc.Benign && !hit {
			t.Errorf("%s: poisoned body must trip ScreenBytes", sc.Tool)
		}
	}
}

// TestPoisonedDescriptionDeniedByStructure: the durable EFFECTS guarantee against a
// poisoned tool description — a never-allowlisted tool is refused by structure, so the
// injection in its description can never make it run (the same deny fak_adjudicate
// returns as POLICY_BLOCK against a policy manifest).
func TestPoisonedDescriptionDeniedByStructure(t *testing.T) {
	for _, ds := range descScenarios {
		if !denyByStructure(ds.Tool) {
			t.Errorf("poisoned tool %q must be denied by the allow-list (effects gated)", ds.Tool)
		}
	}
}

// TestReportIsNonDegenerate guards the A/B's honesty: at least one scenario is
// quarantined AND at least one benign scenario is allowed, so the demo can never read
// as either "nothing happened" or "fak blocks everything".
func TestReportIsNonDegenerate(t *testing.T) {
	rep := simulate(context.Background())
	var quarantined, allowed int
	for _, r := range rep.Results {
		if r.Quarantined {
			quarantined++
		}
		if r.Verdict == "ALLOW" {
			allowed++
		}
	}
	if quarantined == 0 {
		t.Error("no scenario was quarantined — the A/B is degenerate (defense never fired)")
	}
	if allowed == 0 {
		t.Error("no benign scenario was allowed — the demo would read as a blanket block")
	}
}
