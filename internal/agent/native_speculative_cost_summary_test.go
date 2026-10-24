package agent

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/enginestep"
)

// fak-test:runtime fast est=1ms lane=default
func TestNativeSpeculativeCostSummaryFallbackSerialization(t *testing.T) {
	cost := &SpeculativeDecodeCost{
		Scope: "proposal-through-emission-and-advance", Attempts: 2,
		ProposalError: SpeculativeAttemptCost{
			Attempts: 1, MeasuredAttempts: 1,
			DraftElapsed: 2 * time.Millisecond, InclusiveElapsed: 5 * time.Millisecond,
		},
		Cancelled: SpeculativeAttemptCost{Attempts: 1, UnmeasuredAttempts: 1},
	}
	summary := newNativeDecodeSummary(inKernelGenerateResult{gen: 3, specCost: cost})
	raw, err := json.Marshal(summary)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Path        string          `json:"path"`
		Speculative json.RawMessage `json:"speculative"`
		Cost        *struct {
			Scope         string `json:"scope"`
			Attempts      int    `json:"attempts"`
			ProposalError struct {
				Attempts      int   `json:"attempts"`
				EmittedTokens int   `json:"emitted_tokens"`
				Measured      int   `json:"measured_attempts"`
				DraftNS       int64 `json:"draft_elapsed_ns"`
				InclusiveNS   int64 `json:"inclusive_elapsed_ns"`
			} `json:"proposal_error"`
			Cancelled struct {
				Attempts   int `json:"attempts"`
				Measured   int `json:"measured_attempts"`
				Unmeasured int `json:"unmeasured_attempts"`
			} `json:"cancelled"`
		} `json:"speculative_cost"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.Path != NativeDecodePathSerial || len(wire.Speculative) != 0 {
		t.Fatalf("fallback path/tally = %q/%s, want serial with no verified rounds", wire.Path, wire.Speculative)
	}
	if wire.Cost == nil {
		t.Fatal("serial fallback dropped its speculative attempt cost")
	}
	if wire.Cost.Scope != cost.Scope || wire.Cost.Attempts != 2 {
		t.Fatalf("cost scope/attempts = %q/%d", wire.Cost.Scope, wire.Cost.Attempts)
	}
	p := wire.Cost.ProposalError
	if p.Attempts != 1 || p.EmittedTokens != 0 || p.Measured != 1 || p.DraftNS != 2_000_000 || p.InclusiveNS != 5_000_000 {
		t.Fatalf("proposal-error attempts/emitted/measured/draft_ns/inclusive_ns = %d/%d/%d/%d/%d", p.Attempts, p.EmittedTokens, p.Measured, p.DraftNS, p.InclusiveNS)
	}
	c := wire.Cost.Cancelled
	if c.Attempts != 1 || c.Measured != 0 || c.Unmeasured != 1 {
		t.Fatalf("cancelled attempts/measured/unmeasured = %d/%d/%d", c.Attempts, c.Measured, c.Unmeasured)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestNativeSpeculativeCostSummaryRequestIsolation(t *testing.T) {
	cost := &SpeculativeDecodeCost{
		Scope: "proposal-through-emission-and-advance", Attempts: 1,
		Completed: SpeculativeAttemptCost{
			Attempts: 1, EmittedTokens: 3, MeasuredAttempts: 1,
			DraftElapsed: time.Millisecond, InclusiveElapsed: 4 * time.Millisecond,
		},
	}
	want := *cost
	res := inKernelGenerateResult{
		spec:     SpeculativeDecodeTally{Rounds: 1, DraftTokens: 2, AcceptedTokens: 2},
		specCost: cost,
	}
	first, second := newNativeDecodeSummary(res), newNativeDecodeSummary(res)
	if first.SpeculativeCost == nil || second.SpeculativeCost == nil {
		t.Fatal("verified request omitted speculative cost")
	}
	if first.Path != NativeDecodePathSpeculative || first.Speculative == nil || first.Speculative.Rounds != 1 {
		t.Fatal("cost observation changed the verified-round path or tally")
	}
	cost.Completed.InclusiveElapsed = time.Hour
	first.SpeculativeCost.Completed.EmittedTokens = 99
	if *second.SpeculativeCost != want {
		t.Fatal("another summary or source aggregate mutated this request's cost")
	}
	if cost.Completed.EmittedTokens != want.Completed.EmittedTokens {
		t.Fatal("summary mutation escaped into the source aggregate")
	}

	enginestep.Default.ObserveSpeculativeRound(4, 3, 4, time.Second)
	unobserved := newNativeDecodeSummary(inKernelGenerateResult{})
	if unobserved.SpeculativeCost != nil || unobserved.Speculative != nil || unobserved.Path != NativeDecodePathSerial {
		t.Fatal("process-wide speculative observations leaked into an unobserved request")
	}
	raw, err := json.Marshal(unobserved)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		t.Fatal(err)
	}
	if _, present := fields["speculative_cost"]; present {
		t.Fatal("unobserved request serialized a measured-zero speculative cost")
	}
}
