package supervisionpolicy

import (
	"math"
	"reflect"
	"testing"
	"time"
)

func TestRecoveryContracts(t *testing.T) {
	now := time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC)
	base := Request{
		Strategy:           StrategyOneForOne,
		Domain:             "domain-1",
		Session:            "session-1",
		Member:             "member-1",
		Generation:         7,
		ObservedGeneration: 7,
		WriterEpoch:        11,
		EffectCertain:      true,
		Evidence:           []EvidenceRef{"health-probe:42"},
		Now:                now,
		Budget: Budget{
			MaxRestarts: 3,
			Window:      time.Minute,
			BaseBackoff: time.Second,
			MaxBackoff:  10 * time.Second,
		},
	}

	tests := []struct {
		name            string
		mutate          func(*Request)
		action          Action
		outcome         Outcome
		nextWriterEpoch uint64
	}{
		{
			name: "projection reattach preserves logical session",
			mutate: func(r *Request) {
				r.Role = RoleProjection
			},
			action:  ActionReattach,
			outcome: OutcomeRecover,
		},
		{
			name: "adapter restart preserves writer epoch",
			mutate: func(r *Request) {
				r.Role = RoleAdapter
			},
			action:          ActionRestart,
			outcome:         OutcomeRecover,
			nextWriterEpoch: 11,
		},
		{
			name: "helper restart preserves writer epoch",
			mutate: func(r *Request) {
				r.Role = RoleHelper
			},
			action:          ActionRestart,
			outcome:         OutcomeRecover,
			nextWriterEpoch: 11,
		},
		{
			name: "adapter refuses uncertain effect replay",
			mutate: func(r *Request) {
				r.Role = RoleAdapter
				r.EffectCertain = false
			},
			action:  ActionHold,
			outcome: OutcomeUncertainEffect,
		},
		{
			name: "helper refuses uncertain effect replay",
			mutate: func(r *Request) {
				r.Role = RoleHelper
				r.EffectCertain = false
			},
			action:  ActionHold,
			outcome: OutcomeUncertainEffect,
		},
		{
			name: "authority requires durable checkpoint",
			mutate: func(r *Request) {
				r.Role = RoleAuthority
			},
			action:  ActionHold,
			outcome: OutcomeCheckpointRequired,
		},
		{
			name: "authority refuses exhausted writer epoch",
			mutate: func(r *Request) {
				r.Role = RoleAuthority
				r.Checkpoint = "checkpoint:durable-8"
				r.WriterEpoch = math.MaxUint64
			},
			action:  ActionHold,
			outcome: OutcomeWriterEpochExhausted,
		},
		{
			name: "authority restart fences next writer",
			mutate: func(r *Request) {
				r.Role = RoleAuthority
				r.Checkpoint = "checkpoint:durable-8"
			},
			action:          ActionRestart,
			outcome:         OutcomeRecover,
			nextWriterEpoch: 12,
		},
		{
			name: "hold-authority strategy holds authority",
			mutate: func(r *Request) {
				r.Role = RoleAuthority
				r.Strategy = StrategyHoldAuthority
				r.Checkpoint = "checkpoint:durable-8"
			},
			action:  ActionHold,
			outcome: OutcomeStrategyHold,
		},
		{
			name: "escalate-domain strategy escalates",
			mutate: func(r *Request) {
				r.Role = RoleProjection
				r.Strategy = StrategyEscalateDomain
			},
			action:  ActionEscalate,
			outcome: OutcomeStrategyEscalation,
		},
		{
			name: "unknown role holds",
			mutate: func(r *Request) {
				r.Role = Role(255)
			},
			action:  ActionHold,
			outcome: OutcomeUnknownRole,
		},
		{
			name: "stale generation holds",
			mutate: func(r *Request) {
				r.Role = RoleAdapter
				r.ObservedGeneration--
			},
			action:  ActionHold,
			outcome: OutcomeStaleGeneration,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := base
			tt.mutate(&r)
			d := Decide(r)
			if d.Action != tt.action || d.Outcome != tt.outcome {
				t.Fatalf("got action/outcome %v/%v, want %v/%v", d.Action, d.Outcome, tt.action, tt.outcome)
			}
			if d.Session != base.Session || d.Member != base.Member || d.Generation != base.Generation {
				t.Fatalf("logical identity changed: %#v", d)
			}
			if d.NextWriterEpoch != tt.nextWriterEpoch {
				t.Fatalf("next writer epoch = %d, want %d", d.NextWriterEpoch, tt.nextWriterEpoch)
			}
			if d.Role != r.Role || d.Domain != r.Domain || d.Budget.Remaining != 3 || !reflect.DeepEqual(d.Evidence, r.Evidence) {
				t.Fatalf("receipt omitted contract state: %#v", d)
			}
		})
	}
}

func TestIntensityBudget(t *testing.T) {
	now := time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC)
	request := Request{
		Role:               RoleAdapter,
		Strategy:           StrategyOneForOne,
		Domain:             "domain-1",
		Session:            "session-1",
		Member:             "adapter-1",
		Generation:         4,
		ObservedGeneration: 4,
		WriterEpoch:        9,
		EffectCertain:      true,
		Now:                now,
		Budget: Budget{
			MaxRestarts: 3,
			Window:      time.Minute,
			BaseBackoff: 10 * time.Second,
			MaxBackoff:  20 * time.Second,
		},
	}

	tests := []struct {
		name      string
		failures  []time.Time
		now       time.Time
		action    Action
		outcome   Outcome
		used      uint32
		remaining uint32
		retryAt   time.Time
	}{
		{
			name:      "deterministic backoff holds early retry",
			failures:  []time.Time{now.Add(-5 * time.Second)},
			now:       now,
			action:    ActionHold,
			outcome:   OutcomeBackoff,
			used:      1,
			remaining: 2,
			retryAt:   now.Add(5 * time.Second),
		},
		{
			name: "N failures trip escalation",
			failures: []time.Time{
				now.Add(-50 * time.Second),
				now.Add(-30 * time.Second),
				now.Add(-10 * time.Second),
			},
			now:       now,
			action:    ActionEscalate,
			outcome:   OutcomeBudgetExhausted,
			used:      3,
			remaining: 0,
			retryAt:   now.Add(10 * time.Second),
		},
		{
			name: "window expiry allows bounded retry",
			failures: []time.Time{
				now.Add(-2 * time.Minute),
				now.Add(-90 * time.Second),
				now.Add(-61 * time.Second),
			},
			now:       now,
			action:    ActionRestart,
			outcome:   OutcomeRecover,
			used:      0,
			remaining: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := request
			r.Failures = tt.failures
			r.Now = tt.now
			d := Decide(r)
			if d.Action != tt.action || d.Outcome != tt.outcome {
				t.Fatalf("got action/outcome %v/%v, want %v/%v", d.Action, d.Outcome, tt.action, tt.outcome)
			}
			if d.Budget.Used != tt.used || d.Budget.Remaining != tt.remaining || !d.Budget.RetryAt.Equal(tt.retryAt) {
				t.Fatalf("budget = %#v, want used=%d remaining=%d retryAt=%v", d.Budget, tt.used, tt.remaining, tt.retryAt)
			}
		})
	}
}

func TestDecisionCopiesEvidence(t *testing.T) {
	evidence := []EvidenceRef{"probe:1"}
	d := Decide(Request{
		Role:               RoleProjection,
		Strategy:           StrategyOneForOne,
		Domain:             "domain",
		Session:            "session",
		Member:             "projection",
		Generation:         1,
		ObservedGeneration: 1,
		EffectCertain:      true,
		Evidence:           evidence,
		Now:                time.Unix(100, 0),
		Budget:             Budget{MaxRestarts: 1, Window: time.Minute},
	})
	evidence[0] = "mutated"
	if d.Evidence[0] != "probe:1" {
		t.Fatalf("receipt evidence aliases caller memory: %q", d.Evidence[0])
	}
}
