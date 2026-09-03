package coordination

import (
	"reflect"
	"strings"
	"testing"
	"time"
)

func validHarnessWorkflow(h Harness) HarnessWorkflow {
	return HarnessWorkflow{
		Harness:       h,
		Coordination:  true,
		WorkID:        "work-1",
		CorrelationID: "corr-1",
		Workers: []HarnessWorker{
			{ID: "scout", Role: "research", Model: "qwen"},
			{ID: "build", Role: "implement", Model: "qwen"},
		},
		DAG:         []HarnessEdge{{From: "scout", To: "build"}},
		Fanout:      2,
		Concurrency: 2,
		Lease: HarnessLease{
			Lane:      "lease-1",
			Mode:      "exclusive",
			TTL:       time.Minute,
			Renewable: true,
		},
		Budgets: HarnessBudgets{
			Tokens:     1000,
			CostMicros: 100,
			Duration:   time.Minute,
		},
		Interaction:  InteractionNone,
		Cancellation: CancellationDrain,
		Exhaustion:   ExhaustionReduce,
		Witness:      WitnessIndependent,
		Degradation:  DegradationAllowDegraded,
		Capabilities: map[string]CapabilityState{
			"fanout":  CapabilityNative,
			"witness": CapabilityNative,
		},
		Requirements: NeutralRequirements{
			Cache:     []string{"shared-context"},
			Placement: []string{"fak_native"},
			Serve:     []string{"admit"},
		},
	}
}

func fullPressure() HarnessPressure {
	return HarnessPressure{
		Concurrency: 2,
		Tokens:      1000,
		CostMicros:  100,
		Duration:    time.Minute,
		Capabilities: map[string]CapabilityState{
			"fanout":  CapabilityNative,
			"witness": CapabilityNative,
		},
	}
}

func TestHarnessAdapterNormalizeEquivalentHarnesses(t *testing.T) {
	a := NewHarnessAdapter()
	var want NeutralHarnessIntent
	for i, h := range []Harness{HarnessClaude, HarnessCodex, HarnessOpencode, HarnessFakNative} {
		got, err := a.Normalize(validHarnessWorkflow(h))
		if err != nil {
			t.Fatalf("Normalize(%q): %v", h, err)
		}
		if got.WorkID != "work-1" || got.CorrelationID != "corr-1" {
			t.Fatalf("Normalize(%q) IDs = %q, %q", h, got.WorkID, got.CorrelationID)
		}
		if i == 0 {
			want = got
		} else if !reflect.DeepEqual(got, want) {
			t.Fatalf("Normalize(%q) differs from canonical intent\n got: %#v\nwant: %#v", h, got, want)
		}
	}
}

func TestHarnessAdapterCapabilities(t *testing.T) {
	a := NewHarnessAdapter()
	for _, state := range []CapabilityState{CapabilityNative, CapabilityEmulated, CapabilityDegraded, CapabilityUnsupported} {
		t.Run(string(state)+" optional", func(t *testing.T) {
			p := fullPressure()
			p.Capabilities["optional"] = state
			if got := a.Decide(validHarnessWorkflow(HarnessClaude), p); got.Kind != HarnessDecisionExecute {
				t.Fatalf("optional capability %q: got %s (%s)", state, got.Kind, got.Reason)
			}
		})
	}

	p := fullPressure()
	p.Capabilities["witness"] = CapabilityUnsupported
	if got := a.Decide(validHarnessWorkflow(HarnessClaude), p); got.Kind != HarnessDecisionInfeasible {
		t.Fatalf("required unsupported capability: got %s (%s)", got.Kind, got.Reason)
	}
}

func TestHarnessAdapterPressureDecisions(t *testing.T) {
	a := NewHarnessAdapter()
	tests := []struct {
		name   string
		mutate func(*HarnessWorkflow, *HarnessPressure)
		want   HarnessDecisionKind
	}{
		{
			name: "available concurrency reduces",
			mutate: func(_ *HarnessWorkflow, p *HarnessPressure) {
				p.Concurrency = 1
			},
			want: HarnessDecisionReduce,
		},
		{
			name: "saturated queue delays",
			mutate: func(w *HarnessWorkflow, p *HarnessPressure) {
				w.Exhaustion = ExhaustionDelay
				p.Exhausted = true
			},
			want: HarnessDecisionDelay,
		},
		{
			name: "pinned minimum concurrency infeasible",
			mutate: func(w *HarnessWorkflow, p *HarnessPressure) {
				w.Pins.Concurrency = true
				p.Concurrency = 1
			},
			want: HarnessDecisionInfeasible,
		},
		{
			name: "cancelled delays",
			mutate: func(_ *HarnessWorkflow, p *HarnessPressure) {
				p.Cancelled = true
			},
			want: HarnessDecisionDelay,
		},
		{
			name: "budget exhausted checkpoints by reduction",
			mutate: func(_ *HarnessWorkflow, p *HarnessPressure) {
				p.Exhausted = true
			},
			want: HarnessDecisionReduce,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w, p := validHarnessWorkflow(HarnessClaude), fullPressure()
			tt.mutate(&w, &p)
			got := a.Decide(w, p)
			if got.Kind != tt.want {
				t.Fatalf("kind = %s (%s), want %s", got.Kind, got.Reason, tt.want)
			}
			if got.Intent.WorkID != "work-1" || got.Intent.CorrelationID != "corr-1" {
				t.Fatalf("IDs not preserved: %q, %q", got.Intent.WorkID, got.Intent.CorrelationID)
			}
			if tt.want == HarnessDecisionReduce && (!reflect.DeepEqual(got.Intent.DAG, w.DAG) || got.Intent.Witness != w.Witness) {
				t.Fatalf("reduction changed DAG or witness: %#v", got.Intent)
			}
		})
	}
}

func TestHarnessAdapterProject(t *testing.T) {
	a := NewHarnessAdapter()
	intent, err := a.Normalize(validHarnessWorkflow(HarnessFakNative))
	if err != nil {
		t.Fatal(err)
	}
	got, err := a.Project(intent)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Cache, []string{"shared-context"}) ||
		!reflect.DeepEqual(got.Placement, []string{"fak_native"}) ||
		!reflect.DeepEqual(got.Serve, []string{"admit"}) {
		t.Fatalf("projection requirements = %#v", got)
	}
}

func TestHarnessAdapterDisabledDelegates(t *testing.T) {
	w := validHarnessWorkflow(HarnessClaude)
	w.Coordination = false
	got := NewHarnessAdapter().Decide(w, HarnessPressure{})
	if got.Kind != HarnessDecisionDelegate || got.Delegation != "legacy:#5964" {
		t.Fatalf("disabled decision = %#v", got)
	}
}

func TestHarnessAdapterSelfCheck(t *testing.T) {
	got := NewHarnessAdapter().SelfCheck()
	if !got.OK || !got.Deterministic || !got.ContentFree || got.Digest == "" || got.Error != "" {
		t.Fatalf("SelfCheck = %#v", got)
	}
}

func TestHarnessAdapterUnknownValuesFailClosed(t *testing.T) {
	a := NewHarnessAdapter()
	tests := []struct {
		name   string
		mutate func(*HarnessWorkflow, *HarnessPressure)
	}{
		{"harness", func(w *HarnessWorkflow, _ *HarnessPressure) { w.Harness = Harness("unknown") }},
		{"capability", func(w *HarnessWorkflow, _ *HarnessPressure) { w.Capabilities["fanout"] = CapabilityState("unknown") }},
		{"pressure", func(_ *HarnessWorkflow, p *HarnessPressure) { p.Capabilities["fanout"] = CapabilityState("unknown") }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w, p := validHarnessWorkflow(HarnessClaude), fullPressure()
			tt.mutate(&w, &p)
			got := a.Decide(w, p)
			if got.Kind != HarnessDecisionInfeasible || !strings.Contains(got.Reason, "unknown") && !strings.Contains(got.Reason, "invalid") {
				t.Fatalf("decision = %#v", got)
			}
		})
	}
}
