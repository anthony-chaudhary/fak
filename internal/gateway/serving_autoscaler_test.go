package gateway

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestServingAutoscalerNativeScalesPrefillDecodeFromSignals(t *testing.T) {
	ctx := context.Background()
	m := NewFleetMembership(MembershipConfig{Probe: func(context.Context, WorkerSpec) bool { return true }})
	mustAdd(t, m, WorkerSpec{ID: "prefill-0", Role: RolePrefill, Engine: EngineNative})
	mustAdd(t, m, WorkerSpec{ID: "decode-0", Role: RoleDecode, Engine: EngineNative})
	m.ProbeOnce(ctx)

	a := NewServingAutoscaler(m, NewNativePoolActuator(nil), ServingAutoscalerConfig{
		Mode: ScaleModeNative,
		Prefill: PoolObjective{
			MinReplicas:   1,
			MaxReplicas:   4,
			ScaleStep:     1,
			GoodputTarget: 100,
			QueueHigh:     2,
			QueueLow:      0,
			TTFTSLO:       100 * time.Millisecond,
		},
		Decode: PoolObjective{
			MinReplicas:   1,
			MaxReplicas:   4,
			ScaleStep:     1,
			GoodputTarget: 100,
			QueueHigh:     2,
			QueueLow:      0,
			KVUtilHigh:    0.80,
			KVUtilLow:     0.40,
			TPOTSLO:       50 * time.Millisecond,
		},
		HysteresisTicks: 1,
	})

	decisions, err := a.Reconcile(ctx, time.Unix(100, 0), ServingSignals{
		Prefill: PoolSignals{
			Goodput:    80,
			QueueDepth: 4,
			TTFT:       250 * time.Millisecond,
		},
		Decode: PoolSignals{
			Goodput:       70,
			QueueDepth:    3,
			KVUtilization: 0.91,
			TPOT:          90 * time.Millisecond,
		},
		Workers: []WorkerSignals{
			{WorkerID: "prefill-0", Role: RolePrefill, Goodput: 80, QueueDepth: 4, TTFT: 250 * time.Millisecond},
			{WorkerID: "decode-0", Role: RoleDecode, Goodput: 70, QueueDepth: 3, KVUtilization: 0.91, TPOT: 90 * time.Millisecond},
		},
	})
	if err != nil {
		t.Fatalf("Reconcile: %v", err)
	}
	if len(decisions) != 2 {
		t.Fatalf("decisions len = %d, want 2", len(decisions))
	}
	for _, d := range decisions {
		if d.Action != ScaleActionScaleUp {
			t.Fatalf("%s action = %s, want scale_up (%+v)", d.Role, d.Action, d)
		}
		if d.CurrentReplicas != 1 || d.DesiredReplicas != 2 || d.AppliedReplicas != 2 {
			t.Fatalf("%s replicas = current %d desired %d applied %d, want 1/2/2",
				d.Role, d.CurrentReplicas, d.DesiredReplicas, d.AppliedReplicas)
		}
		if len(d.Workers) != 1 {
			t.Fatalf("%s workers = %v, want one started worker", d.Role, d.Workers)
		}
	}

	counts := roleCounts(m)
	if counts[RolePrefill] != 2 || counts[RoleDecode] != 2 {
		t.Fatalf("role counts = %v, want prefill=2 decode=2", counts)
	}
	for _, st := range m.Snapshot() {
		if st.Spec.ID == "prefill-1" || st.Spec.ID == "decode-1" {
			if st.Spec.Engine != EngineNative {
				t.Fatalf("native actuator registered %s as engine %s", st.Spec.ID, st.Spec.Engine)
			}
		}
	}

	trace := a.DecisionJSONLines()
	for _, want := range []string{
		`"role":"prefill"`,
		`"role":"decode"`,
		`"action":"scale_up"`,
		"queue_high",
		"ttft_slo",
		"tpot_slo",
	} {
		if !strings.Contains(trace, want) {
			t.Fatalf("trace missing %q\n%s", want, trace)
		}
	}
}

func TestServingAutoscalerConsumesNormalizedMetricRows(t *testing.T) {
	ctx := context.Background()
	m := NewFleetMembership(MembershipConfig{Probe: func(context.Context, WorkerSpec) bool { return true }})
	mustAdd(t, m, WorkerSpec{ID: "prefill-a", Role: RolePrefill, Engine: EngineExternal})
	mustAdd(t, m, WorkerSpec{ID: "decode-a", Role: RoleDecode, Engine: EngineNative})
	m.ProbeOnce(ctx)

	signals := ServingSignalsFromMetricRows(m, []ServingMetricRow{
		{
			Labels:        ServingMetricLabels{Worker: "prefill-a", Engine: "vllm", Model: "mixtral"},
			Goodput:       ServingGaugeValue(90),
			Waiting:       ServingGaugeValue(4),
			TTFT:          ServingHistogram{Sum: ServingGaugeValue(0.30), Count: ServingGaugeValue(1)},
			KVUtilization: ServingGaugeValue(0.20),
		},
		{
			Labels:        ServingMetricLabels{Worker: "decode-a", Engine: "native", Model: "mixtral"},
			Goodput:       ServingGaugeValue(95),
			Waiting:       ServingGaugeValue(3),
			TPOT:          ServingHistogram{Sum: ServingGaugeValue(0.09), Count: ServingGaugeValue(1)},
			KVUtilization: ServingGaugeValue(0.90),
		},
	})
	if signals.Prefill.QueueDepth != 4 || signals.Prefill.TTFT != 300*time.Millisecond {
		t.Fatalf("prefill signals = %+v, want queue=4 ttft=300ms", signals.Prefill)
	}
	if signals.Decode.QueueDepth != 3 || signals.Decode.TPOT != 90*time.Millisecond || signals.Decode.KVUtilization != 0.90 {
		t.Fatalf("decode signals = %+v, want queue=3 tpot=90ms kv=0.90", signals.Decode)
	}

	a := NewServingAutoscaler(m, NewNativePoolActuator(nil), ServingAutoscalerConfig{
		Mode: ScaleModeNative,
		Prefill: PoolObjective{
			MinReplicas: 1,
			MaxReplicas: 3,
			QueueHigh:   2,
			TTFTSLO:     100 * time.Millisecond,
		},
		Decode: PoolObjective{
			MinReplicas: 1,
			MaxReplicas: 3,
			QueueHigh:   2,
			KVUtilHigh:  0.80,
			TPOTSLO:     50 * time.Millisecond,
		},
		HysteresisTicks: 1,
	})
	decisions, err := a.Reconcile(ctx, time.Unix(150, 0), signals)
	if err != nil {
		t.Fatalf("Reconcile: %v", err)
	}
	for _, d := range decisions {
		if d.Action != ScaleActionScaleUp {
			t.Fatalf("%s action = %s, want scale_up from normalized metrics (%+v)", d.Role, d.Action, d)
		}
		if len(d.WorkerSignals) != 1 {
			t.Fatalf("%s worker signals = %+v, want one source row", d.Role, d.WorkerSignals)
		}
	}
}

func TestServingAutoscalerCanDeferToDynamoPlanner(t *testing.T) {
	ctx := context.Background()
	m := NewFleetMembership(MembershipConfig{Probe: func(context.Context, WorkerSpec) bool { return true }})
	mustAdd(t, m, WorkerSpec{ID: "prefill-0", Role: RolePrefill, Engine: EngineExternal})
	mustAdd(t, m, WorkerSpec{ID: "decode-0", Role: RoleDecode, Engine: EngineExternal})
	m.ProbeOnce(ctx)

	a := NewServingAutoscaler(m, NewNativePoolActuator(nil), ServingAutoscalerConfig{
		Mode:            ScaleModeDynamoDelegated,
		Prefill:         PoolObjective{MinReplicas: 1, MaxReplicas: 5, QueueHigh: 1},
		Decode:          PoolObjective{MinReplicas: 1, MaxReplicas: 5, QueueHigh: 1},
		HysteresisTicks: 1,
	})
	decisions, err := a.Reconcile(ctx, time.Unix(200, 0), ServingSignals{
		Prefill: PoolSignals{QueueDepth: 10},
		Decode:  PoolSignals{QueueDepth: 10},
	})
	if err != nil {
		t.Fatalf("Reconcile: %v", err)
	}
	for _, d := range decisions {
		if d.Action != ScaleActionDeferred {
			t.Fatalf("%s action = %s, want deferred", d.Role, d.Action)
		}
		if !strings.Contains(d.Reason, "dynamo_planner_delegated") {
			t.Fatalf("%s reason = %q, want dynamo delegation", d.Role, d.Reason)
		}
	}
	if counts := roleCounts(m); counts[RolePrefill] != 1 || counts[RoleDecode] != 1 {
		t.Fatalf("deferred controller changed membership: %v", counts)
	}
}

func TestServingAutoscalerHysteresisCooldownAndBoundsAvoidNoisyOscillation(t *testing.T) {
	ctx := context.Background()
	m := NewFleetMembership(MembershipConfig{Probe: func(context.Context, WorkerSpec) bool { return true }})
	mustAdd(t, m, WorkerSpec{ID: "decode-a", Role: RoleDecode, Engine: EngineNative})
	mustAdd(t, m, WorkerSpec{ID: "decode-b", Role: RoleDecode, Engine: EngineNative})
	m.ProbeOnce(ctx)

	a := NewServingAutoscaler(m, NewNativePoolActuator(nil), ServingAutoscalerConfig{
		Mode: ScaleModeNative,
		Prefill: PoolObjective{
			MinReplicas: 0,
			MaxReplicas: 0,
		},
		Decode: PoolObjective{
			MinReplicas:   1,
			MaxReplicas:   3,
			ScaleStep:     1,
			GoodputTarget: 100,
			QueueHigh:     5,
			QueueLow:      0,
			KVUtilLow:     0.40,
			TPOTSLO:       50 * time.Millisecond,
		},
		HysteresisTicks: 2,
		Cooldown:        10 * time.Second,
	})

	high := ServingSignals{Decode: PoolSignals{Goodput: 50, QueueDepth: 8, TPOT: 80 * time.Millisecond}}
	low := ServingSignals{Decode: PoolSignals{Goodput: 200, QueueDepth: 0, KVUtilization: 0.10, TPOT: 10 * time.Millisecond}}

	for i, sig := range []ServingSignals{high, low, high, low} {
		if _, err := a.Reconcile(ctx, time.Unix(int64(300+i), 0), sig); err != nil {
			t.Fatalf("noisy Reconcile(%d): %v", i, err)
		}
	}
	if got := roleCounts(m)[RoleDecode]; got != 2 {
		t.Fatalf("noisy alternating signal changed decode replicas to %d, want 2", got)
	}

	if _, err := a.Reconcile(ctx, time.Unix(310, 0), high); err != nil {
		t.Fatalf("high 1: %v", err)
	}
	if _, err := a.Reconcile(ctx, time.Unix(311, 0), high); err != nil {
		t.Fatalf("high 2: %v", err)
	}
	if got := roleCounts(m)[RoleDecode]; got != 3 {
		t.Fatalf("sustained high signal decode replicas = %d, want max-clamped 3", got)
	}
	if _, err := a.Reconcile(ctx, time.Unix(312, 0), high); err != nil {
		t.Fatalf("high at max: %v", err)
	}
	if got := roleCounts(m)[RoleDecode]; got != 3 {
		t.Fatalf("max clamp allowed decode replicas = %d, want 3", got)
	}

	if _, err := a.Reconcile(ctx, time.Unix(313, 0), low); err != nil {
		t.Fatalf("low cooldown 1: %v", err)
	}
	if _, err := a.Reconcile(ctx, time.Unix(314, 0), low); err != nil {
		t.Fatalf("low cooldown 2: %v", err)
	}
	if got := roleCounts(m)[RoleDecode]; got != 3 {
		t.Fatalf("cooldown allowed immediate scale-down to %d, want 3", got)
	}
}

func TestServingAutoscalerScaleDownDrainsInflightBeforeRemoval(t *testing.T) {
	ctx := context.Background()
	m := NewFleetMembership(MembershipConfig{Probe: func(context.Context, WorkerSpec) bool { return true }})
	mustAdd(t, m, WorkerSpec{ID: "decode-a", Role: RoleDecode, Engine: EngineNative})
	mustAdd(t, m, WorkerSpec{ID: "decode-b", Role: RoleDecode, Engine: EngineNative})
	m.ProbeOnce(ctx)
	if err := m.Acquire("decode-a"); err != nil {
		t.Fatalf("Acquire decode-a: %v", err)
	}
	if err := m.Acquire("decode-b"); err != nil {
		t.Fatalf("Acquire decode-b: %v", err)
	}

	a := NewServingAutoscaler(m, NewNativePoolActuator(nil), ServingAutoscalerConfig{
		Mode: ScaleModeNative,
		Prefill: PoolObjective{
			MinReplicas: 0,
			MaxReplicas: 0,
		},
		Decode: PoolObjective{
			MinReplicas:   1,
			MaxReplicas:   2,
			ScaleStep:     1,
			GoodputTarget: 100,
			QueueLow:      0,
			KVUtilLow:     0.50,
			TPOTSLO:       50 * time.Millisecond,
		},
		HysteresisTicks: 1,
	})

	decisions, err := a.Reconcile(ctx, time.Unix(400, 0), ServingSignals{
		Decode: PoolSignals{
			Goodput:       200,
			QueueDepth:    0,
			KVUtilization: 0.10,
			TPOT:          10 * time.Millisecond,
		},
	})
	if err != nil {
		t.Fatalf("Reconcile: %v", err)
	}
	var down ScaleDecision
	for _, d := range decisions {
		if d.Role == RoleDecode {
			down = d
			break
		}
	}
	if down.Action != ScaleActionScaleDown || len(down.Workers) != 1 {
		t.Fatalf("decode decision = %+v, want one scale_down", down)
	}
	drainingID := down.Workers[0]
	if admissibleIDs(m)[drainingID] {
		t.Fatalf("draining worker %s is still admissible for new work", drainingID)
	}
	if len(m.Snapshot()) != 2 {
		t.Fatalf("drained in-flight worker was removed before release: %+v", m.Snapshot())
	}
	foundDraining := false
	for _, st := range m.Snapshot() {
		if st.Spec.ID == drainingID {
			foundDraining = true
			if !st.Draining || st.Inflight != 1 {
				t.Fatalf("draining worker status = %+v, want draining with one in-flight", st)
			}
		}
	}
	if !foundDraining {
		t.Fatalf("draining worker %s missing from snapshot", drainingID)
	}

	m.Release(drainingID)
	if len(m.Snapshot()) != 1 {
		t.Fatalf("worker not removed after in-flight release: %+v", m.Snapshot())
	}
	m.Release("decode-a")
	m.Release("decode-b")
}

func roleCounts(m *FleetMembership) map[WorkerRole]int {
	out := map[WorkerRole]int{}
	for _, st := range m.Snapshot() {
		if !st.Draining {
			out[st.Spec.Role]++
		}
	}
	return out
}
