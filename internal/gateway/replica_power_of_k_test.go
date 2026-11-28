package gateway

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// fak-test:runtime fast est=1s
func TestReplicaPowerOfKStaleReportIgnored(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	reports := map[string]ReplicaLoadReport{
		// Stale report (>5s) claiming heavy load must be ignored in favor of local floor.
		"r1": {Waiting: 50, Running: 10, ReportedAt: now.Add(-6 * time.Second)},
		// Fresh report (<=5s) accurately reflects moderate load above local floor.
		"r2": {Waiting: 2, Running: 3, ReportedAt: now.Add(-5 * time.Second)},
	}
	pol, err := NewPowerOfKPolicy(PowerOfKConfig{
		K:           2,
		NumGateways: 1,
		StaleWindow: 5 * time.Second,
		Report: func(name string) (ReplicaLoadReport, bool) {
			rep, ok := reports[name]
			return rep, ok
		},
		Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatalf("NewPowerOfKPolicy: %v", err)
	}

	candidates := []PlannerReplica{{Name: "r1"}, {Name: "r2"}}
	local := func(name string) int {
		if name == "r1" {
			return 1
		}
		return 1
	}

	d, ok := pol.PickWithRunnerUp(candidates, []string{"p"}, local)
	if !ok {
		t.Fatal("PickWithRunnerUp returned ok=false")
	}
	if d.Primary.Name != "r1" || d.PrimaryLoad != 1 {
		t.Fatalf("stale report should fall back to local floor 1 on r1, got %+v", d)
	}
	if !d.HasRunnerUp || d.RunnerUp.Name != "r2" || d.RunnerUpLoad != 5 {
		t.Fatalf("fresh report boundary (==5s) should stay active on r2 with load 5, got %+v", d)
	}
}

// fak-test:runtime fast est=1s
func TestReplicaPowerOfKFloorBeatsUnderReportingReplica(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	// under-reporter claims 0 waiting/running, while healthy reports 4;
	// with local inflight = 3 and numGateways = 2, floor = 6 > 4.
	reports := map[string]ReplicaLoadReport{
		"under":  {Waiting: 0, Running: 0, ReportedAt: now.Add(-100 * time.Millisecond)},
		"honest": {Waiting: 1, Running: 3, ReportedAt: now.Add(-100 * time.Millisecond)},
	}
	pol, err := NewPowerOfKPolicy(PowerOfKConfig{
		K:           2,
		NumGateways: 2,
		StaleWindow: 5 * time.Second,
		Report: func(name string) (ReplicaLoadReport, bool) {
			rep, ok := reports[name]
			return rep, ok
		},
		Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatalf("NewPowerOfKPolicy: %v", err)
	}

	candidates := []PlannerReplica{{Name: "under"}, {Name: "honest"}}
	local := func(name string) int {
		if name == "under" {
			return 3 // floor = 3 * 2 = 6
		}
		return 2 // floor = 2 * 2 = 4; max(4, 1+3) = 4
	}

	d, ok := pol.PickWithRunnerUp(candidates, []string{"floor-check"}, local)
	if !ok {
		t.Fatal("PickWithRunnerUp returned ok=false")
	}
	if d.Primary.Name != "honest" || d.PrimaryLoad != 4 {
		t.Fatalf("expected honest (load 4) to beat under-reporting replica (floor 6), got %+v", d)
	}
	if !d.HasRunnerUp || d.RunnerUp.Name != "under" || d.RunnerUpLoad != 6 {
		t.Fatalf("expected under as runner-up with floor load 6, got %+v", d)
	}
}

type failingPowerOfKPlanner struct {
	name  string
	err   error
	calls atomic.Int32
}

func (p *failingPowerOfKPlanner) Model() string { return "Qwen3.8" }

func (p *failingPowerOfKPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls.Add(1)
	if p.err != nil {
		return nil, p.err
	}
	return &agent.Completion{
		Message: agent.Message{Role: agent.RoleAssistant, Content: p.name},
		Model:   "Qwen3.8",
	}, nil
}

// fak-test:runtime fast est=1s
func TestReplicaPowerOfKRunnerUpOnPrimaryFailure(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	// Construct 4 replicas and find a deterministic prefix whose top-2 HRW candidates
	// have loads [1, 3] while the 3rd/4th unsampled candidates have load 0.
	// On primary failure, failover MUST pick the sampled runner-up (load 3) rather
	// than re-sampling into the unsampled replicas.
	names := []string{"w0", "w1", "w2", "w3"}
	prefix := []string{"runnerup-failover-probe"}
	key := replicaRendezvousKey(prefix)

	// Rank names in exact HRW order for prefix so we know top-2 vs unsampled.
	ranked := append([]string(nil), names...)
	for i := 0; i < len(ranked); i++ {
		for j := i + 1; j < len(ranked); j++ {
			si, sj := hrwScore(key, ranked[i]), hrwScore(key, ranked[j])
			if sj > si || (sj == si && ranked[j] < ranked[i]) {
				ranked[i], ranked[j] = ranked[j], ranked[i]
			}
		}
	}
	primaryName, runnerUpName, unsampledA, unsampledB := ranked[0], ranked[1], ranked[2], ranked[3]

	mem := NewFleetMembership(MembershipConfig{
		HealthyAfter:   1,
		UnhealthyAfter: 2,
		Probe:          func(context.Context, WorkerSpec) bool { return true },
	})
	for _, id := range names {
		if err := mem.Add(WorkerSpec{ID: id, Engine: EngineNative, Models: []string{"Qwen3.8"}}); err != nil {
			t.Fatalf("Add(%s): %v", id, err)
		}
	}
	mem.ProbeOnce(context.Background())

	// Seed in-flight loads: primary=1, runnerUp=3, unsampledA=0, unsampledB=0.
	if err := mem.Acquire(primaryName); err != nil {
		t.Fatalf("Acquire primary: %v", err)
	}
	defer mem.Release(primaryName)
	for i := 0; i < 3; i++ {
		if err := mem.Acquire(runnerUpName); err != nil {
			t.Fatalf("Acquire runnerUp: %v", err)
		}
		defer mem.Release(runnerUpName)
	}

	planners := map[string]*failingPowerOfKPlanner{
		primaryName:  {name: primaryName, err: &agent.UpstreamStatusError{Status: http.StatusBadGateway}},
		runnerUpName: {name: runnerUpName},
		unsampledA:   {name: unsampledA},
		unsampledB:   {name: unsampledB},
	}
	replicas := make([]PlannerReplica, 0, len(names))
	for _, id := range names {
		replicas = append(replicas, PlannerReplica{Name: id, Planner: planners[id]})
	}

	router, err := NewReplicaDispatch("Qwen3.8", replicas)
	if err != nil {
		t.Fatalf("NewReplicaDispatch: %v", err)
	}
	if _, err := router.WithPowerOfKPolicy(PowerOfKConfig{
		K:           2,
		NumGateways: 1,
		StaleWindow: 5 * time.Second,
		Now:         func() time.Time { return now },
	}); err != nil {
		t.Fatalf("WithPowerOfKPolicy: %v", err)
	}
	router.WithMembership(mem)

	// Build a message slice whose prefixSegments equals `prefix`? No, Complete derives
	// prefixSegments from messages, so rank for the real message prefix!
	msgs := []agent.Message{{Role: agent.RoleUser, Content: "power-of-k-runnerup-failover"}}
	msgPrefix := prefixSegments(msgs)
	msgKey := replicaRendezvousKey(msgPrefix)
	msgRanked := append([]string(nil), names...)
	for i := 0; i < len(msgRanked); i++ {
		for j := i + 1; j < len(msgRanked); j++ {
			si, sj := hrwScore(msgKey, msgRanked[i]), hrwScore(msgKey, msgRanked[j])
			if sj > si || (sj == si && msgRanked[j] < msgRanked[i]) {
				msgRanked[i], msgRanked[j] = msgRanked[j], msgRanked[i]
			}
		}
	}
	// Reset membership + planners to match msgRanked exactly.
	mem2 := NewFleetMembership(MembershipConfig{
		HealthyAfter:   1,
		UnhealthyAfter: 2,
		Probe:          func(context.Context, WorkerSpec) bool { return true },
	})
	for _, id := range names {
		if err := mem2.Add(WorkerSpec{ID: id, Engine: EngineNative, Models: []string{"Qwen3.8"}}); err != nil {
			t.Fatalf("Add(%s): %v", id, err)
		}
	}
	mem2.ProbeOnce(context.Background())
	if err := mem2.Acquire(msgRanked[0]); err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	defer mem2.Release(msgRanked[0])
	for i := 0; i < 3; i++ {
		if err := mem2.Acquire(msgRanked[1]); err != nil {
			t.Fatalf("Acquire: %v", err)
		}
		defer mem2.Release(msgRanked[1])
	}
	planners2 := map[string]*failingPowerOfKPlanner{
		msgRanked[0]: {name: msgRanked[0], err: &agent.UpstreamStatusError{Status: http.StatusBadGateway}},
		msgRanked[1]: {name: msgRanked[1]},
		msgRanked[2]: {name: msgRanked[2]},
		msgRanked[3]: {name: msgRanked[3]},
	}
	replicas2 := make([]PlannerReplica, 0, len(names))
	for _, id := range names {
		replicas2 = append(replicas2, PlannerReplica{Name: id, Planner: planners2[id]})
	}
	router2, err := NewReplicaDispatch("Qwen3.8", replicas2)
	if err != nil {
		t.Fatalf("NewReplicaDispatch: %v", err)
	}
	if _, err := router2.WithPowerOfKPolicy(PowerOfKConfig{
		K:           2,
		NumGateways: 1,
		StaleWindow: 5 * time.Second,
		Now:         func() time.Time { return now },
	}); err != nil {
		t.Fatalf("WithPowerOfKPolicy: %v", err)
	}
	router2.WithMembership(mem2)

	comp, err := router2.Complete(context.Background(), msgs, nil)
	if err != nil {
		t.Fatalf("Complete failover: %v", err)
	}
	if comp.Message.Content != msgRanked[1] {
		t.Fatalf("failover served by %q, want sampled runner-up %q (unsampled were %q,%q)",
			comp.Message.Content, msgRanked[1], msgRanked[2], msgRanked[3])
	}
	if planners2[msgRanked[0]].calls.Load() != 1 || planners2[msgRanked[1]].calls.Load() != 1 {
		t.Fatalf("unexpected call counts primary=%d runnerUp=%d", planners2[msgRanked[0]].calls.Load(), planners2[msgRanked[1]].calls.Load())
	}
	if planners2[msgRanked[2]].calls.Load() != 0 || planners2[msgRanked[3]].calls.Load() != 0 {
		t.Fatalf("unsampled replicas should not receive runner-up failover traffic: u2=%d u3=%d",
			planners2[msgRanked[2]].calls.Load(), planners2[msgRanked[3]].calls.Load())
	}
}

// fak-test:runtime fast est=1s
func TestReplicaPowerOfKKGreaterOrEqualNLeastLoaded(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	candidates := []PlannerReplica{{Name: "a"}, {Name: "b"}, {Name: "c"}, {Name: "d"}}
	loads := map[string]int{"a": 9, "b": 4, "c": 1, "d": 2}

	for _, k := range []int{4, 10} {
		pol, err := NewPowerOfKPolicy(PowerOfKConfig{
			K:           k,
			NumGateways: 1,
			StaleWindow: 5 * time.Second,
			Now:         func() time.Time { return now },
		})
		if err != nil {
			t.Fatalf("NewPowerOfKPolicy(k=%d): %v", k, err)
		}
		for i := 0; i < 8; i++ {
			d, ok := pol.PickWithRunnerUp(candidates, []string{fmt.Sprintf("req-%d", i)}, func(name string) int {
				return loads[name]
			})
			if !ok {
				t.Fatalf("k=%d i=%d returned ok=false", k, i)
			}
			if d.Primary.Name != "c" || d.PrimaryLoad != 1 {
				t.Fatalf("k=%d i=%d expected global least-loaded c(1), got %+v", k, i, d)
			}
			if !d.HasRunnerUp || d.RunnerUp.Name != "d" || d.RunnerUpLoad != 2 {
				t.Fatalf("k=%d i=%d expected global runner-up d(2), got %+v", k, i, d)
			}
		}
	}
}

// fak-test:runtime fast est=1s
func TestReplicaPowerOfKInvalidParamsAndEmptyReplicasFailClosed(t *testing.T) {
	for _, tc := range []struct {
		name string
		cfg  PowerOfKConfig
	}{
		{name: "negative K", cfg: PowerOfKConfig{K: -1}},
		{name: "negative NumGateways", cfg: PowerOfKConfig{NumGateways: -2}},
		{name: "negative StaleWindow", cfg: PowerOfKConfig{StaleWindow: -time.Second}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewPowerOfKPolicy(tc.cfg); err == nil {
				t.Fatalf("expected error for %s", tc.name)
			}
			r, err := NewReplicaDispatch("m", []PlannerReplica{{Name: "r1", Planner: &failingPowerOfKPlanner{name: "r1"}}})
			if err != nil {
				t.Fatalf("NewReplicaDispatch: %v", err)
			}
			if _, err := r.WithPowerOfKPolicy(tc.cfg); err == nil {
				t.Fatalf("WithPowerOfKPolicy expected error for %s", tc.name)
			}
			if r.policy != nil {
				t.Fatalf("failed WithPowerOfKPolicy mutated dispatch policy")
			}
		})
	}

	var nilDispatch *ReplicaDispatch
	if _, err := nilDispatch.WithPowerOfKPolicy(DefaultPowerOfKConfig()); err == nil {
		t.Fatal("nil ReplicaDispatch.WithPowerOfKPolicy must fail closed")
	}

	var nilPolicy *PowerOfKPolicy
	if _, ok := nilPolicy.Pick([]PlannerReplica{{Name: "r1"}}, nil, nil); ok {
		t.Fatal("nil PowerOfKPolicy.Pick must return ok=false")
	}
	if _, ok := (&PowerOfKPolicy{}).Pick([]PlannerReplica{{Name: "r1"}}, nil, nil); ok {
		t.Fatal("zero-value unconstructed PowerOfKPolicy must fail closed")
	}

	pol, err := NewPowerOfKPolicy(DefaultPowerOfKConfig())
	if err != nil {
		t.Fatalf("NewPowerOfKPolicy: %v", err)
	}
	if _, ok := pol.Pick(nil, []string{"a"}, nil); ok {
		t.Fatal("empty candidates must return ok=false")
	}
	if _, ok := pol.Pick([]PlannerReplica{{Name: ""}}, []string{"a"}, nil); ok {
		t.Fatal("nameless candidates must fail closed")
	}

	// Negative report counts or zero timestamp must be ignored (fall back to local floor).
	now := time.Unix(1_800_000_000, 0)
	if est, used := PowerOfKLoadEstimate(2, 3, ReplicaLoadReport{Waiting: -5, Running: 100, ReportedAt: now}, true, now, 5*time.Second); est != 6 || used {
		t.Fatalf("negative waiting must be ignored: est=%d used=%v", est, used)
	}
	if est, used := PowerOfKLoadEstimate(2, 3, ReplicaLoadReport{Waiting: 50, Running: 50}, true, now, 5*time.Second); est != 6 || used {
		t.Fatalf("zero ReportedAt must be ignored: est=%d used=%v", est, used)
	}

	// Default ReplicaDispatch without WithPowerOfKPolicy remains pure HRW.
	rPlain, err := NewReplicaDispatch("fleet", []PlannerReplica{
		{Name: "w0", Planner: &failingPowerOfKPlanner{name: "w0"}},
		{Name: "w1", Planner: &failingPowerOfKPlanner{name: "w1"}},
		{Name: "w2", Planner: &failingPowerOfKPlanner{name: "w2"}},
	})
	if err != nil {
		t.Fatalf("NewReplicaDispatch: %v", err)
	}
	for i := 0; i < 10; i++ {
		p := []string{fmt.Sprintf("hrw-%d", i)}
		got, err := rPlain.pick(p)
		want, werr := rPlain.pickKeyed(p)
		if !errors.Is(err, werr) || got.Name != want.Name {
			t.Fatalf("default pick diverged from HRW: got=%q,%v want=%q,%v", got.Name, err, want.Name, werr)
		}
	}

	// Verify concurrent PickWithRunnerUp safety under parallel calls.
	var wg sync.WaitGroup
	cands := []PlannerReplica{{Name: "a"}, {Name: "b"}, {Name: "c"}}
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < 20; i++ {
				if _, ok := pol.PickWithRunnerUp(cands, []string{fmt.Sprintf("k-%d-%d", id, i)}, nil); !ok {
					t.Errorf("concurrent PickWithRunnerUp failed")
				}
			}
		}(g)
	}
	wg.Wait()
}
