package servicelease

import (
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

func BenchmarkTableAcquire(b *testing.B) {
	b.Run("ReacquireSameHolder", func(b *testing.B) {
		tb := NewTable(10000)
		inc := Incarnation{Node: "n1", BootID: "b1"}
		tb.RecordIncarnation(inc)
		nowMS := int64(1000)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			l, err := tb.Acquire("bench-workload", inc, nowMS)
			if err != nil || l == nil {
				b.Fatalf("Acquire failed: %v", err)
			}
		}
	})

	b.Run("FreshWorkload", func(b *testing.B) {
		tb := NewTable(10000)
		inc := Incarnation{Node: "n1", BootID: "b1"}
		tb.RecordIncarnation(inc)
		workloads := make([]string, 1024)
		for i := range workloads {
			workloads[i] = fmt.Sprintf("wl-%d", i)
		}
		nowMS := int64(1000)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			wl := workloads[i%1024]
			if _, err := tb.Acquire(wl, inc, nowMS); err != nil {
				b.Fatalf("Acquire failed: %v", err)
			}
		}
	})

	b.Run("RefusedLeaseHeld", func(b *testing.B) {
		tb := NewTable(10000)
		owner := Incarnation{Node: "n1", BootID: "b1"}
		rival := Incarnation{Node: "n2", BootID: "b1"}
		tb.RecordIncarnation(owner)
		tb.RecordIncarnation(rival)
		if _, err := tb.Acquire("bench-workload", owner, 1000); err != nil {
			b.Fatal(err)
		}
		nowMS := int64(2000)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := tb.Acquire("bench-workload", rival, nowMS); err == nil {
				b.Fatal("expected ErrLeaseHeld")
			}
		}
	})
}

func BenchmarkTableRenew(b *testing.B) {
	tb := NewTable(10000)
	inc := Incarnation{Node: "n1", BootID: "b1"}
	tb.RecordIncarnation(inc)
	l, err := tb.Acquire("bench-workload", inc, 1000)
	if err != nil {
		b.Fatal(err)
	}

	b.Run("Valid", func(b *testing.B) {
		tok := l.Token
		nowMS := int64(2000)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := tb.Renew("bench-workload", inc, tok, nowMS); err != nil {
				b.Fatalf("Renew failed: %v", err)
			}
		}
	})

	b.Run("RefusedFencedToken", func(b *testing.B) {
		staleTok := FencingToken{Generation: l.Token.Generation, LeaseSeq: l.Token.LeaseSeq - 1}
		nowMS := int64(2000)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := tb.Renew("bench-workload", inc, staleTok, nowMS); err == nil {
				b.Fatal("expected ErrFenced")
			}
		}
	})

	b.Run("RefusedExpired", func(b *testing.B) {
		tok := l.Token
		expiredMS := int64(1000000)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := tb.Renew("bench-workload", inc, tok, expiredMS); err == nil {
				b.Fatal("expected ErrLeaseExpired")
			}
		}
	})
}

func BenchmarkTablePublishCompletion(b *testing.B) {
	inc := Incarnation{Node: "n1", BootID: "b1"}

	b.Run("AdvanceSequence", func(b *testing.B) {
		tb := NewTable(10000)
		tb.RecordIncarnation(inc)
		l, err := tb.Acquire("bench-workload", inc, 1000)
		if err != nil {
			b.Fatal(err)
		}
		tok := l.Token
		b.ReportAllocs()
		l.Checkpoint = Checkpoint{}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			cp := Checkpoint{Seq: uint64(i + 1), ID: "ckpt", AtUnixMS: 1000 + int64(i)}
			if err := tb.PublishCompletion("bench-workload", inc, tok, cp); err != nil {
				b.Fatalf("PublishCompletion failed: %v", err)
			}
		}
	})

	b.Run("RegressionRefused", func(b *testing.B) {
		tb := NewTable(10000)
		tb.RecordIncarnation(inc)
		l, err := tb.Acquire("bench-workload", inc, 1000)
		if err != nil {
			b.Fatal(err)
		}
		tok := l.Token
		l.Checkpoint = Checkpoint{Seq: 1000000000}
		staleCp := Checkpoint{Seq: 10}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := tb.PublishCompletion("bench-workload", inc, tok, staleCp); err == nil {
				b.Fatal("expected ErrCheckpointRegression")
			}
		}
	})
}

func BenchmarkTableWouldAccept(b *testing.B) {
	tb := NewTable(10000)
	inc := Incarnation{Node: "n1", BootID: "b1"}
	tb.RecordIncarnation(inc)
	l, err := tb.Acquire("bench-workload", inc, 1000)
	if err != nil {
		b.Fatal(err)
	}
	tok := l.Token
	nowMS := int64(2000)

	b.Run("ValidOwner", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if !tb.WouldAccept("bench-workload", inc, tok, nowMS) {
				b.Fatal("expected WouldAccept=true")
			}
		}
	})

	b.Run("StaleToken", func(b *testing.B) {
		staleTok := FencingToken{Generation: tok.Generation, LeaseSeq: tok.LeaseSeq - 1}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if tb.WouldAccept("bench-workload", inc, staleTok, nowMS) {
				b.Fatal("expected WouldAccept=false")
			}
		}
	})
}

func BenchmarkRecordIncarnation(b *testing.B) {
	tb := NewTable(10000)
	inc := Incarnation{Node: "n1", BootID: "b1"}
	tb.RecordIncarnation(inc)

	b.Run("SameBootIdempotent", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if tb.RecordIncarnation(inc) {
				b.Fatal("expected no change on same boot ID")
			}
		}
	})

	b.Run("NewBootSuperseded", func(b *testing.B) {
		boots := make([]Incarnation, 1024)
		for i := range boots {
			boots[i] = Incarnation{Node: "n2", BootID: fmt.Sprintf("boot-%d", i)}
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = tb.RecordIncarnation(boots[i%1024])
		}
	})
}

func BenchmarkClassify(b *testing.B) {
	fresh := Evidence{
		NowMS:              10000,
		LastHeartbeatMS:    9500,
		HeartbeatTimeoutMS: 2000,
		HeartbeatBootID:    "b1",
		KnownBootID:        "b1",
		ReadBack: &servicespec.Observed{
			Schema: servicespec.ObservedSchemaV1,
			Phase:  servicespec.PhaseReady,
		},
	}

	b.Run("Healthy", func(b *testing.B) {
		ev := fresh
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if got := Classify(ev); got != CondHealthy {
				b.Fatalf("got %v, want healthy", got)
			}
		}
	})

	b.Run("ProcessCrashed", func(b *testing.B) {
		ev := fresh
		ev.ReadBack = &servicespec.Observed{
			Schema: servicespec.ObservedSchemaV1,
			Phase:  servicespec.PhaseFailed,
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if got := Classify(ev); got != CondProcessCrashed {
				b.Fatalf("got %v, want process-crashed", got)
			}
		}
	})

	b.Run("HostRebooted", func(b *testing.B) {
		ev := fresh
		ev.HeartbeatBootID = "b2"
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if got := Classify(ev); got != CondHostRebooted {
				b.Fatalf("got %v, want host-rebooted", got)
			}
		}
	})

	b.Run("NetworkPartitioned", func(b *testing.B) {
		ev := fresh
		ev.LastHeartbeatMS = 1000
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if got := Classify(ev); got != CondNetworkPartitioned {
				b.Fatalf("got %v, want network-partitioned", got)
			}
		}
	})

	b.Run("IntentionallyStopped", func(b *testing.B) {
		ev := fresh
		ev.DesiredStopped = true
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if got := Classify(ev); got != CondIntentionallyStopped {
				b.Fatalf("got %v, want intentionally-stopped", got)
			}
		}
	})
}

func BenchmarkBuildPlan(b *testing.B) {
	tb := NewTable(10000)
	inc := Incarnation{Node: "node-1", BootID: "b1"}
	tb.RecordIncarnation(inc)
	if _, err := tb.Acquire("bench-workload", inc, 1000); err != nil {
		b.Fatal(err)
	}

	evHealthy := Evidence{
		NowMS:              2000,
		LastHeartbeatMS:    2000,
		HeartbeatTimeoutMS: 5000,
		HeartbeatBootID:    "b1",
		KnownBootID:        "b1",
		ReadBack:           &servicespec.Observed{Phase: servicespec.PhaseReady},
	}

	evCrash := Evidence{
		NowMS:              2000,
		LastHeartbeatMS:    2000,
		HeartbeatTimeoutMS: 5000,
		HeartbeatBootID:    "b1",
		KnownBootID:        "b1",
		ReadBack:           &servicespec.Observed{Phase: servicespec.PhaseFailed},
	}

	evPartitionWait := Evidence{
		NowMS:              4000,
		LastHeartbeatMS:    1000,
		HeartbeatTimeoutMS: 2000,
	}

	b.Run("Healthy", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			p := BuildPlan(tb, "bench-workload", evHealthy)
			if p.Action != ActionNone {
				b.Fatalf("unexpected action %s", p.Action)
			}
		}
	})

	b.Run("LocalRecovery", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			p := BuildPlan(tb, "bench-workload", evCrash)
			if p.Action != ActionRestartLocal {
				b.Fatalf("unexpected action %s", p.Action)
			}
		}
	})

	b.Run("WaitLease", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			p := BuildPlan(tb, "bench-workload", evPartitionWait)
			if p.Action != ActionWaitLease {
				b.Fatalf("unexpected action %s", p.Action)
			}
		}
	})

	b.Run("PlanJSON", func(b *testing.B) {
		p := BuildPlan(tb, "bench-workload", evHealthy)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			raw, err := p.JSON()
			if err != nil || len(raw) == 0 {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkReconcilerStep(b *testing.B) {
	b.Run("SteadyStateHealthy", func(b *testing.B) {
		tb := NewTable(30000)
		r := NewReconciler(tb, "ctrl-1", "epoch-1")
		clk := int64(1000)
		if _, err := r.AcquireAuthority("node-1", clk); err != nil {
			b.Fatal(err)
		}
		spec := &servicespec.Spec{
			Schema:   servicespec.SchemaV1,
			Identity: servicespec.Identity{Node: "node-1", Service: "svc"},
			Kind:     servicespec.KindService,
			Desired:  servicespec.DesiredRunning,
		}
		spec.Normalize()
		ev := Evidence{
			LastHeartbeatMS:    1000,
			HeartbeatBootID:    "b1",
			HeartbeatTimeoutMS: 5000,
			KnownBootID:        "b1",
			ReadBack: &servicespec.Observed{
				Schema:   servicespec.ObservedSchemaV1,
				Identity: servicespec.Identity{Node: "node-1", Service: "svc"},
				Phase:    servicespec.PhaseReady,
			},
		}
		in := StepInput{
			Spec:     spec,
			Evidence: ev,
			NowMS:    1000,
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res, err := r.Step(in)
			if err != nil || res.Plan.Action != ActionNone {
				b.Fatalf("err=%v action=%s", err, res.Plan.Action)
			}
		}
	})

	b.Run("CrashRecoveryPaced", func(b *testing.B) {
		tb := NewTable(30000)
		r := NewReconciler(tb, "ctrl-1", "epoch-1")
		clk := int64(1000)
		if _, err := r.AcquireAuthority("node-1", clk); err != nil {
			b.Fatal(err)
		}
		spec := &servicespec.Spec{
			Schema:   servicespec.SchemaV1,
			Identity: servicespec.Identity{Node: "node-1", Service: "svc"},
			Kind:     servicespec.KindService,
			Desired:  servicespec.DesiredRunning,
		}
		spec.Normalize()
		ev := Evidence{
			LastHeartbeatMS:    1000,
			HeartbeatBootID:    "b1",
			HeartbeatTimeoutMS: 5000,
			KnownBootID:        "b1",
			ReadBack: &servicespec.Observed{
				Schema:   servicespec.ObservedSchemaV1,
				Identity: servicespec.Identity{Node: "node-1", Service: "svc"},
				Phase:    servicespec.PhaseFailed,
			},
		}
		in := StepInput{
			Spec:     spec,
			Evidence: ev,
			NowMS:    1000,
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			p := r.paces["svc"]
			if p != nil {
				p.attempt = 0
				p.restarts = p.restarts[:0]
			}
			res, err := r.Step(in)
			if err != nil || res.Plan.Action != ActionRestartLocal {
				b.Fatalf("err=%v action=%s", err, res.Plan.Action)
			}
		}
	})
}

func BenchmarkSimStep(b *testing.B) {
	for _, nNodes := range []int{2, 5, 10} {
		b.Run(fmt.Sprintf("%d_Nodes", nNodes), func(b *testing.B) {
			names := make([]string, nNodes)
			for i := range names {
				names[i] = fmt.Sprintf("node-%d", i)
			}
			s := NewSim("sim-workload", 100000000, 1000, 5000, names...)
			if err := s.Grant("node-0"); err != nil {
				b.Fatal(err)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				owners := s.Step()
				if owners > 1 {
					b.Fatalf("invariant broken: %d owners", owners)
				}
			}
		})
	}
}
