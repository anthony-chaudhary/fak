package main

// kvmmu_pressure_bridge_test.go — exercises the host half of the #1073 capacity wire: the sweeper
// closure newCapacityPressureSweeper builds + the lowerPressureCandidates lowering. It proves the
// bridge runs the REAL engine.RunCapacityPressureSweep over a fake device backend at high pressure
// and demotes the lowered candidate (StageSpan → Evict), so the cmd/fak glue the gateway injects
// is itself witnessed, not just defined.

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

type bridgeFakeBackend struct {
	compute.Backend
	total, free int64
}

func (f bridgeFakeBackend) Caps() compute.Caps {
	return compute.Caps{Async: true, DeviceMemory: true, CapacityProbe: true}
}
func (f bridgeFakeBackend) DeviceMemory() (int64, int64, bool) { return f.total, f.free, true }

type bridgeFakeKV struct {
	stageCalls int
	evicts     []struct{ from, n int }
}

func (f *bridgeFakeKV) Len() int                    { return 4096 }
func (f *bridgeFakeKV) Prefill(ids []int) []float32 { return nil }
func (f *bridgeFakeKV) ModelID() string             { return "bridge-model" }
func (f *bridgeFakeKV) Evict(from, n int) int {
	f.evicts = append(f.evicts, struct{ from, n int }{from, n})
	return n
}
func (f *bridgeFakeKV) StageSpan(_ context.Context, digest string, _, n int) (abi.KVResidency, error) {
	f.stageCalls++
	return abi.KVResidency{Outcome: abi.KVResidencyOK, Digest: digest, Positions: n}, nil
}
func (f *bridgeFakeKV) RestoreSpan(_ context.Context, digest string) (abi.KVResidency, error) {
	return abi.KVResidency{Outcome: abi.KVResidencyMiss, Digest: digest}, nil
}

// fakeKVPressureProvider is the stand-in for the not-yet-built resident-span enumerator (#1074 /
// #987): it hands the installer one fat resident span under near-full residency, so a wire test can
// prove the serve-host installer arms a sweep that fires — without depending on the durable ledger
// that does not exist yet.
type fakeKVPressureProvider struct {
	residentBytes int64
	cands         []gateway.KVPressureCandidate
}

func (p fakeKVPressureProvider) PressuredCandidates() (int64, []gateway.KVPressureCandidate) {
	return p.residentBytes, p.cands
}

// TestWireKVPressureReliefReachableWithProvider proves the #1094 installer is REACHABLE end to end:
// wireKVPressureRelief installs (provider, sweeper) on a real gateway.Server, and with the gate on
// the gateway's post-decode hook fires the sweep through that wire — demoting the fat span (StageSpan
// then Evict) instead of dropping it. This witnesses the call site serve.go now drives, with the
// fake provider standing in for the fenced follow-on enumerator. It is the "a test proving the wire
// is reachable with a fake provider" acceptance criterion for the partial.
func TestWireKVPressureReliefReachableWithProvider(t *testing.T) {
	t.Setenv("FAK_INKERNEL_KVMMU", "on")
	const total = 100 << 20
	kv := &bridgeFakeKV{}
	srv := &gateway.Server{}
	provider := fakeKVPressureProvider{
		residentBytes: 90 << 20,
		cands: []gateway.KVPressureCandidate{{
			SpanDigest:           "span-wire",
			From:                 16,
			N:                    4000,
			ModelID:              "bridge-model",
			SizeBytes:            64 << 20,
			Tokens:               4000,
			PerTokenPrefillNanos: 2_000_000,
		}},
	}
	wireKVPressureRelief(srv,
		bridgeFakeBackend{Backend: compute.Default(), total: total, free: compute.FreeUnknown},
		kv, provider)

	relief, fired := srv.RelieveKVPressure(context.Background())
	if !fired {
		t.Fatal("post-decode sweep did not fire through the installed wire")
	}
	if relief.AppliedMoves != 1 {
		t.Fatalf("want exactly one demote applied through the wire, got %+v", relief)
	}
	if kv.stageCalls != 1 || len(kv.evicts) != 1 || kv.evicts[0].from != 16 || kv.evicts[0].n != 4000 {
		t.Fatalf("installed wire did not stage-then-evict the span: stage=%d evicts=%+v", kv.stageCalls, kv.evicts)
	}
}

// TestWireKVPressureReliefNilProviderInert proves the production posture: the installer with a nil
// provider (what serve.go passes today) arms the sweeper but leaves the edge a clean no-op — no
// stage, no evict — so the live serve path is byte-identical to pre-#1094 until the span enumerator
// lands.
func TestWireKVPressureReliefNilProviderInert(t *testing.T) {
	t.Setenv("FAK_INKERNEL_KVMMU", "on")
	kv := &bridgeFakeKV{}
	srv := &gateway.Server{}
	wireKVPressureRelief(srv,
		bridgeFakeBackend{Backend: compute.Default(), total: 100 << 20, free: compute.FreeUnknown},
		kv, nil)
	if _, fired := srv.RelieveKVPressure(context.Background()); fired {
		t.Fatal("nil-provider install fired the sweep — should be inert (production posture)")
	}
	if kv.stageCalls != 0 || len(kv.evicts) != 0 {
		t.Fatalf("nil-provider install ran a demote: stage=%d evicts=%+v", kv.stageCalls, kv.evicts)
	}
}

func TestNewCapacityPressureSweeperDemotes(t *testing.T) {
	const total = 100 << 20
	kv := &bridgeFakeKV{}
	rec := engine.NewCacheEventRecorder()
	sweeper := newCapacityPressureSweeper(
		bridgeFakeBackend{Backend: compute.Default(), total: total, free: compute.FreeUnknown},
		&engine.CapacityAdapter{KV: kv, Recorder: rec},
	)
	relief := sweeper(context.Background(), 90<<20, 0.80, []gateway.KVPressureCandidate{{
		SpanDigest:           "span-bridge",
		From:                 16,
		N:                    4000,
		ModelID:              "bridge-model",
		SizeBytes:            64 << 20,
		Tokens:               4000,
		PerTokenPrefillNanos: 2_000_000,
	}})
	if !relief.Known || relief.AppliedMoves != 1 {
		t.Fatalf("bridge sweeper did not apply one demote: %+v", relief)
	}
	if relief.FinalPressure >= 0.80 {
		t.Fatalf("bridge sweeper did not relieve pressure below target: %+v", relief)
	}
	if kv.stageCalls != 1 || len(kv.evicts) != 1 || kv.evicts[0].from != 16 || kv.evicts[0].n != 4000 {
		t.Fatalf("bridge did not stage-then-evict the lowered span: stage=%d evicts=%+v", kv.stageCalls, kv.evicts)
	}
}

// TestLowerPressureCandidatesUsesProbedProfilesAndLivePressure is the #1468 acceptance (Phase-2
// child of #1463): lowerPressureCandidates must plan against cachemeta.ProbedTierProfiles + the
// live per-tier TierPressure (HostDRAMPressure/DiskPressure/DeviceHBMPressure), not
// cachemeta.DefaultTierProfiles' representative placeholders with an empty (zero-pressure)
// request — the exact gap the issue names in lowerPressureCandidates around cmd/fak/
// kvmmu_pressure_bridge.go:101.
//
// Structured as the issue's own refute guard: first assert the OLD construction (DefaultTierProfiles
// + a probed ladder that OMITS HBM, i.e. a no-GPU box, with DRAM maximally pressured) would still
// have planned a demote INTO DRAM — proving the placeholder target really is what the unfixed code
// produces. Then assert the NEW construction (ProbedTierProfiles for the same no-GPU box + real
// DRAM pressure 1.0 folded into the request) picks a DIFFERENT, correct ToTier — the coldest-colder-
// with-room walk must skip the full DRAM tier. Finally, calls the REAL lowerPressureCandidates (not
// a hand-rolled stand-in) to prove the shipped wiring produces the same shape: Profiles keyed by
// ProbedTierProfiles' tier set (no TierHBM against a backend that cannot probe device memory) and a
// non-nil Pressure map, which is what makes the flip possible on a real box.
func TestLowerPressureCandidatesUsesProbedProfilesAndLivePressure(t *testing.T) {
	const sizeBytes = 64 << 20
	const tokens = 4000
	const perTokenPrefillNanos = 2_000_000

	// The no-GPU ladder #1468 calls out: ProbedTierProfiles with HBMPresent=false drops TierHBM
	// entirely, exactly as a box with no device would. DRAM/Disk stay representative-sized here
	// (0 probe reading keeps the default) — only the PRESSURE differs between old and new below.
	noHBMProbedProfiles := cachemeta.ProbedTierProfiles(cachemeta.CapacityProbe{})

	// Candidates start resident in HBM, exactly as lowerPressureCandidates builds every live
	// candidate's Lifecycle (NewLifecycle(cachemeta.TierHBM, 0).MarkResident(...)) — the sweep only
	// ever fires when HBM is under pressure, so HBM pressure 1.0 puts both requests below in the
	// same demote-or-evict branch PlanPlacement takes on the real serve path.
	baseReq := func(profiles map[cachemeta.ResidencyTier]cachemeta.TierProfile) cachemeta.PlacementRequest {
		lc := cachemeta.NewLifecycle(cachemeta.TierHBM, 0).MarkResident(profiles, 0)
		return cachemeta.PlacementRequest{
			Lifecycle:            lc,
			SizeBytes:            sizeBytes,
			Tokens:               tokens,
			Profiles:             profiles,
			Policy:               cachemeta.LifecyclePolicy{DemoteOnExpiry: true},
			PerTokenPrefillNanos: perTokenPrefillNanos,
		}
	}

	// REFUTE GUARD: the OLD construction — DefaultTierProfiles (the placeholder ladder, which
	// still advertises a TierHBM+TierDRAM a no-GPU/full-DRAM box does not really have room in) with
	// only HBM pressured (DRAM left at its unfixed, never-injected empty pressure) — must plan the
	// demote INTO DRAM, because the placeholder ladder assumes DRAM has room. If this ever fails,
	// the "placeholder target" this test flips away from was never the placeholder target, so the
	// flip below proves nothing.
	oldReq := baseReq(cachemeta.DefaultTierProfiles())
	oldReq.Pressure = cachemeta.TierPressure{cachemeta.TierHBM: 1.0}
	oldDecision := cachemeta.PlanPlacement(oldReq)
	if oldDecision.Action != cachemeta.ActionDemote || oldDecision.ToTier != cachemeta.TierDRAM {
		t.Fatalf("refute guard: DefaultTierProfiles + HBM-only pressure must demote into DRAM, got %s->%s",
			oldDecision.Action, oldDecision.ToTier)
	}

	// NEW construction: the probed (no-HBM-tier) ladder + real per-tier pressure with BOTH HBM and
	// DRAM maximally pressured (1.0), exactly as DeviceHBMPressure+HostDRAMPressure would report on
	// a pressured, no-GPU-relief host. The coldest-colder-with-room walk must skip the full DRAM tier
	// and land on the next colder tier the probed ladder still carries: Disk (NUMA-far/CXL are not
	// local-probeable, so ProbedTierProfiles leaves them out of the proved ladder entirely — see its
	// own doc comment — and coldestColderWithRoom skips any tier absent from Profiles).
	newReq := baseReq(noHBMProbedProfiles)
	newReq.Pressure = cachemeta.TierPressure{cachemeta.TierHBM: 1.0, cachemeta.TierDRAM: 1.0}
	newDecision := cachemeta.PlanPlacement(newReq)
	if newDecision.ToTier == oldDecision.ToTier {
		t.Fatalf("probed profiles + live DRAM pressure did not move the target off the placeholder's DRAM pick: got %s->%s",
			newDecision.Action, newDecision.ToTier)
	}
	if newDecision.Action != cachemeta.ActionSpill || newDecision.ToTier != cachemeta.TierDisk {
		t.Fatalf("with HBM and DRAM full on the probed no-GPU ladder, the candidate should spill to disk (the only colder tier the ladder proves), got %s->%s",
			newDecision.Action, newDecision.ToTier)
	}

	// Now prove the SHIPPED wiring (the real lowerPressureCandidates, not a hand-rolled stand-in)
	// actually builds requests shaped this way: on a backend that cannot probe device memory
	// (cpu-ref-like — bridgeFakeBackend with CapacityProbe:false below reports known=false, so HBM
	// is correctly dropped from the ladder) the lowered candidate's Profiles must NOT contain
	// TierHBM (the no-GPU fix) and its Pressure must be non-nil (populated from the live probes,
	// not the old unset zero value) so a real box's fullness — not an assumed-empty placeholder —
	// is what the planner sees.
	noProbeBackend := bridgeNoCapacityBackend{}
	lowered := lowerPressureCandidates(noProbeBackend, 90<<20, []gateway.KVPressureCandidate{{
		SpanDigest:           "span-1468",
		From:                 16,
		N:                    tokens,
		ModelID:              "bridge-model",
		SizeBytes:            sizeBytes,
		Tokens:               tokens,
		PerTokenPrefillNanos: perTokenPrefillNanos,
	}})
	if len(lowered) != 1 {
		t.Fatalf("want exactly one lowered candidate, got %d", len(lowered))
	}
	got := lowered[0].Request
	if _, hasHBM := got.Profiles[cachemeta.TierHBM]; hasHBM {
		t.Fatalf("lowerPressureCandidates kept TierHBM in the ladder against a backend that cannot probe device memory: %+v", got.Profiles)
	}
	if got.Pressure == nil {
		t.Fatal("lowerPressureCandidates left Pressure nil — live probes were never folded in (the pre-#1468 gap)")
	}
}

// bridgeNoCapacityBackend is a compute.Backend that explicitly cannot probe device memory (its
// Caps().CapacityProbe is false), the cpu-ref/no-GPU-box shape #1468 names: DeviceMemoryInfo must
// report known=false against it, so probedTierProfilesForHost drops TierHBM from the ladder.
type bridgeNoCapacityBackend struct{ compute.Backend }

func (bridgeNoCapacityBackend) Caps() compute.Caps { return compute.Caps{Async: true} }

// TestNewCapacityPressureSweeperNilFailsOpen proves the bridge sweeper is fail-open: a nil backend
// or adapter, or an empty candidate list, reports no relief rather than panicking.
func TestNewCapacityPressureSweeperNilFailsOpen(t *testing.T) {
	nilBackend := newCapacityPressureSweeper(nil, &engine.CapacityAdapter{KV: &bridgeFakeKV{}})
	if r := nilBackend(context.Background(), 1, 0.8, []gateway.KVPressureCandidate{{SpanDigest: "x"}}); r.AppliedMoves != 0 || r.Known {
		t.Fatalf("nil backend should fail open, got %+v", r)
	}
	live := newCapacityPressureSweeper(
		bridgeFakeBackend{Backend: compute.Default(), total: 100 << 20, free: compute.FreeUnknown},
		&engine.CapacityAdapter{KV: &bridgeFakeKV{}},
	)
	if r := live(context.Background(), 90<<20, 0.8, nil); r.AppliedMoves != 0 {
		t.Fatalf("empty candidate list should be a no-op, got %+v", r)
	}
}
