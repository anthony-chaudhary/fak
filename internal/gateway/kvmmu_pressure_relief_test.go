package gateway

// kvmmu_pressure_relief_test.go — issue #1073 KEYSTONE witness: prove that under simulated HBM
// pressure a served turn DEMOTES a span (StageSpan → Evict to the colder tier) instead of
// dropping it, driven through the LIVE gateway call site maybeRelieveKVPressure. The test imports
// internal/engine (the real RunCapacityPressureSweep + CapacityAdapter), so the demote is the
// genuine executor, not a stub — the "a live serve-path call site invokes the executor (not a
// test)" + "a test proves: under simulated HBM pressure, a served turn demotes a span instead of
// dropping it" acceptance criteria, end to end.

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/engine"
)

// pressureFakeBackend is a compute.Backend that advertises a device-capacity probe and reports a
// fixed (total, free) so DeviceHBMPressure is known and high — the test stand-in for a real
// accelerator, mirroring internal/engine/capacity_pressure_test.go's fakeCapBackend.
type pressureFakeBackend struct {
	compute.Backend
	total, free int64
}

func (f pressureFakeBackend) Caps() compute.Caps {
	return compute.Caps{Async: true, DeviceMemory: true, CapacityProbe: true}
}
func (f pressureFakeBackend) DeviceMemory() (int64, int64, bool) { return f.total, f.free, true }

// pressureFakeKV is an abi.KVBackend that records its StageSpan + Evict calls, so the test can
// assert the span was STAGED to the colder tier and THEN evicted from the live tier (a demote),
// never dropped without staging. Mirrors internal/engine/capacity_sweep_test.go's sweepFakeKV.
type pressureFakeKV struct {
	stageCalls int
	evicts     []struct{ from, n int }
}

func (f *pressureFakeKV) Len() int                    { return 4096 }
func (f *pressureFakeKV) Prefill(ids []int) []float32 { return nil }
func (f *pressureFakeKV) ModelID() string             { return "pressure-model" }
func (f *pressureFakeKV) Evict(from, n int) int {
	f.evicts = append(f.evicts, struct{ from, n int }{from, n})
	return n
}
func (f *pressureFakeKV) StageSpan(_ context.Context, digest string, _, n int) (abi.KVResidency, error) {
	f.stageCalls++
	return abi.KVResidency{Outcome: abi.KVResidencyOK, Digest: digest, Positions: n}, nil
}
func (f *pressureFakeKV) RestoreSpan(_ context.Context, digest string) (abi.KVResidency, error) {
	return abi.KVResidency{Outcome: abi.KVResidencyMiss, Digest: digest}, nil
}

// fixedPressureProvider supplies one fat resident span under near-full residency, the candidate
// the sweep should demote.
type fixedPressureProvider struct {
	residentBytes int64
	cands         []KVPressureCandidate
}

func (p fixedPressureProvider) PressuredCandidates() (int64, []KVPressureCandidate) {
	return p.residentBytes, p.cands
}

// newPressureSweeperForTest builds the same sweeper closure the cmd/fak host bridge builds, but
// inline (the bridge lives in package main and is unimportable here). It closes over the fake
// device backend + a real engine.CapacityAdapter over the fake KV, lowering each candidate into a
// resident-on-HBM PlacementRequest so the planner demotes (not evicts) under pressure.
func newPressureSweeperForTest(backend compute.Backend, kv abi.KVBackend, rec *engine.CacheEventRecorder) KVPressureSweeper {
	adapter := &engine.CapacityAdapter{KV: kv, Recorder: rec}
	return func(ctx context.Context, residentBytes int64, target float64, cands []KVPressureCandidate) KVPressureRelief {
		lowered := make([]engine.CapacityPressureCandidate, 0, len(cands))
		for _, c := range cands {
			profiles := cachemeta.DefaultTierProfiles()
			lc := cachemeta.NewLifecycle(cachemeta.TierHBM, 0).MarkResident(profiles, 0)
			lowered = append(lowered, engine.CapacityPressureCandidate{
				Request: cachemeta.PlacementRequest{
					Lifecycle:            lc,
					SizeBytes:            c.SizeBytes,
					Tokens:               int64(c.Tokens),
					Profiles:             profiles,
					Policy:               cachemeta.LifecyclePolicy{DemoteOnExpiry: true},
					PerTokenPrefillNanos: c.PerTokenPrefillNanos,
				},
				Move: engine.PlacementMove{
					SpanDigest:   c.SpanDigest,
					From:         c.From,
					N:            c.N,
					ModelID:      c.ModelID,
					PositionMode: cachemeta.PositionPrefixAligned,
					Owner:        "kv-pressure-sweep",
				},
				ReclaimBytes: c.SizeBytes,
			})
		}
		res, err := engine.RunCapacityPressureSweep(ctx, engine.CapacityPressureSweep{
			Backend:        backend,
			Adapter:        adapter,
			ResidentBytes:  residentBytes,
			TargetPressure: target,
			Candidates:     lowered,
		})
		if err != nil {
			return KVPressureRelief{}
		}
		return KVPressureRelief{
			Known:          res.Known,
			AppliedMoves:   res.AppliedMoves,
			Faults:         res.Faults,
			ReclaimedBytes: res.ReclaimedBytes,
			FinalPressure:  res.FinalPressure,
		}
	}
}

func oneFatCandidate() []KVPressureCandidate {
	return []KVPressureCandidate{{
		SpanDigest:           "span-pressure",
		From:                 32,
		N:                    4000,
		ModelID:              "pressure-model",
		TokenizerID:          "tok",
		SizeBytes:            64 << 20,
		Tokens:               4000,
		PerTokenPrefillNanos: 2_000_000,
	}}
}

// TestMaybeRelieveKVPressureDemotesUnderPressure is the keystone witness (#987): with NO enablement
// env set — proving the DEFAULT-ON policy — and a device-backed provider+sweeper injected, a served
// turn's post-decode hook demotes the hot span (StageSpan then Evict) instead of dropping it, and
// the move lands on the fak_engine_cache_* stream the gateway scrapes onto /metrics.
func TestMaybeRelieveKVPressureDemotesUnderPressure(t *testing.T) {
	// No FAK_KV_PRESSURE_RELIEF (nor FAK_INKERNEL_KVMMU) set: the sweep runs BY DEFAULT (#987).
	const total = 100 << 20
	kv := &pressureFakeKV{}
	rec := engine.NewCacheEventRecorder()
	s := &Server{}
	s.SetKVPressureRelief(
		fixedPressureProvider{residentBytes: 90 << 20, cands: oneFatCandidate()},
		newPressureSweeperForTest(pressureFakeBackend{Backend: compute.Default(), total: total, free: compute.FreeUnknown}, kv, rec),
	)

	relief, fired := s.maybeRelieveKVPressure(context.Background())
	if !fired {
		t.Fatal("pressure-relief edge did not fire with gate on + seams wired + a candidate")
	}
	if relief.AppliedMoves != 1 {
		t.Fatalf("want exactly one demote applied, got %+v", relief)
	}
	if relief.FinalPressure >= 0.80 {
		t.Fatalf("pressure not relieved below the high-water target: %+v", relief)
	}
	// Demote-not-drop: the span was STAGED to the colder tier and THEN evicted from the live one.
	if kv.stageCalls != 1 || len(kv.evicts) != 1 || kv.evicts[0].from != 32 || kv.evicts[0].n != 4000 {
		t.Fatalf("span was not staged-then-evicted (a demote): stage=%d evicts=%+v", kv.stageCalls, kv.evicts)
	}
	// The demote is observable: it lands as a ddr_cache cache-event row in the recorder stream the
	// gateway scrapes onto /metrics (the fak_engine_cache_* family).
	if rows := rec.Metrics().Snapshot().Rows; len(rows) == 0 || rows[0].MemoryClass != string(compute.MemoryDDRCache) {
		t.Fatalf("demote should be visible as a ddr_cache cache-event row, got %+v", rows)
	}
}

// TestMaybeRelieveKVPressureDisabledIsNoOp is the refute guard for the documented disable (#987):
// with FAK_KV_PRESSURE_RELIEF=off the edge is a no-op even with seams wired and a candidate — no
// stage, no evict — proving the disable knob actually gates the default-on policy.
func TestMaybeRelieveKVPressureDisabledIsNoOp(t *testing.T) {
	t.Setenv("FAK_KV_PRESSURE_RELIEF", "off")
	kv := &pressureFakeKV{}
	rec := engine.NewCacheEventRecorder()
	s := &Server{}
	s.SetKVPressureRelief(
		fixedPressureProvider{residentBytes: 90 << 20, cands: oneFatCandidate()},
		newPressureSweeperForTest(pressureFakeBackend{Backend: compute.Default(), total: 100 << 20, free: compute.FreeUnknown}, kv, rec),
	)
	relief, fired := s.maybeRelieveKVPressure(context.Background())
	if fired {
		t.Fatalf("edge fired with the gate off: %+v", relief)
	}
	if kv.stageCalls != 0 || len(kv.evicts) != 0 {
		t.Fatalf("a demote ran with the gate off: stage=%d evicts=%+v", kv.stageCalls, kv.evicts)
	}
}

// TestMaybeRelieveKVPressureNoSeamsFailOpen proves the fail-open fence: default-on policy but no
// provider / sweeper injected (the production serve.go posture today — the resident-span enumerator
// is the gated follow-on #1074) is a clean no-op, byte-identical to pre-#1073.
func TestMaybeRelieveKVPressureNoSeamsFailOpen(t *testing.T) {
	s := &Server{}
	if _, fired := s.maybeRelieveKVPressure(context.Background()); fired {
		t.Fatal("edge fired with no provider/sweeper wired — should fail open")
	}
}

// TestKVPressureReliefEnabled covers the default-on policy and its documented disable (#987): unset
// is ON, off/0/false/no are OFF, and an unrecognized value stays ON so a typo cannot silently
// disable pressure relief.
func TestKVPressureReliefEnabled(t *testing.T) {
	t.Setenv("FAK_KV_PRESSURE_RELIEF", "")
	if !kvPressureReliefEnabled() {
		t.Fatal("unset env: pressure relief must default ON (#987)")
	}
	for _, off := range []string{"off", "0", "false", "no", "OFF", "False"} {
		t.Setenv("FAK_KV_PRESSURE_RELIEF", off)
		if kvPressureReliefEnabled() {
			t.Fatalf("value %q must disable pressure relief", off)
		}
	}
	for _, on := range []string{"on", "1", "true", "yes", "garbage"} {
		t.Setenv("FAK_KV_PRESSURE_RELIEF", on)
		if !kvPressureReliefEnabled() {
			t.Fatalf("value %q must keep pressure relief ON (typo-safe default)", on)
		}
	}
}

// TestKVHighWaterTarget covers the high-water resolution: default, valid override, and the
// fail-safe fallback for an out-of-range or unparseable value.
func TestKVHighWaterTarget(t *testing.T) {
	t.Setenv("FAK_KV_HIGHWATER", "")
	if got := kvHighWaterTarget(); got != defaultKVHighWater {
		t.Fatalf("empty env: want default %v, got %v", defaultKVHighWater, got)
	}
	t.Setenv("FAK_KV_HIGHWATER", "0.65")
	if got := kvHighWaterTarget(); got != 0.65 {
		t.Fatalf("valid override: want 0.65, got %v", got)
	}
	for _, bad := range []string{"0", "1.5", "-0.2", "nan-ish"} {
		t.Setenv("FAK_KV_HIGHWATER", bad)
		if got := kvHighWaterTarget(); got != defaultKVHighWater {
			t.Fatalf("bad override %q: want fallback %v, got %v", bad, defaultKVHighWater, got)
		}
	}
}
