package gateway

// kvmmu_pressure_relief.go — issue #1073, KEYSTONE of epic #1072 (make the OWNED KV cache
// value OBSERVED + default-visible + 2×). It makes the hardware-capacity executor a LIVE
// serve-loop caller for the first time.
//
// The gap (adversarially verified in the issue): internal/engine's RunCapacityPressureSweep →
// CapacityAdapter.Execute → real abi.KVBackend.Evict/StageSpan is a complete, tested
// pressure-relief executor with ZERO live serve-loop callers — every caller is a test. So
// cachemeta.PlanPlacement says "demote to DRAM" and nothing moves the bytes
// (capacity_adapter.go's own "Honest fence": "The live serving loop still has to supply
// pressured candidates and invoke the sweep"). This file is that missing call site: after a
// served decode turn mutates the KV cache, the gateway probes live HBM pressure and runs the
// sweep when it crosses a high-water mark, demoting a hot span to the colder tier instead of
// dropping it.
//
// INJECTION DISCIPLINE (mirrors KVResidencyReclaimer / SetKVResidencyReclaimer in
// kvmmu_slot_reclaim.go). The gateway must not import internal/engine or internal/compute on
// the request path, so the heavy half is injected by the host as two seams:
//
//   - KVPressureCandidateProvider — supplies the live resident KV spans (the candidate list the
//     sweep needs) plus the current resident byte count, as wire-neutral structs.
//   - KVPressureSweeper — a host closure that closes over the compute.Backend + the
//     engine.CapacityAdapter (live KVBackend + CacheEventRecorder), runs
//     engine.RunCapacityPressureSweep at the supplied target, and returns the typed outcome.
//
// FAIL-OPEN / FENCES (the issue's explicit posture, inherited from #915):
//   - Gated behind FAK_INKERNEL_KVMMU (inkernelKVMMUEnabled, default off). Off ⇒ the path is
//     byte-identical to today — no probe, no sweep, no allocation.
//   - A nil provider or sweeper is a no-op. The host injects them ONLY when a device backend is
//     present (a compute.Backend that can report DeviceMemory), so "provider+sweeper wired" IS
//     the "there is a device to relieve pressure on" signal — the gateway needs no compute
//     import to make that call, exactly as the reclaimer needs no kvmmu import.
//   - An empty candidate list is a no-op (nothing resident to move).
//   - Boundary discipline: the sweep runs at the post-decode boundary the SlotEvent-style hooks
//     already fire on (complete()'s success tail), never mid-decode.
//
// The fak_engine_cache_* metric per demote/spill is automatic: the injected sweeper's
// CapacityAdapter.Recorder folds each move into the same cache-event stream the gateway already
// scrapes onto /metrics — no new metric plumbing here.
//
// WHAT THIS DOES NOT CLAIM. The production provider is nil until the persistent in-kernel span
// enumerator lands (the fenced follow-on #1074 / #987): InKernelPlanner keeps residency in a
// radix reuse tree and builds a kvmmu.Context ephemerally per eviction, so there is no durable
// resident-span list to enumerate yet (kvmmu.Segment{From,Len,KV} is the real future source).
// So this ships the LIVE, non-test call site + the demote-not-drop test under synthetic
// pressure; it does not assert the served loop demotes a real span under real GPU pressure today.

import (
	"context"
	"os"
	"strconv"
	"strings"
)

// defaultKVHighWater is the HBM high-water mark (fraction in (0,1]) at which a served turn's
// post-decode pressure sweep begins demoting hot spans to the colder tier. 0.80 matches the
// issue's example and the TargetPressure used throughout the engine sweep tests — demotion
// happens before the allocator is literally out of memory. Override with FAK_KV_HIGHWATER.
const defaultKVHighWater = 0.80

// KVPressureCandidate is the gateway's wire-neutral projection of one live resident KV span the
// pressure sweep may demote — the same import-clean discipline as SlotFreed. The host lowers it
// into an engine.CapacityPressureCandidate (a cachemeta.PlacementRequest + engine.PlacementMove)
// inside the injected sweeper. SizeBytes / Tokens / PerTokenPrefillNanos drive the retain-vs-evict
// economics; SpanDigest / From / N are the span's executable identity in the live KV backend.
type KVPressureCandidate struct {
	SpanDigest           string
	From                 int
	N                    int
	ModelID              string
	TokenizerID          string
	SizeBytes            int64
	Tokens               int
	PerTokenPrefillNanos int64
}

// KVPressureCandidateProvider is the seam the gateway drives after a served decode turn to obtain
// the live pressured KV spans. The host injects an implementation backed by the served residency;
// a server with none wired leaves the edge inert. ResidentBytes is the current live KV residency
// (the byte count the sweep measures pressure against); an empty candidate slice means there is
// nothing resident to move (a clean no-op).
type KVPressureCandidateProvider interface {
	PressuredCandidates() (residentBytes int64, candidates []KVPressureCandidate)
}

// KVPressureRelief is the typed outcome the injected sweeper reports back to the gateway — the
// minimal projection the log line needs, so the gateway never imports engine.CapacityPressureResult.
type KVPressureRelief struct {
	Known          bool
	AppliedMoves   int
	Faults         int
	ReclaimedBytes int64
	FinalPressure  float64
}

// KVPressureSweeper is the host closure that actually relieves pressure: it closes over the
// compute.Backend and the engine.CapacityAdapter (live KVBackend + CacheEventRecorder), runs
// engine.RunCapacityPressureSweep over the lowered candidates at the supplied high-water target,
// and returns the typed outcome. The gateway holds it as a plain func so it stays free of the
// engine/compute imports.
type KVPressureSweeper func(ctx context.Context, residentBytes int64, target float64, candidates []KVPressureCandidate) KVPressureRelief

// SetKVPressureRelief installs the host's KV pressure-relief seams (#1073). Pass a non-nil
// provider AND sweeper to arm the post-decode sweep; pass nil to clear. Settable after New so the
// host can build them once the in-kernel model/residency + device backend are loaded (mirroring
// SetKVResidencyReclaimer). A nil receiver is a no-op.
func (s *Server) SetKVPressureRelief(provider KVPressureCandidateProvider, sweeper KVPressureSweeper) {
	if s == nil {
		return
	}
	s.kvPressureMu.Lock()
	s.kvPressureProvider = provider
	s.kvPressureSweeper = sweeper
	s.kvPressureMu.Unlock()
}

// RelieveKVPressure is the exported, host-drivable form of the post-decode pressure-relief edge
// (#1094) — the symmetric twin of the slot sibling's exported ReclaimKVOnSlotFreed. The gateway
// already calls maybeRelieveKVPressure on its own served tail; this lets the host (or a host-wire
// test) drive the SAME edge through the SAME gating (flag + injected provider + sweeper), so the
// installer wired by serve.go's wireKVPressureRelief is witnessable without reaching into an
// unexported method. It carries no extra authority: every fence (flag-off, nil seams, empty
// candidates) is enforced by the shared maybeRelieveKVPressure body.
func (s *Server) RelieveKVPressure(ctx context.Context) (relief KVPressureRelief, fired bool) {
	return s.maybeRelieveKVPressure(ctx)
}

// maybeRelieveKVPressure is the gateway's post-decode consumer (#1073): after a served turn has
// mutated the KV cache, it probes live pressure and runs the injected sweep when it crosses the
// high-water mark, demoting a hot span instead of dropping it. It is the LIVE serve-path call
// site the keystone exists to add.
//
// Returns (relief, fired): fired reports whether the sweep edge engaged — flag on AND a provider
// AND a sweeper wired AND a non-empty candidate list. Every non-firing path returns
// (KVPressureRelief{}, false): flag off (the default, byte-identical to pre-#1073), no seams
// injected yet, or nothing resident to move (fail-open, exactly as the explainer fences).
func (s *Server) maybeRelieveKVPressure(ctx context.Context) (relief KVPressureRelief, fired bool) {
	if s == nil {
		return KVPressureRelief{}, false
	}
	// Flag-off default: a no-op, byte-identical to the pre-#1073 served path. The gate mirrors
	// the in-kernel KVMMU span bridge so the demote edge engages on exactly the same truthy set.
	if !inkernelKVMMUEnabled() {
		return KVPressureRelief{}, false
	}
	s.kvPressureMu.RLock()
	provider := s.kvPressureProvider
	sweeper := s.kvPressureSweeper
	s.kvPressureMu.RUnlock()
	if provider == nil || sweeper == nil {
		// No device / no host wiring: behave exactly as today (the issue's fail-open fence).
		return KVPressureRelief{}, false
	}
	residentBytes, cands := provider.PressuredCandidates()
	if len(cands) == 0 {
		return KVPressureRelief{}, false
	}
	relief = sweeper(ctx, residentBytes, kvHighWaterTarget(), cands)
	if (relief.AppliedMoves > 0 || relief.Faults > 0) && s.logf != nil {
		s.logf("gateway: KV pressure sweep applied=%d faults=%d reclaimed=%dB final_pressure=%.3f",
			relief.AppliedMoves, relief.Faults, relief.ReclaimedBytes, relief.FinalPressure)
	}
	return relief, true
}

// kvHighWaterTarget resolves the HBM high-water mark: defaultKVHighWater (0.80), overridable via
// FAK_KV_HIGHWATER. An unparseable or out-of-(0,1] value falls back to the default, so a typo can
// never disable pressure relief by pushing the target to 0 or above 1.
func kvHighWaterTarget() float64 {
	v := strings.TrimSpace(os.Getenv("FAK_KV_HIGHWATER"))
	if v == "" {
		return defaultKVHighWater
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil || f <= 0 || f > 1 {
		return defaultKVHighWater
	}
	return f
}
