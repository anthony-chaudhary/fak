package main

// kvmmu_pressure_bridge.go — issue #1073, KEYSTONE of epic #1072: the HOST half of the live
// serve-loop capacity wire. internal/gateway exposes two import-clean seams the served decode
// loop drives post-turn (KVPressureCandidateProvider + KVPressureSweeper, see
// internal/gateway/kvmmu_pressure_relief.go); this bridge supplies the heavy implementation the
// gateway must not import — it closes the sweeper closure over the live compute.Backend and the
// engine.CapacityAdapter (the real abi.KVBackend.StageSpan+Evict executor + the
// CacheEventRecorder that folds each demote into the fak_engine_cache_* metric stream), lowers
// the gateway's wire-neutral candidates into engine.CapacityPressureCandidate, and runs
// engine.RunCapacityPressureSweep. It is the cmd/fak twin of kvmmu_slot_bridge.go.
//
// The native complete-prefix bridge below supplies both halves from the in-kernel
// radix tree: enumerable hot owners and a digest-addressed host-DRAM stage/restore
// backend. It is deliberately direct-native-only; proxy/provider KV does not cross
// this seam and cannot be counted as fak-owned residency.

import (
	"context"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

type inKernelPrefixPressureBridge struct {
	source agent.KVPrefixPressureSource
}

func newInKernelPrefixPressureBridge(source agent.KVPrefixPressureSource) *inKernelPrefixPressureBridge {
	if source == nil {
		return nil
	}
	return &inKernelPrefixPressureBridge{source: source}
}

// nativePrefixStageBreakEvenPrefillNanos supplies the least non-zero recompute
// cost that makes the generic capacity planner prefer a DRAM stage over a bare
// eviction. Native complete-prefix owners deliberately refuse to evict their sole
// copy, so treating an absent measurement as free recompute creates an impossible
// move: ActionEvict skips StageSpan, then EvictDigest correctly returns zero.
//
// This is a capability floor, not a measured speed claim. A future measured
// PerTokenPrefillNanos value can replace it; until then, stage-before-evict is the
// only executable pressure-relief action and the radix tree's host byte budget
// remains the final admission gate.
func nativePrefixStageBreakEvenPrefillNanos(sizeBytes int64, tokens int) int64 {
	if sizeBytes <= 0 || tokens <= 0 {
		return 0
	}
	profile, ok := cachemeta.DefaultTierProfiles()[cachemeta.TierDRAM]
	if !ok || profile.BandwidthMBPerSec <= 0 {
		return 0
	}
	const maxInt64 = int64(^uint64(0) >> 1)
	if sizeBytes > maxInt64/1000 {
		return maxInt64 / int64(tokens)
	}
	transferNanos := sizeBytes * 1000 / profile.BandwidthMBPerSec
	if transferNanos > maxInt64-profile.ReadLatencyNanos {
		return maxInt64 / int64(tokens)
	}
	stageNanos := profile.ReadLatencyNanos + transferNanos
	perToken := stageNanos/int64(tokens) + 1 // planner comparison is strict: stage < recompute.
	if perToken <= 0 {
		return 1
	}
	return perToken
}

func (b *inKernelPrefixPressureBridge) PressuredCandidates() (int64, []gateway.KVPressureCandidate) {
	if b == nil || b.source == nil {
		return 0, nil
	}
	resident, source := b.source.KVPrefixPressuredCandidates()
	out := make([]gateway.KVPressureCandidate, 0, len(source))
	for _, candidate := range source {
		out = append(out, gateway.KVPressureCandidate{
			SpanDigest: candidate.SpanDigest,
			From:       0,
			N:          candidate.Tokens,
			ModelID:    candidate.ModelID,
			SizeBytes:  candidate.SizeBytes,
			Tokens:     candidate.Tokens,
			PerTokenPrefillNanos: nativePrefixStageBreakEvenPrefillNanos(
				candidate.SizeBytes,
				candidate.Tokens,
			),
		})
	}
	return resident, out
}

func (b *inKernelPrefixPressureBridge) Len() int { return 0 }

func (b *inKernelPrefixPressureBridge) Prefill([]int) []float32 { return nil }

func (b *inKernelPrefixPressureBridge) Evict(int, int) int { return 0 }

func (b *inKernelPrefixPressureBridge) EvictDigest(digest string, _, _ int) int {
	if b == nil || b.source == nil {
		return 0
	}
	return b.source.EvictHotKVPrefix(digest)
}

func (b *inKernelPrefixPressureBridge) ModelID() string {
	if b == nil || b.source == nil {
		return ""
	}
	_, candidates := b.source.KVPrefixPressuredCandidates()
	if len(candidates) == 0 {
		return ""
	}
	return candidates[0].ModelID
}

func (b *inKernelPrefixPressureBridge) StageSpan(ctx context.Context, digest string, from, n int) (abi.KVResidency, error) {
	return b.StageSpanTo(ctx, digest, from, n, string(cachemeta.TierDRAM))
}

func (b *inKernelPrefixPressureBridge) StageSpanTo(ctx context.Context, digest string, _, _ int, tier string) (abi.KVResidency, error) {
	if tier != string(cachemeta.TierDRAM) {
		return abi.KVResidency{
			Outcome: abi.KVResidencyMiss,
			Digest:  digest,
			Reason:  "native prefix backend owns only the host DRAM L2",
		}, nil
	}
	if b == nil || b.source == nil {
		return abi.KVResidency{Outcome: abi.KVResidencyMiss, Digest: digest, Reason: "native prefix source absent"}, nil
	}
	return lowerNativePrefixResidency(b.source.StageKVPrefixToHost(ctx, digest)), nil
}

func (b *inKernelPrefixPressureBridge) RestoreSpan(ctx context.Context, digest string) (abi.KVResidency, error) {
	if b == nil || b.source == nil {
		return abi.KVResidency{Outcome: abi.KVResidencyMiss, Digest: digest, Reason: "native prefix source absent"}, nil
	}
	return lowerNativePrefixResidency(b.source.RestoreKVPrefixFromHost(ctx, digest)), nil
}

func lowerNativePrefixResidency(in agent.KVPrefixTransfer) abi.KVResidency {
	outcome := abi.KVResidencyUnknown
	switch in.Outcome {
	case "ok":
		outcome = abi.KVResidencyOK
	case "miss":
		outcome = abi.KVResidencyMiss
	case "fault":
		outcome = abi.KVResidencyFault
	}
	return abi.KVResidency{
		Outcome:    outcome,
		Digest:     in.SpanDigest,
		Positions:  in.Positions,
		BytesMoved: in.BytesMoved,
		Reason:     in.Reason,
	}
}

// wireKVPressureRelief is the serve-host installer for the #1073 post-decode capacity sweep — the
// LIVE, non-test call site of gateway.Server.SetKVPressureRelief (#1094). It builds the host
// sweeper closure over the live device backend + an engine.CapacityAdapter (the supplied
// abi.KVBackend that owns the bytes + a fresh CacheEventRecorder, so each demote folds into the
// fak_engine_cache_* stream the gateway already scrapes) and installs (provider, sweeper) on the
// server. The gateway gates the edge on FAK_INKERNEL_KVMMU and on BOTH seams being non-nil, so
// this installer can be called unconditionally — with a nil provider the edge stays inert
// (fail-open), byte-identical to today.
//
// A nil backend/provider remains an inert no-op for CPU-only and passthrough
// serves. For a native device serve with FAK_INKERNEL_RADIX_HOST_L2_BYTES set,
// serve_stages wires the in-kernel bridge here and the existing post-turn sweep
// stages complete PrefixSnapshot payloads before releasing their hot owners.
func wireKVPressureRelief(srv *gateway.Server, backend compute.Backend, kv abi.KVBackend, provider gateway.KVPressureCandidateProvider) {
	if srv == nil {
		return
	}
	adapter := &engine.CapacityAdapter{KV: kv, Recorder: engine.NewCacheEventRecorder()}
	srv.SetKVPressureRelief(provider, newCapacityPressureSweeper(backend, adapter))
}

// newCapacityPressureSweeper builds the host sweeper closure the gateway drives after a served
// decode turn (#1073). It closes over the live device backend and an engine.CapacityAdapter (the
// KVBackend that owns the bytes + the CacheEventRecorder), lowers the gateway's candidates, and
// runs engine.RunCapacityPressureSweep at the gateway-supplied high-water target — so a hot span
// is DEMOTED to the colder tier (StageSpan then Evict) instead of dropped. The typed result is
// projected back to the gateway's minimal KVPressureRelief. A nil backend or adapter yields a
// closure that always reports an empty (Known=false) relief — fail-open, matching the sweep.
func newCapacityPressureSweeper(backend compute.Backend, adapter *engine.CapacityAdapter) gateway.KVPressureSweeper {
	return func(ctx context.Context, residentBytes int64, target float64, cands []gateway.KVPressureCandidate) gateway.KVPressureRelief {
		if backend == nil || adapter == nil || len(cands) == 0 {
			return gateway.KVPressureRelief{}
		}
		res, err := engine.RunCapacityPressureSweep(ctx, engine.CapacityPressureSweep{
			Backend:        backend,
			Adapter:        adapter,
			ResidentBytes:  residentBytes,
			TargetPressure: target,
			Candidates:     lowerPressureCandidates(backend, residentBytes, cands),
		})
		if err != nil {
			// A sweep error (e.g. a nil adapter slipping through) is fail-open: report no relief
			// rather than failing the served turn the demote was meant to help.
			return gateway.KVPressureRelief{}
		}
		return gateway.KVPressureRelief{
			Known:          res.Known,
			AppliedMoves:   res.AppliedMoves,
			Faults:         res.Faults,
			ReclaimedBytes: res.ReclaimedBytes,
			FinalPressure:  res.FinalPressure,
		}
	}
}

// lowerPressureCandidates translates the gateway's wire-neutral KVPressureCandidate list into the
// engine's CapacityPressureCandidate (a cachemeta.PlacementRequest carrying the retain-vs-evict
// economics + an engine.PlacementMove carrying the span's executable identity).
//
// #1468 (Phase-2 child of #1463): the placement request is built against the box that ACTUALLY
// exists rather than cachemeta.DefaultTierProfiles' representative order-of-magnitude placeholders.
// The tier ladder comes from cachemeta.ProbedTierProfiles (already shipped, MLCACHE4 / #988) sized
// from the live HBM/DRAM/disk capacity probes, and the per-tier fullness comes from the already-
// shipped live pressure probes (DeviceHBMPressure, HostDRAMPressure, the disk free-space probe,
// MLCACHE1/MLCACHE2) instead of an empty (zero-value) cachemeta.TierPressure. Both folds are
// fail-open PER TIER: a probe that reports known=false (no GPU, an unsupported host-memory
// platform, an unprobeable disk path) leaves that one tier at ProbedTierProfiles'/the representative
// default rather than dragging every tier back to the placeholder ladder.
func lowerPressureCandidates(backend compute.Backend, residentBytes int64, cands []gateway.KVPressureCandidate) []engine.CapacityPressureCandidate {
	if len(cands) == 0 {
		return nil
	}
	profiles := probedTierProfilesForHost(backend, residentBytes)
	pressure := liveTierPressure(backend, residentBytes)

	out := make([]engine.CapacityPressureCandidate, 0, len(cands))
	for _, c := range cands {
		lc := cachemeta.NewLifecycle(cachemeta.TierHBM, 0).MarkResident(profiles, 0)
		req := cachemeta.PlacementRequest{
			Lifecycle:            lc,
			SizeBytes:            c.SizeBytes,
			Tokens:               int64(c.Tokens),
			Profiles:             profiles,
			Pressure:             pressure,
			Policy:               cachemeta.LifecyclePolicy{DemoteOnExpiry: true},
			PerTokenPrefillNanos: c.PerTokenPrefillNanos,
		}
		out = append(out, engine.CapacityPressureCandidate{
			Request: req,
			Move: engine.PlacementMove{
				SpanDigest:   c.SpanDigest,
				From:         c.From,
				N:            c.N,
				ModelID:      c.ModelID,
				TokenizerID:  c.TokenizerID,
				PositionMode: cachemeta.PositionPrefixAligned,
				Owner:        "kv-pressure-sweep",
			},
			ReclaimBytes: c.SizeBytes,
		})
	}
	return out
}

// kvPressureSpillPath is the filesystem the disk-tier capacity probe reads free space from when
// lowering a candidate's tier ladder. There is no dedicated spill-directory config yet (pure
// wiring per #1468's fence — no new policy), so this uses the same generic scratch filesystem
// cmd/fak already probes for other host-capacity purposes (os.TempDir()).
func kvPressureSpillPath() string { return os.TempDir() }

// probedTierProfilesForHost builds the tier ladder cachemeta plans candidates against from the box
// THIS process can prove it has: HBM sized from backend's device-memory probe (dropped entirely
// when the backend cannot report one, e.g. cpu-ref — the no-GPU-box fix #1468 calls out), DRAM
// sized from the host's real physical memory, disk sized from the scratch filesystem's real
// free space, and the far tiers (NUMA-far / CXL) entering the ladder only when the NUMA-topology
// probe confirmed them (#1470 — before that probe existed they were unconditionally out). Each
// capacity reading is independently fail-open (cachemeta.ProbedTierProfiles keeps the
// representative default for an always-present tier whose probe reads non-positive/absent, and
// keeps an unconfirmed far tier out of the ladder).
func probedTierProfilesForHost(backend compute.Backend, residentBytes int64) map[cachemeta.ResidencyTier]cachemeta.TierProfile {
	probe := cachemeta.CapacityProbe{}
	if hbmTotal, _, ok := compute.DeviceMemoryInfo(backend); ok && hbmTotal > 0 {
		probe.HBMPresent = true
		probe.HBMBytes = hbmTotal
	}
	if dramTotal, _, ok := compute.HostSystemMemoryInfo(); ok && dramTotal > 0 {
		probe.DRAMBytes = dramTotal
	}
	if diskTotal, _, ok := compute.DiskInfo(kvPressureSpillPath()); ok && diskTotal > 0 {
		probe.DiskBytes = diskTotal
	}
	if farTotal, _, ok := compute.NUMAFarMemoryInfo(); ok && farTotal > 0 {
		probe.NUMAFarPresent = true
		probe.NUMAFarBytes = farTotal
	}
	if cxlTotal, _, ok := compute.CXLMemoryInfo(); ok && cxlTotal > 0 {
		probe.CXLPresent = true
		probe.CXLBytes = cxlTotal
	}
	// The peer-DRAM-over-RDMA rung (#4306/#5083) enters the ladder only when a neighbor has
	// registered a lendable region under an active lease. The roster lives ABOVE cachemeta so
	// the policy plane never imports the RDMA fabric HAL; a lapsed/reclaimed lease folds to
	// zero and drops the rung (fail-closed), same prove-it-or-drop-it rule as the far tiers.
	probe = applyPeerDRAMLenders(probe, defaultPeerDRAMRoster.snapshot(), time.Now().UnixMilli())
	return cachemeta.ProbedTierProfiles(probe)
}

// liveTierPressure assembles the per-tier fullness the planner's coldest-colder-with-room walk
// reads, from the same already-shipped live probes the engine's PlanPlacementForLocalLadder
// wire folds (capacity_pressure.go / capacity_dram.go / capacity_disk.go / capacity_far.go) — reused here rather than
// reinvented, since RunCapacityPressureSweep independently re-derives HBM pressure from Backend for
// its own high-water gate. A probe that reports known=false leaves that tier absent from the
// returned TierPressure, which cachemeta.TierPressure.HasRoom treats as "has room" (unknown != full)
// — the fail-open contract every capacity plank in this codebase honors.
func liveTierPressure(backend compute.Backend, residentBytes int64) cachemeta.TierPressure {
	pressure := cachemeta.TierPressure{}
	if p, _, known := engine.DeviceHBMPressure(backend, residentBytes); known {
		pressure[cachemeta.TierHBM] = p
	}
	if p, _, known := engine.HostDRAMPressure(residentBytes); known {
		pressure[cachemeta.TierDRAM] = p
	}
	if p, _, known := engine.DiskPressure(kvPressureSpillPath()); known {
		pressure[cachemeta.TierDisk] = p
	}
	// The far tiers (#1470) finish the per-tier ladder. residentBytes 0: fak keeps no
	// far-resident counter, and the topology probe always reports free when it reports
	// at all, so the resident-bytes fallback never engages; an unconfirmed tier stays
	// absent (unknown != full — the same fail-open contract as the three above).
	if p, _, known := engine.NUMAFarPressure(0); known {
		pressure[cachemeta.TierNUMAFar] = p
	}
	if p, _, known := engine.CXLPressure(0); known {
		pressure[cachemeta.TierCXL] = p
	}
	return pressure
}
