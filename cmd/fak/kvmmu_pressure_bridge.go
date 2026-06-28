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
// SCOPE / FENCES. This only WIRES the existing executor onto the existing seam — no new eviction
// policy, no flag handling of its own (the gateway owns FAK_INKERNEL_KVMMU; off, the edge is a
// byte-identical no-op). The production provider is left nil at the serve.go call site (the real
// in-kernel resident-span enumerator is the fenced follow-on #1074 / #987, over the persistent
// kvmmu.Segment{From,Len,KV} ledger that InKernelPlanner does not surface yet); what this ships
// is the LIVE sweeper closure + the lowering, so the executor has a real, non-test serve-path
// caller for the first time.

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// wireKVPressureRelief is the serve-host installer for the #1073 post-decode capacity sweep — the
// LIVE, non-test call site of gateway.Server.SetKVPressureRelief (#1094). It builds the host
// sweeper closure over the live device backend + an engine.CapacityAdapter (the supplied
// abi.KVBackend that owns the bytes + a fresh CacheEventRecorder, so each demote folds into the
// fak_engine_cache_* stream the gateway already scrapes) and installs (provider, sweeper) on the
// server. The gateway gates the edge on FAK_INKERNEL_KVMMU and on BOTH seams being non-nil, so
// this installer can be called unconditionally — with a nil provider the edge stays inert
// (fail-open), byte-identical to today.
//
// WHAT IS WIRED vs WHAT IS STILL MISSING (#1094, honest fence). The sweeper half is real: the
// closure runs the genuine engine.RunCapacityPressureSweep executor (witnessed by
// newCapacityPressureSweeper's tests + the gateway keystone test). The PROVIDER half — the live
// resident-span enumerator over kvmmu.Segment{From,Len,KV} that yields the candidate list — is
// passed in. In production serve.go passes nil for it (so the installed sweep is inert: a nil
// provider is a clean no-op), because the durable resident-span ledger does not exist yet: the
// InKernelPlanner builds a kvmmu.Context EPHEMERALLY per eviction (internal/agent's EvictKVSpan /
// ElideKVSpans rebuild a fresh bridge over a fresh session and discard it) and keeps cross-turn
// state only as the radixkv prefix-reuse tree, which is keyed by prefix node, not as enumerable
// per-span retain-vs-evict candidates. Building that enumerator is the fenced follow-on #1074 /
// #987. So this installer makes the executor ONE provider-construction away from live: the day the
// span enumerator lands, serve.go passes it here instead of nil and the post-decode sweep fires on
// real residency with no further wiring. A nil backend or KV leaves the sweeper a no-op too, so a
// CPU-only / passthrough serve installs nothing live.
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
			Candidates:     lowerPressureCandidates(cands),
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
// economics + an engine.PlacementMove carrying the span's executable identity). The placement
// request is built resident-on-HBM so the planner, under device pressure, demotes the span one
// tier colder rather than evicting it — the exact shape the engine sweep tests exercise.
func lowerPressureCandidates(cands []gateway.KVPressureCandidate) []engine.CapacityPressureCandidate {
	out := make([]engine.CapacityPressureCandidate, 0, len(cands))
	for _, c := range cands {
		profiles := cachemeta.DefaultTierProfiles()
		lc := cachemeta.NewLifecycle(cachemeta.TierHBM, 0).MarkResident(profiles, 0)
		req := cachemeta.PlacementRequest{
			Lifecycle:            lc,
			SizeBytes:            c.SizeBytes,
			Tokens:               int64(c.Tokens),
			Profiles:             profiles,
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
