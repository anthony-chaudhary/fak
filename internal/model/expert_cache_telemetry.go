package model

import "github.com/anthony-chaudhary/fak/pkg/moecache"

// expert_cache_telemetry.go — the #1305 live-engine expert-cache telemetry contract, bound to the
// operator report. MoEResidencyReport (expert_residency_report.go) is the human/operator view, rich
// but internal; ExpertCacheTelemetry is the SHARED wire shape a private gateway can also import
// (fak-private imports fak/pkg/* only, never fak/internal/*). One fold, not a second channel: every
// number here is read from the SAME components' own accounting the report already gathers, and the
// metric NAMES match the benchmark receipt (internal/model/expert_cache_receipt.go) so a live series
// and a benchmark series join on one vocabulary.
//
// The load-bearing rule is the reason vocabulary. A ringless session, a window with no measured
// access, a synchronous backend, and an unasked regret replay each report UNKNOWN-with-reason -
// never a fabricated 0. The conversion lives here, on the internal side, so the shared package stays
// free of any dependency on internal/model: moecache does not import internal/model; this file
// imports moecache. One-way.

// Telemetry binds this report into the shared expert-cache telemetry contract. It is safe to call on
// any report, including the zero value: every metric whose source is absent becomes
// Unknown(reason) rather than a phantom 0.
func (rep MoEResidencyReport) Telemetry() moecache.ExpertCacheTelemetry {
	tel := moecache.New()
	tel.Shape = moecache.ModelShape{
		Experts:           rep.Shape.Experts,
		TopK:              rep.Shape.ExpertsPerToken,
		Layers:            rep.Shape.Layers,
		ActivatedFraction: rep.Shape.ActivatedFraction,
	}
	tel.Coverage = moecache.Coverage{
		ActivatedExperts: ringMetric(rep, func(s ExpertRingStats) int { return s.ActivatedExperts }),
		ActivatedCovered: ringMetric(rep, func(s ExpertRingStats) int { return s.ActivatedCovered }),
		Refusals:         ringMetric(rep, func(s ExpertRingStats) int { return s.Refusals }),
		Lookups:          ringMetric(rep, func(s ExpertRingStats) int { return s.Lookups }),
		AsyncOverlap:     ringAsyncOverlap(rep),
	}

	// The DRAM tier is present whenever the waterfall has a resident ring at all; a session without
	// one reports no tier rather than a zeroed one.
	if rep.Ring.Enabled {
		dram := tel.Tier(moecache.TierDRAM)
		*dram = ringTierTelemetry(rep)
	}

	// The checkpoint tier is the host/host-cache rung one below the ring (R5), present only when the
	// model actually has one. Its NVMe bytes are the backing-store reads #1305 asks for, named
	// consistently with the benchmark receipt's bytes_read_nvme.
	if rep.Checkpoint.Enabled {
		ck := tel.Tier(moecache.TierCheckpoint)
		*ck = checkpointTierTelemetry(rep)
	}

	return tel
}

// ringMetric reads a ring-derived count, or Unknown("no-ring") when this session has no ring. The
// reason is the closed moecache vocabulary so a consumer can branch on it.
func ringMetric[T int | int64 | float64](rep MoEResidencyReport, read func(ExpertRingStats) T) moecache.Metric[T] {
	if !rep.Ring.Enabled {
		return moecache.Unknown[T](moecache.ReasonNoRing)
	}
	return moecache.Known(read(rep.Ring))
}

// ringTierTelemetry fills the DRAM tier's #1305 axes from the ring's own ledger.
func ringTierTelemetry(rep MoEResidencyReport) moecache.TierTelemetry {
	ring := rep.Ring
	tt := moecache.TierTelemetry{
		Tier:          moecache.TierDRAM,
		CapacityBytes: moecache.Known(ring.BudgetBytes),
		ResidentBytes: moecache.Known(ring.ResidentBytes),
		ResidentCount: moecache.Known(ring.ResidentCount),
		EvictionCount: moecache.Known(int64(ring.Evictions)),
		// MissBytes is the bytes cold uploads moved - the streamed traffic the ladder exists to
		// reduce. NVMeBytes on this tier is deliberately left unknown: the ring's page-in bytes are
		// the miss bytes above, and re-labelling them as a separate NVMe read would double-count.
		MissBytes: moecache.Known(ring.PageInBytes),
		NVMeBytes: moecache.Unknown[int64](moecache.ReasonNotDerivable),
		// HitBytes: the ring keeps no per-outcome byte counter, so the byte-weighted rate has no
		// numerator on the live surface. Reporting it would mean inventing one.
		HitBytes:            moecache.Unknown[int64](moecache.ReasonNotDerivable),
		ByteWeightedHitRate: moecache.Unknown[float64](moecache.ReasonNotDerivable),
		// PrefetchPrecision: Prefetched counts weights staged ahead; there is no counter of how many
		// were activated versus evicted-before-demand, so a real precision is not derivable here.
		PrefetchPrecision: moecache.Unknown[float64](moecache.ReasonNotDerivable),
	}

	// HitRate is Hits/(Hits+PageIns), the same rate the report derives. A zero denominator means
	// nothing was accessed, so it is Unknown("no-measured-access"), never 0%.
	if served := ring.Hits + ring.PageIns; served > 0 {
		tt.HitRate = moecache.Known(float64(ring.Hits) / float64(served))
	} else {
		tt.HitRate = moecache.Unknown[float64](moecache.ReasonNoMeasuredAccess)
	}

	// BeladyRegret: the offline oracle comparison (#4233) is present only when a regret replay was
	// asked for AND the window produced a replayable trace. GoodDecisionRatio=1 is zero regret.
	if rep.Regret != nil {
		tt.BeladyRegret = moecache.Known(1 - rep.Regret.LRUGoodDecisionRatio)
	} else {
		tt.BeladyRegret = moecache.Unknown[float64](moecache.ReasonNoReplayTrace)
	}
	return tt
}

// checkpointTierTelemetry fills the checkpoint tier from R5's host-IO ledger. HitRate is
// Hits/Reads over the tier's own fault counters; NVMeBytes is the backing-store read the tier paid.
func checkpointTierTelemetry(rep MoEResidencyReport) moecache.TierTelemetry {
	ck := rep.Checkpoint
	tt := moecache.TierTelemetry{
		Tier:          moecache.TierCheckpoint,
		CapacityBytes: moecache.Known(ck.BudgetBytes),
		ResidentBytes: moecache.Known(ck.ResidentBytes),
		ResidentCount: moecache.Known(ck.ResidentCount),
		EvictionCount: moecache.Known(int64(ck.Evictions)),
		// On the checkpoint tier a fault's bytes ARE the backing-store read, so MissBytes and
		// NVMeBytes are the same measured quantity under the two #1305 names.
		MissBytes: moecache.Known(ck.BytesRead),
		NVMeBytes: moecache.Known(ck.BytesRead),
		// HitBytes and the byte-weighted rate are not derivable from the host-cache counters.
		HitBytes:            moecache.Unknown[int64](moecache.ReasonNotDerivable),
		ByteWeightedHitRate: moecache.Unknown[float64](moecache.ReasonNotDerivable),
		PrefetchPrecision:   moecache.Unknown[float64](moecache.ReasonNotDerivable),
		// Eviction regret is a ring-policy gauge; the checkpoint tier has no replay.
		BeladyRegret: moecache.Unknown[float64](moecache.ReasonNotDerivable),
	}
	if ck.Reads > 0 {
		tt.HitRate = moecache.Known(float64(ck.Hits) / float64(ck.Reads))
	} else {
		tt.HitRate = moecache.Unknown[float64](moecache.ReasonNoMeasuredAccess)
	}
	return tt
}

// ringAsyncOverlap reads the #5627 overlap meter. It is Unknown("backend-not-async") when nothing was
// fenced, which is the honest reading for a synchronous backend - not "no overlap was achieved".
func ringAsyncOverlap(rep MoEResidencyReport) moecache.Metric[float64] {
	if !rep.Ring.Enabled {
		return moecache.Unknown[float64](moecache.ReasonNoRing)
	}
	if rep.Ring.AsyncOverlapped+rep.Ring.AsyncWaited <= 0 {
		return moecache.Unknown[float64](moecache.ReasonBackendNotAsync)
	}
	return moecache.Known(rep.Ring.AsyncOverlapFraction())
}
