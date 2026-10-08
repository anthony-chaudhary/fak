package radixkv

import (
	"time"
)

// Stats is a snapshot of the cache's structural state for reporting.
type Stats struct {
	CPUCacheBytes              int64        `json:"cpu_cache_bytes"`
	MaxCPUCacheBytes           int64        `json:"max_cpu_cache_bytes"`
	CPUCacheBypasses           int64        `json:"cpu_cache_bypasses"`
	CPUCacheLastBypass         string       `json:"cpu_cache_last_bypass,omitempty"`
	Tokens                     int          // total cached tokens (Σ edge lengths) — the LRU-budget metric
	PrefixTokens               int          // Σ node.plen over nodes holding a kv — TRUE resident KV positions
	Nodes                      int          // non-root nodes
	ProtectedTokens            int          `json:"protected_tokens"` // Σ edge tokens on the root path of every leased node
	EvictableTokens            int          `json:"evictable_tokens"` // Tokens - ProtectedTokens
	SnapshotBytes              int64        `json:"snapshot_bytes"`
	MaxSnapshotBytes           int64        `json:"max_snapshot_bytes"`
	HostSnapshotBytes          int64        `json:"host_snapshot_bytes"`
	MaxHostSnapshotBytes       int64        `json:"max_host_snapshot_bytes"`
	DeviceSnapshotBytes        int64        `json:"device_snapshot_bytes"`
	DeviceSnapshotHostBytes    int64        `json:"device_snapshot_host_bytes"`
	DeviceSnapshotTokens       int          `json:"device_snapshot_tokens"`
	L1Hits                     int          `json:"l1_hits"`
	L1Misses                   int          `json:"l1_misses"`
	L1Faults                   int          `json:"l1_faults"`
	L1HitTokens                int          `json:"l1_hit_tokens"`
	L2Hits                     int          `json:"l2_hits"`
	L2Misses                   int          `json:"l2_misses"`
	L2Faults                   int          `json:"l2_faults"`
	L2HitTokens                int          `json:"l2_hit_tokens"`
	L2StageBytes               int64        `json:"l2_stage_bytes"`
	L2RestoreBytes             int64        `json:"l2_restore_bytes"`
	L2Evictions                int          `json:"l2_evictions"`
	L3Enabled                  bool         `json:"l3_enabled"`
	L3ReferencedBytes          int64        `json:"l3_referenced_bytes"`
	L3Hits                     int          `json:"l3_hits"`
	L3Misses                   int          `json:"l3_misses"`
	L3Faults                   int          `json:"l3_faults"`
	L3HitTokens                int          `json:"l3_hit_tokens"`
	L3StageBytes               int64        `json:"l3_stage_bytes"`
	L3RestoreBytes             int64        `json:"l3_restore_bytes"`
	L3StageNanos               int64        `json:"l3_stage_nanos"`
	L3RestoreNanos             int64        `json:"l3_restore_nanos"`
	L3StageFaults              int          `json:"l3_stage_faults"`
	L3RestoreFaults            int          `json:"l3_restore_faults"`
	L3Breaker                  BreakerStats `json:"l3_breaker"`
	L3BreakerState             BreakerState `json:"l3_breaker_state,omitempty"`
	L3BreakerConsecutiveFaults int          `json:"l3_breaker_consecutive_faults,omitempty"`
	L3BreakerTotalFaults       int          `json:"l3_breaker_total_faults,omitempty"`
	L3BreakerOpenSkips         int          `json:"l3_breaker_open_skips,omitempty"`
	L3BreakerProbesAttempted   int          `json:"l3_breaker_probes_attempted,omitempty"`
	L3BreakerProbeRecoveries   int          `json:"l3_breaker_probe_recoveries,omitempty"`
	L3BreakerProbeFailures     int          `json:"l3_breaker_probe_failures,omitempty"`
	L3BreakerOpenedAt          time.Time    `json:"l3_breaker_opened_at,omitempty"`
	Leaves                     int          // leaf nodes
	MaxDepthTokens             int          // longest cached prefix
	Evictions                  int          // LRU leaf evictions performed
	CostEvictions              int          // cost-aware leaf evictions performed
	PageEvictions              int          `json:"page_evictions,omitempty"` // page-aware leaf evictions performed
	PolicyEvictions            int          // EvictNode calls
	Splits                     int          // edge splits performed
	MaxTokens                  int          // configured LRU budget (0 = unbounded)
	EvictionPolicy             string
	ReuseHits                  int

	LastEvictPolicy       string
	LastEvictCandidates   int
	LastEvictLocked       int
	LastEvictVictimCost   float64
	LastEvictVictimHits   int
	LastEvictVictimTokens int
	LastEvictVictimPrefix int

	// Evict→reuse thrash detector (#3393, thrash.go). ThrashReuses counts budget
	// evictions whose exact key was re-demanded within thrashWindow logical ticks —
	// premature evictions, the live "budget/policy too aggressive" signal. Gaps are in
	// logical-clock ticks (deterministic, wall-clock-free).
	ThrashReuses   int    // premature evictions detected (evict→reuse within the window)
	ThrashTokens   int    // Σ victim edge tokens over those thrashes — wasted recompute
	ThrashGapTotal uint64 // Σ evict→reuse gap over those thrashes
	ThrashGapLast  uint64 // gap of the most recent thrash
	ThrashGapMax   uint64 // largest gap counted
	ThrashTracked  int    // just-evicted keys currently probe-able (≤ thrashCap)

	// Bounded-eviction plane (#3387, plan.go). BoundedEvictions counts victims staged
	// (and detached) by PlanBoundedEviction — a subset of Evictions, which counts BOTH
	// planes. LedgerConfirmed lags BoundedEvictions by exactly LedgerPending: staged
	// lifetime == BoundedEvictions, and every staged victim is confirmed exactly once.
	BoundedEvictions      int // victims evicted via the ratio-capped bounded plane
	LedgerConfirmed       int // confirmed deletes settled by ConfirmEvictions
	LedgerConfirmedTokens int // Σ victim edge tokens over confirmed deletes
	LedgerPending         int // staged-unconfirmed ledger entries (≤ evictionLedgerCap)

	// Prefix-cache lifecycle events (#5804, lookupobs.go). These count DEMAND, not
	// residency: they are monotonic for the tree's lifetime and eviction never rolls them
	// back. The three lookup buckets partition every probe exactly —
	// Lookups == LookupHits + LookupMissCold + LookupMissDivergent — which is what makes a
	// 0% hit rate diagnosable: all-cold is warmup, all-divergent is genuine non-overlap.
	Lookups             int // LookupNS calls (demand reuse probes; MatchLenNS is not counted)
	LookupHits          int // probes that matched at least one leading token
	LookupMissCold      int // misses where nothing could have matched (empty namespace / empty request)
	LookupMissDivergent int // misses against a populated namespace that shared no leading token
	Fills               int // leaves attached (demand Insert + prewarm WarmInsert)

	// Value/frequency admission (#9311, admission.go). Candidates count only inserts
	// that would displace token or snapshot residency; no-pressure fills stay off the
	// decision hot path. The fixed sketch and bounded pending journal make overhead
	// explicit, while rejects/recoveries show pollution refused and later heat earned.
	AdmissionEnabled        bool `json:"admission_enabled"`
	AdmissionSketchCells    int  `json:"admission_sketch_cells"`
	AdmissionObservations   int  `json:"admission_observations"`
	AdmissionCandidates     int  `json:"admission_candidates"`
	AdmissionAdmitted       int  `json:"admission_admitted"`
	AdmissionRejected       int  `json:"admission_rejected"`
	AdmissionRejectedTokens int  `json:"admission_rejected_tokens"` // candidate token writes bypassed
	// Candidate value-axis footprint bypassed: exact resident bytes for snapshots and
	// radixkv's existing full-prefix token proxy for regular KV nodes.
	AdmissionRejectedBytes      int64  `json:"admission_rejected_bytes"`
	AdmissionTelemetryFallbacks int    `json:"admission_telemetry_fallbacks"`
	AdmissionHotProtected       int    `json:"admission_hot_protected"`
	AdmissionRecoveries         int    `json:"admission_recoveries"`
	AdmissionJournalPending     int    `json:"admission_journal_pending"`
	AdmissionJournalDropped     int    `json:"admission_journal_dropped"`
	AdmissionRecoveryGapLast    uint64 `json:"admission_recovery_gap_last"`
	AdmissionRecoveryGapMax     uint64 `json:"admission_recovery_gap_max"`
	LastAdmissionFrequency      int    `json:"last_admission_frequency"`
	LastAdmissionReason         string `json:"last_admission_reason"`
}

// Stats walks the tree and returns its current shape.
//
// Note the deliberate Tokens vs PrefixTokens split. The LRU budget (MaxTokens) bounds
// Tokens = Σ edge lengths — SGLang's per-segment accounting, the apples-to-apples
// hit-rate metric this package exists to compare. But each node stores the FULL prefix
// root→node KV (length plen), so the TRUE resident KV footprint is PrefixTokens = Σ plen,
// which can exceed the configured budget by an O(prefix-depth) factor on a deep/narrow
// tree (a single N-token chain holds N·(N+1)/2 positions while Tokens reports only N).
// PrefixTokens makes that gap measurable instead of silent (see TestBudgetVsTrueKVFootprint).
func (t *Tree) Stats() Stats {
	var bStats BreakerStats
	if t.remoteBreaker != nil {
		bStats = t.remoteBreaker.Stats()
	} else {
		bStats = BreakerStats{
			State:          BreakerClosed,
			FaultThreshold: DefaultBreakerFaultThreshold,
			Cooldown:       DefaultBreakerCooldown,
		}
	}
	s := Stats{
		CPUCacheBytes:              t.cpuCacheBytes(),
		MaxCPUCacheBytes:           t.maxCPUCacheBytes,
		CPUCacheBypasses:           t.cpuCacheBypasses,
		CPUCacheLastBypass:         t.cpuCacheLastBypass,
		Evictions:                  t.evictions,
		CostEvictions:              t.costEvictions,
		PageEvictions:              t.pageEvictions,
		PolicyEvictions:            t.policyEvictions,
		Splits:                     t.splits,
		MaxTokens:                  t.effectiveMaxTokens(),
		SnapshotBytes:              t.snapshotBytes,
		MaxSnapshotBytes:           t.maxSnapshotBytes,
		HostSnapshotBytes:          t.hostSnapshotBytes,
		MaxHostSnapshotBytes:       t.maxHostSnapshotBytes,
		L1Hits:                     t.l1Hits,
		L1Misses:                   t.l1Misses,
		L1Faults:                   t.l1Faults,
		L1HitTokens:                t.l1HitTokens,
		L2Hits:                     t.l2Hits,
		L2Misses:                   t.l2Misses,
		L2Faults:                   t.l2Faults,
		L2HitTokens:                t.l2HitTokens,
		L2StageBytes:               t.l2StageBytes,
		L2RestoreBytes:             t.l2RestoreBytes,
		L2Evictions:                t.l2Evictions,
		L3Enabled:                  t.RemoteSnapshotEnabled(),
		L3ReferencedBytes:          t.remoteSnapshotBytes,
		L3Hits:                     t.l3Hits,
		L3Misses:                   t.l3Misses,
		L3Faults:                   t.l3Faults,
		L3HitTokens:                t.l3HitTokens,
		L3StageBytes:               t.l3StageBytes,
		L3RestoreBytes:             t.l3RestoreBytes,
		L3StageNanos:               t.l3StageNanos,
		L3RestoreNanos:             t.l3RestoreNanos,
		L3StageFaults:              t.l3StageFaults,
		L3RestoreFaults:            t.l3RestoreFaults,
		L3Breaker:                  bStats,
		L3BreakerState:             bStats.State,
		L3BreakerConsecutiveFaults: bStats.ConsecutiveFaults,
		L3BreakerTotalFaults:       bStats.TotalFaults,
		L3BreakerOpenSkips:         bStats.OpenSkips,
		L3BreakerProbesAttempted:   bStats.ProbesAttempted,
		L3BreakerProbeRecoveries:   bStats.ProbeRecoveries,
		L3BreakerProbeFailures:     bStats.ProbeFailures,
		L3BreakerOpenedAt:          bStats.OpenedAt,
		EvictionPolicy:             t.evictionStrategy().Name(),
		LastEvictPolicy:            t.lastEvictPolicyName(),
		LastEvictCandidates:        t.lastEvictCandidates,
		LastEvictLocked:            t.lastEvictLocked,
		LastEvictVictimCost:        t.lastEvictVictimCost,
		LastEvictVictimHits:        t.lastEvictVictimHits,
		LastEvictVictimTokens:      t.lastEvictVictimTokens,
		LastEvictVictimPrefix:      t.lastEvictVictimPrefix,
		ThrashReuses:               t.thrashReuses,
		ThrashTokens:               t.thrashTokens,
		ThrashGapTotal:             t.thrashGapTotal,
		ThrashGapLast:              t.thrashGapLast,
		ThrashGapMax:               t.thrashGapMax,
		ThrashTracked:              len(t.thrashIndex),
		BoundedEvictions:           t.boundedEvictions,
		LedgerConfirmed:            t.ledgerConfirmed,
		LedgerConfirmedTokens:      t.ledgerConfirmedTokens,
		LedgerPending:              len(t.evictLedger),
		Lookups:                    t.lookups,
		LookupHits:                 t.lookupHits,
		LookupMissCold:             t.lookupMissCold,
		LookupMissDivergent:        t.lookupMissDivergent,
		Fills:                      t.fills,
		AdmissionEnabled:           t.admissionEnabled,
		AdmissionSketchCells: func() int {
			if t.admissionSketch == nil {
				return 0
			}
			return t.admissionSketch.Cells()
		}(),
		AdmissionObservations:       t.admissionObservations,
		AdmissionCandidates:         t.admissionCandidates,
		AdmissionAdmitted:           t.admissionAdmitted,
		AdmissionRejected:           t.admissionRejected,
		AdmissionRejectedTokens:     t.admissionRejectedTokens,
		AdmissionRejectedBytes:      t.admissionRejectedBytes,
		AdmissionTelemetryFallbacks: t.admissionTelemetryFallbacks,
		AdmissionHotProtected:       t.admissionHotProtected,
		AdmissionRecoveries:         t.admissionRecoveries,
		AdmissionJournalPending:     t.admissionJournalPending(),
		AdmissionJournalDropped:     t.admissionJournalDropped,
		AdmissionRecoveryGapLast:    t.admissionRecoveryGapLast,
		AdmissionRecoveryGapMax:     t.admissionRecoveryGapMax,
		LastAdmissionFrequency:      t.lastAdmissionFrequency,
		LastAdmissionReason:         t.lastAdmissionReason,
	}
	var visit func(n *node) bool
	visit = func(n *node) bool {
		leased := n.refs > 0
		if n.parent != nil { // skip every namespace root (parent==nil); count real nodes once
			s.Nodes++
			s.Tokens += len(n.key)
			if n.kv != nil {
				s.PrefixTokens += n.plen
			}
			if n.snapshot != nil {
				host, device := n.snapshot.ResidencyBytes()
				s.DeviceSnapshotHostBytes += host
				s.DeviceSnapshotBytes += device
				if device > 0 {
					s.DeviceSnapshotTokens += n.plen
				}
			}
			if n.plen > s.MaxDepthTokens {
				s.MaxDepthTokens = n.plen
			}
			s.ReuseHits += n.hits
			if len(n.children) == 0 {
				s.Leaves++
			}
		}
		for _, c := range n.children {
			if visit(c) {
				leased = true
			}
		}
		if leased && n.parent != nil {
			s.ProtectedTokens += len(n.key)
		}
		return leased
	}
	t.forEachRoot(func(r *node) { visit(r) }) // one Stats snapshot across every namespace's subtree
	s.EvictableTokens = s.Tokens - s.ProtectedTokens
	return s
}
