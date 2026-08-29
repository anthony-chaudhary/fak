package cacheobs

// tiers.go — the EXPLICIT-TIER axis of cache observation (#6422).
//
// Borrow (study-repo, LMCache @a976ce09dd98): `docs/design/usage_telemetry/l2_metrics.md:20-109`
// specifies that L2 usage is reported as structured PER-TIER facts — request / hit / miss /
// bytes / latency / backend dimensions — rather than inferred from aggregate cache traffic,
// and `lmcache/usage_telemetry/l2_usage.py` keeps the L1 and L2 collection paths separate so
// each tier exposes its own truth.
//
// The fak seam this closes: every counter in this package is TIER-BLIND. The depth axis
// (prompt / cacheable / eligible / reused) and the provenance axis (bysource.go) both fold a
// multi-tier reality into one aggregate, so a managed provider cache, a shared KV store, and
// the in-process prefix tree can be blended into a single flattering hit-rate — and a tier
// that is not implemented at all is indistinguishable from a tier that is implemented and
// idle. Both read as "0".
//
// This axis fixes exactly that:
//
//   - TierAccess is the typed record of ONE cache access: which tier, which operation, what
//     outcome, how many bytes moved, how long it took, and the coarse backend class behind it.
//   - Every dimension is a CLOSED vocabulary or a number. The record deliberately carries no
//     string field at all, so a cache key, a prompt fragment, a tenant/account identifier, or
//     a provider response body cannot ride this axis into an exported report even by mistake
//     (TestTierReportCarriesOnlyClosedVocabularyStrings pins it).
//   - Bytes and latency are only counted when the tap actually WITNESSED them (BytesKnown /
//     LatencyKnown). An uninstrumented dimension books zero SIZED accesses instead of a
//     zero-byte access, so "we never measured it" cannot masquerade as "it moved no bytes".
//   - TierSnapshot reports EVERY tier in the vocabulary, each carrying its status: a tier with
//     no collector in this build renders "unsupported", never a silent zero row that reads as
//     an implemented tier earning nothing. An observation always beats the declaration —
//     evidence of a live tap promotes the tier to "supported".
//
// The existing taps feed it: observeAttributed (behind Observe / ObserveSplit /
// ObservePreempted / ObserveLabeled — i.e. every in-kernel turn the planner books) emits one
// local-prefix read per turn inside the SAME critical section as the aggregate counters, so
// the tier totals can never desync from the aggregate they decompose. TierSnapshot().Total
// .Requests == Snapshot().Turns is the reconciliation witness.
//
// Route: inspire (clean-room Go; both Apache-2.0). Source cited, no bytes vendored.

import "time"

// CacheTier names WHICH cache served (or failed to serve) an access. The vocabulary is
// closed and ordered by distance from the compute: an out-of-range tier is ignored at
// observe time rather than bucketed to a catch-all, so a summed tier report can only
// ever under-count — it can never mis-attribute one tier's value to another.
type CacheTier int

const (
	// TierLocalPrefix is the in-process KV-prefix cache (the RadixAttention tree): resident
	// in this process's memory, no hop, the tier every in-kernel turn consults first.
	TierLocalPrefix CacheTier = iota
	// TierSharedStore is a shared KV store outside this process — the L2/L3 disaggregated
	// tier a fleet reads across the fabric. Distinct from TierLocalPrefix because its hits
	// cost a transfer, which is the whole reason its latency must be booked separately.
	TierSharedStore
	// TierProviderManaged is an upstream PROVIDER's managed prompt cache: value fak only
	// OBSERVES (relayed cache_read), never authors. Kept apart from the two fak-owned tiers
	// so a provider's cache can never be summed into fak's own witnessed reuse.
	TierProviderManaged
	// numCacheTiers bounds the closed vocabulary. Not a tier.
	numCacheTiers
)

// String renders the label spelling of a tier. An out-of-range value renders "unknown" (it
// is never booked, so it can only appear if a caller stringifies a raw int).
func (t CacheTier) String() string {
	switch t {
	case TierLocalPrefix:
		return "local_prefix"
	case TierSharedStore:
		return "shared_store"
	case TierProviderManaged:
		return "provider_managed"
	default:
		return "unknown"
	}
}

func (t CacheTier) valid() bool { return t >= 0 && t < numCacheTiers }

// AllTiers returns the closed tier vocabulary in report order. A report always carries a row
// for every entry — that is what makes an unsupported tier explicit rather than absent.
//
//enumlint:exempt numCacheTiers is the exclusive bound sentinel, not a reportable tier.
func AllTiers() []CacheTier {
	return []CacheTier{TierLocalPrefix, TierSharedStore, TierProviderManaged}
}

// TierOp names WHICH operation the access performed against the tier. Read and write are
// kept apart because their costs and their outcomes mean different things: a read's miss is
// lost reuse, a write's miss is an admission refusal.
type TierOp int

const (
	// OpRead is a lookup/retrieve against the tier.
	OpRead TierOp = iota
	// OpWrite is a store/admit into the tier.
	OpWrite
	// numTierOps bounds the closed vocabulary. Not an operation.
	numTierOps
)

// String renders the label spelling of an operation; "unknown" out of range.
func (op TierOp) String() string {
	switch op {
	case OpRead:
		return "read"
	case OpWrite:
		return "write"
	default:
		return "unknown"
	}
}

func (op TierOp) valid() bool { return op >= 0 && op < numTierOps }

// AllOps returns the closed operation vocabulary in report order.
//
//enumlint:exempt numTierOps is the exclusive bound sentinel, not a reportable operation.
func AllOps() []TierOp { return []TierOp{OpRead, OpWrite} }

// TierOutcome names HOW the access ended. For a read, hit/miss is the cache verdict; for a
// write, hit means the tier ADMITTED the entry and miss means admission refused it. Error is
// neither — a tier that failed returned no verdict at all, so it is counted apart and left
// out of the hit ratio's denominator (a flapping backend must not read as a cache miss).
type TierOutcome int

const (
	// OutcomeHit is a read served from the tier, or a write the tier admitted.
	OutcomeHit TierOutcome = iota
	// OutcomeMiss is a read the tier could not serve, or a write admission refused.
	OutcomeMiss
	// OutcomeError is an access the tier failed to complete (no cache verdict).
	OutcomeError
	// numTierOutcomes bounds the closed vocabulary. Not an outcome.
	numTierOutcomes
)

// String renders the label spelling of an outcome; "unknown" out of range.
func (o TierOutcome) String() string {
	switch o {
	case OutcomeHit:
		return "hit"
	case OutcomeMiss:
		return "miss"
	case OutcomeError:
		return "error"
	default:
		return "unknown"
	}
}

func (o TierOutcome) valid() bool { return o >= 0 && o < numTierOutcomes }

// BackendClass is the COARSE storage class behind a tier — deliberately coarse: it says what
// kind of medium the access paid for (and therefore what latency to expect), never which
// vendor, host, bucket, or endpoint served it. Naming a concrete backend would leak
// deployment topology into a telemetry record; a class does not.
type BackendClass int

const (
	// BackendMemory is process- or host-resident RAM.
	BackendMemory BackendClass = iota
	// BackendDisk is local persistent storage.
	BackendDisk
	// BackendRemote is anything reached over the network.
	BackendRemote
	// numBackendClasses bounds the closed vocabulary. Not a backend.
	numBackendClasses
)

// String renders the label spelling of a backend class; "unknown" out of range.
func (b BackendClass) String() string {
	switch b {
	case BackendMemory:
		return "memory"
	case BackendDisk:
		return "disk"
	case BackendRemote:
		return "remote"
	default:
		return "unknown"
	}
}

func (b BackendClass) valid() bool { return b >= 0 && b < numBackendClasses }

// AllBackends returns the closed backend-class vocabulary in report order.
//
//enumlint:exempt numBackendClasses is the exclusive bound sentinel, not a backend.
func AllBackends() []BackendClass { return []BackendClass{BackendMemory, BackendDisk, BackendRemote} }

// TierStatus says whether a tier is even COLLECTED in this build. It is the difference
// between the two readings a bare zero row conflates: "implemented, earned nothing" and
// "not implemented, so of course nothing".
type TierStatus int

const (
	// TierStatusUndeclared is the default: nobody declared this tier and nothing has been
	// observed on it. Its zeros mean "unknown", not "idle" and not "absent".
	TierStatusUndeclared TierStatus = iota
	// TierStatusSupported means a collector feeds this tier in this build, so zeros on it
	// are genuine idle-cache zeros.
	TierStatusSupported
	// TierStatusUnsupported means no collector exists for this tier in this build. Its
	// zeros are STRUCTURAL — they are not cache misses and must never be read as lost value.
	TierStatusUnsupported
	// numTierStatuses bounds the closed vocabulary. Not a status.
	numTierStatuses
)

// String renders the label spelling of a status; "unknown" out of range.
func (s TierStatus) String() string {
	switch s {
	case TierStatusUndeclared:
		return "undeclared"
	case TierStatusSupported:
		return "supported"
	case TierStatusUnsupported:
		return "unsupported"
	default:
		return "unknown"
	}
}

func (s TierStatus) valid() bool { return s >= 0 && s < numTierStatuses }

// TierAccess is the typed record of one cache access against one tier (#6422). Every field is
// a closed-vocabulary enum, a number, or the boolean that says whether that number was
// measured at all — there is no free-form string, so no cache key, prompt fragment, tenant
// identifier, or provider payload can ride it.
//
// The two *Known flags carry the honesty this axis exists for: a tap that does not measure
// payload size leaves BytesKnown false and its access books into Requests but NOT into Bytes
// or SizedAccesses, so the report shows "N accesses, 0 of them sized" rather than "N accesses
// moving 0 bytes". A zero Bytes with BytesKnown true is a real measured zero (a miss moves
// nothing) and is counted as such.
type TierAccess struct {
	// Tier is which cache was consulted; Op what was asked of it; Outcome how it ended.
	Tier    CacheTier
	Op      TierOp
	Outcome TierOutcome
	// Backend is the coarse storage class the tier is backed by for this access — the same
	// tier can be memory-backed on one box and remote on another.
	Backend BackendClass
	// Bytes is the payload this access moved, counted only when BytesKnown. A negative
	// value is treated as no witness at all (a negative byte count is a bug, and booking it
	// as a measured zero would fabricate a measurement that never happened).
	Bytes      int64
	BytesKnown bool
	// Latency is the wall time this access took, counted only when LatencyKnown. Negative
	// values are dropped like negative Bytes.
	Latency      time.Duration
	LatencyKnown bool
}

// valid reports whether every dimension of the access is inside its closed vocabulary. An
// invalid access is ignored WHOLE — never partially booked and never bucketed to a catch-all
// row, so a caller passing a raw int cannot silently mis-attribute another tier's value.
func (a TierAccess) valid() bool {
	return a.Tier.valid() && a.Op.valid() && a.Outcome.valid() && a.Backend.valid()
}

// tierKey is the row key of the per-tier breakdown: the three dimensions that are cut
// (tier x operation x backend class). The outcome is not a key — it folds into the
// hit/miss/error counters of the row.
type tierKey struct {
	tier    CacheTier
	op      TierOp
	backend BackendClass
}

// tierTotals is one row's raw counters. Kept unexported and additive; every derived rate is
// computed at snapshot time from these, so a rate can never drift from its base counters.
type tierTotals struct {
	requests      uint64
	hits          uint64
	misses        uint64
	errors        uint64
	bytes         uint64
	sizedAccesses uint64
	latencyNanos  uint64
	timedAccesses uint64
}

// add folds another row's counters into this one (saturating, like every counter here).
func (t tierTotals) add(u tierTotals) tierTotals {
	return tierTotals{
		requests:      saturatingAddU64(t.requests, u.requests),
		hits:          saturatingAddU64(t.hits, u.hits),
		misses:        saturatingAddU64(t.misses, u.misses),
		errors:        saturatingAddU64(t.errors, u.errors),
		bytes:         saturatingAddU64(t.bytes, u.bytes),
		sizedAccesses: saturatingAddU64(t.sizedAccesses, u.sizedAccesses),
		latencyNanos:  saturatingAddU64(t.latencyNanos, u.latencyNanos),
		timedAccesses: saturatingAddU64(t.timedAccesses, u.timedAccesses),
	}
}

// TierCounters is the reported counter block of a tier row (and of the grand total). The
// additive fields are the truth; the three ratios below them are pure functions of those
// fields, computed at snapshot time.
type TierCounters struct {
	Requests uint64 `json:"requests"`
	Hits     uint64 `json:"hits"`
	Misses   uint64 `json:"misses"`
	// Errors is accesses the tier failed to complete. Deliberately outside the hit ratio's
	// denominator: an error returned no cache verdict, so counting it as a miss would blame
	// the cache for a backend outage.
	Errors uint64 `json:"errors"`
	// Bytes is the payload moved by the accesses that carried a byte witness.
	Bytes uint64 `json:"bytes"`
	// SizedAccesses is how many accesses carried that witness. Bytes without it is
	// unreadable: 0 bytes over 0 sized accesses means "never measured", while 0 bytes over
	// 100 sized accesses means "measured, and nothing moved".
	SizedAccesses uint64 `json:"sized_accesses"`
	// LatencyNanos is the summed wall time of the accesses that carried a latency witness,
	// and TimedAccesses is how many did — the same explicit-denominator discipline.
	LatencyNanos  uint64 `json:"latency_ns"`
	TimedAccesses uint64 `json:"timed_accesses"`
	// HitRatio is Hits / (Hits + Misses) — the tier's own hit rate, over the accesses that
	// actually returned a verdict. 0 when no verdict has been observed.
	HitRatio float64 `json:"hit_ratio"`
	// MeanBytes is Bytes / SizedAccesses and MeanLatencyNanos is LatencyNanos /
	// TimedAccesses — averaged over the WITNESSED denominator, never over Requests, so an
	// uninstrumented tap cannot dilute a measured mean toward zero. 0 when nothing was
	// witnessed.
	MeanBytes        float64 `json:"mean_bytes"`
	MeanLatencyNanos float64 `json:"mean_latency_ns"`
}

// counters derives the reported block from the raw row.
func (t tierTotals) counters() TierCounters {
	c := TierCounters{
		Requests:      t.requests,
		Hits:          t.hits,
		Misses:        t.misses,
		Errors:        t.errors,
		Bytes:         t.bytes,
		SizedAccesses: t.sizedAccesses,
		LatencyNanos:  t.latencyNanos,
		TimedAccesses: t.timedAccesses,
	}
	if verdicts := saturatingAddU64(t.hits, t.misses); verdicts > 0 {
		c.HitRatio = float64(t.hits) / float64(verdicts)
	}
	if t.sizedAccesses > 0 {
		c.MeanBytes = float64(t.bytes) / float64(t.sizedAccesses)
	}
	if t.timedAccesses > 0 {
		c.MeanLatencyNanos = float64(t.latencyNanos) / float64(t.timedAccesses)
	}
	return c
}

// TierOpStats is one (operation, backend class) row inside a tier — the sub-cut that shows
// whether a tier's latency came from its reads or its writes, and from which medium.
type TierOpStats struct {
	Op      string `json:"op"`
	Backend string `json:"backend"`
	TierCounters
}

// TierStats is one tier's reported row: its status, its per-(op, backend) rows, and its own
// totals (the sum of those rows).
type TierStats struct {
	Tier   string `json:"tier"`
	Status string `json:"status"`
	// Ops is the (operation, backend class) breakdown, in vocabulary order. Absent when the
	// tier has seen no access — the Status field, not an empty row list, is what says why.
	Ops []TierOpStats `json:"ops,omitempty"`
	TierCounters
}

// TierReport is the JSON/metrics surface of the tier axis: one row per tier in the closed
// vocabulary (ALWAYS all of them, so an unsupported tier is visible rather than missing) plus
// the grand total across tiers. Total is exactly the sum of the per-tier rows, which is what
// lets a reader reconcile a tier-split view against the tier-blind aggregate this package has
// always reported.
type TierReport struct {
	Tiers []TierStats  `json:"tiers"`
	Total TierCounters `json:"total"`
	// RejectedTierAccesses is observer-owned admission telemetry, not a tier row.
	RejectedTierAccesses uint64 `json:"rejected_tier_accesses"`
}

// defaultTierStatus is the tier support this BUILD ships, and it is a statement about the
// binary rather than about any one observer, so every New() observer starts from it. The
// in-process KV-prefix tree is live (observeAttributed feeds it on every turn); no shared
// KV store and no provider-managed cache reports into this package yet, so both are declared
// unsupported — their zeros are structural, exactly as the planner's by-source tap documents
// its structurally-zero external-transfer bucket. An actual observation on either promotes
// it to supported at snapshot time (evidence beats declaration), so the day a tap lands the
// report tells the truth without this table being edited first.
//
//enumlint:exempt numCacheTiers is the exclusive bound sentinel, not a status-bearing tier.
func defaultTierStatus() map[CacheTier]TierStatus {
	return map[CacheTier]TierStatus{
		TierLocalPrefix:     TierStatusSupported,
		TierSharedStore:     TierStatusUnsupported,
		TierProviderManaged: TierStatusUnsupported,
	}
}

// DeclareTier records whether a tier is collected in this build (#6422). A deployment that
// wires a shared KV store declares it supported; one that compiles it out declares it
// unsupported, so its zero row cannot be read as lost cache value. An out-of-vocabulary tier
// or status is ignored. Safe for concurrent use, like the rest of the observer.
func (o *Observer) DeclareTier(tier CacheTier, status TierStatus) {
	if o == nil || !tier.valid() || !status.valid() {
		return
	}
	o.mu.Lock()
	if o.tierStatus == nil {
		o.tierStatus = make(map[CacheTier]TierStatus, int(numCacheTiers))
	}
	o.tierStatus[tier] = status
	o.mu.Unlock()
}

// ObserveTier books one cache access against its explicit tier (#6422). An access whose
// tier, operation, outcome, or backend class falls outside the closed vocabulary is ignored
// WHOLE — never bucketed to a catch-all — so the report can under-count but never
// mis-attribute. Bytes and latency are folded only when the access says it measured them.
// Safe for concurrent use.
func (o *Observer) ObserveTier(a TierAccess) {
	if o == nil {
		return
	}
	o.mu.Lock()
	if !a.valid() {
		o.rejectedTierAccesses = saturatingAddU64(o.rejectedTierAccesses, 1)
		o.mu.Unlock()
		return
	}
	o.observeTierLocked(a)
	o.mu.Unlock()
}

// observeTierLocked is the accumulation core. Caller holds o.mu and has already validated
// the access — observeAttributed calls it from INSIDE its own critical section so a turn's
// tier record and its aggregate counters are booked atomically together and can never desync.
func (o *Observer) observeTierLocked(a TierAccess) {
	if o.tierRows == nil {
		o.tierRows = make(map[tierKey]*tierTotals)
	}
	k := tierKey{tier: a.Tier, op: a.Op, backend: a.Backend}
	row := o.tierRows[k]
	if row == nil {
		row = &tierTotals{}
		o.tierRows[k] = row
	}
	row.requests = saturatingAddU64(row.requests, 1)
	//enumlint:exempt numTierOutcomes is the exclusive bound sentinel rejected by TierAccess.valid.
	switch a.Outcome {
	case OutcomeHit:
		row.hits = saturatingAddU64(row.hits, 1)
	case OutcomeMiss:
		row.misses = saturatingAddU64(row.misses, 1)
	case OutcomeError:
		row.errors = saturatingAddU64(row.errors, 1)
	}
	if a.BytesKnown && a.Bytes >= 0 {
		row.bytes = saturatingAddU64(row.bytes, uint64(a.Bytes))
		row.sizedAccesses = saturatingAddU64(row.sizedAccesses, 1)
	}
	if a.LatencyKnown && a.Latency >= 0 {
		row.latencyNanos = saturatingAddU64(row.latencyNanos, uint64(a.Latency))
		row.timedAccesses = saturatingAddU64(row.timedAccesses, 1)
	}
}

// TierSnapshot returns the per-tier report (#6422): one row for EVERY tier in the closed
// vocabulary — a tier with no collector renders "unsupported" rather than vanishing or
// showing a bare zero row — each with its (operation, backend class) breakdown, its own
// totals, and the grand total across tiers.
//
// A tier that has actually been observed reports "supported" whatever the declaration says:
// a live access is proof the collector exists, and a stale declaration must not be able to
// label real traffic as structurally impossible. Nil-safe like Snapshot: a nil observer
// reports the full vocabulary with everything undeclared and zero, never a phantom split.
func (o *Observer) TierSnapshot() TierReport {
	rows := make(map[tierKey]tierTotals)
	status := make(map[CacheTier]TierStatus, int(numCacheTiers))
	var rejected uint64
	if o != nil {
		o.mu.Lock()
		for k, v := range o.tierRows {
			rows[k] = *v
		}
		for t, s := range o.tierStatus {
			status[t] = s
		}
		rejected = o.rejectedTierAccesses
		o.mu.Unlock()
	}
	rep := TierReport{Tiers: make([]TierStats, 0, int(numCacheTiers)), RejectedTierAccesses: rejected}
	var grand tierTotals
	for _, tier := range AllTiers() {
		ts := TierStats{Tier: tier.String(), Status: status[tier].String()}
		var perTier tierTotals
		for _, op := range AllOps() {
			for _, backend := range AllBackends() {
				row, ok := rows[tierKey{tier: tier, op: op, backend: backend}]
				if !ok {
					continue
				}
				ts.Ops = append(ts.Ops, TierOpStats{Op: op.String(), Backend: backend.String(), TierCounters: row.counters()})
				perTier = perTier.add(row)
			}
		}
		if perTier.requests > 0 {
			ts.Status = TierStatusSupported.String() // evidence beats declaration
		}
		ts.TierCounters = perTier.counters()
		grand = grand.add(perTier)
		rep.Tiers = append(rep.Tiers, ts)
	}
	rep.Total = grand.counters()
	return rep
}

// Tier returns the report row for one tier and whether the vocabulary contains it, so a
// caller can read a single tier's totals without scanning TierReport.Tiers by string.
func (r TierReport) Tier(tier CacheTier) (TierStats, bool) {
	if !tier.valid() {
		return TierStats{}, false
	}
	name := tier.String()
	for _, ts := range r.Tiers {
		if ts.Tier == name {
			return ts, true
		}
	}
	return TierStats{}, false
}

// The tier-axis fields of the declarative metric registry (metricspec.go). They cover the
// GRAND totals across tiers, which is what a second collection path has to reproduce for the
// tier split to be trustworthy: if the per-tier accumulation and a flat reduction of the same
// access stream disagree on any of these, the split is losing or inventing accesses.
const (
	FieldTierRequests     = "tier_requests"
	FieldTierHits         = "tier_hits"
	FieldTierMisses       = "tier_misses"
	FieldTierErrors       = "tier_errors"
	FieldTierBytes        = "tier_bytes"
	FieldTierLatencyNanos = "tier_latency_ns"
)

// TierFields is the tier-axis counter schema TierSpecs must cover exactly once — the sibling
// of SnapshotFields for the depth axis.
func TierFields() []string {
	return []string{
		FieldTierRequests,
		FieldTierHits,
		FieldTierMisses,
		FieldTierErrors,
		FieldTierBytes,
		FieldTierLatencyNanos,
	}
}

// TierSpecs is the declarative registry for the tier-axis totals: each field declared once,
// every one reduced by summation over EventTierAccess events. It is the single definition
// that SpecFold (a literal reduction of the access stream) and TierObserverFold (the
// imperative per-tier accumulator read back through TierSnapshot) both reproduce, so
// Reconcile can prove the two paths agree field-for-field.
func TierSpecs() []MetricSpec {
	return []MetricSpec{
		{Event: EventTierAccess, Field: FieldTierRequests, Extract: func(Event) float64 { return 1 }, Reduce: sumSamples},
		{Event: EventTierAccess, Field: FieldTierHits, Extract: outcomeCount(OutcomeHit), Reduce: sumSamples},
		{Event: EventTierAccess, Field: FieldTierMisses, Extract: outcomeCount(OutcomeMiss), Reduce: sumSamples},
		{Event: EventTierAccess, Field: FieldTierErrors, Extract: outcomeCount(OutcomeError), Reduce: sumSamples},
		{Event: EventTierAccess, Field: FieldTierBytes, Extract: witnessedBytes, Reduce: sumSamples},
		{Event: EventTierAccess, Field: FieldTierLatencyNanos, Extract: witnessedLatency, Reduce: sumSamples},
	}
}

// outcomeCount builds the Extract that emits 1 for an access with the given outcome, 0
// otherwise — the literal definition of "count the hits" as a map-reduce row.
func outcomeCount(want TierOutcome) func(Event) float64 {
	return func(e Event) float64 {
		if e.Access.Outcome == want {
			return 1
		}
		return 0
	}
}

// witnessedBytes / witnessedLatency emit a sample ONLY for an access that measured the
// dimension, mirroring the observer's rule exactly: an unmeasured dimension contributes
// nothing rather than a zero, so the two paths agree on what "0 bytes" means.
func witnessedBytes(e Event) float64 {
	if !e.Access.BytesKnown || e.Access.Bytes < 0 {
		return 0
	}
	return float64(e.Access.Bytes)
}

func witnessedLatency(e Event) float64 {
	if !e.Access.LatencyKnown || e.Access.Latency < 0 {
		return 0
	}
	return float64(e.Access.Latency)
}

// TierObserverFold is the SECOND collection path for the tier registry: it routes each access
// event through the real per-tier accumulator (ObserveTier) and reads the totals back off
// TierSnapshot. It reproduces the registry's fields by the per-tier map the production code
// actually uses, so reconciling it against SpecFold proves the tier split neither loses nor
// invents accesses. A spec whose field the report does not expose reduces to 0 here and
// surfaces as a divergence rather than as silence.
func TierObserverFold(specs []MetricSpec, events []Event) Report {
	o := New()
	for _, e := range events {
		if e.Kind != EventTierAccess {
			continue
		}
		o.ObserveTier(e.Access)
	}
	total := o.TierSnapshot().Total
	byField := map[string]float64{
		FieldTierRequests:     float64(total.Requests),
		FieldTierHits:         float64(total.Hits),
		FieldTierMisses:       float64(total.Misses),
		FieldTierErrors:       float64(total.Errors),
		FieldTierBytes:        float64(total.Bytes),
		FieldTierLatencyNanos: float64(total.LatencyNanos),
	}
	out := make(Report, len(specs))
	for _, spec := range specs {
		out[spec.Field] = byField[spec.Field] // absent -> 0, surfaced as a divergence
	}
	return out
}
