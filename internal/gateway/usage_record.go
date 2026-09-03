package gateway

// usage_record.go — the ONE stable per-request provider-usage record (issue
// #10670). The gateway_inference_turn JSON event answers "what did this turn
// cost" but pairs it with per-delta token_count streams that need offline FIFO
// reconstruction; nothing at run time told an agent whether a request actually
// hit the provider prompt cache, or whether that agreed with fak's own native
// warm state. This file supplies the three missing surfaces, all counts and
// ratios only (never prompts or content):
//
//  1. UsageRecord — emitted exactly once per COMPLETED request from the shared
//     per-turn chokepoint (logInferenceTurnWithContextEvent), carrying the
//     request ordinal (per session/trace), the provider's token axes, and the
//     cached ratio with the canonical alignment threshold. It rides the JSON
//     log stream as the gateway_usage_record sibling event and is retained in
//     a bounded window so it can be queried live.
//  2. /v1/fak/usage/cache-alignment — the agent-visible read: "last N requests:
//     X% cache-aligned", threshold visible in the response.
//  3. The join — each record reconciled against the native warm-state receipt
//     already flowing through observeVCacheTurn's governor journal for the same
//     trace: aligned / misaligned / warm_unknown. The provider side is the
//     OBSERVED cache_read the provider relayed; the native side is the
//     DECISION-class governor posture for the family. Divergence ("fak expected
//     this prefix warm" vs "the provider re-prefilled it") is the expensive
//     misalignment the record makes visible per request instead of days later.
//
// Law A2: purely observational. Nothing in the request path reads any of it.

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/vcachegov"
)

// UsageRecordEventName is the JSON log event name the per-request record rides
// under — the sibling of gateway_inference_turn on the same --log stream.
const UsageRecordEventName = "gateway_usage_record"

// usageAlignmentSchema is the versioned contract of the query endpoint.
const usageAlignmentSchema = "fak.gateway.usage-alignment.v1"

// CacheAlignedThreshold is the canonical cached-ratio above which a completed
// request counts as cache-aligned: at least 80% of the resident prompt served
// from the provider's cache. Exposed in every query response so a reader never
// has to guess where the bit came from.
const CacheAlignedThreshold = 0.80

// usageRecordCap bounds the retained record window (drop-oldest) so a 24/7
// gateway stays flat in memory; the query surface reads "last N" out of it.
const usageRecordCap = 1024

// usageAlignmentDefaultN is the default "last N" the query endpoint reports.
const usageAlignmentDefaultN = 50

// Join agreement classes for the per-request native-warm-state reconciliation.
const (
	usageJoinAligned     = "aligned"
	usageJoinMisaligned  = "misaligned"
	usageJoinWarmUnknown = "warm_unknown"
)

// UsageRecord is the stable per-request usage row. Token axes are the
// provider's own counters, normalized the same way gateway_inference_turn
// normalizes them (CachedPromptTokens/UncachedPromptTokens) so every provider
// shape reads identically.
type UsageRecord struct {
	// Ordinal is the 1-based request ordinal within the session/trace — the
	// pairing key that makes "which request was this" trivial instead of a
	// FIFO reconstruction over per-delta records.
	Ordinal uint64 `json:"request_ordinal"`
	// Family is the normalized session/trace id (the join key against the
	// native vcache/governor planes; "unknown" when the wire carried none).
	Family string `json:"family"`
	// Wire names the served surface (anthropic_messages, openai_chat_completions, …).
	Wire   string `json:"wire"`
	Stream bool   `json:"stream"`
	// Effort is the reasoning-effort tier the request ran at. NOT threaded on
	// this surface today: no wire hands the turn chokepoint an effort value
	// (the only effort on the gateway is the deepseek-anthropic profile
	// control, which never reaches the turn log), so it stays honestly absent
	// (omitempty) rather than fabricated. Threading it is a follow-on that
	// must touch the wire call sites.
	Effort string `json:"effort,omitempty"`

	// InputTokens is the provider's own prompt/input counter, verbatim.
	InputTokens int64 `json:"input_tokens"`
	// OutputTokens is the provider's completion counter, verbatim.
	OutputTokens int64 `json:"output_tokens"`
	// ReasoningTokens is the provider-reported reasoning subcounter when it
	// reports one (0 otherwise; completion already includes it).
	ReasoningTokens int64 `json:"reasoning_tokens"`
	// CachedTokens is the provider-normalized prompt-cache hit (OBSERVED).
	CachedTokens int64 `json:"cached_tokens"`
	// CacheWriteTokens is the cache-creation axis — the write the turn paid
	// to seed the provider cache (OBSERVED).
	CacheWriteTokens int64 `json:"cache_write_tokens"`

	// CacheRatio is CachedTokens over the full resident prompt
	// (uncached + cached, provider-normalized); 0 when the prompt was empty.
	CacheRatio float64 `json:"cache_ratio"`
	// CacheAligned reports CacheRatio >= CacheAlignedThreshold at emission.
	CacheAligned bool `json:"cache_aligned"`

	// UnixMillis is the same turn timestamp the native vcache plane recorded
	// for this request, so record and native row join exactly.
	UnixMillis int64 `json:"unix_millis"`
}

// usageJoinView is the per-request reconciliation against fak's native
// warm-state receipt.
type usageJoinView struct {
	// Class is the agreement verdict: aligned, misaligned, or warm_unknown.
	Class string `json:"class"`
	// NativeWarmExpected reports whether the family's native governor posture
	// holds the prefix should be warm (ride_natural / heartbeat_pin).
	NativeWarmExpected bool `json:"native_warm_expected"`
	// ProviderServedFromCache is the OBSERVED side: the provider relayed a
	// non-zero cache_read for the request.
	ProviderServedFromCache bool `json:"provider_served_from_cache"`
	// NativeGovernorDecision names the native posture the class was
	// reconciled against (empty when no receipt exists for the family).
	NativeGovernorDecision string `json:"native_governor_decision,omitempty"`
	// Provenance labels both sides of the join with their trust class, in the
	// same vocabulary the vcache planes use.
	Provenance map[string]string `json:"provenance"`
}

// usageRecordView is one query-response row: the stable record plus its join.
type usageRecordView struct {
	UsageRecord
	Join usageJoinView `json:"join"`
}

// usageAlignmentSummary aggregates the queried window.
type usageAlignmentSummary struct {
	Count             int     `json:"count"`
	CacheAligned      int     `json:"cache_aligned"`
	CacheAlignedRatio float64 `json:"cache_aligned_ratio"`
	JoinAligned       int     `json:"join_aligned"`
	JoinMisaligned    int     `json:"join_misaligned"`
	JoinWarmUnknown   int     `json:"join_warm_unknown"`
}

// usageAlignmentWindow describes the queried slice of the retained window.
type usageAlignmentWindow struct {
	Requests    int  `json:"requests"`
	Capped      bool `json:"capped"`
	RetainedCap int  `json:"retained_cap"`
}

// usageAlignmentReport is the /v1/fak/usage/cache-alignment response.
type usageAlignmentReport struct {
	Schema           string                `json:"schema"`
	AlignedThreshold float64               `json:"aligned_threshold"`
	Window           usageAlignmentWindow  `json:"window"`
	Summary          usageAlignmentSummary `json:"summary"`
	Records          []usageRecordView     `json:"records"`
}

// recordUsageTurn builds and retains the per-request usage record. It must be
// called after observeVCacheTurn for the same turn so the native plane is
// populated first; unixMillis is the SAME timestamp observeVCacheTurn recorded,
// which is what makes the record join the native row exactly. Returns ok=false
// on a nil metrics object (the caller then emits no log event).
func (m *gatewayMetrics) recordUsageTurn(traceID, wire string, stream bool, usage agent.Usage, unixMillis int64) (UsageRecord, bool) {
	if m == nil {
		return UsageRecord{}, false
	}
	family := strings.TrimSpace(traceID)
	if family == "" {
		family = "unknown"
	}
	cached := int64(clampNonNeg(usage.CachedPromptTokens()))
	uncached := int64(clampNonNeg(usage.UncachedPromptTokens()))
	rec := UsageRecord{
		Family:           family,
		Wire:             wire,
		Stream:           stream,
		InputTokens:      int64(clampNonNeg(usage.PromptTokens)),
		OutputTokens:     int64(clampNonNeg(usage.CompletionTokens)),
		ReasoningTokens:  int64(clampNonNeg(usageReasoningTokens(usage))),
		CachedTokens:     cached,
		CacheWriteTokens: int64(clampNonNeg(usage.CacheCreationInputTokens)),
		UnixMillis:       unixMillis,
	}
	rec.CacheRatio = usageCacheRatio(cached, uncached)
	rec.CacheAligned = rec.CacheRatio >= CacheAlignedThreshold

	m.usageMu.Lock()
	if m.usageOrdinals == nil {
		m.usageOrdinals = map[string]uint64{}
	}
	m.usageOrdinals[family]++
	rec.Ordinal = m.usageOrdinals[family]
	m.usageRecords = append(m.usageRecords, rec)
	if len(m.usageRecords) > usageRecordCap {
		// Drop-oldest onto a fresh backing array so the dropped head can be
		// GC'd instead of pinned by the cap (the same shape the vcache window keeps).
		trimmed := make([]UsageRecord, usageRecordCap)
		copy(trimmed, m.usageRecords[len(m.usageRecords)-usageRecordCap:])
		m.usageRecords = trimmed
		m.usageRecordsDropped = true
	}
	m.usageMu.Unlock()
	return rec, true
}

// usageRecordsSnapshot returns a copy of the retained records (oldest first)
// and whether drop-oldest has trimmed the window.
func (m *gatewayMetrics) usageRecordsSnapshot() ([]UsageRecord, bool) {
	if m == nil {
		return nil, false
	}
	m.usageMu.Lock()
	defer m.usageMu.Unlock()
	out := make([]UsageRecord, len(m.usageRecords))
	copy(out, m.usageRecords)
	return out, m.usageRecordsDropped
}

// usageReasoningTokens peels the provider-reported reasoning subcounter off the
// completion details without assuming the block is present.
func usageReasoningTokens(usage agent.Usage) int {
	if usage.CompletionTokensDetails == nil {
		return 0
	}
	return usage.CompletionTokensDetails.ReasoningTokens
}

// usageCacheRatio is the cached share of the full resident prompt. The
// normalized pair guarantees uncached + cached == the resident prompt on every
// provider shape, so one formula reads the same everywhere; an empty prompt
// (no evidence of any resident prefix) ratios 0 and is never aligned.
func usageCacheRatio(cached, uncached int64) float64 {
	denom := cached + uncached
	if denom <= 0 {
		return 0
	}
	return float64(cached) / float64(denom)
}

// logEvent renders the record as the gateway_usage_record sibling event — the
// record's own fields at the top level plus the event name, so a log reader
// pairs it with gateway_inference_turn by trace_id + request_ordinal.
func (r UsageRecord) logEvent() map[string]any {
	b, err := json.Marshal(r)
	if err != nil {
		return nil
	}
	ev := map[string]any{}
	if err := json.Unmarshal(b, &ev); err != nil {
		return nil
	}
	ev["event"] = UsageRecordEventName
	return ev
}

// emitUsageRecordEvent writes the sibling event to the same JSON log sink as
// gateway_inference_turn, carrying the same trace/principal attribution.
func (s *Server) emitUsageRecordEvent(traceID string, rec UsageRecord) {
	if s == nil || s.logf == nil {
		return
	}
	ev := rec.logEvent()
	if ev == nil {
		return
	}
	if trace := strings.TrimSpace(traceID); trace != "" {
		ev["trace_id"] = trace
		if owner, ok := s.traceOwnerOf(trace); ok && owner != "" {
			ev["principal"] = owner
		}
	}
	b, err := json.Marshal(ev)
	if err != nil {
		return
	}
	s.logf("%s", b)
}

// handleFakUsageCacheAlignment serves the agent-visible cache-alignment read:
// the last N completed requests, the share that hit the provider prompt cache
// at the canonical threshold, and each request's reconciliation against fak's
// native warm-state receipt. Read-only, GET, no prompt or content fields.
func (s *Server) handleFakUsageCacheAlignment(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	writeJSON(w, http.StatusOK, s.usageAlignmentReport(usageAlignmentNFromRequest(r)))
}

// usageAlignmentNFromRequest parses the ?n= "last N" bound, clamped to the
// retained window so a caller can never ask the report to fabricate history it
// does not hold; an absent or non-positive n takes the default.
func usageAlignmentNFromRequest(r *http.Request) int {
	n := usageAlignmentDefaultN
	if r != nil {
		if parsed, err := strconv.Atoi(strings.TrimSpace(r.URL.Query().Get("n"))); err == nil && parsed > 0 {
			n = parsed
		}
	}
	if n > usageRecordCap {
		n = usageRecordCap
	}
	return n
}

// usageAlignmentReport folds the last n retained records into the report. The
// join reconciles each record against the family's LATEST native governor
// posture — the DECISION-class receipt the observe plane journals per turn —
// gated on the family having actually engaged the cache plane in the retained
// window (a family with no cache activity has no warm-state evidence to
// reconcile against and reports warm_unknown rather than a guess).
func (s *Server) usageAlignmentReport(n int) usageAlignmentReport {
	rep := usageAlignmentReport{
		Schema:           usageAlignmentSchema,
		AlignedThreshold: CacheAlignedThreshold,
		Window:           usageAlignmentWindow{RetainedCap: usageRecordCap},
	}
	if s == nil || s.metrics == nil {
		return rep
	}
	records, capped := s.metrics.usageRecordsSnapshot()
	rep.Window.Capped = capped
	if len(records) > n {
		records = records[len(records)-n:]
	}
	rep.Window.Requests = len(records)
	latestPosture := s.metrics.usageNativeWarmReceipts()
	familyActive := s.metrics.usageFamiliesWithCacheActivity()
	for _, rec := range records {
		view := usageRecordView{UsageRecord: rec}
		dec := latestPosture[rec.Family]
		if !familyActive[rec.Family] || dec == nil {
			view.Join = usageJoinView{
				Class:                   usageJoinWarmUnknown,
				ProviderServedFromCache: rec.CachedTokens > 0,
				Provenance:              usageJoinProvenance(),
			}
			if dec != nil {
				view.Join.NativeGovernorDecision = dec.Decision
			}
		} else {
			view.Join = usageJoinFor(rec, dec)
		}
		switch view.Join.Class {
		case usageJoinAligned:
			rep.Summary.JoinAligned++
		case usageJoinMisaligned:
			rep.Summary.JoinMisaligned++
		default:
			rep.Summary.JoinWarmUnknown++
		}
		if rec.CacheAligned {
			rep.Summary.CacheAligned++
		}
		rep.Records = append(rep.Records, view)
	}
	rep.Summary.Count = len(records)
	if rep.Summary.Count > 0 {
		rep.Summary.CacheAlignedRatio = float64(rep.Summary.CacheAligned) / float64(rep.Summary.Count)
	}
	return rep
}

// usageJoinProvenance is the provenance label set every join row carries, in
// the vcache planes' vocabulary: the provider side is relayed (OBSERVED), the
// native posture is fak's own verdict (DECISION).
func usageJoinProvenance() map[string]string {
	return map[string]string{
		"provider_cache":    "OBSERVED",
		"native_warm_state": "DECISION",
	}
}

// usageJoinFor reconciles one record against its family's native posture. The
// agreement rule mirrors the governor journal's own keep-bit (the posture the
// decision implies must match the cache activity witnessed): a warm posture is
// vindicated by a provider cache read and breached by a re-prefill; a lapse or
// never-warm posture is vindicated by silence and breached by either a read or
// (for the Law-D4 never-warm classes) a cache write.
func usageJoinFor(rec UsageRecord, dec *vcacheGovernorDecisionRecord) usageJoinView {
	view := usageJoinView{
		Class:                   usageAlignmentClass(rec, dec),
		ProviderServedFromCache: rec.CachedTokens > 0,
		NativeGovernorDecision:  dec.Decision,
		Provenance:              usageJoinProvenance(),
	}
	switch vcachegov.GovernorDecision(dec.Decision) {
	case vcachegov.DecisionRideNatural, vcachegov.DecisionHeartbeatPin:
		view.NativeWarmExpected = true
	}
	return view
}

// usageAlignmentClass is the pure agreement rule: the record's OBSERVED provider
// axes vs the family's DECISION-class native posture. No receipt = warm_unknown.
func usageAlignmentClass(rec UsageRecord, dec *vcacheGovernorDecisionRecord) string {
	if dec == nil {
		return usageJoinWarmUnknown
	}
	served := rec.CachedTokens > 0
	switch vcachegov.GovernorDecision(dec.Decision) {
	case vcachegov.DecisionRideNatural, vcachegov.DecisionHeartbeatPin:
		if served {
			return usageJoinAligned
		}
		return usageJoinMisaligned
	case vcachegov.DecisionNoCache, vcachegov.DecisionExplicitCache:
		if !served && rec.CacheWriteTokens == 0 {
			return usageJoinAligned
		}
		return usageJoinMisaligned
	default:
		// Lapse posture (lazy_rebuild / evict): the prefix was meant to go cold.
		if !served {
			return usageJoinAligned
		}
		return usageJoinMisaligned
	}
}

// usageNativeWarmReceipts indexes the retained governor journal by family,
// keeping each family's LATEST row (the journal is seq-ordered). Pointers into
// the returned copy are safe: recordsCopy hands back a fresh slice.
func (m *gatewayMetrics) usageNativeWarmReceipts() map[string]*vcacheGovernorDecisionRecord {
	if m == nil || m.vcacheGovernor == nil {
		return nil
	}
	records := m.vcacheGovernorDecisionRecords()
	latest := make(map[string]*vcacheGovernorDecisionRecord, len(records))
	for i := range records {
		latest[records[i].Family] = &records[i]
	}
	return latest
}

// usageFamiliesWithCacheActivity reports which families actually engaged the
// provider cache plane (any read or write) in the retained vcache window. A
// family that never did has no native warm-state evidence for the join, whatever
// posture the classifier projects over a no-cache workload.
func (m *gatewayMetrics) usageFamiliesWithCacheActivity() map[string]bool {
	turns, _ := m.vcacheTurnsSnapshot()
	active := make(map[string]bool, len(turns))
	for _, t := range turns {
		if t.CacheRead > 0 || t.CacheCreation > 0 {
			active[t.Family] = true
		}
	}
	return active
}
