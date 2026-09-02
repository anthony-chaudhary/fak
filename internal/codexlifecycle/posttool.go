// posttool.go — the #10662 post-tool model-latency span: one first-class
// `tool_result_recorded → next_model_item` interval per tool result, so post-tool
// model latency growth across long sessions (median ~11s at ordinal 1–20 growing
// to ~21s at 101–200 in the audited Codex corpus) becomes a queryable, bucketed,
// regression-witnessed quantity instead of an unnamed hole in the timeline.
//
// THE SPAN. For every function_call_output / custom_tool_call_output record (in
// file order), the span is:
//
//	call ──ToolMS──▶ output ──GapMS──▶ next model-emitted record
//
// A "model-emitted record" is the decomposeTimeline set — function_call,
// custom_tool_call, token_count, task_complete, turn_aborted — custom calls
// already folded into Kind function_call at ingestion and recovered here from
// ARecord.PayloadKind (no JSON re-parsing).
//
// DISJOINTNESS (the no-double-counting witness). ToolMS measures call → its own
// output; GapMS measures that output → the next model-emitted record. The two
// intervals share exactly one endpoint and cannot overlap by construction, so
// ToolMS + GapMS always equals the call → next-model-record interval: a slow tool
// can never be booked as post-tool model latency and vice versa.
//
// CORRELATION, NOT CAUSATION. Journal timestamps cannot separate provider TTFT
// from gateway queueing or harness scheduling inside GapMS. The Attribution
// vocabulary is a closed CORRELATION aid over observable structure only; it never
// claims a causal split. Live-path emit belongs to #10636 and the timing
// inventory to #10621.
//
// Like every report in this package, spans carry ids, names, numbers, and closed
// tokens only — no commands, no result bodies, no prompts, no paths.
package codexlifecycle

import (
	"os"
	"time"
)

// PostToolSpan is one observed `tool_result_recorded → next_model_item` interval.
type PostToolSpan struct {
	Session        string            `json:"session"`                // rollout UUID (opaque)
	TurnID         string            `json:"turn_id"`                // owning task_started turn ("" when unattributable)
	Ordinal        int               `json:"ordinal"`                // 1-based function/custom-tool-result ordinal within the session
	CallKind       string            `json:"call_kind"`              // function_call | custom_tool_call ("" when the call record is absent)
	Tool           string            `json:"tool,omitempty"`         // tool name ("" when unmatched)
	ToolClass      ToolClass         `json:"tool_class"`             // typed outcome class of the result envelope
	ToolMS         int64             `json:"tool_ms"`                // call → output (tool execution; disjoint from GapMS by construction)
	GapMS          int64             `json:"gap_ms"`                 // output → next model-emitted record
	NextRecordKind string            `json:"next_record_kind"`       // function_call | custom_tool_call | token_count | task_complete | turn_aborted
	Subspans       []PostToolSubspan `json:"subspans,omitempty"`     // observable interior segments (empty when none)
	Compactions    int               `json:"compactions,omitempty"`  // compacted records inside the gap
	InputTokens    int               `json:"input_tokens,omitempty"` // first token_count with InputTokens>0 at/after gap end (0 = unobserved)
	ContextBand    string            `json:"context_band"`           // closed band token (contextBand tokens below)
	Attribution    string            `json:"attribution"`            // closed CORRELATION token (attribution tokens below)
	Stall          bool              `json:"stall,omitempty"`        // GapMS > stallGapMS (idle split applies)
}

// PostToolSubspan is one observable interior segment of a span's gap. Subspans
// tile the gap exactly (their MS values sum to GapMS) — the same
// no-double-counting property as the span itself.
type PostToolSubspan struct {
	Kind string `json:"kind"` // "pre_compaction" | "compaction"
	MS   int64  `json:"ms"`
}

// Context bands, closed tokens over the first observed post-gap input-token
// reading (0 = unobserved). Boundaries are the audited hotspot edges.
const (
	BandUnobserved = "unobserved"
	BandLT10K      = "lt10k"
	Band10K25K     = "10k_25k"
	Band25K50K     = "25k_50k"
	Band50K100K    = "50k_100k"
	BandGTE100K    = "gte100k"
)

// bandOrder is the canonical report order for context bands.
var bandOrder = []string{BandUnobserved, BandLT10K, Band10K25K, Band25K50K, Band50K100K, BandGTE100K}

// contextBand maps an input-token reading to its closed band token. Bands are
// half-open on the upper edge: [10k,25k) is 10k_25k, and 100k lands in gte100k.
func contextBand(tokens int) string {
	switch {
	case tokens <= 0:
		return BandUnobserved
	case tokens < 10_000:
		return BandLT10K
	case tokens < 25_000:
		return Band10K25K
	case tokens < 50_000:
		return Band25K50K
	case tokens < 100_000:
		return Band50K100K
	default:
		return BandGTE100K
	}
}

// Attribution tokens — closed CORRELATION aids, explicitly not causal labels.
const (
	AttrToolSlow        = "tool_slow"         // the wait was the tool (ToolMS >= GapMS), not the model
	AttrStallCapped     = "stall_capped"      // GapMS > stallGapMS: only the first 300s is model-active, the remainder is idle (decomposeTimeline's rule)
	AttrCompactionInGap = "compaction_in_gap" // a compaction fired inside the gap (not a stall)
	AttrModelReasoning  = "model_reasoning"   // none of the above: the gap is attributable to the model step
)

// postToolAttribution resolves one span's correlation token. Precedence follows
// the evidence: an observed tool-dominated wait outranks the stall cap; the stall
// cap outranks compaction (a >300s gap is already idle-capped regardless of what
// fired inside it); a zero-ToolMS span (unmatched call) never claims tool_slow.
func postToolAttribution(toolMS, gapMS int64, compactions int, stall bool) string {
	switch {
	case toolMS > 0 && toolMS >= gapMS:
		return AttrToolSlow
	case gapMS > stallGapMS:
		return AttrStallCapped
	case compactions > 0:
		return AttrCompactionInGap
	default:
		return AttrModelReasoning
	}
}

// Ordinal buckets, closed tokens over the per-session tool-result ordinal. These
// are the audited growth edges (median gap ~11s at 1–20 → ~21s at 101–200).
const (
	OrdBucket1_20    = "1_20"
	OrdBucket21_50   = "21_50"
	OrdBucket51_100  = "51_100"
	OrdBucket101_200 = "101_200"
	OrdBucketGTE201  = "gte201"
)

// ordinalOrder is the canonical report order for ordinal buckets.
var ordinalOrder = []string{OrdBucket1_20, OrdBucket21_50, OrdBucket51_100, OrdBucket101_200, OrdBucketGTE201}

// postToolOrdinalBucket maps a 1-based tool-result ordinal to its closed bucket.
func postToolOrdinalBucket(ordinal int) string {
	switch {
	case ordinal >= 201:
		return OrdBucketGTE201
	case ordinal >= 101:
		return OrdBucket101_200
	case ordinal >= 51:
		return OrdBucket51_100
	case ordinal >= 21:
		return OrdBucket21_50
	default:
		return OrdBucket1_20
	}
}

// modelEmitted reports whether a record can end a post-tool gap — the same
// model-emitted set decomposeTimeline anchors on (custom calls already fold into
// Kind function_call at ingestion).
func modelEmitted(r ARecord) bool {
	return r.Kind == kindToolCall || r.Kind == kindTokens ||
		r.Kind == KindComplete || r.Kind == KindAborted
}

// nextKindLabel names the gap-ending record by its PAYLOAD type so custom calls
// stay distinguishable from function calls without re-parsing any JSON.
func nextKindLabel(r ARecord) string {
	switch r.Kind {
	case kindToolCall:
		if r.PayloadKind == "custom_tool_call" {
			return "custom_tool_call"
		}
		return "function_call"
	case kindTokens:
		return "token_count"
	case KindComplete:
		return "task_complete"
	case KindAborted:
		return "turn_aborted"
	}
	return r.Kind
}

// walkPostToolSpans is the pure span walk behind AnalyzePostToolSpans. It returns
// the spans in file order plus the count of trailing tool results with no next
// model-emitted record (the live tail), which are skipped as unmeasurable.
func walkPostToolSpans(meta Meta, records []ARecord) (spans []PostToolSpan, tailSkipped int) {
	// The same call→output join AnalyzeRollout uses, walked in the output
	// direction: call_id -> record index of the call (first wins).
	callIdx := map[string]int{}
	for i, r := range records {
		if r.Kind == kindToolCall && r.CallID != "" {
			if _, dup := callIdx[r.CallID]; !dup {
				callIdx[r.CallID] = i
			}
		}
	}

	turnID := ""
	ordinal := 0
	for oi := range records {
		out := &records[oi]
		if out.Kind != "function_call_output" {
			if out.Kind == KindStarted {
				turnID = out.TurnID
			}
			continue
		}
		if out.TS.IsZero() {
			continue // no measurable timestamps: an ingestion artifact, not a span
		}
		ordinal++

		// Next model-emitted record after the output; compacted records inside
		// the gap become observable subspans.
		next := -1
		var compIdxs []int
		for ni := oi + 1; ni < len(records); ni++ {
			if modelEmitted(records[ni]) {
				next = ni
				break
			}
			if records[ni].Kind == kindCompacted {
				compIdxs = append(compIdxs, ni)
			}
		}
		if next < 0 {
			tailSkipped++ // live tail: the interval never closed, so it measures nothing
			continue
		}
		nextTS := records[next].TS

		span := PostToolSpan{
			Session:     meta.RolloutID,
			TurnID:      turnID,
			Ordinal:     ordinal,
			ToolClass:   ToolOK,
			GapMS:       gapMS(out.TS, nextTS),
			Compactions: len(compIdxs),
		}
		if ci, ok := callIdx[out.CallID]; ok && out.CallID != "" {
			call := &records[ci]
			span.CallKind = call.PayloadKind
			if span.CallKind == "" {
				span.CallKind = "function_call"
			}
			span.Tool = call.Tool
			span.ToolMS = gapMS(call.TS, out.TS)
			span.ToolClass, _, _ = ClassifyOutcome(call.Head, out.Env)
		} else {
			// Unmatched call: the output's own payload type still says which call
			// kind produced it; the tool name is unobservable, so it stays "".
			switch out.PayloadKind {
			case "custom_tool_call_output":
				span.CallKind = "custom_tool_call"
			case "function_call_output":
				span.CallKind = "function_call"
			}
			span.ToolClass, _, _ = ClassifyOutcome("", out.Env)
		}
		span.NextRecordKind = nextKindLabel(records[next])
		span.InputTokens = firstTokensAtOrAfter(records, next)
		span.ContextBand = contextBand(span.InputTokens)
		span.Stall = span.GapMS > stallGapMS
		span.Attribution = postToolAttribution(span.ToolMS, span.GapMS, span.Compactions, span.Stall)

		// Interior subspans: with k compactions inside the gap, pre_compaction
		// runs output → first compaction, then one compaction segment per edge
		// through to the gap end. Sum == GapMS (monotone timestamps).
		if len(compIdxs) > 0 {
			prev := out.TS
			for k, ci := range compIdxs {
				kind := "compaction"
				if k == 0 {
					kind = "pre_compaction"
				}
				span.Subspans = append(span.Subspans, PostToolSubspan{Kind: kind, MS: gapMS(prev, records[ci].TS)})
				prev = records[ci].TS
			}
			span.Subspans = append(span.Subspans, PostToolSubspan{Kind: "compaction", MS: gapMS(prev, nextTS)})
		}
		spans = append(spans, span)
	}
	return spans, tailSkipped
}

// gapMS is the signed-safe millisecond delta between two record timestamps:
// positive intervals only, 0 otherwise (decomposeTimeline never books a
// non-positive gap either).
func gapMS(from, to time.Time) int64 {
	if to.Before(from) {
		return 0
	}
	return to.Sub(from).Milliseconds()
}

// firstTokensAtOrAfter returns the first token_count record at or after index
// from carrying a positive input-token reading. When the gap ends at that
// token_count, it reports the usage of the request that consumed this tool
// result; otherwise the first later reading is the honest nearest observation.
func firstTokensAtOrAfter(records []ARecord, from int) int {
	for i := from; i < len(records); i++ {
		if records[i].Kind == kindTokens && records[i].InputTokens > 0 {
			return records[i].InputTokens
		}
	}
	return 0
}

// AnalyzePostToolSpans walks one rollout's analytics records in file order and
// returns one PostToolSpan per completed `tool_result_recorded → next_model_item`
// interval. Pure: no IO, no clock. Trailing results with no next model-emitted
// record are skipped (ScanPostToolCorpus reports them as TailSkipped).
func AnalyzePostToolSpans(meta Meta, records []ARecord) []PostToolSpan {
	spans, _ := walkPostToolSpans(meta, records)
	return spans
}

// PostToolBucket is one context-band or ordinal-bucket row of the aggregate.
// Percentiles are seconds; Over30s counts gaps over 30s; ToolP50 is the
// tool-execution control that separates genuine tool slowness from post-tool
// model latency.
type PostToolBucket struct {
	Key     string  `json:"key"`
	N       int     `json:"n"`
	P50     float64 `json:"p50"`
	P90     float64 `json:"p90"`
	P95     float64 `json:"p95"`
	Over30s int     `json:"over30s"`
	ToolP50 float64 `json:"tool_p50"`
}

// PostToolSchema identifies the report shape.
const PostToolSchema = "fak-codex-posttool/1"

// PostToolReport is the corpus-wide post-tool latency report. Everything it
// exports is stable-and-scrubbed: opaque session ids, tool names, closed tokens,
// and numbers — the analytics.go privacy contract applies unchanged.
type PostToolReport struct {
	Schema          string           `json:"schema"`
	Root            string           `json:"root"`
	Sessions        int              `json:"sessions"`
	Unreadable      int              `json:"unreadable,omitempty"`
	Spans           int              `json:"spans"`
	TailSkipped     int              `json:"tail_skipped"`
	Gap             Percentiles      `json:"gap"`     // overall, seconds
	ToolMS          Percentiles      `json:"tool_ms"` // overall tool-execution control, seconds
	Over30sShare    float64          `json:"over30s_share"`
	StallSpans      int              `json:"stall_spans"`
	CompactionInGap int              `json:"compaction_in_gap"`
	ByBand          []PostToolBucket `json:"by_band"`    // canonical band order, N>0 only
	ByOrdinal       []PostToolBucket `json:"by_ordinal"` // canonical ordinal order, N>0 only
}

// ScanPostToolCorpus folds every rollout under root through the post-tool span
// walk. Unreadable files are counted, never fatal; opt.CWD scopes to one
// repository's sessions; opt.Limit caps files scanned, newest first — the same
// durability and scrubbing contract as ScanAnalyticsCorpus.
func ScanPostToolCorpus(root string, opt ScanOptions) (PostToolReport, error) {
	rep := PostToolReport{Schema: PostToolSchema, Root: root}

	paths, err := rolloutPaths(root, opt.Limit)
	if err != nil {
		return rep, err
	}
	var gaps, tools []float64
	over30s := 0
	bandSpans := map[string][]PostToolSpan{}
	ordinalSpans := map[string][]PostToolSpan{}

	for _, p := range paths {
		if _, statErr := os.Stat(p); statErr != nil {
			rep.Unreadable++
			continue
		}
		fh, openErr := os.Open(p)
		if openErr != nil {
			rep.Unreadable++
			continue
		}
		meta, records, parseErr := ReadAnalyticsRollout(fh)
		_ = fh.Close()
		if parseErr != nil {
			rep.Unreadable++
			continue
		}
		if opt.CWD != "" && !sameDir(meta.CWD, opt.CWD) {
			continue
		}
		spans, tailSkipped := walkPostToolSpans(meta, records)
		if len(spans) == 0 && tailSkipped == 0 {
			continue // no post-tool signal in this rollout
		}
		rep.Sessions++
		rep.TailSkipped += tailSkipped
		for _, s := range spans {
			rep.Spans++
			gS, tS := float64(s.GapMS)/1000.0, float64(s.ToolMS)/1000.0
			gaps = append(gaps, gS)
			tools = append(tools, tS)
			if s.GapMS > 30_000 {
				over30s++
			}
			if s.Stall {
				rep.StallSpans++
			}
			if s.Compactions > 0 {
				rep.CompactionInGap++
			}
			bandSpans[s.ContextBand] = append(bandSpans[s.ContextBand], s)
			ordinalSpans[postToolOrdinalBucket(s.Ordinal)] = append(ordinalSpans[postToolOrdinalBucket(s.Ordinal)], s)
		}
	}

	rep.Gap = percentiles(gaps)
	rep.ToolMS = percentiles(tools)
	if rep.Spans > 0 {
		rep.Over30sShare = float64(over30s) / float64(rep.Spans)
	}
	rep.ByBand = postToolBuckets(bandSpans, bandOrder)
	rep.ByOrdinal = postToolBuckets(ordinalSpans, ordinalOrder)
	return rep, nil
}

// postToolBuckets renders one bucket table in canonical key order (deterministic
// across runs), keeping only non-empty buckets.
func postToolBuckets(byKey map[string][]PostToolSpan, order []string) []PostToolBucket {
	var rows []PostToolBucket
	for _, key := range order {
		spans := byKey[key]
		if len(spans) == 0 {
			continue
		}
		var gVals, tVals []float64
		over30s := 0
		for _, s := range spans {
			gVals = append(gVals, float64(s.GapMS)/1000.0)
			tVals = append(tVals, float64(s.ToolMS)/1000.0)
			if s.GapMS > 30_000 {
				over30s++
			}
		}
		g, t := percentiles(gVals), percentiles(tVals)
		rows = append(rows, PostToolBucket{
			Key: key, N: len(spans),
			P50: g.P50, P90: g.P90, P95: g.P95,
			Over30s: over30s, ToolP50: t.P50,
		})
	}
	return rows
}
