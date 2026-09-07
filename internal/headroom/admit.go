package headroom

import (
	"context"
	"strconv"
	"strings"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/mcpbroker"
)

func init() {
	mcpbroker.RegisterContentScreener(func(body []byte) bool {
		_, held := ctxmmu.ScreenBytes(body)
		return held
	})
}

// AdmitRank places the compression gate AFTER the de-obfuscation rescan (normgate,
// rank 5) and as a peer just ahead of the context-MMU (ctxmmu, rank 10). At rank 8
// a clean BENIGN result is offered to the selected compressor before ctxmmu would
// page an oversize one out to an OPAQUE pointer — compression keeps the bytes
// MODEL-READABLE ("same answer, fewer tokens"), where a page-out makes the model
// retrieve. Because Transform (the compress verdict) outranks Allow but is
// outranked by Quarantine in the fold, a poisoned result still loses to the
// security gates; and because this gate SCREENS the raw bytes itself (ScreenBytes)
// and declines to touch poison, it can never hide an injection from the gates that
// screen raw bytes downstream.
const AdmitRank = 8

// Gate is the result-admission driver that folds the selected Compressor into the
// kernel result path (it is also reached from the gateway's inbound result
// admission, so `fak guard -- claude` compresses tool results in-stream with the
// same one registration). Construct with NewGate; Default is the registered one.
type Gate struct {
	considered int64
	compressed int64
	bytesIn    int64
	bytesOut   int64
	// The "when NOT to compress" decision breakdown — so the governance is
	// auditable, not just the savings. Each counts a considered result the gate
	// declined to compress, by reason.
	skippedEmpty    int64 // resolved to nothing
	skippedPoison   int64 // left raw for the security gates (the load-bearing skip)
	skippedNoSaving int64 // the compressor found no smaller rendering
	skippedNotWorth int64 // a real saving that did not clear the worth-it floor

	// Dedicated MCP normalization counters (separate from native headroom)
	mcpConsidered      int64
	mcpNormalized      int64
	mcpBytesIn         int64
	mcpBytesOut        int64
	mcpSavedBytes      int64
	mcpSkippedPoison   int64
	mcpSkippedOptOut   int64
	mcpSkippedAlready  int64
	mcpSkippedNoSaving int64
}

// NewGate builds a fresh gate (its counters are independent).
func NewGate() *Gate { return &Gate{} }

// Default is the registered gate; its counters are the process-wide compression
// KPI surfaced by `fak headroom status`.
var Default = NewGate()

func (g *Gate) Caps() []abi.Capability { return nil }

// Admit offers a benign tool result to the selected Compressor and, on a genuine
// saving, rewrites the payload to the smaller MODEL-READABLE rendering
// (VerdictTransform) while preserving the original in the shared CAS so it stays
// retrievable (the reversible-compression / CCR promise). It returns Allow
// (admit-as-is — the ResultAdmitter fold identity) whenever it must not act:
// compression is off (noop selected), the bytes screen as poison (left for the
// security gates), the result is empty, the compressor found no saving, or the
// saving is real but not worth the indirection (the worth-it floor in policy.go).
func (g *Gate) Admit(ctx context.Context, c *abi.ToolCall, r *abi.Result) abi.Verdict {
	if res, ok := reservesFromToolCall(c); ok {
		v, _ := g.AdmitWithReserves(ctx, c, r, res)
		return v
	}
	return g.admitStandard(ctx, c, r)
}

func (g *Gate) admitStandard(ctx context.Context, c *abi.ToolCall, r *abi.Result) abi.Verdict {
	if r == nil {
		return admitAsIs()
	}
	if isMCPCall(c, r) {
		if admRes, err := g.AdmitMCP(ctx, MCPAdmissionRequest{Call: c, Result: r}); err == nil {
			return admRes.Verdict
		}
	}
	comp := Selected()
	if comp.Name() == NoopName {
		return admitAsIs() // compression disabled -> zero overhead, no resolve
	}
	atomic.AddInt64(&g.considered, 1)

	body := resolveBytes(ctx, r.Payload)
	if len(body) == 0 {
		atomic.AddInt64(&g.skippedEmpty, 1)
		return admitAsIs()
	}
	// Security FIRST: never compress what the gates would quarantine. A poisoned
	// result is left raw so ctxmmu/normgate screen the REAL bytes and seal it.
	if _, poison := ctxmmu.ScreenBytes(body); poison {
		atomic.AddInt64(&g.skippedPoison, 1)
		return admitAsIs()
	}

	out, err := comp.Compress(ctx, Input{
		Tool:  toolName(c),
		Kind:  Detect(body),
		Model: modelHint(c),
		Bytes: body,
	})
	if err != nil || !out.Compressed || len(out.Bytes) == 0 || len(out.Bytes) >= len(body) {
		atomic.AddInt64(&g.skippedNoSaving, 1)
		return admitAsIs()
	}
	// The "when to compress" floor: a real but marginal saving on a small result is
	// not worth the preserve-write + the codec annotation the model must read, so
	// leave it raw (the model gets the verbatim bytes, nothing is spent). See
	// policy.go — this is fak deciding WHEN compression pays, not just HOW.
	if !worthCompressing(len(body), len(out.Bytes)) {
		atomic.AddInt64(&g.skippedNotWorth, 1)
		return admitAsIs()
	}

	// Preserve the original for retrieval (reversible CCR), then rewrite the
	// payload to the compressed, model-readable rendering. Tool-aware distillers
	// carry the exact model-callable restore handle inline; the generic metadata
	// remains authoritative for every compressor.
	origin := preserveOriginal(ctx, body)
	if origin != "" && strings.HasSuffix(out.Codec, "-distill") {
		out.Bytes = appendRestoreHint(out.Bytes, origin)
		out.NewLen = len(out.Bytes)
		if len(out.Bytes) >= len(body) {
			atomic.AddInt64(&g.skippedNoSaving, 1)
			return admitAsIs()
		}
	}
	atomic.AddInt64(&g.compressed, 1)
	atomic.AddInt64(&g.bytesIn, int64(len(body)))
	atomic.AddInt64(&g.bytesOut, int64(len(out.Bytes)))

	meta := map[string]string{
		"compressed":  "true",
		"compressor":  comp.Name(),
		"codec":       out.Codec,
		"saved_ratio": strconv.FormatFloat(out.SavedRatio(), 'f', 3, 64),
		"orig_len":    strconv.Itoa(len(body)),
		"new_len":     strconv.Itoa(len(out.Bytes)),
	}
	if out.Retrieval != "" {
		meta["ccr"] = out.Retrieval // external-service retrieval handle(s)
	}
	if origin != "" {
		meta["origin"] = origin // in-CAS digest of the original bytes
	}
	ref := abi.Ref{Kind: abi.RefInline, Inline: out.Bytes, Len: int64(len(out.Bytes))}
	return abi.Verdict{
		Kind:    abi.VerdictTransform,
		By:      "headroom",
		Payload: abi.TransformPayload{NewArgs: ref},
		Meta:    meta,
	}
}

// AdmitWithReserves offers a benign tool result to the selected Compressor and applies
// deterministic context headroom budget limits based on live context reserves and risk.
// When context headroom is Critical, tool result rendering is clamped to a strict
// minimal/compact budget. When context headroom is Healthy, standard budget is allowed.
func (g *Gate) AdmitWithReserves(ctx context.Context, c *abi.ToolCall, r *abi.Result, reserves ReserveState) (abi.Verdict, BudgetReceipt) {
	receipt := ComputeBudget(reserves)
	bMeta := budgetMeta(receipt)
	if r == nil {
		return admitAsIs(bMeta), receipt
	}
	if isMCPCall(c, r) {
		if admRes, err := g.AdmitMCP(ctx, MCPAdmissionRequest{Call: c, Result: r, Reserves: reserves}); err == nil {
			return admRes.Verdict, admRes.BudgetReceipt
		}
	}
	comp := Selected()
	if comp.Name() == NoopName {
		// When noop is selected, if context headroom is critical and result exceeds minimal budget,
		// we clamp to protect the model context window.
		if receipt.Status == HeadroomStatusCritical {
			body := resolveBytes(ctx, r.Payload)
			if clampedBytes, didClamp := ClampToolResult(body, receipt.Budget); didClamp {
				origin := preserveOriginal(ctx, body)
				atomic.AddInt64(&g.considered, 1)
				atomic.AddInt64(&g.compressed, 1)
				atomic.AddInt64(&g.bytesIn, int64(len(body)))
				atomic.AddInt64(&g.bytesOut, int64(len(clampedBytes)))
				meta := map[string]string{
					"compressed":      "true",
					"compressor":      "budget-gate",
					"codec":           "budget-clamp",
					"orig_len":        strconv.Itoa(len(body)),
					"new_len":         strconv.Itoa(len(clampedBytes)),
					"headroom_status": string(receipt.Status),
					"budget_tier":     receipt.Budget.Tier,
					"budget_receipt":  receipt.String(),
					"clamped":         "true",
				}
				if origin != "" {
					meta["origin"] = origin
				}
				ref := abi.Ref{Kind: abi.RefInline, Inline: clampedBytes, Len: int64(len(clampedBytes))}
				return abi.Verdict{
					Kind:    abi.VerdictTransform,
					By:      "headroom",
					Payload: abi.TransformPayload{NewArgs: ref},
					Meta:    meta,
				}, receipt
			}
		}
		return admitAsIs(bMeta), receipt // compression disabled -> zero overhead, no resolve
	}
	atomic.AddInt64(&g.considered, 1)

	body := resolveBytes(ctx, r.Payload)
	if len(body) == 0 {
		atomic.AddInt64(&g.skippedEmpty, 1)
		return admitAsIs(bMeta), receipt
	}
	// Security FIRST: never compress what the gates would quarantine. A poisoned
	// result is left raw so ctxmmu/normgate screen the REAL bytes and seal it.
	if _, poison := ctxmmu.ScreenBytes(body); poison {
		atomic.AddInt64(&g.skippedPoison, 1)
		return admitAsIs(bMeta), receipt
	}

	out, err := comp.Compress(ctx, Input{
		Tool:  toolName(c),
		Kind:  Detect(body),
		Model: modelHint(c),
		Bytes: body,
	})
	if err != nil || !out.Compressed || len(out.Bytes) == 0 || len(out.Bytes) >= len(body) {
		// When context headroom is Critical or explicitly clamped, clamp tool result rendering
		// even if the compressor found no structural saving.
		if receipt.Status == HeadroomStatusCritical || (receipt.Clamped && receipt.Status != HeadroomStatusUnknown) {
			clampedBytes, didClamp := ClampToolResult(body, receipt.Budget)
			if didClamp {
				origin := preserveOriginal(ctx, body)
				atomic.AddInt64(&g.compressed, 1)
				atomic.AddInt64(&g.bytesIn, int64(len(body)))
				atomic.AddInt64(&g.bytesOut, int64(len(clampedBytes)))
				meta := map[string]string{
					"compressed":      "true",
					"compressor":      comp.Name(),
					"codec":           "budget-clamp",
					"saved_ratio":     strconv.FormatFloat(float64(len(body)-len(clampedBytes))/float64(len(body)), 'f', 3, 64),
					"orig_len":        strconv.Itoa(len(body)),
					"new_len":         strconv.Itoa(len(clampedBytes)),
					"headroom_status": string(receipt.Status),
					"budget_tier":     receipt.Budget.Tier,
					"budget_receipt":  receipt.String(),
					"clamped":         "true",
				}
				if origin != "" {
					meta["origin"] = origin
				}
				ref := abi.Ref{Kind: abi.RefInline, Inline: clampedBytes, Len: int64(len(clampedBytes))}
				return abi.Verdict{
					Kind:    abi.VerdictTransform,
					By:      "headroom",
					Payload: abi.TransformPayload{NewArgs: ref},
					Meta:    meta,
				}, receipt
			}
		}
		atomic.AddInt64(&g.skippedNoSaving, 1)
		return admitAsIs(bMeta), receipt
	}

	// Clamp compressed output if it still exceeds the computed budget.
	if receipt.Budget.ByteLimit > 0 || receipt.Budget.MaxItems > 0 {
		if clampedBytes, didClamp := ClampToolResult(out.Bytes, receipt.Budget); didClamp {
			out.Bytes = clampedBytes
			out.NewLen = len(clampedBytes)
			out.Codec += "+budget-clamp"
			receipt.Clamped = true
		}
	}

	// The "when to compress" floor: a real but marginal saving on a small result is
	// not worth the preserve-write + the codec annotation the model must read, so
	// leave it raw (the model gets the verbatim bytes, nothing is spent). See
	// policy.go — this is fak deciding WHEN compression pays, not just HOW.
	// Critical headroom or clamped results bypass the skip floor because every byte counts.
	if !worthCompressing(len(body), len(out.Bytes)) && receipt.Status != HeadroomStatusCritical && !receipt.Clamped {
		atomic.AddInt64(&g.skippedNotWorth, 1)
		return admitAsIs(bMeta), receipt
	}

	// Preserve the original for retrieval (reversible CCR), then rewrite the
	// payload to the compressed, model-readable rendering. Tool-aware distillers
	// carry the exact model-callable restore handle inline; the generic metadata
	// remains authoritative for every compressor.
	origin := preserveOriginal(ctx, body)
	if origin != "" && strings.HasSuffix(out.Codec, "-distill") {
		out.Bytes = appendRestoreHint(out.Bytes, origin)
		out.NewLen = len(out.Bytes)
		if len(out.Bytes) >= len(body) {
			atomic.AddInt64(&g.skippedNoSaving, 1)
			return admitAsIs(bMeta), receipt
		}
	}
	atomic.AddInt64(&g.compressed, 1)
	atomic.AddInt64(&g.bytesIn, int64(len(body)))
	atomic.AddInt64(&g.bytesOut, int64(len(out.Bytes)))

	meta := map[string]string{
		"compressed":  "true",
		"compressor":  comp.Name(),
		"codec":       out.Codec,
		"saved_ratio": strconv.FormatFloat(out.SavedRatio(), 'f', 3, 64),
		"orig_len":    strconv.Itoa(len(body)),
		"new_len":     strconv.Itoa(len(out.Bytes)),
	}
	if receipt.Status != HeadroomStatusUnknown || receipt.Clamped {
		meta["headroom_status"] = string(receipt.Status)
		meta["budget_tier"] = receipt.Budget.Tier
		meta["budget_receipt"] = receipt.String()
		if receipt.Clamped {
			meta["clamped"] = "true"
		}
	}
	if out.Retrieval != "" {
		meta["ccr"] = out.Retrieval // external-service retrieval handle(s)
	}
	if origin != "" {
		meta["origin"] = origin // in-CAS digest of the original bytes
	}
	ref := abi.Ref{Kind: abi.RefInline, Inline: out.Bytes, Len: int64(len(out.Bytes))}
	return abi.Verdict{
		Kind:    abi.VerdictTransform,
		By:      "headroom",
		Payload: abi.TransformPayload{NewArgs: ref},
		Meta:    meta,
	}, receipt
}

// Stats is the gate's lifetime compression KPI, including the "when NOT to
// compress" decision breakdown — so the governance is auditable, not just the
// savings (Considered == Compressed + every Skipped* reason).
type Stats struct {
	Considered      int64 `json:"considered"`
	Compressed      int64 `json:"compressed"`
	BytesIn         int64 `json:"bytes_in"`
	BytesOut        int64 `json:"bytes_out"`
	SkippedEmpty    int64 `json:"skipped_empty"`
	SkippedPoison   int64 `json:"skipped_poison"`
	SkippedNoSaving int64 `json:"skipped_no_saving"`
	SkippedNotWorth int64 `json:"skipped_not_worth"`
}

// Stats snapshots the gate's counters.
func (g *Gate) Stats() Stats {
	return Stats{
		Considered:      atomic.LoadInt64(&g.considered),
		Compressed:      atomic.LoadInt64(&g.compressed),
		BytesIn:         atomic.LoadInt64(&g.bytesIn),
		BytesOut:        atomic.LoadInt64(&g.bytesOut),
		SkippedEmpty:    atomic.LoadInt64(&g.skippedEmpty),
		SkippedPoison:   atomic.LoadInt64(&g.skippedPoison),
		SkippedNoSaving: atomic.LoadInt64(&g.skippedNoSaving),
		SkippedNotWorth: atomic.LoadInt64(&g.skippedNotWorth),
	}
}

func admitAsIs(meta ...map[string]string) abi.Verdict {
	v := abi.Verdict{Kind: abi.VerdictAllow, By: "headroom"}
	if len(meta) > 0 && meta[0] != nil {
		v.Meta = meta[0]
	}
	return v
}

func budgetMeta(receipt BudgetReceipt) map[string]string {
	if receipt.Status == HeadroomStatusUnknown && !receipt.Clamped {
		return nil
	}
	m := map[string]string{
		"headroom_status": string(receipt.Status),
		"budget_tier":     receipt.Budget.Tier,
		"budget_receipt":  receipt.String(),
	}
	if receipt.Clamped {
		m["clamped"] = "true"
	}
	return m
}

func resolveBytes(ctx context.Context, r abi.Ref) []byte {
	if r.Kind == abi.RefInline {
		return r.Inline
	}
	if res := abi.ActiveResolver(); res != nil {
		if b, err := res.Resolve(ctx, r); err == nil {
			return b
		}
	}
	return nil
}

// preserveOriginal stores the pre-compression bytes in the shared content-
// addressed store and returns their digest, so a future read can retrieve the
// exact original. Best-effort: if no blob codec is registered it returns "" and
// the compression still proceeds (the external plugin's own CCR may still hold it).
func preserveOriginal(ctx context.Context, body []byte) string {
	b, ok := abi.PageOut("blob")
	if !ok {
		return ""
	}
	h, err := b.PageOut(ctx, abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body))})
	if err != nil {
		return ""
	}
	return h.Digest
}

func toolName(c *abi.ToolCall) string {
	if c == nil {
		return ""
	}
	return c.Tool
}

func modelHint(c *abi.ToolCall) string {
	if c == nil || c.Meta == nil {
		return ""
	}
	return c.Meta["model"]
}

func reservesFromToolCall(c *abi.ToolCall) (ReserveState, bool) {
	if c == nil || c.Meta == nil {
		return ReserveState{}, false
	}
	var res ReserveState
	has := false
	if s := c.Meta["headroom_status"]; s != "" {
		res.Status = ParseHeadroomStatus(s)
		has = true
	} else if s := c.Meta["context_status"]; s != "" {
		res.Status = ParseHeadroomStatus(s)
		has = true
	}
	if r := c.Meta["compression_risk"]; r != "" {
		res.CompressionRisk = ParseCompressionRisk(r)
		has = true
	}
	if tok := c.Meta["remaining_tokens"]; tok != "" {
		if n, err := strconv.Atoi(tok); err == nil {
			res.RemainingTokens = n
			has = true
		}
	}
	if tok := c.Meta["reserve_tokens"]; tok != "" {
		if n, err := strconv.Atoi(tok); err == nil {
			res.ReserveTokens = n
			has = true
		}
	}
	if tok := c.Meta["total_tokens"]; tok != "" {
		if n, err := strconv.Atoi(tok); err == nil {
			res.TotalTokens = n
			has = true
		}
	}
	return res, has
}

func isMCPCall(c *abi.ToolCall, r *abi.Result) bool {
	if c != nil {
		if strings.HasPrefix(c.Tool, "mcp__") {
			return true
		}
		if c.Meta != nil {
			if c.Meta["mcp"] == "true" || c.Meta["protocol"] == "mcp" {
				return true
			}
		}
	}
	if r != nil && r.Meta != nil {
		if r.Meta["mcp"] == "true" || r.Meta["protocol"] == "mcp" {
			return true
		}
	}
	return false
}

func init() {
	abi.RegisterResultAdmitter(AdmitRank, Default)
	abi.RegisterCapability("headroom.v1")
}
