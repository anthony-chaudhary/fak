package headroom

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/mcpbroker"
)

// MCPAdmissionRequest configures an MCP tool result admission evaluation.
type MCPAdmissionRequest struct {
	Call        *abi.ToolCall `json:"call,omitempty"`
	Result      *abi.Result   `json:"result,omitempty"`
	RawEnvelope []byte        `json:"raw_envelope,omitempty"`
	RawContent  []byte        `json:"raw_content,omitempty"`
	Reserves    ReserveState  `json:"reserves,omitempty"`
	OptOut      bool          `json:"opt_out,omitempty"`
	MemoryEntry []byte        `json:"memory_entry,omitempty"`
}

// StageAttribution captures byte accounting and disposition for a single pipeline stage.
type StageAttribution struct {
	Stage      string        `json:"stage"` // e.g. "screening", "mcp_normalization", "headroom_compression", "budget_clamp", "memory_admit"
	Executed   bool          `json:"executed"`
	Skipped    bool          `json:"skipped"`
	SkipReason string        `json:"skip_reason,omitempty"` // e.g. "poison", "opt_out", "already_applied", "no_saving", "noop_selected"
	Codec      string        `json:"codec,omitempty"`       // e.g. "json-min", "native", "budget-clamp"
	BytesIn    int           `json:"bytes_in"`
	BytesOut   int           `json:"bytes_out"`
	BytesSaved int           `json:"bytes_saved"`
	Duration   time.Duration `json:"duration,omitempty"`
}

// MCPAdmissionReceipt is the explicit attribution receipt for an MCP admission pass.
type MCPAdmissionReceipt struct {
	Schema          string `json:"schema"` // "fak-mcp-admission-receipt/1"
	RawBytes        int    `json:"raw_bytes"`
	FinalBytes      int    `json:"final_bytes"`
	TotalBytesSaved int    `json:"total_bytes_saved"`

	ScreeningPassed  bool           `json:"screening_passed"`
	Quarantined      bool           `json:"quarantined"`
	QuarantineReason abi.ReasonCode `json:"quarantine_reason,omitempty"`

	MCPNormalized bool   `json:"mcp_normalized"`
	MCPBytesSaved int    `json:"mcp_bytes_saved"`
	MCPCodec      string `json:"mcp_codec,omitempty"`

	HeadroomCompressed bool   `json:"headroom_compressed"`
	HeadroomBytesSaved int    `json:"headroom_bytes_saved"`
	HeadroomCodec      string `json:"headroom_codec,omitempty"`

	BudgetClamped    bool `json:"budget_clamped"`
	BudgetBytesSaved int  `json:"budget_bytes_saved"`

	OriginDigest string             `json:"origin_digest,omitempty"`
	Stages       []StageAttribution `json:"stages"`
	Metadata     map[string]string  `json:"metadata,omitempty"`
}

// MCPAdmissionResult is the complete outcome of an MCP admission evaluation.
type MCPAdmissionResult struct {
	Verdict       abi.Verdict         `json:"verdict"`
	AdmittedBytes []byte              `json:"admitted_bytes"`
	MemoryBytes   []byte              `json:"memory_bytes,omitempty"`
	Receipt       MCPAdmissionReceipt `json:"receipt"`
	BudgetReceipt BudgetReceipt       `json:"budget_receipt"`
	OriginDigest  string              `json:"origin_digest,omitempty"`
}

// MCPStats holds the gate's lifetime counters for MCP structured compression decisions.
type MCPStats struct {
	Considered      int64 `json:"considered"`
	Normalized      int64 `json:"normalized"`
	BytesIn         int64 `json:"bytes_in"`
	BytesOut        int64 `json:"bytes_out"`
	BytesSaved      int64 `json:"bytes_saved"`
	SkippedPoison   int64 `json:"skipped_poison"`
	SkippedOptOut   int64 `json:"skipped_opt_out"`
	SkippedAlready  int64 `json:"skipped_already"`
	SkippedNoSaving int64 `json:"skipped_no_saving"`
}

// MCPStats snapshots the gate's MCP-specific counters.
func (g *Gate) MCPStats() MCPStats {
	return MCPStats{
		Considered:      atomic.LoadInt64(&g.mcpConsidered),
		Normalized:      atomic.LoadInt64(&g.mcpNormalized),
		BytesIn:         atomic.LoadInt64(&g.mcpBytesIn),
		BytesOut:        atomic.LoadInt64(&g.mcpBytesOut),
		BytesSaved:      atomic.LoadInt64(&g.mcpSavedBytes),
		SkippedPoison:   atomic.LoadInt64(&g.mcpSkippedPoison),
		SkippedOptOut:   atomic.LoadInt64(&g.mcpSkippedOptOut),
		SkippedAlready:  atomic.LoadInt64(&g.mcpSkippedAlready),
		SkippedNoSaving: atomic.LoadInt64(&g.mcpSkippedNoSaving),
	}
}

// AdmitMCP delegates MCP admission to Default.AdmitMCP.
func AdmitMCP(ctx context.Context, req MCPAdmissionRequest) (*MCPAdmissionResult, error) {
	return Default.AdmitMCP(ctx, req)
}

// AdmitMCP executes the integrated MCP admission pipeline:
// raw MCP result -> screening -> safe normalization -> optional headroom -> budgeting/page-out -> cache and memory.
func (g *Gate) AdmitMCP(ctx context.Context, req MCPAdmissionRequest) (*MCPAdmissionResult, error) {
	start := time.Now()
	atomic.AddInt64(&g.mcpConsidered, 1)

	if req.Call == nil {
		req.Call = &abi.ToolCall{Tool: "mcp_tool"}
	}

	// 1. Resolve raw content and raw envelope
	var envelopeBytes, contentBytes []byte
	var primaryIsEnvelope bool

	if len(req.RawEnvelope) > 0 {
		envelopeBytes = req.RawEnvelope
		if len(req.RawContent) > 0 {
			contentBytes = req.RawContent
		} else {
			contentBytes = extractMCPContent(envelopeBytes)
		}
		if req.Result == nil || bytes.Equal(resolveBytes(ctx, req.Result.Payload), req.RawEnvelope) {
			primaryIsEnvelope = true
		}
	} else if req.Result != nil {
		resolved := resolveBytes(ctx, req.Result.Payload)
		if isMCPEnvelope(resolved) {
			envelopeBytes = resolved
			contentBytes = extractMCPContent(resolved)
			primaryIsEnvelope = true
		} else {
			contentBytes = resolved
		}
	} else if len(req.RawContent) > 0 {
		contentBytes = req.RawContent
	}

	if len(contentBytes) == 0 && len(envelopeBytes) > 0 {
		contentBytes = envelopeBytes
	}

	rawBytes := contentBytes
	if primaryIsEnvelope && len(envelopeBytes) > 0 {
		rawBytes = envelopeBytes
	}
	rawLen := len(rawBytes)

	receipt := MCPAdmissionReceipt{
		Schema:   "fak-mcp-admission-receipt/1",
		RawBytes: rawLen,
		Metadata: make(map[string]string),
	}

	// Stage 1: Screening (raw poison handling)
	// Screening happens on the RAW bytes BEFORE any normalization.
	screenStart := time.Now()
	var reason abi.ReasonCode
	var isPoison bool

	if len(contentBytes) > 0 {
		reason, isPoison = ctxmmu.ScreenBytes(contentBytes)
	}
	if !isPoison && len(envelopeBytes) > 0 {
		reason, isPoison = ctxmmu.ScreenBytes(envelopeBytes)
	}

	receipt.Stages = append(receipt.Stages, StageAttribution{
		Stage:    "screening",
		Executed: true,
		Skipped:  false,
		BytesIn:  rawLen,
		BytesOut: rawLen,
		Duration: time.Since(screenStart),
	})

	if isPoison {
		atomic.AddInt64(&g.mcpSkippedPoison, 1)
		receipt.ScreeningPassed = false
		receipt.Quarantined = true
		receipt.QuarantineReason = reason
		receipt.FinalBytes = rawLen
		receipt.TotalBytesSaved = 0

		receipt.Stages = append(receipt.Stages,
			StageAttribution{Stage: "mcp_normalization", Executed: false, Skipped: true, SkipReason: "poison"},
			StageAttribution{Stage: "headroom_compression", Executed: false, Skipped: true, SkipReason: "poison"},
			StageAttribution{Stage: "budget_clamp", Executed: false, Skipped: true, SkipReason: "poison"},
			StageAttribution{Stage: "memory_admit", Executed: false, Skipped: true, SkipReason: "poison"},
		)

		verdict := abi.Verdict{
			Kind:    abi.VerdictQuarantine,
			Reason:  reason,
			By:      "ctxmmu",
			Payload: abi.QuarantinePayload{PageOut: false},
			Meta: map[string]string{
				"quarantined":       "true",
				"quarantine_reason": abi.ReasonName(reason),
				"screening_passed":  "false",
			},
		}

		return &MCPAdmissionResult{
			Verdict:       verdict,
			AdmittedBytes: rawBytes,
			MemoryBytes:   nil, // Poison is withheld from durable memory
			Receipt:       receipt,
			BudgetReceipt: ComputeBudget(req.Reserves),
		}, nil
	}

	receipt.ScreeningPassed = true

	// Stage 2: Safe Normalization
	normStart := time.Now()
	workingBytes := rawBytes
	workingContent := contentBytes

	// Check identity precedence (opt-out)
	isOptOut := req.OptOut || mcpbroker.IsOperatorForcedIdentity()
	if !isOptOut && req.Call != nil && len(req.Call.Meta) > 0 {
		if mcpbroker.IsCompressionMetadataOptOut(req.Call.Meta) || mcpbroker.IsCompressionOptOut(req.Call.Meta["compression"]) {
			isOptOut = true
		}
	}
	if !isOptOut && ctx != nil {
		if mcpbroker.IsCompressionOptOut(string(mcpbroker.ResolveEffectiveCompression(ctx, mcpbroker.CallRequest{}, ""))) {
			isOptOut = true
		}
	}

	// Check one-time stage application (already normalized)
	alreadyNormalized := false
	if req.Result != nil && len(req.Result.Meta) > 0 {
		if req.Result.Meta["mcp_normalized"] == "true" || req.Result.Meta["stage"] == mcpbroker.CompressionStageIdentity || req.Result.Meta["mcp_stage"] == "normalized" {
			alreadyNormalized = true
		}
	}
	if !alreadyNormalized && req.Call != nil && len(req.Call.Meta) > 0 {
		if req.Call.Meta["mcp_normalized"] == "true" || req.Call.Meta["stage"] == mcpbroker.CompressionStageIdentity || req.Call.Meta["mcp_stage"] == "normalized" {
			alreadyNormalized = true
		}
	}

	var mcpSaved int
	if isOptOut {
		atomic.AddInt64(&g.mcpSkippedOptOut, 1)
		receipt.Stages = append(receipt.Stages, StageAttribution{
			Stage:      "mcp_normalization",
			Executed:   false,
			Skipped:    true,
			SkipReason: "opt_out",
			BytesIn:    rawLen,
			BytesOut:   rawLen,
			Duration:   time.Since(normStart),
		})
	} else if alreadyNormalized {
		atomic.AddInt64(&g.mcpSkippedAlready, 1)
		receipt.Stages = append(receipt.Stages, StageAttribution{
			Stage:      "mcp_normalization",
			Executed:   false,
			Skipped:    true,
			SkipReason: "already_applied",
			BytesIn:    rawLen,
			BytesOut:   rawLen,
			Duration:   time.Since(normStart),
		})
	} else if len(envelopeBytes) > 0 && len(contentBytes) > 0 {
		compacted, mReceipt := mcpbroker.CompactStructuredContentWithReceipt(envelopeBytes, contentBytes, mcpbroker.WithCompressionContext(ctx))
		if mReceipt != nil && mReceipt.Reason == mcpbroker.ReasonSaved {
			mcpSaved = mReceipt.BytesSaved
			receipt.MCPNormalized = true
			receipt.MCPBytesSaved = mcpSaved
			receipt.MCPCodec = mReceipt.Codec

			atomic.AddInt64(&g.mcpNormalized, 1)
			atomic.AddInt64(&g.mcpBytesIn, int64(mReceipt.InputBytes))
			atomic.AddInt64(&g.mcpBytesOut, int64(mReceipt.OutputBytes))
			atomic.AddInt64(&g.mcpSavedBytes, int64(mcpSaved))

			receipt.Stages = append(receipt.Stages, StageAttribution{
				Stage:      "mcp_normalization",
				Executed:   true,
				Codec:      mReceipt.Codec,
				BytesIn:    mReceipt.InputBytes,
				BytesOut:   mReceipt.OutputBytes,
				BytesSaved: mcpSaved,
				Duration:   mReceipt.Duration,
			})

			workingContent = compacted
			if primaryIsEnvelope {
				workingBytes = replaceContentInEnvelope(envelopeBytes, contentBytes, compacted)
			} else {
				workingBytes = compacted
			}
		} else {
			atomic.AddInt64(&g.mcpSkippedNoSaving, 1)
			skipReason := "no_saving"
			if mReceipt != nil {
				skipReason = string(mReceipt.Reason)
			}
			receipt.Stages = append(receipt.Stages, StageAttribution{
				Stage:      "mcp_normalization",
				Executed:   false,
				Skipped:    true,
				SkipReason: skipReason,
				BytesIn:    len(contentBytes),
				BytesOut:   len(contentBytes),
				Duration:   time.Since(normStart),
			})
		}
	} else {
		atomic.AddInt64(&g.mcpSkippedNoSaving, 1)
		receipt.Stages = append(receipt.Stages, StageAttribution{
			Stage:      "mcp_normalization",
			Executed:   false,
			Skipped:    true,
			SkipReason: "noneligible",
			BytesIn:    rawLen,
			BytesOut:   rawLen,
			Duration:   time.Since(normStart),
		})
	}

	// Stage 3: Optional Headroom Compression
	// Operates on workingBytes (already normalized).
	headroomStart := time.Now()
	comp := Selected()
	var headroomSaved int
	var headroomCodec string

	if comp.Name() != NoopName {
		preHeadroomLen := len(workingBytes)
		out, err := comp.Compress(ctx, Input{
			Tool:  toolName(req.Call),
			Kind:  Detect(workingBytes),
			Model: modelHint(req.Call),
			Bytes: workingBytes,
		})
		if err == nil && out.Compressed && len(out.Bytes) > 0 && len(out.Bytes) < preHeadroomLen {
			if worthCompressing(preHeadroomLen, len(out.Bytes)) || req.Reserves.Status == HeadroomStatusCritical {
				headroomSaved = preHeadroomLen - len(out.Bytes)
				headroomCodec = out.Codec
				receipt.HeadroomCompressed = true
				receipt.HeadroomBytesSaved = headroomSaved
				receipt.HeadroomCodec = headroomCodec

				atomic.AddInt64(&g.compressed, 1)
				atomic.AddInt64(&g.bytesIn, int64(preHeadroomLen))
				atomic.AddInt64(&g.bytesOut, int64(len(out.Bytes)))

				receipt.Stages = append(receipt.Stages, StageAttribution{
					Stage:      "headroom_compression",
					Executed:   true,
					Codec:      headroomCodec,
					BytesIn:    preHeadroomLen,
					BytesOut:   len(out.Bytes),
					BytesSaved: headroomSaved,
					Duration:   time.Since(headroomStart),
				})
				workingBytes = out.Bytes
			} else {
				receipt.Stages = append(receipt.Stages, StageAttribution{
					Stage:      "headroom_compression",
					Executed:   false,
					Skipped:    true,
					SkipReason: "not_worth",
					BytesIn:    preHeadroomLen,
					BytesOut:   preHeadroomLen,
					Duration:   time.Since(headroomStart),
				})
			}
		} else {
			receipt.Stages = append(receipt.Stages, StageAttribution{
				Stage:      "headroom_compression",
				Executed:   false,
				Skipped:    true,
				SkipReason: "no_saving",
				BytesIn:    preHeadroomLen,
				BytesOut:   preHeadroomLen,
				Duration:   time.Since(headroomStart),
			})
		}
	} else {
		receipt.Stages = append(receipt.Stages, StageAttribution{
			Stage:      "headroom_compression",
			Executed:   false,
			Skipped:    true,
			SkipReason: "noop_selected",
			BytesIn:    len(workingBytes),
			BytesOut:   len(workingBytes),
			Duration:   time.Since(headroomStart),
		})
	}

	// Stage 4: Budgeting and Page-out
	budgetStart := time.Now()
	bReceipt := ComputeBudget(req.Reserves)
	var budgetSaved int
	if bReceipt.Status == HeadroomStatusCritical || bReceipt.Clamped {
		preBudgetLen := len(workingBytes)
		clamped, didClamp := ClampToolResult(workingBytes, bReceipt.Budget)
		if didClamp && len(clamped) < preBudgetLen {
			budgetSaved = preBudgetLen - len(clamped)
			receipt.BudgetClamped = true
			receipt.BudgetBytesSaved = budgetSaved
			receipt.Stages = append(receipt.Stages, StageAttribution{
				Stage:      "budget_clamp",
				Executed:   true,
				Codec:      "budget-clamp",
				BytesIn:    preBudgetLen,
				BytesOut:   len(clamped),
				BytesSaved: budgetSaved,
				Duration:   time.Since(budgetStart),
			})
			workingBytes = clamped
		} else {
			receipt.Stages = append(receipt.Stages, StageAttribution{
				Stage:      "budget_clamp",
				Executed:   false,
				Skipped:    true,
				SkipReason: "within_budget",
				BytesIn:    preBudgetLen,
				BytesOut:   preBudgetLen,
				Duration:   time.Since(budgetStart),
			})
		}
	} else {
		receipt.Stages = append(receipt.Stages, StageAttribution{
			Stage:      "budget_clamp",
			Executed:   false,
			Skipped:    true,
			SkipReason: "not_clamped",
			BytesIn:    len(workingBytes),
			BytesOut:   len(workingBytes),
			Duration:   time.Since(budgetStart),
		})
	}

	// Preserve original uncompressed raw bytes in CAS for reversible restoration (CCR)
	origin := preserveOriginal(ctx, rawBytes)
	receipt.OriginDigest = origin

	// Stage 5: Cache and Memory Consistency
	memoryStart := time.Now()
	var memoryBytes []byte
	if len(req.MemoryEntry) > 0 {
		if _, mPoison := ctxmmu.ScreenBytes(req.MemoryEntry); mPoison {
			memoryBytes = nil // Refuse poisoned memory write
			receipt.Stages = append(receipt.Stages, StageAttribution{
				Stage:      "memory_admit",
				Executed:   false,
				Skipped:    true,
				SkipReason: "poison_refused",
				Duration:   time.Since(memoryStart),
			})
		} else {
			// Memory entry is clean: keep consistent representation
			mEntry := req.MemoryEntry
			if receipt.MCPNormalized && json.Valid(mEntry) {
				var compactedMem bytes.Buffer
				if err := json.Compact(&compactedMem, mEntry); err == nil && compactedMem.Len() > 0 {
					mEntry = compactedMem.Bytes()
				}
			}
			if len(mEntry) > ctxmmu.MemoryWriteMaxBytes {
				mEntry = mEntry[:ctxmmu.MemoryWriteMaxBytes]
			}
			memoryBytes = mEntry
			receipt.Stages = append(receipt.Stages, StageAttribution{
				Stage:    "memory_admit",
				Executed: true,
				BytesIn:  len(req.MemoryEntry),
				BytesOut: len(memoryBytes),
				Duration: time.Since(memoryStart),
			})
		}
	} else {
		// Derive memory representation consistently from admitted working bytes
		memCandidate := workingBytes
		if len(memCandidate) > ctxmmu.MemoryWriteMaxBytes {
			memCandidate = memCandidate[:ctxmmu.MemoryWriteMaxBytes]
		}
		memoryBytes = memCandidate
		receipt.Stages = append(receipt.Stages, StageAttribution{
			Stage:    "memory_admit",
			Executed: true,
			BytesIn:  len(workingBytes),
			BytesOut: len(memoryBytes),
			Duration: time.Since(memoryStart),
		})
	}

	receipt.FinalBytes = len(workingBytes)
	receipt.TotalBytesSaved = rawLen - len(workingBytes)

	// Build metadata
	meta := make(map[string]string)
	meta["stage"] = mcpbroker.CompressionStageIdentity
	meta["mcp_admitted"] = "true"
	meta["raw_len"] = strconv.Itoa(rawLen)
	meta["final_len"] = strconv.Itoa(len(workingBytes))
	meta["total_saved"] = strconv.Itoa(receipt.TotalBytesSaved)

	if receipt.MCPNormalized {
		meta["mcp_normalized"] = "true"
		meta["mcp_codec"] = receipt.MCPCodec
		meta["mcp_saved_bytes"] = strconv.Itoa(receipt.MCPBytesSaved)
	}
	if receipt.HeadroomCompressed {
		meta["compressed"] = "true"
		meta["compressor"] = comp.Name()
		meta["codec"] = receipt.HeadroomCodec
		meta["headroom_saved_bytes"] = strconv.Itoa(receipt.HeadroomBytesSaved)
	}
	if receipt.BudgetClamped {
		meta["clamped"] = "true"
		meta["budget_tier"] = bReceipt.Budget.Tier
		meta["budget_receipt"] = bReceipt.String()
	}
	if origin != "" {
		meta["origin"] = origin
		meta["ccr"] = origin
	}
	receipt.Metadata = meta

	// Construct verdict
	var verdict abi.Verdict
	if receipt.MCPNormalized || receipt.HeadroomCompressed || receipt.BudgetClamped {
		verdict = abi.Verdict{
			Kind: abi.VerdictTransform,
			By:   "headroom",
			Payload: abi.TransformPayload{
				NewArgs: abi.Ref{
					Kind:   abi.RefInline,
					Inline: workingBytes,
					Len:    int64(len(workingBytes)),
				},
			},
			Meta: meta,
		}
	} else {
		verdict = abi.Verdict{
			Kind:    abi.VerdictAllow,
			By:      "headroom",
			Payload: abi.WitnessPayload{Claim: "mcp admitted as-is"},
			Meta:    meta,
		}
	}

	_ = start
	_ = workingContent
	return &MCPAdmissionResult{
		Verdict:       verdict,
		AdmittedBytes: workingBytes,
		MemoryBytes:   memoryBytes,
		Receipt:       receipt,
		BudgetReceipt: bReceipt,
		OriginDigest:  origin,
	}, nil
}

// RestoreOriginal retrieves the original pre-normalization bytes from the content-addressed
// store using the origin digest recorded during admission.
func RestoreOriginal(ctx context.Context, origin string) ([]byte, error) {
	if origin == "" {
		return nil, fmt.Errorf("headroom: empty origin digest")
	}
	if res := abi.ActiveResolver(); res != nil {
		if b, err := res.Resolve(ctx, abi.Ref{Kind: abi.RefBlob, Digest: origin}); err == nil {
			return b, nil
		}
	}
	backend, ok := abi.PageOut("blob")
	if !ok {
		return nil, fmt.Errorf("headroom: blob store not available for restore")
	}
	ref, err := backend.PageIn(ctx, abi.Ref{Kind: abi.RefBlob, Digest: origin})
	if err != nil {
		return nil, fmt.Errorf("headroom: restore from blob store: %w", err)
	}
	return ref.Inline, nil
}

func extractMCPContent(envelope []byte) []byte {
	if len(envelope) == 0 || !json.Valid(envelope) {
		return nil
	}
	var env struct {
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(envelope, &env); err == nil && len(env.Content) > 0 {
		return env.Content
	}
	return nil
}

func isMCPEnvelope(data []byte) bool {
	if len(data) == 0 || !json.Valid(data) {
		return false
	}
	var env struct {
		Content           json.RawMessage `json:"content"`
		StructuredContent json.RawMessage `json:"structuredContent"`
	}
	if err := json.Unmarshal(data, &env); err == nil && len(env.Content) > 0 && len(env.StructuredContent) > 0 {
		return true
	}
	return false
}

func replaceContentInEnvelope(envelope, oldContent, newContent []byte) []byte {
	idx := bytes.Index(envelope, oldContent)
	if idx < 0 {
		return envelope
	}
	out := make([]byte, 0, len(envelope)-len(oldContent)+len(newContent))
	out = append(out, envelope[:idx]...)
	out = append(out, newContent...)
	out = append(out, envelope[idx+len(oldContent):]...)
	return out
}
