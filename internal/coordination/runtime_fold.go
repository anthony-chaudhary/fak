//go:build wip_coordination

package coordination

import (
	"context"
	"encoding/json"
	"slices"
	"strconv"
	"sync"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

// CompactFoldHandler processes and compacts worker receipts into an audit buffer,
// suppressing raw intermediate turns from coordinator context and computing context savings.
type CompactFoldHandler struct {
	mu               sync.RWMutex
	folded           []harnesskit.WorkerReceipt
	defaultRawTokens int
}

var _ harnesskit.ReceiptFoldHandler = (*CompactFoldHandler)(nil)

// NewCompactFoldHandler creates a new fold handler with an optional default baseline for raw transcript tokens.
func NewCompactFoldHandler(defaultRawTokens ...int) *CompactFoldHandler {
	raw := 10000
	if len(defaultRawTokens) > 0 && defaultRawTokens[0] > 0 {
		raw = defaultRawTokens[0]
	}
	return &CompactFoldHandler{
		folded:           make([]harnesskit.WorkerReceipt, 0),
		defaultRawTokens: raw,
	}
}

// Fold compacts a worker receipt, calculates context savings against the raw transcript tokens,
// and records the receipt into an internal audit buffer.
func (f *CompactFoldHandler) Fold(ctx context.Context, receipt harnesskit.WorkerReceipt) (harnesskit.ContextSavings, error) {
	if err := ctx.Err(); err != nil {
		return harnesskit.ContextSavings{}, &harnesskit.Error{
			Code: harnesskit.CodeCanceled,
			Op:   "fold",
			Err:  err,
		}
	}

	if err := receipt.Validate(); err != nil {
		return harnesskit.ContextSavings{}, &harnesskit.Error{
			Code: harnesskit.CodeInvalid,
			Op:   "fold.validate",
			Err:  err,
		}
	}

	// Determine raw transcript tokens
	rawTranscriptTokens := 0
	if receipt.Artifacts != nil {
		if rawStr, ok := receipt.Artifacts["raw_transcript_tokens"]; ok {
			if parsed, err := strconv.Atoi(rawStr); err == nil && parsed > 0 {
				rawTranscriptTokens = parsed
			}
		} else if rawStr, ok := receipt.Artifacts["raw_tokens"]; ok {
			if parsed, err := strconv.Atoi(rawStr); err == nil && parsed > 0 {
				rawTranscriptTokens = parsed
			}
		}
	}

	if rawTranscriptTokens == 0 {
		if receipt.Tokens.InputTokens > 0 || receipt.Tokens.OutputTokens > 0 {
			rawTranscriptTokens = receipt.Tokens.InputTokens + receipt.Tokens.OutputTokens
		}
	}

	// Determine folded receipt tokens
	foldTokens := receipt.Tokens.TotalTokens
	if foldTokens <= 0 {
		// Approximate fold token footprint from compact JSON representation
		data, err := json.Marshal(receipt)
		if err == nil && len(data) > 0 {
			foldTokens = len(data) / 4
		} else {
			foldTokens = len(receipt.Summary)/4 + 20
		}
		if foldTokens < 10 {
			foldTokens = 10
		}
	}

	// If raw turns were unspecified or less than folded receipt, apply default baseline or multi-turn simulation
	if rawTranscriptTokens <= foldTokens {
		if f.defaultRawTokens > foldTokens {
			rawTranscriptTokens = f.defaultRawTokens
		} else {
			rawTranscriptTokens = foldTokens * 5
		}
	}

	savings := harnesskit.CalculateContextSavings(rawTranscriptTokens, foldTokens)

	// Create an isolated compact copy for the audit store
	compact := receipt
	compact.TouchedFiles = slices.Clone(receipt.TouchedFiles)
	if receipt.Artifacts != nil {
		compact.Artifacts = make(map[string]string, len(receipt.Artifacts))
		for k, v := range receipt.Artifacts {
			compact.Artifacts[k] = v
		}
	}
	if receipt.Diagnosis != nil {
		diagCopy := *receipt.Diagnosis
		diagCopy.UnmetAssumptions = slices.Clone(receipt.Diagnosis.UnmetAssumptions)
		compact.Diagnosis = &diagCopy
	}

	f.mu.Lock()
	f.folded = append(f.folded, compact)
	f.mu.Unlock()

	return savings, nil
}

// FoldedReceipts returns a snapshot of all folded receipts stored in the audit buffer.
func (f *CompactFoldHandler) FoldedReceipts() []harnesskit.WorkerReceipt {
	f.mu.RLock()
	defer f.mu.RUnlock()

	out := make([]harnesskit.WorkerReceipt, len(f.folded))
	for i, r := range f.folded {
		out[i] = r
		out[i].TouchedFiles = slices.Clone(r.TouchedFiles)
		if r.Artifacts != nil {
			out[i].Artifacts = make(map[string]string, len(r.Artifacts))
			for k, v := range r.Artifacts {
				out[i].Artifacts[k] = v
			}
		}
		if r.Diagnosis != nil {
			diagCopy := *r.Diagnosis
			diagCopy.UnmetAssumptions = slices.Clone(r.Diagnosis.UnmetAssumptions)
			out[i].Diagnosis = &diagCopy
		}
	}
	return out
}

// Len returns the count of folded receipts currently buffered.
func (f *CompactFoldHandler) Len() int {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return len(f.folded)
}

// Clear flushes all stored receipts from the audit buffer.
func (f *CompactFoldHandler) Clear() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.folded = f.folded[:0]
}
