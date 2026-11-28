package vdso

import (
	"bytes"
	"context"
	"encoding/json"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// PrefetchExecutor executes an admitted speculatable tool call to warm the
// Tier-2 vDSO cache before the model stream finishes.
type PrefetchExecutor func(ctx context.Context, call *abi.ToolCall) (*abi.Result, error)

// StreamToolDelta represents one incremental tool-call fragment emitted during
// a streaming model response (prior to finish_reason=tool_calls).
type StreamToolDelta struct {
	Index        int               `json:"index"`
	ID           string            `json:"id,omitempty"`
	Tool         string            `json:"tool,omitempty"`
	Name         string            `json:"name,omitempty"`
	ArgsFragment string            `json:"args_fragment,omitempty"`
	Meta         map[string]string `json:"meta,omitempty"`
	FinishReason string            `json:"finish_reason,omitempty"`
}

// PrefetchReceipt records the deterministic admission and Tier-2 warm-prefetch
// outcome for a single streamed tool-call slot.
type PrefetchReceipt struct {
	Index        int        `json:"index"`
	ID           string     `json:"id,omitempty"`
	Tool         string     `json:"tool"`
	Args         []byte     `json:"args,omitempty"`
	Reason       SpecReason `json:"reason"`
	Admitted     bool       `json:"admitted"`
	Warmed       bool       `json:"warmed"`
	CacheHit     bool       `json:"cache_hit"`
	WorldVersion uint64     `json:"world_version"`
	Error        string     `json:"error,omitempty"`
}

type streamSlot struct {
	index     int
	id        string
	tool      string
	argsBuf   bytes.Buffer
	meta      map[string]string
	evaluated bool
	receipt   PrefetchReceipt
}

// StreamingPrefetcher deterministically recognizes complete JSON tool-call
// fragments mid-stream, gates them through Speculatable() (default-deny on
// effects), and warms the Tier-2 vDSO cache before finish_reason=tool_calls.
type StreamingPrefetcher struct {
	mu           sync.Mutex
	vdso         *VDSO
	exec         PrefetchExecutor
	slots        map[int]*streamSlot
	slotOrder    []int
	finishReason string
}

// NewStreamingPrefetcher constructs a StreamingPrefetcher bound to v (or Default
// if v is nil) and the provided executor.
func NewStreamingPrefetcher(v *VDSO, exec PrefetchExecutor) *StreamingPrefetcher {
	if v == nil {
		v = Default
	}
	return &StreamingPrefetcher{
		vdso:  v,
		exec:  exec,
		slots: make(map[int]*streamSlot),
	}
}

// Feed ingests one incremental stream delta. As soon as a slot's accumulated
// JSON arguments form a complete JSON object and a tool name is known, the call
// is gated by Speculatable() and, if SpecOK, warmed into Tier-2 vDSO.
func (sp *StreamingPrefetcher) Feed(ctx context.Context, delta StreamToolDelta) (PrefetchReceipt, bool) {
	if sp == nil {
		return PrefetchReceipt{}, false
	}

	var (
		readyCall *abi.ToolCall
		slotID    string
		slotTool  string
		argsBytes []byte
		slotIdx   = delta.Index
	)

	sp.mu.Lock()
	if delta.FinishReason != "" {
		sp.finishReason = delta.FinishReason
	}
	slot, exists := sp.slots[slotIdx]
	if !exists {
		slot = &streamSlot{
			index: slotIdx,
			meta:  make(map[string]string),
		}
		sp.slots[slotIdx] = slot
		sp.slotOrder = append(sp.slotOrder, slotIdx)
	}
	if delta.ID != "" {
		slot.id = delta.ID
	}
	if toolName := firstNonEmpty(delta.Tool, delta.Name); toolName != "" {
		slot.tool = toolName
	}
	for k, val := range delta.Meta {
		if slot.meta == nil {
			slot.meta = make(map[string]string)
		}
		slot.meta[k] = val
	}
	if delta.ArgsFragment != "" {
		slot.argsBuf.WriteString(delta.ArgsFragment)
	}

	if slot.evaluated {
		rec := slot.receipt
		sp.mu.Unlock()
		return rec, true
	}

	if slot.tool == "" || !isCompleteJSONObject(slot.argsBuf.Bytes()) {
		sp.mu.Unlock()
		return PrefetchReceipt{}, false
	}

	// Mark evaluated before releasing mu so concurrent/duplicate feeds never
	// double-dispatch the same slot.
	slot.evaluated = true
	slotID = slot.id
	slotTool = slot.tool
	argsBytes = append([]byte(nil), bytes.TrimSpace(slot.argsBuf.Bytes())...)
	metaCopy := make(map[string]string, len(slot.meta))
	for k, val := range slot.meta {
		metaCopy[k] = val
	}
	readyCall = &abi.ToolCall{
		Tool: slotTool,
		Args: abi.Ref{
			Kind:   abi.RefInline,
			Inline: argsBytes,
			Len:    int64(len(argsBytes)),
		},
		Meta: metaCopy,
	}
	sp.mu.Unlock()

	rec := sp.evaluateAndWarm(ctx, slotIdx, slotID, readyCall, argsBytes)

	sp.mu.Lock()
	if s, ok := sp.slots[slotIdx]; ok {
		s.receipt = rec
	}
	sp.mu.Unlock()

	return rec, true
}

func (sp *StreamingPrefetcher) evaluateAndWarm(
	ctx context.Context,
	index int,
	id string,
	call *abi.ToolCall,
	argsBytes []byte,
) PrefetchReceipt {
	reason, ok := Speculatable(call)
	rec := PrefetchReceipt{
		Index:        index,
		ID:           id,
		Tool:         call.Tool,
		Args:         argsBytes,
		Reason:       reason,
		Admitted:     ok,
		WorldVersion: sp.vdso.WorldVersion(),
	}
	if !ok {
		return rec
	}

	if _, hit := sp.vdso.Lookup(ctx, call); hit {
		rec.CacheHit = true
		return rec
	}

	if sp.exec == nil {
		rec.Error = "no prefetch executor configured"
		return rec
	}

	res, err := sp.exec(ctx, call)
	if err != nil {
		rec.Error = err.Error()
		return rec
	}
	if res == nil || res.Status != abi.StatusOK {
		rec.Error = "prefetch execution did not return StatusOK"
		return rec
	}

	sp.vdso.Emit(abi.Event{
		Kind:   abi.EvComplete,
		Call:   call,
		Result: res,
	})
	rec.Warmed = true
	rec.WorldVersion = sp.vdso.WorldVersion()
	return rec
}

// FinishReason returns the last non-empty finish_reason observed from the stream.
func (sp *StreamingPrefetcher) FinishReason() string {
	if sp == nil {
		return ""
	}
	sp.mu.Lock()
	defer sp.mu.Unlock()
	return sp.finishReason
}

// Receipt returns the recorded receipt for a given stream slot index, if evaluated.
func (sp *StreamingPrefetcher) Receipt(index int) (PrefetchReceipt, bool) {
	if sp == nil {
		return PrefetchReceipt{}, false
	}
	sp.mu.Lock()
	defer sp.mu.Unlock()
	s, ok := sp.slots[index]
	if !ok || !s.evaluated {
		return PrefetchReceipt{}, false
	}
	return s.receipt, true
}

// Receipts returns all evaluated slot receipts in first-seen stream order.
func (sp *StreamingPrefetcher) Receipts() []PrefetchReceipt {
	if sp == nil {
		return nil
	}
	sp.mu.Lock()
	defer sp.mu.Unlock()
	out := make([]PrefetchReceipt, 0, len(sp.slotOrder))
	for _, idx := range sp.slotOrder {
		if s := sp.slots[idx]; s != nil && s.evaluated {
			out = append(out, s.receipt)
		}
	}
	return out
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// isCompleteJSONObject reports whether b (after trimming whitespace) is a single
// balanced JSON object {...} and passes json.Valid.
func isCompleteJSONObject(b []byte) bool {
	trimmed := bytes.TrimSpace(b)
	if len(trimmed) < 2 || trimmed[0] != '{' || trimmed[len(trimmed)-1] != '}' {
		return false
	}
	depth := 0
	inString := false
	escaped := false
	for i, ch := range trimmed {
		if inString {
			if escaped {
				escaped = false
				continue
			}
			if ch == '\\' {
				escaped = true
				continue
			}
			if ch == '"' {
				inString = false
			}
			continue
		}
		switch ch {
		case '"':
			inString = true
		case '{':
			depth++
		case '}':
			depth--
			if depth < 0 {
				return false
			}
			if depth == 0 && i != len(trimmed)-1 {
				return false
			}
		}
	}
	if depth != 0 || inString {
		return false
	}
	return json.Valid(trimmed)
}
