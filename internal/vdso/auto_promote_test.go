package vdso

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func inlineRef(b []byte) abi.Ref {
	return abi.Ref{Kind: abi.RefInline, Inline: b, Len: int64(len(b))}
}

// TestVDSO_AutoPromoteIdempotencyReceipts is the primary witness test required by #12525.
// It verifies that an unhinted custom tool call executes cold on turn 1, registers an
// idempotency receipt on r.Meta, and serves turn 2 as a zero-latency vdso_hits cache hit
// without invoking the underlying engine. It also verifies that subsequent state drift
// or mutation invalidates the dynamic promotion.
func TestVDSO_AutoPromoteIdempotencyReceipts(t *testing.T) {
	ctx := context.Background()
	v := New(100)

	call := &abi.ToolCall{
		Tool: "custom_mcp_query",
		Args: inlineRef([]byte(`{"query":"cluster_status"}`)),
	}

	// Baseline: custom_mcp_query is unhinted and not promoted.
	if v.IsPromotedReadOnly("custom_mcp_query") {
		t.Fatal("expected custom_mcp_query to NOT be promoted initially")
	}

	// Turn 1: Lookup must miss (cold execution).
	res1, ok1 := v.Lookup(ctx, call)
	if ok1 || res1 != nil {
		t.Fatalf("turn 1: expected cold miss, got ok=%v, res=%v", ok1, res1)
	}

	// Engine executes the tool and produces an idempotency receipt on r.Meta.
	turn1Result := &abi.Result{
		Call:    call,
		Status:  abi.StatusOK,
		Payload: inlineRef([]byte(`{"nodes":3,"status":"healthy"}`)),
		Meta: map[string]string{
			"read_only": "true",
			"outcome":   "verified_fresh_reuse",
		},
	}

	// vDSO observes completion via Emit (or StoreResult).
	v.Emit(abi.Event{
		Kind:   abi.EvComplete,
		Call:   call,
		Result: turn1Result,
	})

	// After observing the verified receipt, the tool signature must be auto-promoted.
	if !v.IsPromotedReadOnly("custom_mcp_query") {
		t.Fatal("expected custom_mcp_query to be auto-promoted to read-only")
	}

	// Turn 2: Identical unhinted call must now hit Tier-2 vDSO cache!
	res2, ok2 := v.Lookup(ctx, call)
	if !ok2 || res2 == nil {
		t.Fatalf("turn 2: expected cache hit, got ok=%v, res=%v", ok2, res2)
	}
	if res2.Meta["served_by"] != "vdso" {
		t.Fatalf("turn 2: served_by=%q, want \"vdso\"", res2.Meta["served_by"])
	}
	if res2.Meta["tier"] != "2" {
		t.Fatalf("turn 2: tier=%q, want \"2\"", res2.Meta["tier"])
	}
	if string(res2.Payload.Inline) != `{"nodes":3,"status":"healthy"}` {
		t.Fatalf("turn 2 payload = %s, want original output", string(res2.Payload.Inline))
	}

	_, hits, _, _ := v.Stats()
	if hits != 1 {
		t.Fatalf("expected hits=1, got %d", hits)
	}

	// Turn 3: Observe state drift / mutation on the tool.
	// A subsequent call or event observing mutation/drift must invalidate the promotion.
	mutEvent := abi.Event{
		Kind: abi.EvComplete,
		Call: call,
		Result: &abi.Result{
			Call:   call,
			Status: abi.StatusOK,
			Meta: map[string]string{
				"state_drift": "true",
			},
		},
	}
	v.Emit(mutEvent)

	if v.IsPromotedReadOnly("custom_mcp_query") {
		t.Fatal("expected custom_mcp_query to be invalidated after state drift observation")
	}
}

func TestVDSO_AutoPromoteReceiptVariants(t *testing.T) {
	ctx := context.Background()

	cases := []struct {
		name string
		meta map[string]string
		want bool
	}{
		{"read_only_true", map[string]string{"read_only": "true"}, true},
		{"readOnly_true", map[string]string{"readOnly": "true"}, true},
		{"idempotent_true", map[string]string{"idempotent": "true"}, true},
		{"idempotency_true", map[string]string{"idempotency": "true"}, true},
		{"outcome_verified_fresh_reuse", map[string]string{"outcome": "verified_fresh_reuse"}, true},
		{"outcome_executed_cold_read", map[string]string{"outcome": "executed_cold_read"}, true},
		{"no_meta", nil, false},
		{"empty_meta", map[string]string{}, false},
		{"read_only_false", map[string]string{"read_only": "false"}, false},
		{"idempotent_false", map[string]string{"idempotent": "false"}, false},
		{"arbitrary_meta", map[string]string{"foo": "bar"}, false},
	}

	for i, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := New(50)
			toolName := fmt.Sprintf("mcp_sample_tool_%d", i)
			call := &abi.ToolCall{
				Tool: toolName,
				Args: inlineRef([]byte(`{"x":1}`)),
			}
			result := &abi.Result{
				Call:    call,
				Status:  abi.StatusOK,
				Payload: inlineRef([]byte(`{"result":42}`)),
				Meta:    tc.meta,
			}

			_, _ = v.StoreResult(ctx, call, result)

			got := v.IsPromotedReadOnly(toolName)
			if got != tc.want {
				t.Fatalf("tool %q: IsPromotedReadOnly=%v, want %v", toolName, got, tc.want)
			}

			// If promoted, verify lookup hits. If not, verify lookup misses.
			_, hit := v.Lookup(ctx, call)
			if hit != tc.want {
				t.Fatalf("tool %q: Lookup hit=%v, want %v", toolName, hit, tc.want)
			}
		})
	}
}

func TestVDSO_AutoPromoteDestructiveInvalidation(t *testing.T) {
	v := New(50)
	toolName := "mcp_data_service"

	readCall := &abi.ToolCall{
		Tool: toolName,
		Args: inlineRef([]byte(`{"op":"read"}`)),
	}
	readResult := &abi.Result{
		Call:    readCall,
		Status:  abi.StatusOK,
		Payload: inlineRef([]byte(`{"data":"abc"}`)),
		Meta:    map[string]string{"read_only": "true"},
	}

	// Turn 1 promotes tool
	v.Emit(abi.Event{Kind: abi.EvComplete, Call: readCall, Result: readResult})
	if !v.IsPromotedReadOnly(toolName) {
		t.Fatal("expected tool to be promoted")
	}

	// Turn 2: tool is called destructively
	writeCall := &abi.ToolCall{
		Tool: toolName,
		Args: inlineRef([]byte(`{"op":"mutate","val":"new"}`)),
		Meta: map[string]string{"destructive": "true"},
	}
	writeResult := &abi.Result{
		Call:    writeCall,
		Status:  abi.StatusOK,
		Payload: inlineRef([]byte(`{"status":"ok"}`)),
	}

	v.Emit(abi.Event{Kind: abi.EvComplete, Call: writeCall, Result: writeResult})

	// Promotion must be invalidated by the destructive execution
	if v.IsPromotedReadOnly(toolName) {
		t.Fatal("expected tool to be invalidated after destructive execution")
	}
}

func TestVDSO_AutoPromoteRegistryAPI(t *testing.T) {
	reg := NewPromotedRegistry()
	if reg.Len() != 0 {
		t.Fatalf("expected empty registry, got %d", reg.Len())
	}

	reg.Promote("alpha")
	reg.Promote("beta")
	reg.Promote("gamma")

	if !reg.IsPromoted("alpha") || !reg.IsPromoted("beta") || !reg.IsPromoted("gamma") {
		t.Fatal("expected alpha, beta, gamma to be promoted")
	}
	if reg.IsPromoted("delta") {
		t.Fatal("delta should not be promoted")
	}
	if reg.Len() != 3 {
		t.Fatalf("expected len 3, got %d", reg.Len())
	}

	tools := reg.Tools()
	if len(tools) != 3 || tools[0] != "alpha" || tools[1] != "beta" || tools[2] != "gamma" {
		t.Fatalf("unexpected tools list: %v", tools)
	}

	reg.Invalidate("beta")
	if reg.IsPromoted("beta") {
		t.Fatal("beta should be invalidated")
	}
	if reg.Len() != 2 {
		t.Fatalf("expected len 2, got %d", reg.Len())
	}

	reg.InvalidateAll()
	if reg.Len() != 0 {
		t.Fatalf("expected len 0 after InvalidateAll, got %d", reg.Len())
	}
}

func TestVDSO_AutoPromoteConcurrencyRace(t *testing.T) {
	ctx := context.Background()
	v := New(200)

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			tool := fmt.Sprintf("dynamic_tool_%d", workerID%5)
			call := &abi.ToolCall{
				Tool: tool,
				Args: inlineRef([]byte(fmt.Sprintf(`{"worker":%d}`, workerID))),
			}

			// Turn 1: lookup cold
			v.Lookup(ctx, call)

			// Store / promote
			res := &abi.Result{
				Call:    call,
				Status:  abi.StatusOK,
				Payload: inlineRef([]byte(fmt.Sprintf(`{"res":%d}`, workerID))),
				Meta:    map[string]string{"read_only": "true"},
			}
			_, _ = v.StoreResult(ctx, call, res)

			// Turn 2: lookup warm
			v.Lookup(ctx, call)

			if workerID%3 == 0 {
				v.InvalidatePromoted(tool)
			}
		}(i)
	}
	wg.Wait()
}
