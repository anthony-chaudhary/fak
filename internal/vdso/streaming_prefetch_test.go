package vdso

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// fak-test:runtime fast est=100ms
func TestStreamingToolPrefetch(t *testing.T) {
	ctx := context.Background()

	t.Run("read_only_warms_tier2_before_stream_end", func(t *testing.T) {
		v := New(16)
		var execCalls int64

		sp := NewStreamingPrefetcher(v, func(_ context.Context, call *abi.ToolCall) (*abi.Result, error) {
			atomic.AddInt64(&execCalls, 1)
			return &abi.Result{
				Call:    call,
				Status:  abi.StatusOK,
				Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"status":"ok","doc":"prefetched"}`)},
			}, nil
		})

		// Fragment 1: tool name + partial JSON args -> must not execute yet.
		if _, ready := sp.Feed(ctx, StreamToolDelta{
			Index:        0,
			ID:           "call_ro_1",
			Tool:         "fetch_doc",
			ArgsFragment: `{"path":`,
			Meta: map[string]string{
				"readOnlyHint":   "true",
				"idempotentHint": "true",
			},
		}); ready {
			t.Fatalf("partial JSON fragment should not trigger prefetch")
		}
		if got := atomic.LoadInt64(&execCalls); got != 0 {
			t.Fatalf("execCalls after partial fragment = %d, want 0", got)
		}

		// Fragment 2: completes JSON object BEFORE finish_reason=tool_calls.
		rec, ready := sp.Feed(ctx, StreamToolDelta{
			Index:        0,
			ArgsFragment: `"docs/spec.md"}`,
		})
		if !ready {
			t.Fatalf("complete JSON fragment should trigger evaluation")
		}
		if rec.Reason != SpecOK || !rec.Admitted || !rec.Warmed {
			t.Fatalf("unexpected prefetch receipt: %+v", rec)
		}
		if got := atomic.LoadInt64(&execCalls); got != 1 {
			t.Fatalf("execCalls before stream end = %d, want 1", got)
		}
		if sp.FinishReason() != "" {
			t.Fatalf("expected empty finish_reason before stream end, got %q", sp.FinishReason())
		}

		// Verify Tier-2 cache is ALREADY warm before finish_reason=tool_calls arrives.
		probeCall := &abi.ToolCall{
			Tool: "fetch_doc",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"path":"docs/spec.md"}`)},
			Meta: map[string]string{
				"readOnlyHint":   "true",
				"idempotentHint": "true",
			},
		}
		res, hit := v.Lookup(ctx, probeCall)
		if !hit || res == nil {
			t.Fatalf("expected Tier-2 vDSO cache hit before finish_reason=tool_calls")
		}
		if res.Meta["served_by"] != "vdso" || res.Meta["tier"] != "2" {
			t.Fatalf("expected tier-2 vdso serve, got meta=%v", res.Meta)
		}

		// Fragment 3: stream finishes with finish_reason=tool_calls; no duplicate exec.
		sp.Feed(ctx, StreamToolDelta{
			Index:        0,
			FinishReason: "tool_calls",
		})
		if sp.FinishReason() != "tool_calls" {
			t.Fatalf("FinishReason() = %q, want tool_calls", sp.FinishReason())
		}
		if got := atomic.LoadInt64(&execCalls); got != 1 {
			t.Fatalf("execCalls after stream end = %d, want 1", got)
		}
	})

	t.Run("write_destructive_and_unstamped_never_execute", func(t *testing.T) {
		v := New(16)
		var execCalls int64

		sp := NewStreamingPrefetcher(v, func(_ context.Context, call *abi.ToolCall) (*abi.Result, error) {
			atomic.AddInt64(&execCalls, 1)
			return &abi.Result{
				Call:    call,
				Status:  abi.StatusOK,
				Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)},
			}, nil
		})

		cases := []struct {
			name       string
			index      int
			tool       string
			args       string
			meta       map[string]string
			wantReason SpecReason
		}{
			{
				name:       "unstamped_no_hints",
				index:      0,
				tool:       "fetch_doc",
				args:       `{"path":"a.md"}`,
				meta:       nil,
				wantReason: SpecRefusedNotReadOnly,
			},
			{
				name:       "readonly_without_idempotent",
				index:      1,
				tool:       "fetch_doc",
				args:       `{"path":"b.md"}`,
				meta:       map[string]string{"readOnlyHint": "true"},
				wantReason: SpecRefusedNotIdempotent,
			},
			{
				name:       "write_shaped_with_ro_hints",
				index:      2,
				tool:       "write_file",
				args:       `{"path":"c.md","content":"x"}`,
				meta:       map[string]string{"readOnlyHint": "true", "idempotentHint": "true"},
				wantReason: SpecRefusedDestructive,
			},
			{
				name:       "explicit_destructive_hint",
				index:      3,
				tool:       "fetch_doc",
				args:       `{"path":"d.md"}`,
				meta:       map[string]string{"readOnlyHint": "true", "idempotentHint": "true", "destructive": "true"},
				wantReason: SpecRefusedDestructive,
			},
			{
				name:       "non_readonly_bash",
				index:      4,
				tool:       "Bash",
				args:       `{"command":"rm -rf /tmp/foo"}`,
				meta:       nil,
				wantReason: SpecRefusedNotReadOnly,
			},
		}

		for _, tc := range cases {
			rec, ready := sp.Feed(ctx, StreamToolDelta{
				Index:        tc.index,
				Tool:         tc.tool,
				ArgsFragment: tc.args,
				Meta:         tc.meta,
			})
			if !ready {
				t.Fatalf("%s: expected complete fragment to be evaluated", tc.name)
			}
			if rec.Admitted || rec.Warmed {
				t.Fatalf("%s: expected refusal, got admitted=%v warmed=%v", tc.name, rec.Admitted, rec.Warmed)
			}
			if rec.Reason != tc.wantReason || !rec.Reason.Refused() {
				t.Fatalf("%s: reason = %s, want %s", tc.name, rec.Reason, tc.wantReason)
			}
		}

		if got := atomic.LoadInt64(&execCalls); got != 0 {
			t.Fatalf("refused calls executed %d times, want 0", got)
		}
	})

	t.Run("world_version_mutation_invalidates_prefetched_entries", func(t *testing.T) {
		v := New(16)
		sp := NewStreamingPrefetcher(v, func(_ context.Context, call *abi.ToolCall) (*abi.Result, error) {
			return &abi.Result{
				Call:    call,
				Status:  abi.StatusOK,
				Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"val":42}`)},
			}, nil
		})

		rec, ready := sp.Feed(ctx, StreamToolDelta{
			Index:        0,
			Tool:         "read_state",
			ArgsFragment: `{"key":"k1"}`,
			Meta: map[string]string{
				"readOnlyHint":   "true",
				"idempotentHint": "true",
			},
		})
		if !ready || !rec.Warmed {
			t.Fatalf("expected warmed prefetch, got ready=%v rec=%+v", ready, rec)
		}
		vBefore := v.WorldVersion()

		probeCall := &abi.ToolCall{
			Tool: "read_state",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"key":"k1"}`)},
			Meta: map[string]string{
				"readOnlyHint":   "true",
				"idempotentHint": "true",
			},
		}
		if _, hit := v.Lookup(ctx, probeCall); !hit {
			t.Fatalf("expected Tier-2 cache hit before world-version mutation")
		}

		// Simulate a write completion that bumps world-version.
		writeCall := &abi.ToolCall{
			Tool: "write_state",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"key":"k1","val":99}`)},
		}
		v.Emit(abi.Event{
			Kind: abi.EvComplete,
			Call: writeCall,
			Result: &abi.Result{
				Call:   writeCall,
				Status: abi.StatusOK,
			},
		})

		if vAfter := v.WorldVersion(); vAfter <= vBefore {
			t.Fatalf("expected world version to advance (%d -> %d)", vBefore, vAfter)
		}
		if _, hit := v.Lookup(ctx, probeCall); hit {
			t.Fatalf("expected Tier-2 prefetched entry to be invalidated after world-version mutation")
		}
	})
}
