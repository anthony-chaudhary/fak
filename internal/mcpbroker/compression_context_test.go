package mcpbroker

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// TestStdioCallToolIdentityContext witnesses the transport-level identity context signal.
// It verifies:
// 1. Exact original bytes are preserved when identity context is carried.
// 2. Automatic compaction occurs on independent context without identity signal.
// 3. Operator FAK_COMPRESSOR=noop and FAK_COMPRESSOR=none dominate independent contexts.
// 4. Emitted opt-out receipts are strictly payload-free with unchanged byte accounting.
// 5. Monotonic narrowing: inherited identity context cannot be overridden back to auto.
func TestStdioCallToolIdentityContext(t *testing.T) {
	pretty, compact, wantOriginal, wantCompact := testPrettyAndCompact()

	t.Run("ExactOriginalBytes_IdentityContext", func(t *testing.T) {
		transport, cleanup := setupMockTransport(t, pretty, compact)
		defer cleanup()

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		ctx = WithIdentityContext(ctx)
		if !HasIdentityContext(ctx) {
			t.Fatalf("expected HasIdentityContext to be true")
		}

		resp, err := transport.CallTool(ctx, "get_data", nil)
		if err != nil {
			t.Fatalf("CallTool failed: %v", err)
		}

		if string(resp.Content) != wantOriginal {
			t.Fatalf("expected exact original bytes, got: %s", string(resp.Content))
		}

		rcpt := resp.CompressionReceipt
		if rcpt == nil {
			t.Fatalf("expected non-nil CompressionReceipt")
		}
		if rcpt.Reason != ReasonOptOut {
			t.Errorf("expected reason %q, got %q", ReasonOptOut, rcpt.Reason)
		}
		if rcpt.BytesSaved != 0 {
			t.Errorf("expected BytesSaved == 0, got %d", rcpt.BytesSaved)
		}
		if rcpt.InputBytes != len(wantOriginal) {
			t.Errorf("expected InputBytes == %d, got %d", len(wantOriginal), rcpt.InputBytes)
		}
		if rcpt.OutputBytes != len(wantOriginal) {
			t.Errorf("expected OutputBytes == %d, got %d", len(wantOriginal), rcpt.OutputBytes)
		}
		if rcpt.Stage != CompressionStageIdentity {
			t.Errorf("expected Stage == %q, got %q", CompressionStageIdentity, rcpt.Stage)
		}
		if rcpt.Metadata["decision"] != "skipped" {
			t.Errorf("expected decision=skipped, got %q", rcpt.Metadata["decision"])
		}
		if rcpt.Metadata["reason"] != string(ReasonOptOut) {
			t.Errorf("expected reason metadata=%q, got %q", ReasonOptOut, rcpt.Metadata["reason"])
		}
	})

	t.Run("AutoCompaction_IndependentContext", func(t *testing.T) {
		transport, cleanup := setupMockTransport(t, pretty, compact)
		defer cleanup()

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		if HasIdentityContext(ctx) {
			t.Fatalf("independent context should not have identity context")
		}

		resp, err := transport.CallTool(ctx, "get_data", nil)
		if err != nil {
			t.Fatalf("CallTool failed: %v", err)
		}

		if string(resp.Content) != wantCompact {
			t.Fatalf("expected compacted bytes, got: %s", string(resp.Content))
		}

		rcpt := resp.CompressionReceipt
		if rcpt == nil {
			t.Fatalf("expected non-nil CompressionReceipt")
		}
		if rcpt.Reason != ReasonSaved {
			t.Errorf("expected reason %q, got %q", ReasonSaved, rcpt.Reason)
		}
		if rcpt.BytesSaved <= 0 {
			t.Errorf("expected positive BytesSaved, got %d", rcpt.BytesSaved)
		}
		if rcpt.OutputBytes >= rcpt.InputBytes {
			t.Errorf("expected OutputBytes < InputBytes, got %d >= %d", rcpt.OutputBytes, rcpt.InputBytes)
		}
		if rcpt.Metadata["decision"] != "saved" {
			t.Errorf("expected decision=saved, got %q", rcpt.Metadata["decision"])
		}
	})

	t.Run("OperatorDominance_Noop", func(t *testing.T) {
		t.Setenv("FAK_COMPRESSOR", "noop")

		transport, cleanup := setupMockTransport(t, pretty, compact)
		defer cleanup()

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		resp, err := transport.CallTool(ctx, "get_data", nil)
		if err != nil {
			t.Fatalf("CallTool failed: %v", err)
		}

		if string(resp.Content) != wantOriginal {
			t.Fatalf("operator noop must force exact original bytes, got: %s", string(resp.Content))
		}
		if resp.CompressionReceipt == nil || resp.CompressionReceipt.Reason != ReasonOptOut {
			t.Fatalf("expected ReasonOptOut under FAK_COMPRESSOR=noop, got: %v", resp.CompressionReceipt)
		}
		if resp.CompressionReceipt.BytesSaved != 0 {
			t.Fatalf("expected BytesSaved=0 under operator dominance, got: %d", resp.CompressionReceipt.BytesSaved)
		}
	})

	t.Run("OperatorDominance_None", func(t *testing.T) {
		t.Setenv("FAK_COMPRESSOR", "none")

		transport, cleanup := setupMockTransport(t, pretty, compact)
		defer cleanup()

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		resp, err := transport.CallTool(ctx, "get_data", nil)
		if err != nil {
			t.Fatalf("CallTool failed: %v", err)
		}

		if string(resp.Content) != wantOriginal {
			t.Fatalf("operator none must force exact original bytes, got: %s", string(resp.Content))
		}
		if resp.CompressionReceipt == nil || resp.CompressionReceipt.Reason != ReasonOptOut {
			t.Fatalf("expected ReasonOptOut under FAK_COMPRESSOR=none, got: %v", resp.CompressionReceipt)
		}
		if resp.CompressionReceipt.BytesSaved != 0 {
			t.Fatalf("expected BytesSaved=0 under operator dominance, got: %d", resp.CompressionReceipt.BytesSaved)
		}
	})

	t.Run("PayloadFree_Receipt", func(t *testing.T) {
		transport, cleanup := setupMockTransport(t, pretty, compact)
		defer cleanup()

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		ctx = WithIdentityContext(ctx)
		resp, err := transport.CallTool(ctx, "get_data", nil)
		if err != nil {
			t.Fatalf("CallTool failed: %v", err)
		}

		rcpt := resp.CompressionReceipt
		if rcpt == nil {
			t.Fatalf("expected non-nil receipt")
		}

		assertZeroRawPayloadInMetadata(t, rcpt, pretty, wantOriginal)

		data, err := json.Marshal(rcpt)
		if err != nil {
			t.Fatalf("json.Marshal failed: %v", err)
		}
		serialized := string(data)
		if strings.Contains(serialized, pretty) {
			t.Fatalf("raw payload leaked into serialized receipt JSON: %s", serialized)
		}
		if strings.Contains(serialized, "9007199254740993") {
			t.Fatalf("raw payload tokens leaked into serialized receipt JSON: %s", serialized)
		}
	})

	t.Run("MonotonicNarrowing_InheritedIdentity", func(t *testing.T) {
		transport, cleanup := setupMockTransport(t, pretty, compact)
		defer cleanup()

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		parent := WithIdentityContext(ctx)
		if !HasIdentityContext(parent) {
			t.Fatalf("parent must have identity context")
		}

		// Attempting to override with auto must not widen back to auto
		child1 := WithCompressionPolicy(parent, CompressionAuto)
		if !HasIdentityContext(child1) {
			t.Fatalf("child1 must inherit identity context")
		}

		resp1, err := transport.CallTool(child1, "get_data", nil)
		if err != nil {
			t.Fatalf("CallTool failed: %v", err)
		}
		if string(resp1.Content) != wantOriginal {
			t.Fatalf("inherited identity must not widen to auto, got: %s", string(resp1.Content))
		}

		// Attempting to pass Identity: false on an existing identity context must not clear it
		child2 := WithIdentityContext(parent, IdentityContext{Identity: false})
		if !HasIdentityContext(child2) {
			t.Fatalf("child2 must inherit identity context")
		}

		resp2, err := transport.CallTool(child2, "get_data", nil)
		if err != nil {
			t.Fatalf("CallTool failed: %v", err)
		}
		if string(resp2.Content) != wantOriginal {
			t.Fatalf("inherited identity must remain identity, got: %s", string(resp2.Content))
		}
	})

	t.Run("ContextAliases", func(t *testing.T) {
		ctx1 := WithIdentity(context.Background())
		if !IsIdentityContext(ctx1) {
			t.Errorf("expected IsIdentityContext to be true for WithIdentity")
		}
		sig1, ok1 := IdentityContextFromContext(ctx1)
		if !ok1 || !sig1.Identity {
			t.Errorf("expected valid IdentityContext from WithIdentity, got: %v, ok=%v", sig1, ok1)
		}

		ctx2 := WithCompressionIdentity(context.Background())
		if !HasIdentityContext(ctx2) {
			t.Errorf("expected HasIdentityContext to be true for WithCompressionIdentity")
		}
		sig2, ok2 := IdentityContextFromContext(ctx2)
		if !ok2 || !sig2.Identity {
			t.Errorf("expected valid IdentityContext from WithCompressionIdentity, got: %v, ok=%v", sig2, ok2)
		}

		ctx3 := context.Background()
		if HasIdentityContext(ctx3) {
			t.Errorf("expected HasIdentityContext to be false for Background()")
		}
		_, ok3 := IdentityContextFromContext(ctx3)
		if ok3 {
			t.Errorf("expected ok=false for Background()")
		}
	})
}
