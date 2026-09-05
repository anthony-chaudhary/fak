package gateway

import (
	"context"
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kv"
)

func TestGatewayIslandsWiring(t *testing.T) {
	// Initialize default server
	srv, err := New(Config{})
	if err != nil {
		t.Fatalf("gateway.New failed: %v", err)
	}

	t.Run("CompactDenySummary wired with FormatCompactRefusalNote", func(t *testing.T) {
		adjs := []ToolAdjudication{
			{
				Tool: "rm_rf",
				Verdict: WireVerdict{
					Kind:        "DENY",
					Reason:      "POLICY_BLOCK",
					Disposition: "TERMINAL",
					Detail:      map[string]string{"remedy": "use git clean instead"},
				},
			},
		}

		compact := srv.CompactDenySummary(adjs)
		if !strings.Contains(compact, "[FAK GATE: POLICY_BLOCK]") {
			t.Fatalf("expected [FAK GATE: POLICY_BLOCK] in compact refusal note, got: %q", compact)
		}
		if !strings.Contains(compact, "Next Action:") {
			t.Fatalf("expected Next Action line in compact refusal note, got: %q", compact)
		}
	})

	t.Run("KVStore wired into gateway Server", func(t *testing.T) {
		store := srv.KVStore()
		if store == nil {
			t.Fatal("expected non-nil kv.Store from srv.KVStore()")
		}

		key := kv.CacheKey{
			SessionID:   "session-test-kv",
			Turn:        1,
			Layer:       0,
			TokenOffset: 0,
			NumTokens:   16,
		}

		page, err := store.AllocatePage(key)
		if err != nil {
			t.Fatalf("store.AllocatePage failed: %v", err)
		}
		if page == nil {
			t.Fatal("expected non-nil page")
		}

		data := []byte("tensor-kv-cache-data")
		putPage, err := store.Put(key, data)
		if err != nil {
			t.Fatalf("store.Put failed: %v", err)
		}
		if putPage.BytesUsed != len(data) {
			t.Fatalf("expected %d bytes used, got %d", len(data), putPage.BytesUsed)
		}

		gotPage, err := store.Get(key)
		if err != nil {
			t.Fatalf("store.Get failed: %v", err)
		}
		if string(gotPage.Bytes()) != string(data) {
			t.Fatalf("got data %q, want %q", string(gotPage.Bytes()), string(data))
		}
	})

	t.Run("Negative verification methods wired into gateway Server", func(t *testing.T) {
		// 1. VerifyNominalHostCallbacks
		variants := []ExecutionVariant{
			{StructuredOutput: false, SpeculativeDecoding: false, SkipHostCallback: false},
			{StructuredOutput: true, SpeculativeDecoding: false, SkipHostCallback: false},
		}
		receipt, err := srv.VerifyNominalHostCallbacks(variants, false)
		if err != nil {
			t.Fatalf("VerifyNominalHostCallbacks failed: %v", err)
		}
		if receipt.ModesEvaluated != 2 {
			t.Fatalf("expected 2 modes evaluated, got %d", receipt.ModesEvaluated)
		}

		// 2. ApplyGrammarMask
		logits := []float32{1.0, 2.0, float32(math.Inf(-1)), 3.0}
		grammarAllowed := map[int]bool{0: true, 2: true}
		maskReceipt, err := srv.ApplyGrammarMask(logits, grammarAllowed)
		if err != nil {
			t.Fatalf("ApplyGrammarMask failed: %v", err)
		}
		if !maskReceipt.DomainMonotonicityOK {
			t.Fatal("expected domain monotonicity to be OK")
		}

		// 3. EvaluateSpeculativeRunaway
		prompts := []string{"repeat this phrase endlessly", "simple prompt"}
		runawayReceipt, err := srv.EvaluateSpeculativeRunaway(prompts, 30)
		if err != nil {
			t.Fatalf("EvaluateSpeculativeRunaway failed: %v", err)
		}
		if runawayReceipt.PromptsEvaluated != 2 {
			t.Fatalf("expected 2 prompts evaluated, got %d", runawayReceipt.PromptsEvaluated)
		}
	})
	_ = context.Background()
}
