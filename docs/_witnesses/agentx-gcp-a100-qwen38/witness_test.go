package agentxgcpa100witness_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agentxbench"
)

const pinnedReceiptSHA256 = "1af84415c89f94df31214b0970a9a95befc0c9e2afbea9febf40a5ca1e74ae9d"

func TestAgentXReceiptIntegrity(t *testing.T) {
	data, err := os.ReadFile("receipt.json")
	if err != nil {
		t.Fatalf("failed to read receipt.json: %v", err)
	}

	sum := sha256.Sum256(data)
	gotHash := hex.EncodeToString(sum[:])
	if gotHash != pinnedReceiptSHA256 {
		t.Fatalf("hash mismatch: got %s, want %s", gotHash, pinnedReceiptSHA256)
	}

	var receipt agentxbench.AgentXReceipt
	if err := json.Unmarshal(data, &receipt); err != nil {
		t.Fatalf("failed to parse receipt: %v", err)
	}

	if receipt.Schema != agentxbench.SchemaIdentifier {
		t.Fatalf("unexpected schema: %s", receipt.Schema)
	}
	if receipt.ValidationStatus != "VERIFIED_PASS" {
		t.Fatalf("expected VERIFIED_PASS, got %s", receipt.ValidationStatus)
	}
	if receipt.Aggregated.SuccessRate != 1.0 {
		t.Fatalf("expected 1.0 success rate, got %f", receipt.Aggregated.SuccessRate)
	}
	if receipt.Aggregated.PrefixSpeedupRatio < 4.0 {
		t.Fatalf("expected prefix speedup >= 4.0x, got %f", receipt.Aggregated.PrefixSpeedupRatio)
	}
	if receipt.Aggregated.NormalizedInteractivity < 50.0 {
		t.Fatalf("expected normalized interactivity >= 50 tok/s, got %f", receipt.Aggregated.NormalizedInteractivity)
	}

	errs := agentxbench.ValidateReceipt(&receipt)
	if len(errs) > 0 {
		t.Fatalf("expected 0 validation errors, got %d: %v", len(errs), errs)
	}
}
