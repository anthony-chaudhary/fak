package agentxgcpa100witness_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agentxbench"
)

const pinnedReceiptSHA256 = "1af84415c89f94df31214b0970a9a95befc0c9e2afbea9febf40a5ca1e74ae9d"

// TestAgentXReceiptIsRetracted pins the retracted receipt and proves why it is
// not evidence: it is self-check output that carries no execution mode.
func TestAgentXReceiptIsRetracted(t *testing.T) {
	data, err := os.ReadFile("receipt.json")
	if err != nil {
		t.Fatalf("failed to read receipt.json: %v", err)
	}

	sum := sha256.Sum256(data)
	if got := hex.EncodeToString(sum[:]); got != pinnedReceiptSHA256 {
		t.Fatalf("hash mismatch: got %s, want %s", got, pinnedReceiptSHA256)
	}

	var receipt agentxbench.AgentXReceipt
	if err := json.Unmarshal(data, &receipt); err != nil {
		t.Fatalf("failed to parse receipt: %v", err)
	}

	for _, req := range receipt.Requests {
		if req.ResponseText != agentxbench.MockResponseText {
			t.Fatalf("%s: response %q is not the mock sentence; the retraction premise no longer holds", req.RequestID, req.ResponseText)
		}
		if want := 1024 + 256*req.TurnIndex; req.PromptTokens != want {
			t.Fatalf("%s: prompt_tokens %d, mock formula gives %d", req.RequestID, req.PromptTokens, want)
		}
	}

	errs := agentxbench.EvidenceErrors(&receipt)
	if !hasCode(errs, "EXECUTION_MODE_MISSING") {
		t.Fatalf("retracted receipt must be refused for its missing execution mode, got %v", errs)
	}
}

func hasCode(errs []string, code string) bool {
	for _, e := range errs {
		if strings.HasPrefix(e, code+":") {
			return true
		}
	}
	return false
}
