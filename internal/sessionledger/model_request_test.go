package sessionledger

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/promptaudit"
)

func TestModelRequestReceiptReconstructsAfterReopenAndDetectsDivergence(t *testing.T) {
	dir := t.TempDir()
	ledger, err := Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	largeSystem := strings.Repeat("immutable-system-byte-", MaxContentBytes/8)
	largeSystemJSON, err := json.Marshal(map[string]string{"role": "system", "content": largeSystem})
	if err != nil {
		t.Fatalf("marshal large system segment: %v", err)
	}
	if len(largeSystemJSON) <= MaxContentBytes {
		t.Fatalf("large fixture = %d bytes, want > MaxContentBytes", len(largeSystemJSON))
	}
	wire := ModelRequest{
		Identity: ModelRequestIdentity{
			RequestID: "native-receipt/turn/1",
			Model:     "fixture-model",
			Turn:      1,
			MaxTokens: 256,
		},
		Segments: []ModelRequestSegment{
			{Kind: "system", Source: promptaudit.SourceFakPolicy, Content: largeSystemJSON},
			{Kind: "user_input", Source: promptaudit.SourceUserConfig, Content: json.RawMessage(`{"role":"user","content":"look up order A-1"}`)},
			{Kind: "injected_directive", Source: promptaudit.SourceUserConfig, Content: json.RawMessage(`{"role":"user","content":"switch to the receipt plan"}`)},
		},
		Tools: json.RawMessage(`[{"type":"function","function":{"name":"lookup_order","description":"look up an order","parameters":{"type":"object"}}}]`),
	}

	receipt, err := ledger.AppendModelRequest("native-receipt", wire)
	if err != nil {
		t.Fatalf("AppendModelRequest: %v", err)
	}
	wantDigest, err := CanonicalModelRequestDigest(wire)
	if err != nil {
		t.Fatalf("CanonicalModelRequestDigest: %v", err)
	}
	if receipt.RequestSHA256 != wantDigest || receipt.PromptSHA256 == "" {
		t.Fatalf("receipt digests = request %q prompt %q, want request %q and a prompt digest", receipt.RequestSHA256, receipt.PromptSHA256, wantDigest)
	}

	logBytes, err := os.ReadFile(filepath.Join(dir, LogName))
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	if bytes.Contains(logBytes, []byte("immutable-system-byte-")) || bytes.Contains(logBytes, []byte("look up order A-1")) || bytes.Contains(logBytes, []byte("lookup_order")) {
		t.Fatalf("ledger duplicated model-visible payload instead of retaining only bounded content references: %s", logBytes)
	}

	reopened, err := Open(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	rebuilt, rebuiltReceipt, err := reopened.ReconstructModelRequest("native-receipt", wire.Identity.RequestID)
	if err != nil {
		t.Fatalf("ReconstructModelRequest after reopen: %v", err)
	}
	if rebuiltReceipt.RequestSHA256 != receipt.RequestSHA256 {
		t.Fatalf("reopened receipt digest = %q, want %q", rebuiltReceipt.RequestSHA256, receipt.RequestSHA256)
	}
	if err := VerifyModelRequest(wire, rebuilt); err != nil {
		t.Fatalf("reconstructed request differs from model boundary: %v", err)
	}

	wireOnly := cloneModelRequest(wire)
	wireOnly.Segments = append(wireOnly.Segments, ModelRequestSegment{
		Kind: "injected_directive", Source: promptaudit.SourceFakPolicy,
		Content: json.RawMessage(`{"role":"system","content":"wire-only directive"}`),
	})
	assertModelRequestAxis(t, VerifyModelRequest(wireOnly, rebuilt), AxisWireOnly)
	assertModelRequestAxis(t, VerifyModelRequest(wire, wireOnly), AxisLedgerOnly)

	changedContent := cloneModelRequest(wire)
	changedContent.Segments[1].Content = json.RawMessage(`{"role":"user","content":"look up order B-2"}`)
	assertModelRequestAxis(t, VerifyModelRequest(changedContent, rebuilt), AxisContent)
	changedTools := cloneModelRequest(wire)
	changedTools.Tools = json.RawMessage(`[{"type":"function","function":{"name":"other_tool","parameters":{"type":"object"}}}]`)
	assertModelRequestAxis(t, VerifyModelRequest(changedTools, rebuilt), AxisTools)
}

func TestInputClaimReopensAndBindsExactModelRequest(t *testing.T) {
	dir := t.TempDir()
	ledger, err := Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	input := json.RawMessage(`{"role":"user","content":"A"}`)
	claimReceipt, err := ledger.AppendInputClaim("trace", InputClaim{Turn: 1, Inputs: []json.RawMessage{input}})
	if err != nil {
		t.Fatalf("AppendInputClaim: %v", err)
	}
	binding := &InputClaimBinding{
		ClaimID: claimReceipt.ClaimID, InputSHA256: claimReceipt.InputSHA256, InputCount: claimReceipt.InputCount,
	}
	if _, err := ledger.AppendModelRequest("trace", ModelRequest{
		Identity: ModelRequestIdentity{Model: "test-model", Turn: 1},
		Segments: []ModelRequestSegment{{Kind: "user_input", Source: promptaudit.SourceUserConfig, Content: json.RawMessage(`{"role":"user","content":"B"}`)}},
		Tools:    json.RawMessage(`[]`), InputClaim: binding,
	}); err == nil || !strings.Contains(err.Error(), "exact claimed inputs") {
		t.Fatalf("AppendModelRequest without A = %v, want exact-claim refusal", err)
	}
	requestReceipt, err := ledger.AppendModelRequest("trace", ModelRequest{
		Identity: ModelRequestIdentity{Model: "test-model", Turn: 1},
		Segments: []ModelRequestSegment{{Kind: "user_input", Source: promptaudit.SourceUserConfig, Content: input}},
		Tools:    json.RawMessage(`[]`), InputClaim: binding,
	})
	if err != nil {
		t.Fatalf("AppendModelRequest: %v", err)
	}
	if requestReceipt.InputClaim == nil || *requestReceipt.InputClaim != *binding {
		t.Fatalf("request receipt input claim = %+v, want %+v", requestReceipt.InputClaim, binding)
	}

	reopened, err := Open(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	claim, rebuiltClaimReceipt, err := reopened.ReconstructInputClaim("trace", claimReceipt.ClaimID)
	if err != nil {
		t.Fatalf("ReconstructInputClaim: %v", err)
	}
	if rebuiltClaimReceipt.State != InputClaimClaimed || rebuiltClaimReceipt.InputSHA256 != claimReceipt.InputSHA256 ||
		len(claim.Inputs) != 1 || !bytes.Equal(claim.Inputs[0], input) {
		t.Fatalf("rebuilt claim = %+v receipt=%+v", claim, rebuiltClaimReceipt)
	}
	rebuiltRequest, rebuiltRequestReceipt, err := reopened.ReconstructModelRequest("trace", requestReceipt.RequestID)
	if err != nil {
		t.Fatalf("ReconstructModelRequest: %v", err)
	}
	if rebuiltRequest.InputClaim == nil || rebuiltRequestReceipt.InputClaim == nil ||
		*rebuiltRequest.InputClaim != *binding || *rebuiltRequestReceipt.InputClaim != *binding {
		t.Fatalf("rebuilt request binding = request %+v receipt %+v, want %+v", rebuiltRequest.InputClaim, rebuiltRequestReceipt.InputClaim, binding)
	}
}

func TestInputClaimReleaseIsIdempotentAcrossRestart(t *testing.T) {
	dir := t.TempDir()
	ledger, err := Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	receipt, err := ledger.AppendInputClaim("trace", InputClaim{
		Turn: 1, Inputs: []json.RawMessage{json.RawMessage(`{"role":"user","content":"A"}`)},
	})
	if err != nil {
		t.Fatalf("AppendInputClaim: %v", err)
	}
	const releasers = 8
	results := make(chan InputClaimReceipt, releasers)
	errs := make(chan error, releasers)
	var wg sync.WaitGroup
	for i := 0; i < releasers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			result, err := ledger.ReleaseInputClaim("trace", receipt.ClaimID, "PROMPT_ASSEMBLY_FAILED")
			results <- result
			errs <- err
		}()
	}
	wg.Wait()
	close(results)
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("concurrent ReleaseInputClaim: %v", err)
		}
	}
	var releaseEntry Hash
	for result := range results {
		if releaseEntry == "" {
			releaseEntry = result.LedgerEntry
		}
		if result.LedgerEntry != releaseEntry {
			t.Fatalf("duplicate release appended: first=%s later=%s", releaseEntry, result.LedgerEntry)
		}
	}

	reopened, err := Open(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	_, terminal, err := reopened.ReconstructInputClaim("trace", receipt.ClaimID)
	if err != nil {
		t.Fatalf("ReconstructInputClaim: %v", err)
	}
	if terminal.State != InputClaimReleased || terminal.Reason != "PROMPT_ASSEMBLY_FAILED" {
		t.Fatalf("terminal claim receipt = %+v", terminal)
	}
	chain, err := reopened.Chain("trace")
	if err != nil {
		t.Fatalf("Chain: %v", err)
	}
	var claimRows int
	for _, entry := range chain {
		if entry.Kind == KindInputClaimReceipt {
			claimRows++
		}
	}
	if claimRows != 2 {
		t.Fatalf("input claim rows = %d, want one CLAIMED plus one RELEASED", claimRows)
	}
}

func cloneModelRequest(in ModelRequest) ModelRequest {
	out := in
	out.Segments = append([]ModelRequestSegment(nil), in.Segments...)
	out.Tools = bytes.Clone(in.Tools)
	return out
}

func assertModelRequestAxis(t *testing.T, err error, want ModelRequestMismatchAxis) {
	t.Helper()
	var mismatch *ModelRequestMismatch
	if !errors.As(err, &mismatch) {
		t.Fatalf("VerifyModelRequest error = %v, want typed mismatch on %s", err, want)
	}
	if mismatch.Axis != want {
		t.Fatalf("VerifyModelRequest axis = %s, want %s (%v)", mismatch.Axis, want, err)
	}
}
