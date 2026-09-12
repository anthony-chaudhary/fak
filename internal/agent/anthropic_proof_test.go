package agent

import (
	"bytes"
	"strings"
	"testing"
)

func TestAnthropicRewriteProofStructuralAnchors(t *testing.T) {
	before := deepCompactionFixture("retain originating task", false)
	// Preserve a tail field after messages; it is not part of the selected array.
	before = append(append([]byte{}, before[:len(before)-1]...), []byte(`,"proof_tail":"TAIL-SENTINEL"}`)...)
	after, outcome := CompactAnthropicHistoryWithOptions(before, CompactOptions{Budget: 700, Anchor: CompactAnchorHead, ColdCache: true})
	if len(after) >= len(before) {
		t.Fatalf("fixture did not compact: %+v", outcome)
	}
	proof, err := VerifyAnthropicRewrite(before, after, CompactAnchorHead)
	if err != nil {
		t.Fatal(err)
	}
	if proof.ShedBytes != len(before)-len(after) || proof.PrefixBytes <= 0 || proof.SuffixBytes <= 0 || proof.PrePrefixSHA256 != proof.PostPrefixSHA256 || proof.PreSuffixSHA256 != proof.PostSuffixSHA256 {
		t.Fatalf("incorrect structural proof: %+v", proof)
	}
	tamperedTail := bytes.Replace(after, []byte("TAIL-SENTINEL"), []byte("FAIL-SENTINEL"), 1)
	if _, err := VerifyAnthropicRewrite(before, tamperedTail, CompactAnchorHead); err == nil {
		t.Fatal("accepted changed protected tail")
	}
	tamperedHead := append([]byte{}, after...)
	tamperedHead[0] = ' '
	if _, err := VerifyAnthropicRewrite(before, tamperedHead, CompactAnchorHead); err == nil {
		t.Fatal("accepted invalid/prefix-mutated body")
	}
	if _, err := VerifyAnthropicRewrite(before, before, CompactAnchorHead); err == nil {
		t.Fatal("no-op claimed savings")
	}
}

func TestAnthropicElisionProofUsesCacheAnchor(t *testing.T) {
	before := elideWireBody(t, strings.Repeat("old-result ", 1000), strings.Repeat("cached-result ", 200), strings.Repeat("recent-result ", 200))
	after, outcome := ElideAnthropicResultsWithOutcome(before, 512)
	if len(after) >= len(before) {
		t.Fatalf("fixture did not elide: %+v", outcome)
	}
	proof, err := VerifyAnthropicElision(before, after)
	if err != nil {
		t.Fatal(err)
	}
	if proof.ShedBytes != len(before)-len(after) || proof.PrePrefixSHA256 != proof.PostPrefixSHA256 {
		t.Fatalf("wrong cache-anchor proof: %+v", proof)
	}
	tampered := bytes.Replace(after, []byte("cached head context"), []byte("broken head context"), 1)
	if bytes.Equal(tampered, after) {
		t.Fatal("fixture lacks cached head")
	}
	if _, err := VerifyAnthropicElision(before, tampered); err == nil {
		t.Fatal("accepted cached prefix drift")
	}
}
