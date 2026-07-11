package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/guideddecode"
)

// byteTokenBytes / byteEncode are the trivial 1-byte-per-token tokenizer over the synthetic
// 256-vocab (id == byte value). In this regime token boundaries are byte boundaries, so
// byte-level schema determinism IS token-level determinism — which is what lets the
// end-to-end test prove 100% structural acceptance exactly rather than empirically.
func byteTokenBytes(id int) []byte {
	if id < 0 || id >= 256 {
		return nil
	}
	return []byte{byte(id)}
}

func byteEncode(bs []byte) []int {
	ids := make([]int, len(bs))
	for i, b := range bs {
		ids[i] = int(b)
	}
	return ids
}

const envelopeSkeleton = `{"name":"get_weather","arguments":`

// TestFastForwardSpanSingleNameIsFullSkeleton: with one declared tool the whole envelope
// skeleton up to the (unconstrained) value is schema-forced, so the span is the full
// `{"name":"<name>","arguments":`.
func TestFastForwardSpanSingleNameIsFullSkeleton(t *testing.T) {
	schema := guideddecode.ToolSchema{Names: []string{"get_weather"}}
	if got := string(FastForwardSpan(nil, schema)); got != envelopeSkeleton {
		t.Fatalf("span = %q, want %q", got, envelopeSkeleton)
	}
}

// TestFastForwardSpanStopsAtEnumBranch: two names sharing a prefix force the skeleton plus
// the shared name prefix, then STOP where the enum diverges (the model must pick the tool).
func TestFastForwardSpanStopsAtEnumBranch(t *testing.T) {
	schema := guideddecode.ToolSchema{Names: []string{"get_weather", "get_time"}}
	want := `{"name":"get_` // shared prefix "get_"; 'w' vs 't' is the branch
	if got := string(FastForwardSpan(nil, schema)); got != want {
		t.Fatalf("span = %q, want %q (stop at the enum branch)", got, want)
	}
}

// TestFastForwardSpanEmptyInValueRegion: once the skeleton is consumed the FSM is
// UNCONSTRAINED (the argument value), so there is nothing deterministic to draft.
func TestFastForwardSpanEmptyInValueRegion(t *testing.T) {
	schema := guideddecode.ToolSchema{Names: []string{"get_weather"}}
	if got := FastForwardSpan([]byte(envelopeSkeleton), schema); len(got) != 0 {
		t.Fatalf("value-region span = %q, want empty (unconstrained ⇒ draft nothing)", got)
	}
}

// TestFastForwardSpanIsSchemaDeterministic is the load-bearing invariant behind the
// ~100%-acceptance claim: replaying the span byte-by-byte, EVERY byte was the UNIQUE
// admissible byte at its position. A schema-masked target therefore has exactly one legal
// choice at each step and can only reproduce the span.
func TestFastForwardSpanIsSchemaDeterministic(t *testing.T) {
	schema := guideddecode.ToolSchema{Names: []string{"get_weather", "get_time"}}
	span := FastForwardSpan(nil, schema)
	if len(span) == 0 {
		t.Fatal("empty span")
	}
	cur := []byte{}
	for i, b := range span {
		allowed := guideddecode.AllowedNextBytes(cur, schema)
		if len(allowed) != 1 || !allowed[b] {
			t.Fatalf("span byte %d (%q) not the unique admissible byte: allowed=%v", i, b, allowed)
		}
		cur = append(cur, b)
	}
}

// TestFastForwardDraftGateOffIsInert: with the native guided-decode flag off the drafter is
// a dormant no-op (nil draft), mirroring constraint.go's flag-off mask.
func TestFastForwardDraftGateOffIsInert(t *testing.T) {
	t.Setenv("FAK_NATIVE_GUIDED_DECODE", "0")
	d := &FastForwardDrafter{
		Schema:     guideddecode.ToolSchema{Names: []string{"get_weather"}},
		TokenBytes: byteTokenBytes,
		Encode:     byteEncode,
	}
	if got := d.Draft(nil); got != nil {
		t.Fatalf("gate off: Draft = %v, want nil (dormant)", got)
	}
}

// TestFastForwardDraftLiftsSpanToTokens: with the gate on, the drafted token ids decode back
// to exactly the deterministic byte span.
func TestFastForwardDraftLiftsSpanToTokens(t *testing.T) {
	t.Setenv("FAK_NATIVE_GUIDED_DECODE", "1")
	d := &FastForwardDrafter{
		Schema:     guideddecode.ToolSchema{Names: []string{"get_weather"}},
		TokenBytes: byteTokenBytes,
		Encode:     byteEncode,
	}
	ids := d.Draft(nil)
	var got []byte
	for _, id := range ids {
		got = append(got, byteTokenBytes(id)...)
	}
	if string(got) != envelopeSkeleton {
		t.Fatalf("draft decodes to %q, want %q", got, envelopeSkeleton)
	}
}

// TestFastForwardDraftDropsNonMatchingToken: a lossy proposer whose trailing id does not
// decode within the span is dropped, so the draft stays byte-identical to the span (every
// drafted token schema-forced).
func TestFastForwardDraftDropsNonMatchingToken(t *testing.T) {
	t.Setenv("FAK_NATIVE_GUIDED_DECODE", "1")
	badEncode := func(bs []byte) []int { return append(byteEncode(bs), 300) } // id 300: undecodable (>=256)
	d := &FastForwardDrafter{
		Schema:     guideddecode.ToolSchema{Names: []string{"get_weather"}},
		TokenBytes: byteTokenBytes,
		Encode:     badEncode,
	}
	ids := d.Draft(nil)
	var got []byte
	for _, id := range ids {
		got = append(got, byteTokenBytes(id)...)
	}
	if string(got) != envelopeSkeleton {
		t.Fatalf("draft with a bogus trailing token decodes to %q, want %q (bogus token dropped)", got, envelopeSkeleton)
	}
}

// TestFastForwardDraftAcceptedBySchemaMaskedVerify is the end-to-end proof that the draft
// plugs into the real verify substrate with 100% structural acceptance. It runs the drafted
// chain through a real synthetic model's VerifyForward, then checks — at every draft position
// — that a schema-masked greedy target (the SAME GuidedByteMask from constraint.go) argmaxes
// exactly the drafted token. Because the mask admits a single legal id at each span position,
// the target has no other choice: every structural draft token is accepted, independent of
// the model's raw logits. VerifyForward remains the sole authority (verify.go), so this is
// lossless by construction.
func TestFastForwardDraftAcceptedBySchemaMaskedVerify(t *testing.T) {
	t.Setenv("FAK_NATIVE_GUIDED_DECODE", "1")
	schema := guideddecode.ToolSchema{Names: []string{"get_weather"}}
	drafter := &FastForwardDrafter{Schema: schema, TokenBytes: byteTokenBytes, Encode: byteEncode}
	draft := drafter.Draft(nil)
	if len(draft) == 0 {
		t.Fatal("expected a non-empty structural draft")
	}

	m := NewSynthetic(cfgV(64, 4, 4, 2, 16, 128))
	s := m.NewSession()
	// Arbitrary committed context (the assistant turn before the tool call). The envelope FSM
	// spans only the drafted bytes, not this prompt.
	preLogits := s.Prefill([]int{1, 2, 3, 4, 5})
	verLogits := s.VerifyForward(draft, nil, nil)
	if len(verLogits) != len(draft) {
		t.Fatalf("VerifyForward returned %d logit vecs, want %d", len(verLogits), len(draft))
	}

	mask := &GuidedByteMask{Schema: schema, TokenBytes: byteTokenBytes}
	for j := 0; j < len(draft); j++ {
		// The distribution that predicts draft[j]: the prefill for j==0, else the verify
		// logits after draft[j-1]. history is the envelope bytes emitted so far.
		logits, history := preLogits, []int(nil)
		if j > 0 {
			logits, history = verLogits[j-1], draft[:j]
		}
		masked := append([]float32(nil), logits...)
		mask.MaskLogits(history, masked)
		if got := argmaxV(masked); got != draft[j] {
			t.Fatalf("position %d: schema-masked target argmax=%d, drafted=%d — not 100%% accepted", j, got, draft[j])
		}
	}
}
