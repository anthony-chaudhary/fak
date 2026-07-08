package microagent_test

import (
	"bytes"
	"errors"
	"fmt"
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

// TestContextBoundedTokenCapAcrossLongRun is the #2004 acceptance witness for the
// hard token ceiling: across a long synthetic run the history's token cost NEVER
// exceeds the configured cap, even though the number of appended turns grows far
// beyond what the cap can hold. That is the O(cap)-memory guarantee — residency is
// bounded by the cap, not by run length — and it is proven by checking the
// invariant AFTER every single Append, not just at the end.
func TestContextBoundedTokenCapAcrossLongRun(t *testing.T) {
	const cap = 500 // tokens (~2000 chars)
	ctx := microagent.NewContext(cap)

	rng := rand.New(rand.NewSource(1)) // fixed seed: the "run" is deterministic
	const steps = 20000
	var totalEvicted, everEvictedTurns, maxTokens, maxLen int

	for i := 0; i < steps; i++ {
		// Bounded per-message content (1..200 chars) — comfortably smaller than the
		// cap, so drop-oldest can always bring the history back under the ceiling.
		content := fmt.Sprintf("turn-%d:%s", i, string(bytes.Repeat([]byte("x"), 1+rng.Intn(200))))
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		if ev := ctx.Append(role, content); ev > 0 {
			totalEvicted += ev
			everEvictedTurns++
		}

		// The hard-ceiling invariant, checked on every step.
		if got := ctx.Tokens(); got > cap {
			t.Fatalf("step %d: history %d tokens exceeded cap %d", i, got, cap)
		}
		if ctx.Overflow() {
			t.Fatalf("step %d: unexpected Overflow with per-message content <= cap", i)
		}
		if n := ctx.Tokens(); n > maxTokens {
			maxTokens = n
		}
		if n := ctx.Len(); n > maxLen {
			maxLen = n
		}
	}

	// The run must actually have hit the cap and evicted — otherwise the test is
	// vacuous (it never exercised the ceiling).
	if totalEvicted == 0 {
		t.Fatal("no messages were ever evicted — the cap was never exercised")
	}
	// O(cap): the retained history stayed small while the run grew to 20k turns.
	if maxLen >= steps/10 {
		t.Fatalf("retained history grew to %d messages over %d steps — not O(cap)", maxLen, steps)
	}
	t.Logf("cap=%d tok · %d steps · evicted %d msgs across %d evicting appends · peak resident %d tok / %d msgs",
		cap, steps, totalEvicted, everEvictedTurns, maxTokens, maxLen)
}

// TestContextEncodeDecodeRoundTripIdentical is the #2004 acceptance witness for
// deterministic serialization: Encode produces canonical bytes, Decode restores the
// context, and re-Encoding the restored context reproduces the ORIGINAL bytes
// exactly. This is the property M12 hibernation stands on (a parked context resumes
// byte-identically), and it exercises the round trip at three states: fresh/empty,
// populated, and drained-past-the-cap.
func TestContextEncodeDecodeRoundTripIdentical(t *testing.T) {
	build := func() *microagent.Context {
		c := microagent.NewContext(300)
		c.Append("system", "you are a linear-history microagent")
		c.Append("user", "step one")
		c.Append("assistant", "did step one")
		// Force at least one eviction so the serialized state is a drained suffix.
		for i := 0; i < 50; i++ {
			c.Append("user", fmt.Sprintf("filler turn %d %s", i, string(bytes.Repeat([]byte("y"), 40))))
		}
		return c
	}

	for _, tc := range []struct {
		name string
		ctx  *microagent.Context
	}{
		{"empty", microagent.NewContext(128)},
		{"populated-and-drained", build()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b1, err := tc.ctx.Encode()
			if err != nil {
				t.Fatalf("Encode: %v", err)
			}

			restored := microagent.NewContext(1) // deliberately different cap
			if err := restored.Decode(b1); err != nil {
				t.Fatalf("Decode: %v", err)
			}
			if restored.Cap() != tc.ctx.Cap() {
				t.Errorf("Cap not restored: got %d, want %d", restored.Cap(), tc.ctx.Cap())
			}
			if restored.Len() != tc.ctx.Len() {
				t.Errorf("Len not restored: got %d, want %d", restored.Len(), tc.ctx.Len())
			}
			if restored.Tokens() != tc.ctx.Tokens() {
				t.Errorf("Tokens not restored: got %d, want %d", restored.Tokens(), tc.ctx.Tokens())
			}

			b2, err := restored.Encode()
			if err != nil {
				t.Fatalf("re-Encode: %v", err)
			}
			if !bytes.Equal(b1, b2) {
				t.Errorf("round trip not byte-identical:\n first %s\n again %s", b1, b2)
			}

			// Messages must survive in order.
			om, rm := tc.ctx.Messages(), restored.Messages()
			if len(om) != len(rm) {
				t.Fatalf("message count diverged: %d vs %d", len(om), len(rm))
			}
			for i := range om {
				if om[i] != rm[i] {
					t.Errorf("message %d diverged: %+v vs %+v", i, om[i], rm[i])
				}
			}
		})
	}
}

// TestContextTokenizerMatchesGuardBootPath is the compatibility witness the
// gen/second-next frame asks for: the ceiling is priced with the EXACT tokenizer
// the guard boot / served gateway path uses (agent.EstimateAnthropicTokens), not a
// second estimator that could silently drift. Context.Tokens() must equal the
// estimate over the equivalent Anthropic request.
func TestContextTokenizerMatchesGuardBootPath(t *testing.T) {
	c := microagent.NewContext(1 << 20) // huge cap: no eviction, so all turns remain
	msgs := []microagent.Msg{
		{Role: "system", Content: "system prompt of some length"},
		{Role: "user", Content: "a user turn with a tool result payload pasted in"},
		{Role: "assistant", Content: "the assistant reply that references it"},
	}
	for _, m := range msgs {
		c.Append(m.Role, m.Content)
	}

	req := &agent.AnthropicMessagesRequest{}
	for _, m := range msgs {
		req.Messages = append(req.Messages, agent.Message{Role: m.Role, Content: m.Content})
	}
	want := agent.EstimateAnthropicTokens(req)
	if got := c.Tokens(); got != want {
		t.Fatalf("Context.Tokens()=%d drifted from guard-path EstimateAnthropicTokens=%d", got, want)
	}
}

// TestContextDegenerateOverflowIsM25Boundary pins the honest edge the drop policy
// cannot cover: a single message larger than the whole cap. Append keeps it (a
// linear history cannot silently discard its only turn), the ceiling is exceeded,
// and Overflow reports true — the boundary the M25 compactor owns. This test
// asserts the type does not FAKE a pass by dropping the only message.
func TestContextDegenerateOverflowIsM25Boundary(t *testing.T) {
	c := microagent.NewContext(10) // 10 tokens (~40 chars)
	c.Append("user", string(bytes.Repeat([]byte("z"), 4000)))
	if c.Len() != 1 {
		t.Fatalf("oversized lone message should be kept, got Len=%d", c.Len())
	}
	if !c.Overflow() {
		t.Fatal("a lone message larger than the cap should report Overflow")
	}
	if c.Tokens() <= c.Cap() {
		t.Fatalf("degenerate case should exceed cap: tokens=%d cap=%d", c.Tokens(), c.Cap())
	}
}

// TestContextDecodeRefusals pins the serialization refusal edges: a bad-version
// blob is refused with ErrContextVersion, and malformed JSON errors rather than
// panicking. NewContext(0) selects the default cap.
func TestContextDecodeRefusals(t *testing.T) {
	if got := microagent.NewContext(0).Cap(); got != microagent.DefaultContextCap {
		t.Errorf("NewContext(0).Cap()=%d, want DefaultContextCap %d", got, microagent.DefaultContextCap)
	}
	c := microagent.NewContext(64)
	if err := c.Decode([]byte(`{"v":999,"cap":64,"msgs":[]}`)); !errors.Is(err, microagent.ErrContextVersion) {
		t.Errorf("Decode(bad version)=%v, want ErrContextVersion", err)
	}
	if err := c.Decode([]byte(`{not json`)); err == nil {
		t.Error("Decode(malformed) should error")
	}
}
