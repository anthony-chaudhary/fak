package microagent_test

import (
	"bytes"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/microagent"
)

func TestManagedContextLongHorizonBeatsNaiveTruncation(t *testing.T) {
	const cap = 100
	managed := microagent.NewManagedContext(cap)
	naive := microagent.NewContext(cap)
	artifact := microagent.ArtifactPointer{Kind: "commit", URI: "git:abc123"}
	managed.Append("user", strings.Repeat("setup ", 30), artifact)
	naive.Append("user", strings.Repeat("setup ", 30)+" durable=git:abc123")
	for i := 0; i < 30; i++ {
		content := fmt.Sprintf("turn-%02d %s", i, strings.Repeat("noise ", 12))
		managed.Append("assistant", content)
		naive.Append("assistant", content)
	}
	if managed.Tokens() > cap || managed.PeakTokens() > cap {
		t.Fatalf("managed context exceeded cap: tokens=%d peak=%d cap=%d", managed.Tokens(), managed.PeakTokens(), cap)
	}
	if managed.Compactions() == 0 {
		t.Fatal("long run never compacted")
	}
	managedText := contextText(managed.Messages())
	naiveText := contextText(naive.Messages())
	if !strings.Contains(managedText, "git:abc123") {
		t.Fatalf("managed context lost durable pointer: %s", managedText)
	}
	if strings.Contains(naiveText, "git:abc123") {
		t.Fatalf("naive truncation unexpectedly retained stale evidence: %s", naiveText)
	}
	// Completion oracle: the final action can name the witnessed commit only
	// when the pointer survives. This is the deterministic long-horizon A/B.
	managedComplete := strings.Contains(managedText, "git:abc123")
	naiveComplete := strings.Contains(naiveText, "git:abc123")
	if !managedComplete || naiveComplete {
		t.Fatalf("completion A/B managed=%v naive=%v", managedComplete, naiveComplete)
	}
}

func TestManagedContextDeterministicPointerOnlyRecap(t *testing.T) {
	build := func() *microagent.ManagedContext {
		ctx := microagent.NewManagedContext(65)
		ctx.Append("user", strings.Repeat("secret prose ", 20),
			microagent.ArtifactPointer{Kind: "test", URI: "artifact://z"},
			microagent.ArtifactPointer{Kind: "commit", URI: "git:a"},
			microagent.ArtifactPointer{Kind: "commit", URI: "git:a"})
		ctx.Append("assistant", strings.Repeat("new turn ", 15))
		return ctx
	}
	first, second := build(), build()
	if !bytes.Equal(mustEncodeManaged(t, first), mustEncodeManaged(t, second)) {
		t.Fatal("compaction is not byte deterministic")
	}
	text := contextText(first.Messages())
	if strings.Contains(text, "secret prose") {
		t.Fatalf("stale prose leaked into recap: %s", text)
	}
	if strings.Count(text, "git:a") != 1 || !strings.Contains(text, "artifact://z") {
		t.Fatalf("pointer recap incorrect: %s", text)
	}
	if strings.Index(text, "git:a") > strings.Index(text, "artifact://z") {
		t.Fatalf("pointers are not canonical: %s", text)
	}
}

func TestManagedContextBoundsOversizedLatestTurn(t *testing.T) {
	ctx := microagent.NewManagedContext(24)
	ctx.Append("user", strings.Repeat("oversized ", 1000))
	if ctx.Tokens() > ctx.Cap() || ctx.PeakTokens() > ctx.Cap() {
		t.Fatalf("oversized latest turn escaped cap: tokens=%d peak=%d cap=%d", ctx.Tokens(), ctx.PeakTokens(), ctx.Cap())
	}
	if !strings.Contains(contextText(ctx.Messages()), "[latest-turn-elided]") {
		t.Fatalf("oversized-turn elision is not explicit: %+v", ctx.Messages())
	}
}

func TestManagedContextEncodeDecodePreservesState(t *testing.T) {
	ctx := microagent.NewManagedContext(75)
	for i := 0; i < 8; i++ {
		ctx.Append("user", strings.Repeat(fmt.Sprintf("%d ", i), 20), microagent.ArtifactPointer{Kind: "file", URI: fmt.Sprintf("artifact://%d", i)})
	}
	encoded := mustEncodeManaged(t, ctx)
	var restored microagent.ManagedContext
	if err := restored.Decode(encoded); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(encoded, mustEncodeManaged(t, &restored)) {
		t.Fatalf("roundtrip changed state\nwant=%s\ngot=%s", encoded, mustEncodeManaged(t, &restored))
	}
	if restored.Tokens() > restored.Cap() || restored.Compactions() != ctx.Compactions() {
		t.Fatalf("restored state invalid: tokens=%d cap=%d compactions=%d", restored.Tokens(), restored.Cap(), restored.Compactions())
	}
}

func mustEncodeManaged(t *testing.T, ctx *microagent.ManagedContext) []byte {
	t.Helper()
	b, err := ctx.Encode()
	if err != nil {
		t.Fatal(err)
	}
	return b
}
func contextText(messages []microagent.Msg) string {
	var b strings.Builder
	for _, message := range messages {
		b.WriteString(message.Content)
		b.WriteByte('\n')
	}
	return b.String()
}
