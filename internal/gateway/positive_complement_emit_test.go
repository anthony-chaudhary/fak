package gateway

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestPositiveComplementEmit(t *testing.T) {
	input := []agent.Message{{Role: "user", Content: "Use not shared access and not global routing."}}
	got := applyPositiveComplementEmit(cloneMessages(t, input), true)
	want := "Use exclusive access and cluster or keyword routing."
	if got[0].Content != want {
		t.Fatalf("emit content = %q, want %q", got[0].Content, want)
	}
}

func TestPositiveComplementEmitDefaultOffByteExact(t *testing.T) {
	t.Setenv("FAK_POSITIVE_COMPLEMENT", "")
	input := []agent.Message{{Role: "user", Content: "Use not shared access."}}
	before, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	got := applyPositiveComplementEmit(input, positiveComplementEmitEnabled())
	after, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != string(before) {
		t.Fatalf("default-off request changed:\nbefore %s\nafter  %s", before, after)
	}
}

func TestPositiveComplementEmitPreservesOpaqueAndUnknown(t *testing.T) {
	input := []agent.Message{{Role: "user", Content: "not shared\n`not true`\n```text\nnot global\n```\nnot imaginary"}}
	got := applyPositiveComplementEmit(cloneMessages(t, input), true)[0].Content
	want := "exclusive\n`not true`\n```text\nnot global\n```\nnot imaginary"
	if got != want {
		t.Fatalf("opaque/unknown emit = %q, want %q", got, want)
	}
}

func TestPositiveComplementEmitMetric(t *testing.T) {
	before := positiveComplementSubstitutions.Load()
	applyPositiveComplementEmit([]agent.Message{{Role: "user", Content: "not false"}}, true)
	if got := positiveComplementSubstitutions.Load(); got != before+1 {
		t.Fatalf("substitutions = %d, want %d", got, before+1)
	}
	var metrics strings.Builder
	writePositiveComplementMetrics(&metrics)
	want := fmt.Sprintf("fak_positive_complement_substitutions_total %d", before+1)
	if !strings.Contains(metrics.String(), want) {
		t.Fatalf("metric sample absent: want %q in %q", want, metrics.String())
	}
}

func cloneMessages(t *testing.T, input []agent.Message) []agent.Message {
	t.Helper()
	raw, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	var out []agent.Message
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	return out
}
