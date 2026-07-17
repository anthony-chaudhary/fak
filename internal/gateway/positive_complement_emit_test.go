package gateway

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/negframe"
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

func TestNegframeRewriteRequest(t *testing.T) {
	input := []agent.Message{{Role: "user", Content: "Don't forget to stamp the commit."}}
	got := applyMechanicalNegframeRewrite(cloneMessages(t, input), true)
	if want := "remember to stamp the commit."; got[0].Content != want {
		t.Fatalf("rewrite content = %q, want %q", got[0].Content, want)
	}
}

func TestNegframeRewriteDefaultOffByteExact(t *testing.T) {
	t.Setenv("FAK_NEGFRAME_REWRITE", "")
	input := []agent.Message{{Role: "user", Content: "Don't forget to stamp the commit."}}
	before, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	got := applyMechanicalNegframeRewrite(input, negframeRewriteEnabled())
	after, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != string(before) {
		t.Fatalf("default-off request changed:\nbefore %s\nafter  %s", before, after)
	}
}

func TestNegframeRewritePreservesJudgementAndFences(t *testing.T) {
	input := []agent.Message{{Role: "user", Content: "Never delete evidence.\n```text\nDon't forget to mutate.\n```\nDon't forget to push."}}
	got := applyMechanicalNegframeRewrite(cloneMessages(t, input), true)[0].Content
	want := "Never delete evidence.\n```text\nDon't forget to mutate.\n```\nremember to push."
	if got != want {
		t.Fatalf("judgement/fence rewrite = %q, want %q", got, want)
	}
}

func TestNegframeRewriteMetric(t *testing.T) {
	before := negframeRewriteSubstitutions.Load()
	applyMechanicalNegframeRewrite([]agent.Message{{Role: "user", Content: "No need to wait."}}, true)
	if got := negframeRewriteSubstitutions.Load(); got != before+1 {
		t.Fatalf("rewrite substitutions = %d, want %d", got, before+1)
	}
	var metrics strings.Builder
	writePositiveComplementMetrics(&metrics)
	want := fmt.Sprintf("fak_negframe_rewrite_substitutions_total %d", before+1)
	if !strings.Contains(metrics.String(), want) {
		t.Fatalf("rewrite metric sample absent: want %q in %q", want, metrics.String())
	}
}

func TestEmitNegationPassDefaultOn(t *testing.T) {
	t.Setenv("FAK_NEGATION_OP", "")
	t.Setenv("FAK_POSITIVE_COMPLEMENT", "")
	t.Setenv("FAK_NEGFRAME_REWRITE", "")
	input := []agent.Message{{Role: "user", Content: "Use not shared access."}}
	got := applyNegframeRequestPass(cloneMessages(t, input), "trace-on")
	if got[0].Content != "Use exclusive access." {
		t.Fatalf("default-on content = %q", got[0].Content)
	}
}

func TestEmitNegationPassOffIsByteExact(t *testing.T) {
	t.Setenv("FAK_NEGATION_OP", "off")
	// The master ablation wins over legacy soak flags.
	t.Setenv("FAK_POSITIVE_COMPLEMENT", "on")
	t.Setenv("FAK_NEGFRAME_REWRITE", "on")
	input := []agent.Message{{Role: "user", Content: "Never disable checks; use not shared access."}}
	got := applyNegframeRequestPass(cloneMessages(t, input), "trace-off")
	if got[0].Content != input[0].Content {
		t.Fatalf("off content = %q, want byte exact %q", got[0].Content, input[0].Content)
	}
}

func TestEmitNegationPassPolarityAndIdempotence(t *testing.T) {
	t.Setenv("FAK_NEGATION_OP", "on")
	input := []agent.Message{{Role: "system", Content: "Never delete records. Use not shared access. `not shared`"}}
	once := applyNegframeRequestPass(cloneMessages(t, input), "trace-one")
	twice := applyNegframeRequestPass(cloneMessages(t, once), "trace-two")
	want := "Never delete records. Use exclusive access. `not shared`"
	if once[0].Content != want || twice[0].Content != want {
		t.Fatalf("once=%q twice=%q want=%q", once[0].Content, twice[0].Content, want)
	}
}

func TestEmitNegationPassJournalCounts(t *testing.T) {
	path := filepath.Join(t.TempDir(), "operator.jsonl")
	t.Setenv("FAK_NEGATION_OP", "on")
	t.Setenv("FAK_NEGFRAME_JOURNAL", path)
	input := []agent.Message{{Role: "user", Content: "Use not shared access; never invent facts."}}
	_ = applyNegframeRequestPass(cloneMessages(t, input), "journal-trace")
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var rows []negframe.ReframeJournalRow
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		var row negframe.ReframeJournalRow
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			t.Fatal(err)
		}
		rows = append(rows, row)
	}
	if len(rows) != 1 || rows[0].Site != "gateway.negation_operator" || rows[0].Applied != 1 || rows[0].Refused != 0 || rows[0].ResidualNegatives < 1 {
		t.Fatalf("journal rows = %+v", rows)
	}
}
