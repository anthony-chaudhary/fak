// fak-test:runtime fast est=1s lane=default
package trajectory

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDetectDegenerateCycleThresholds(t *testing.T) {
	phrase := "I need to inspect the file and verify the exact line numbers before editing anything here "
	if len(tokenizeForCycle(phrase)) < 16 {
		t.Fatalf("fixture phrase too short: %d tokens", len(tokenizeForCycle(phrase)))
	}

	t.Run("consecutive 4x triggers with exact offset", func(t *testing.T) {
		prefix := "Let me check. "
		text := prefix + strings.Repeat(phrase, 4)
		got, ok := DetectDegenerateCycle(2, "reasoning", text)
		if !ok {
			t.Fatal("expected degenerate cycle to be detected")
		}
		if got.TurnIndex != 2 || got.Channel != "reasoning" {
			t.Fatalf("unexpected metadata: %+v", got)
		}
		if got.SpanTokens < 16 || got.RepeatCount != 4 || got.FirstOffset != len(prefix) {
			t.Fatalf("unexpected cycle metrics: %+v (want span>=16, repeat=4, offset=%d)", got, len(prefix))
		}
	})

	t.Run("spaceless 64-char span repeating 4x triggers", func(t *testing.T) {
		spaceless := strings.Repeat("abcd1234", 8) // 64 non-whitespace chars
		text := strings.Repeat(spaceless, 4)
		got, ok := DetectDegenerateCycle(0, "content", text)
		if !ok {
			t.Fatal("expected 64-char spaceless degenerate cycle to be detected")
		}
		if got.RepeatCount != 4 || got.SpanTokens < 16 {
			t.Fatalf("unexpected spaceless cycle: %+v", got)
		}
	})

	t.Run("below repeat threshold does not trigger", func(t *testing.T) {
		text := strings.Repeat(phrase, 3) + "Done."
		if got, ok := DetectDegenerateCycle(0, "content", text); ok {
			t.Fatalf("unexpected cycle on 3x repeat: %+v", got)
		}
	})

	t.Run("short span below 16 tokens does not trigger", func(t *testing.T) {
		short := "ok wait let me check "
		if got, ok := DetectDegenerateCycle(0, "content", strings.Repeat(short, 6)); ok {
			t.Fatalf("unexpected cycle on short span: %+v", got)
		}
	})

	t.Run("non-consecutive 4x with >0.5 overlap triggers", func(t *testing.T) {
		sep := "step "
		text := phrase + sep + phrase + sep + phrase + sep + phrase
		got, ok := DetectDegenerateCycle(1, "content", text)
		if !ok {
			t.Fatal("expected non-consecutive >0.5 overlap cycle to be detected")
		}
		if got.RepeatCount != 4 || got.SpanTokens < 16 {
			t.Fatalf("unexpected non-consecutive cycle: %+v", got)
		}
	})
}

func TestAuditEmitsDegenerateCyclesAndExemplars(t *testing.T) {
	root := t.TempDir()
	loopPhrase := "Let me check the configuration file and verify every single flag in the handler before proceeding. "
	jsonl := strings.Join([]string{
		`{"type":"assistant","sessionId":"ses-loop","message":{"id":"m1","model":"inkling2-mini","content":[{"type":"text","text":"Starting check."}],"usage":{"input_tokens":10,"output_tokens":3}}}`,
		`{"type":"assistant","sessionId":"ses-loop","message":{"id":"m2","model":"inkling2-mini","content":[{"type":"thinking","thinking":"First thought. ` + strings.Repeat(loopPhrase, 5) + `"},{"type":"text","text":"Done."}],"usage":{"input_tokens":20,"output_tokens":100}}}`,
	}, "\n") + "\n"
	if err := os.WriteFile(filepath.Join(root, "ses-loop.jsonl"), []byte(jsonl), 0o644); err != nil {
		t.Fatal(err)
	}

	res, err := RunAudit(AuditOptions{
		Sources: []AuditSource{{Name: AuditSourceClaude, Root: root, RootLabel: "claude/projects"}},
	})
	if err != nil {
		t.Fatalf("Audit: %v", err)
	}
	if res.Summary.DegenerateCycles != 1 {
		t.Fatalf("Summary.DegenerateCycles = %d, want 1", res.Summary.DegenerateCycles)
	}
	if len(res.Transcripts) != 1 {
		t.Fatalf("Transcripts = %d, want 1", len(res.Transcripts))
	}
	tr := res.Transcripts[0]
	if tr.DegenerateCycles != 1 || len(tr.DegenerateCycleExemplars) != 1 {
		t.Fatalf("transcript degenerate cycles = %d exemplars=%+v, want 1", tr.DegenerateCycles, tr.DegenerateCycleExemplars)
	}
	ex := tr.DegenerateCycleExemplars[0]
	if ex.TurnIndex != 1 || ex.Channel != "reasoning" || ex.SpanTokens < 16 || ex.RepeatCount != 5 || ex.FirstOffset <= 0 {
		t.Fatalf("unexpected exemplar: %+v", ex)
	}
}
