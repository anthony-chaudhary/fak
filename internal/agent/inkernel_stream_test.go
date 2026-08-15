package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestInKernelPlannerProjectsCompletedTurnAsSemanticStream(t *testing.T) {
	m := model.NewSynthetic(tinyConcurrencyConfig())
	m.Quantize()
	p := NewInKernelPlanner(m, loadProbeTok(t), "tiny-stream", false, nil, false)
	if !p.StreamingSupported() {
		t.Fatal("in-kernel planner did not advertise semantic streaming")
	}
	var chunks []string
	completion, err := p.CompleteStream(context.Background(), func(chunk string) error {
		chunks = append(chunks, chunk)
		return nil
	}, []Message{{Role: RoleUser, Content: "hello"}}, nil, WithMaxTokens(2))
	if err != nil {
		t.Fatal(err)
	}
	if completion.Message.Content == "" {
		t.Fatal("empty completion")
	}
	if got := strings.Join(chunks, ""); got != completion.Message.Content {
		t.Fatalf("streamed content %q != completion content %q", got, completion.Message.Content)
	}
}
