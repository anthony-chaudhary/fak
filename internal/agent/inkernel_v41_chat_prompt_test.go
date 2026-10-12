package agent_test

import (
	"context"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// A two-turn V4.1 chat with a system prompt encodes as the GGUF template's
// non-thinking form: BOS, the system prompt, <｜User｜> turns, and
// <｜Assistant｜></think> turns closed by EOS, ending on the open generation turn.
// The text segments are golden reference-oracle lines of the committed vocab slice.
// fak-test:justify why=contract when=changed:internal/agent/**
// fak-test:runtime medium est=2s lane=default
func TestV41ChatPromptTwoTurnIDs(t *testing.T) {
	m := v41CompressorHTTPModel(t)
	tok := v41OracleSubsetTokenizer(t, map[int]string{128803: "<｜User｜>", 128804: "<｜Assistant｜>", 128822: "</think>"})
	planner := agent.NewInKernelPlanner(m, tok, "native", false, nil, false)
	got, err := planner.EncodePrompt(context.Background(), []agent.Message{
		{Role: agent.RoleSystem, Content: "Say hello."},
		{Role: agent.RoleUser, Content: "The chemical symbol for gold is"},
		{Role: agent.RoleAssistant, Content: "1234567890"},
		{Role: agent.RoleUser, Content: "The capital of France is"},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := []int{
		0,                // <｜begin▁of▁sentence｜>
		63006, 44388, 16, // Say hello.
		128803,                          // <｜User｜>
		671, 7658, 7380, 362, 8495, 344, // The chemical symbol for gold is
		128804, 128822, // <｜Assistant｜></think>
		6895, 18009, 25744, 18, // 1234567890
		1,                         // <｜end▁of▁sentence｜>
		128803,                    // <｜User｜>
		671, 6102, 294, 8760, 344, // The capital of France is
		128804, 128822, // open generation turn
	}
	if !reflect.DeepEqual(got.TokenIDs, want) || got.RendererID != "deepseek41-chat-nothink-v2" {
		t.Fatalf("V4.1 two-turn chat = %s %v, want deepseek41-chat-nothink-v2 %v", got.RendererID, got.TokenIDs, want)
	}
}
