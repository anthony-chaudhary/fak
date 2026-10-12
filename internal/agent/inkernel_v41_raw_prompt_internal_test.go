package agent

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// Only a raw-marked request on DeepSeek V4.1 skips the chat template; every other
// family, and V4.1 chat, keep their renderer.
// fak-test:runtime fast est=1ms
func TestV41RawPromptOnlyForRawDeepSeekV41(t *testing.T) {
	v41, qwen := model.Config{ModelType: "deepseek41"}, model.Config{ModelType: "qwen3"}
	for _, c := range []struct {
		cfg  model.Config
		sp   SampleParams
		want bool
	}{
		{v41, SampleParams{RawPrompt: true}, true},
		{v41, SampleParams{}, false},
		{qwen, SampleParams{RawPrompt: true}, false},
	} {
		if got := inKernelRawV41Prompt(c.cfg, c.sp); got != c.want {
			t.Errorf("%s raw=%v: raw path %v, want %v", c.cfg.ModelType, c.sp.RawPrompt, got, c.want)
		}
		if raw := inKernelPromptRendererID(c.cfg, c.sp) == "deepseek41-raw-bos-v1"; raw != c.want {
			t.Errorf("%s raw=%v: renderer raw=%v, want %v", c.cfg.ModelType, c.sp.RawPrompt, raw, c.want)
		}
	}
}

// The V4.1 template closes each assistant reply with EOS and opens the next
// generation as a non-thinking assistant turn.
// fak-test:runtime fast est=1ms
func TestRenderDeepSeekV41ChatMultiTurn(t *testing.T) {
	got := renderDeepSeekV41Chat([]Message{
		{Role: RoleSystem, Content: "S"},
		{Role: RoleUser, Content: "U1"},
		{Role: RoleAssistant, Content: "A1"},
		{Role: RoleUser, Content: "U2"},
	})
	want := deepSeekV41BOS + "S" + deepSeekV41User + "U1" + deepSeekV41Assistant + "</think>A1" + deepSeekV41EOS +
		deepSeekV41User + "U2" + deepSeekV41Assistant + "</think>"
	if got != want {
		t.Fatalf("render = %q, want %q", got, want)
	}
	if id := inKernelPromptRendererID(model.Config{ModelType: "deepseek41"}, SampleParams{}); id != deepSeekV41ChatRendererID {
		t.Fatalf("V4.1 chat renderer = %q", id)
	}
	if id := inKernelPromptRendererID(model.Config{ModelType: "qwen3"}, SampleParams{}); id == deepSeekV41ChatRendererID {
		t.Fatal("non-V4.1 config took the V4.1 chat renderer")
	}
}
