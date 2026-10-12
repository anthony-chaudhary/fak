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
