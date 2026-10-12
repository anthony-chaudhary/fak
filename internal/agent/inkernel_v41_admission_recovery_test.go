package agent_test

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// Under the default (non-device-only) policy a V4.1 stage-admission refusal is
// panicked by Session.Prefill; Complete must return it as the request's typed
// error so `fak serve` warmup reports a failure and the process stays up.
// fak-test:justify why=regression when=changed:internal/agent/**
// fak-test:runtime medium est=2s lane=default
func TestV41StageAdmissionRefusalFailsRequestNotProcess(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	t.Setenv("FAK_INKERNEL_MAX_TOKENS", "1")
	t.Setenv("FAK_SESSION_LEDGER_DIR", t.TempDir())
	m := v41CompressorHTTPModel(t)
	d41 := *m.Cfg.DeepSeekV41
	d41.CompressRopeTheta = 0
	m.Cfg.DeepSeekV41 = &d41
	const prompt = "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>hi<\uff5cAssistant\uff5c></think>"
	tok, err := tokenizer.FromGGML([]string{prompt, "a", "b", "c", "d", "e", "f", "g"}, []string{"a b"}, []int32{3, 1, 1, 1, 1, 1, 1, 1}, "gpt2")
	if err != nil {
		t.Fatal(err)
	}
	planner := agent.NewInKernelPlanner(m, tok, "native", false, nil, false)
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("V4.1 admission refusal escaped Complete as a panic: %v", r)
			}
		}()
		_, err = planner.Complete(context.Background(), []agent.Message{{Role: "user", Content: "hi"}}, nil)
	}()
	var stage *model.V41ForwardError
	if !errors.As(err, &stage) || !errors.Is(err, model.ErrV41ForwardStage) || stage.Layer != 0 {
		t.Fatalf("Complete error = %v, want the layer-0 V4.1 stage refusal", err)
	}
}
