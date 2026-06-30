package agent

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestInKernelGenerateReusedHonorsCancellationPerStep(t *testing.T) {
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	p := reusePlanner(false, false, cfg)
	ids := synthIDs(cfg.VocabSize, 8, 32)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	emitted := 0
	gen, _, _, _, _, stopped, err := p.generateReusedContext(ctx, ids, 8, 0, 0, 0, map[int]bool{}, func(int) bool {
		emitted++
		if emitted == 2 {
			cancel()
		}
		return false
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("generateReusedContext err = %v, want context.Canceled", err)
	}
	if stopped {
		t.Fatal("cancellation must not be reported as a model stop")
	}
	if gen != emitted || gen != 2 {
		t.Fatalf("generated/emitted = %d/%d, want cancellation after 2 emitted tokens", gen, emitted)
	}
}

func TestInKernelCompleteThreadsContextToDecode(t *testing.T) {
	p := NewInKernelPlanner(model.NewSynthetic(tinyConcurrencyConfig()), loadProbeTok(t), "tiny-cancel", false, nil, false)
	p.quant = false
	p.maxNew = 4

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := p.Complete(ctx, []Message{{Role: RoleUser, Content: "hello"}}, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Complete err = %v, want context.Canceled", err)
	}
}
