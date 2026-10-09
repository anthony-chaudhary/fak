package agent

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// A boot probe runs through the real in-kernel planner but must not book a turn on
// the serving KV-prefix tap; a served turn on the same planner still does.
//
// fak-test:runtime fast est=2s lane=default
func TestBootProbeSkipsKVPrefixTap(t *testing.T) {
	cfg := tinyConcurrencyConfig()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	m.Quantize()
	p := NewInKernelPlanner(m, loadProbeTok(t), "synthetic-boot-probe", false, nil, false)
	p.maxNew = 2
	p.batchDecode = false
	msgs := []Message{{Role: RoleUser, Content: "warmup"}}

	before := cacheobs.Default.Snapshot()
	if _, err := p.Complete(WithBootProbe(context.Background()), msgs, nil); err != nil {
		t.Fatal(err)
	}
	afterProbe := cacheobs.Default.Snapshot()
	if afterProbe.Turns != before.Turns || afterProbe.PromptTokens != before.PromptTokens {
		t.Fatalf("boot probe booked kv-prefix turns %d->%d prompt %d->%d, want unchanged",
			before.Turns, afterProbe.Turns, before.PromptTokens, afterProbe.PromptTokens)
	}

	if _, err := p.Complete(context.Background(), msgs, nil); err != nil {
		t.Fatal(err)
	}
	if got := cacheobs.Default.Snapshot().Turns - afterProbe.Turns; got != 1 {
		t.Fatalf("served turn booked %d kv-prefix turns, want 1", got)
	}
}

func TestIsBootProbe(t *testing.T) {
	if IsBootProbe(context.Background()) || IsBootProbe(nil) { //nolint:staticcheck // nil ctx is a supported input
		t.Fatal("unmarked context reported as boot probe")
	}
	if !IsBootProbe(WithBootProbe(context.Background())) {
		t.Fatal("marked context not reported as boot probe")
	}
}
