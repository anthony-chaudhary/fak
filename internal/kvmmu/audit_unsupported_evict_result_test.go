package kvmmu_test

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/kvmmu"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type auditQuarantineGate struct{}

func (auditQuarantineGate) Admit(context.Context, *abi.ToolCall, *abi.Result) abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictQuarantine, By: "audit"}
}

type auditUnsupportedEvictBackend struct{ abi.KVBackend }

func (b *auditUnsupportedEvictBackend) CanEvict() error {
	return errors.New("audit backend cannot evict its recurrent state")
}

func TestAuditHybridQuarantineDoesNotClaimSkippedEviction(t *testing.T) {
	m := model.NewSynthetic(model.Config{
		ModelType: "llama", VocabSize: 16, HiddenSize: 16, IntermediateSize: 32,
		NumLayers: 1, NumHeads: 2, NumKVHeads: 2, HeadDim: 8,
		RMSNormEps: 1e-5, RopeTheta: 10000, EOSTokenID: -1,
	})
	inner, ok := model.KVBackend(m.NewSession())
	if !ok {
		t.Fatal("model.KVBackend: ok=false")
	}
	backend := &auditUnsupportedEvictBackend{KVBackend: inner}
	c := kvmmu.NewBackendWithGate(backend, auditQuarantineGate{})
	c.Append("sys", "system", []int{1, 2, 3})

	v, evicted, _ := c.AdmitResult(context.Background(), "result", "lookup", []int{4, 5}, []byte("ordinary cached result"))
	if v.Kind != abi.VerdictQuarantine {
		t.Fatalf("verdict=%v, want QUARANTINE", v.Kind)
	}
	if evicted {
		t.Errorf("AdmitResult evicted=true, but backend rejected eviction and cache still has %d positions", c.CacheLen())
	}
	if n, ok := c.Quarantine("result"); n != 0 || ok {
		t.Errorf("Quarantine returned (%d,%v), want (0,false) when backend rejects eviction", n, ok)
	}
	if got := c.EvictColdest(1); len(got) != 0 {
		t.Errorf("EvictColdest reported skipped eviction as %+v, want no evicted spans", got)
	}
}
