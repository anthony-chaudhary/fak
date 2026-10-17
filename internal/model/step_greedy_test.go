package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func stepGreedyTestModel() *Model {
	return NewSynthetic(Config{
		HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 97, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	})
}

func TestStepGreedyBackendSessionReturnsDeviceArgmaxOnly(t *testing.T) {
	m := stepGreedyTestModel()
	be := compute.Pick("cpu-ref")
	prompt := []int{3, 17, 5, 23}
	full := m.NewBackendSession(be)
	greedy := m.NewBackendSession(be)
	id := argmaxF32(full.Prefill(prompt))
	if got := argmaxF32(greedy.Prefill(prompt)); got != id {
		t.Fatalf("prefill argmax %d want %d", got, id)
	}
	for i := 0; i < 6; i++ {
		want := argmaxF32(full.Step(id))
		next, logits := greedy.StepGreedy(id)
		if logits != nil {
			t.Fatalf("step %d: backend StepGreedy returned %d host logits, want token only", i, len(logits))
		}
		if next != want {
			t.Fatalf("step %d: StepGreedy=%d want full-logit argmax %d", i, next, want)
		}
		if greedy.halKV.Len() != full.halKV.Len() {
			t.Fatalf("step %d: KV len %d want %d", i, greedy.halKV.Len(), full.halKV.Len())
		}
		id = next
	}
}

func TestStepGreedyRetainedLogitsReadBackOnceAndMatchStep(t *testing.T) {
	m := stepGreedyTestModel()
	be := compute.Pick("cpu-ref")
	prompt := []int{3, 17, 5, 23}
	full := m.NewBackendSession(be)
	greedy := m.NewBackendSession(be)
	id := argmaxF32(full.Prefill(prompt))
	greedy.Prefill(prompt)
	greedy.SetRetainGreedyLogits(true)
	for i := 0; i < 3; i++ {
		want := full.Step(id)
		next, logits := greedy.StepGreedy(id)
		if logits != nil || next != argmaxF32(want) {
			t.Fatalf("step %d: retained StepGreedy next=%d logits=%d, want token %d only", i, next, len(logits), argmaxF32(want))
		}
		if i == 2 {
			assertSameF32(t, "retained greedy logits", want, greedy.TakeGreedyLogits())
			if again := greedy.TakeGreedyLogits(); again != nil {
				t.Fatalf("second TakeGreedyLogits returned %d logits, want nil", len(again))
			}
		}
		id = next
	}
	greedy.StepGreedy(id)
	greedy.Step(id)
	if got := greedy.TakeGreedyLogits(); got != nil {
		t.Fatalf("Step left a stale retained distribution of %d logits", len(got))
	}
}

func TestStepGreedyHostSessionReturnsStepLogits(t *testing.T) {
	m := stepGreedyTestModel()
	prompt := []int{3, 17, 5, 23}
	ref := m.NewSession()
	host := m.NewSession()
	ref.Prefill(prompt)
	host.Prefill(prompt)
	for i, id := range []int{11, 29, 7} {
		want := ref.Step(id)
		next, logits := host.StepGreedy(id)
		if next != 0 {
			t.Fatalf("step %d: host StepGreedy next=%d, want 0 with logits", i, next)
		}
		assertSameF32(t, "host StepGreedy step "+itoa(i), want, logits)
	}
}

func TestStepGreedyVulkanReadsBackTokenNotLogits(t *testing.T) {
	be := compute.Pick("vulkan")
	if be.Name() != "vulkan" {
		t.Skip("vulkan backend not registered (no reachable Vulkan device)")
	}
	m := NewSynthetic(Config{
		HiddenSize: 64, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 128, VocabSize: 4096, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	})
	prompt := []int{3, 9, 44, 1}
	full := m.NewBackendSession(be)
	greedy := m.NewBackendSession(be)
	full.Prefill(prompt)
	greedy.Prefill(prompt)
	d2h := func(step func()) uint64 {
		t.Helper()
		window, ok, err := compute.BeginBackendExecutionObservation(be)
		if err != nil || !ok {
			t.Skipf("vulkan execution observation unavailable: ok=%t err=%v", ok, err)
		}
		step()
		obs, err := window.End()
		if err != nil || !obs.TransferCountersObserved {
			t.Fatalf("vulkan execution observation end: observed=%t err=%v", obs.TransferCountersObserved, err)
		}
		return obs.Counters.D2HBytes
	}
	id := 17
	for i := 0; i < 3; i++ {
		var want, got int
		fullBytes := d2h(func() { want = argmaxF32(full.Step(id)) })
		greedyBytes := d2h(func() {
			var logits []float32
			got, logits = greedy.StepGreedy(id)
			if logits != nil {
				t.Fatalf("step %d: vulkan StepGreedy returned host logits", i)
			}
		})
		if got != want {
			t.Fatalf("step %d: StepGreedy=%d want full-logit argmax %d", i, got, want)
		}
		if i == 2 {
			greedy.SetRetainGreedyLogits(true)
		}
		if fullBytes < uint64(4*m.Cfg.VocabSize) || greedyBytes > 64 {
			t.Fatalf("step %d: D2H bytes full=%d greedy=%d, want full>=%d greedy<=64", i, fullBytes, greedyBytes, 4*m.Cfg.VocabSize)
		}
		t.Logf("step %d: D2H bytes full=%d greedy=%d token=%d", i, fullBytes, greedyBytes, got)
		id = got
	}
	want := full.Step(id)
	var retainedBytes uint64
	retainedBytes = d2h(func() {
		if next, logits := greedy.StepGreedy(id); logits != nil || next != argmaxF32(want) {
			t.Fatalf("retained vulkan StepGreedy next=%d logits=%d, want token %d only", next, len(logits), argmaxF32(want))
		}
	})
	if retainedBytes > 64 {
		t.Fatalf("retained greedy step D2H bytes=%d, want token only", retainedBytes)
	}
	assertSameF32(t, "vulkan retained greedy logits", want, greedy.TakeGreedyLogits())
}

type deviceCapsBackend struct{ compute.Backend }

func (b deviceCapsBackend) Caps() compute.Caps {
	caps := b.Backend.Caps()
	caps.DeviceMemory = true
	return caps
}

func hostFallbackCount(t *testing.T, stage string) uint64 {
	t.Helper()
	for _, c := range compute.HostFallbackCounts() {
		if c.Stage == stage {
			return c.Count
		}
	}
	t.Fatalf("host-fallback stage %q is not declared", stage)
	return 0
}

func TestStepGreedyCountsHostFallbackOnDeviceBackend(t *testing.T) {
	m := stepGreedyTestModel()
	s := m.NewBackendSession(deviceCapsBackend{compute.Pick("cpu-ref")})
	s.Prefill([]int{3, 17, 5})
	s.SetRetainGreedyLogits(true)
	argmaxBefore := hostFallbackCount(t, hostFallbackGreedyArgmax)
	embedBefore := hostFallbackCount(t, hostFallbackEmbedRowUpload)
	if _, logits := s.StepGreedy(11); logits == nil {
		t.Fatal("non-cloning device backend retained greedy logits, want host logits")
	}
	if got := hostFallbackCount(t, hostFallbackGreedyArgmax) - argmaxBefore; got != 1 {
		t.Fatalf("greedy_argmax fallbacks +%d, want +1", got)
	}
	if got := hostFallbackCount(t, hostFallbackEmbedRowUpload) - embedBefore; got != 1 {
		t.Fatalf("embedding_row_upload fallbacks +%d, want +1", got)
	}
}
