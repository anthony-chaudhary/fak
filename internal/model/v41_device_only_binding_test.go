package model

import (
	"errors"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func v41HostOpNames(ops []v41HostOp) []string {
	names := make([]string, len(ops))
	for i, op := range ops {
		names[i] = op.name
	}
	return names
}

// Ref #13668. An empty static inventory is not admission: a backend that
// leaves a used op's seam unbound would run it on the host, so device-only
// refuses with the typed inventory error naming that op. A backend without
// score admission stays refused even when another backend can bind the seam.
// fak-test:runtime medium est=2s lane=default
func TestV41DeviceOnlyRefusesUnboundIndexerScore(t *testing.T) {
	saved := v41DeviceOnlyHostOps
	t.Cleanup(func() { v41DeviceOnlyHostOps = saved })
	v41DeviceOnlyHostOps = nil

	m := v41IndexerTestFixture(t)
	if !v41UsesIndexer(m.Cfg) || !v41UsesCompressedKV(m.Cfg) {
		t.Fatal("fixture must run the compressed indexer path")
	}
	without := v41DenseTestSession(t, m, newV41DenseTestBackend())
	if !slices.Contains(v41HostOpNames(without.v41DeviceOnlyUnboundOps()), "attention-indexer-score") {
		t.Fatalf("unbound ops %v omit the indexer score", v41HostOpNames(without.v41DeviceOnlyUnboundOps()))
	}
	with := v41DenseTestSession(t, m, newV41ScoreTestBackend())
	if slices.Contains(v41HostOpNames(with.v41DeviceOnlyUnboundOps()), "attention-indexer-score") {
		t.Fatal("a backend that admits the indexer score is still refused for it")
	}

	without.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	err := recoverError(func() { without.validateDeviceOnlyExecution("decode") })
	var refused *BackendForwardOperationError
	var hostOps *v41DeviceOnlyHostOpError
	if !errors.As(err, &refused) || refused.Path != "device-only" || !errors.Is(err, errV41DeviceOnlyHostOp) || !errors.As(err, &hostOps) {
		t.Fatalf("device-only admission = %v, want typed host-op refusal", err)
	}
	if !slices.Contains(v41HostOpNames(hostOps.ops), "attention-indexer-score") || !without.BackendSessionClosed() {
		t.Fatalf("refusal %v does not name the unbound indexer score or left the session open", v41HostOpNames(hostOps.ops))
	}
}

// fak-test:runtime fast est=1s lane=default
func TestV41DeviceOnlyAdmissionJoinsInventoryAndBindings(t *testing.T) {
	savedOps, savedBindings := v41DeviceOnlyHostOps, v41DeviceOnlyBindings
	t.Cleanup(func() { v41DeviceOnlyHostOps, v41DeviceOnlyBindings = savedOps, savedBindings })

	seen := map[string]bool{}
	for _, b := range savedBindings {
		if b.op.name == "" || b.op.site == "" || b.bound == nil || seen[b.op.name] {
			t.Fatalf("binding %+v is blank or duplicated", b.op)
		}
		seen[b.op.name] = true
	}

	s := &Session{M: &Model{Cfg: Config{}}}
	bound := func(*Session) bool { return true }
	unbound := func(*Session) bool { return false }
	never := func(Config) bool { return false }
	v41DeviceOnlyHostOps = nil
	v41DeviceOnlyBindings = []v41DeviceOnlyBinding{
		{op: v41HostOp{name: "a", site: "A"}, bound: bound},
		{op: v41HostOp{name: "unused", site: "U"}, uses: never, bound: unbound},
	}
	if err := s.v41DeviceOnlyAdmissionRefusal(); err != nil {
		t.Fatalf("every used op bound and inventory empty: %v, want admission", err)
	}

	v41DeviceOnlyHostOps = []v41HostOp{{name: "static", site: "S"}, {name: "dup", site: "S2"}}
	v41DeviceOnlyBindings = append(v41DeviceOnlyBindings,
		v41DeviceOnlyBinding{op: v41HostOp{name: "missing", site: "M"}, bound: unbound},
		v41DeviceOnlyBinding{op: v41HostOp{name: "dup", site: "D"}, bound: unbound})
	err := s.v41DeviceOnlyAdmissionRefusal()
	var typed *v41DeviceOnlyHostOpError
	if !errors.Is(err, errV41DeviceOnlyHostOp) || !errors.As(err, &typed) {
		t.Fatalf("refusal = %v, want typed host-op error", err)
	}
	if got := v41HostOpNames(typed.ops); !slices.Equal(got, []string{"static", "dup", "missing"}) {
		t.Fatalf("refused ops = %v, want static inventory then unbound ops, deduplicated", got)
	}
}

// This witnesses admission and teardown with a cpu-ref recorder. It does not
// qualify any device kernel. Other inventory/bindings are cleared only inside
// this test to isolate the conditional LoRA blocker after checking static rows.
// fak-test:runtime fast est=100ms lane=default
func TestV41DeviceOnlyHostInventoryAndLoRAAdmission(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("admission witness requires the cpu-ref recorder; no physical device is qualified")
	}
	t.Run("static-inventory", func(t *testing.T) {
		want := []v41HostOp{
			{name: "bf16-boundary", site: "v41RoundBF16"},
			{name: "compressed-publication-rope", site: "Model.v41CompressedPublicationRoPE"},
			{name: "indexer-query-rope", site: "Model.v41IndexRoPE"},
			{name: "logit-scale", site: "Model.v41HeadWithFinalNorm"},
		}
		last := -1
		for _, op := range want {
			at := slices.Index(v41DeviceOnlyHostOps, op)
			if at <= last {
				t.Fatalf("host operation %+v missing or out of forward order", op)
			}
			last = at
		}
	})
	savedOps, savedBindings := v41DeviceOnlyHostOps, v41DeviceOnlyBindings
	t.Cleanup(func() { v41DeviceOnlyHostOps, v41DeviceOnlyBindings = savedOps, savedBindings })
	v41DeviceOnlyHostOps, v41DeviceOnlyBindings = nil, nil
	for _, packed := range []bool{false, true} {
		name := "dense"
		if packed {
			name = "packed-tied-head"
		}
		t.Run(name, func(t *testing.T) {
			m := &Model{Cfg: Config{ModelType: "deepseek41", HiddenSize: 256, VocabSize: 1}}
			target := layerName(0, "attn.wq_a.weight")
			if packed {
				var err error
				m.Q2KEmbedding, err = NewQ2KEmbedding(q2kFixtureTensor(1, 256, 13818).raw, 1, 256)
				if err != nil {
					t.Fatal(err)
				}
				target = m.headName()
				if target != "model.embed_tokens.weight" || m.v41PackedHead(target) == nil {
					t.Fatal("fixture must select the real dedicated packed tied head")
				}
			}
			control := &Session{M: m, Backend: newV41DenseTestBackend()}
			control.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
			if err := recoverError(func() { control.validateDeviceOnlyExecution("decode") }); err != nil {
				t.Fatalf("unadapted admission gained a LoRA blocker: %v", err)
			}
			control.Close()
			set := NewLoRASet()
			if err := set.Add(&LoRAAdapter{Adapter: "test", Target: target, In: 256, Out: 1, Rank: 1, Alpha: 1, A: make([]float32, 256), B: []float32{1}}); err != nil {
				t.Fatal(err)
			}
			m.SetLoRA(set)
			portable := &Session{M: m, Backend: newV41DenseTestBackend()}
			if err := recoverError(func() { portable.validateDeviceOnlyExecution("decode") }); err != nil || portable.BackendSessionClosed() {
				t.Fatalf("portable adapter admission changed: %v", err)
			}
			portable.Close()
			for _, stage := range []string{"prefill", "decode"} {
				t.Run(stage, func(t *testing.T) {
					b := newV41DenseTestBackend()
					s := &Session{M: m, Backend: b}
					s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
					err := recoverError(func() {
						if stage == "prefill" {
							s.Prefill([]int{0})
						} else {
							s.Step(0)
						}
					})
					var closed *BackendForwardOperationError
					var host *v41DeviceOnlyHostOpError
					if !errors.As(err, &closed) || closed.Path != "device-only" || !errors.Is(err, errV41DeviceOnlyHostOp) || !errors.As(err, &host) {
						t.Fatalf("installed LoRA returned %v; want typed device-only refusal", err)
					}
					if !slices.Equal(v41HostOpNames(host.ops), []string{"dense-lora-apply"}) || !s.BackendSessionClosed() || !s.HostFallbackObserved() {
						t.Fatalf("refusal %v did not isolate LoRA or retire the session", err)
					}
					if len(b.ops) != 0 || len(b.uploads) != 0 || len(b.reads) != 0 {
						t.Fatal("adapter admission reached forward device work")
					}
				})
			}
			m.ClearLoRA()
			if err := (&Session{M: m}).v41DeviceOnlyAdmissionRefusal(); err != nil {
				t.Fatalf("cleared adapter retained a LoRA blocker: %v", err)
			}
		})
	}
}
