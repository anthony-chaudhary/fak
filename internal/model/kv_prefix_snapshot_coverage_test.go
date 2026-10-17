package model

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime medium est=2s lane=default
// fak-test:usefulness protects=v41-snapshot-logit-coverage-ownership-and-identity
func TestV41PrefixSnapshotOwnsForwardCoverage(t *testing.T) {
	s := prefixCoverageSession(t)
	t.Cleanup(s.Close)
	want := s.NativeForwardCoverage()
	if want == nil || want.Validate() != nil {
		t.Fatalf("real prefill did not produce valid coverage: %+v", want)
	}
	capture := func() *PrefixSnapshot {
		t.Helper()
		snapshot, err := s.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(snapshot.Close)
		return snapshot
	}
	target := func(m *Model) *Session {
		t.Helper()
		next, err := m.NewBackendSessionChecked(s.Backend)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(next.Close)
		return next
	}
	snapshot := capture()
	clone, err := snapshot.Clone()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(clone.Close)
	reuseClone, err := snapshot.CloneForRestore()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(reuseClone.Close)

	// Mutating the live source after capture must not change either clone.
	s.v41Forward.forwardCoverage.Complete = false
	first := target(s.M)
	if err := snapshot.Restore(first); err != nil {
		t.Fatal(err)
	}
	if got := first.NativeForwardCoverage(); got == nil || *got != *want {
		t.Fatalf("restored cached-logit coverage = %+v, want %+v", got, want)
	}
	first.v41Forward.forwardCoverage.ExecutedLayers = 0
	for _, owned := range []*PrefixSnapshot{clone, reuseClone} {
		branch := target(s.M)
		if err := owned.Restore(branch); err != nil {
			t.Fatal(err)
		}
		got := branch.NativeForwardCoverage()
		if got == nil || *got != *want {
			t.Fatalf("clone aliases another coverage owner: got %+v want %+v", got, want)
		}
		got.Complete = false
		if !branch.NativeForwardCoverage().Complete {
			t.Fatal("coverage reader mutated session state")
		}
		if logits := branch.Step(7); len(logits) == 0 {
			t.Fatal("restored branch could not produce fresh incremental logits")
		}
		if got := branch.NativeForwardCoverage(); got == nil || got.LogitsSource != NativeForwardLogitsSourceV41Incremental {
			t.Fatalf("fresh step did not replace cached provenance: %+v", got)
		}
	}

	for _, tc := range []struct {
		name   string
		mutate func(*NativeForwardCoverage)
	}{
		{"incomplete", func(c *NativeForwardCoverage) { c.Complete = false }},
		{"model type", func(c *NativeForwardCoverage) { c.ModelType = "deepseek_v41_text" }},
		{"layers", func(c *NativeForwardCoverage) { c.ConfiguredLayers++; c.ExecutedLayers++ }},
		{"head dimension", func(c *NativeForwardCoverage) { c.HeadDim++ }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			coverage := *want
			tc.mutate(&coverage)
			s.v41Forward.forwardCoverage = &coverage
			bad := capture()
			branch := target(s.M)
			cache := branch.Cache
			if err := bad.Restore(branch); !errors.Is(err, ErrNativeForwardCoverageInvalid) {
				t.Fatalf("invalid provenance restore error = %v", err)
			}
			if bad.Cache == nil || branch.Cache != cache || branch.NativeForwardCoverage() != nil {
				t.Fatal("invalid provenance partially consumed or installed snapshot state")
			}
		})
	}
	s.v41Forward.forwardCoverage = want
	host := s.M.NewSession()
	t.Cleanup(host.Close)
	for _, other := range []*Session{target(prefixCoverageModel(t)), host} {
		bad := capture()
		cache := other.Cache
		if err := bad.Restore(other); err == nil {
			t.Fatal("foreign model or backend accepted cached-logit provenance")
		}
		if bad.Cache == nil || other.Cache != cache || other.NativeForwardCoverage() != nil {
			t.Fatal("identity refusal partially consumed or installed snapshot state")
		}
	}

	// No-logits state is valid continuation state, but it cannot invent evidence
	// for a sampled output. A subsequent logits-producing call must supply it.
	s.PrefillNoLogits([]int{7})
	noLogits := capture()
	branch := target(s.M)
	if err := noLogits.Restore(branch); err != nil {
		t.Fatal(err)
	}
	if branch.NativeForwardCoverage() != nil {
		t.Fatal("snapshot fabricated coverage for a no-logits prefill")
	}
}

// fak-test:runtime medium est=1s lane=default
// fak-test:usefulness protects=v41-failed-step-continuation-and-cached-logit-provenance
func TestV41PrefixCoverageSurvivesFailedStep(t *testing.T) {
	s := prefixCoverageSession(t)
	t.Cleanup(s.Close)
	want := s.NativeForwardCoverage()
	if want == nil || want.Validate() != nil {
		t.Fatalf("real prefill did not produce valid coverage: %+v", want)
	}
	snapshot, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(snapshot.Close)
	before := captureV41ForwardSnapshot(s.v41Forward)
	sentinel := errors.New("selected final norm failure after all decoder layers")
	calls, advancedLayers := 0, 0
	s.v41Forward.finalNorm = func([]float32) ([]float32, error) {
		calls++
		for _, layer := range s.v41Forward.layers {
			if layer.nextWindowPos == len(before.history)+1 {
				advancedLayers++
			}
		}
		return nil, sentinel
	}
	var fault any
	returned := false
	func() {
		defer func() { fault = recover() }()
		_, _, _ = s.M.forwardV41Step(7, s.v41Forward, nil)
		returned = true
	}()
	cause, ok := fault.(error)
	var selected *V41ProjectionOperationError
	if returned || !ok || !errors.Is(cause, sentinel) || !errors.As(cause, &selected) || calls != 1 || advancedLayers != s.M.Cfg.NumLayers {
		t.Fatalf("real failed Step did not roll back its selected head failure: returned=%t calls=%d advancedLayers=%d panic=%v", returned, calls, advancedLayers, fault)
	}
	// Rollback restores arithmetic continuation, but a failed attempt must not
	// advertise old logits as its result. These are deliberately separate owners.
	if s.NativeForwardCoverage() != nil || !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), before) {
		t.Fatal("failed Step changed retained continuation or retained stale coverage")
	}
	branch, err := s.M.NewBackendSessionChecked(s.Backend)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(branch.Close)
	if err := snapshot.Restore(branch); err != nil {
		t.Fatal(err)
	}
	if got := branch.NativeForwardCoverage(); got == nil || *got != *want {
		t.Fatalf("failed source Step changed independently cached logits provenance: got %+v want %+v", got, want)
	}
	if !reflect.DeepEqual(captureV41ForwardSnapshot(branch.v41Forward), before) {
		t.Fatal("prefix restore changed the cached arithmetic continuation")
	}
}

// prefixCoverageModel keeps this ownership witness independent of unrelated
// oracle/test helper files. Its two real decoder layers retain the published
// 384-expert/top-6 contract with reduced F32 axes; temporal compression, Engram
// and cross-layer sources are explicitly absent. Coverage is produced only by
// the actual prefill and incremental traversal, never by this fixture builder.
func prefixCoverageModel(t *testing.T) *Model {
	t.Helper()
	raw, err := os.ReadFile("testdata/deepseek_v41_flash_config.json")
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	if hex.EncodeToString(sum[:]) != "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879" {
		t.Fatal("official config fixture changed")
	}
	var cfg Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	// Select the canonical native identity, as the public F32 loader requires.
	cfg.ModelType = "deepseek41"
	cfg.NumLayers, cfg.NumExperts, cfg.NumExpertsPerTok = 2, 384, 6
	cfg.HiddenSize, cfg.NumHeads, cfg.HeadDim = 64, 2, 32
	cfg.QKNopeHeadDim, cfg.QKRopeHeadDim, cfg.QLoraRank = 16, 16, 32
	cfg.OLoraRank, cfg.OGroups, cfg.MoEIntermediateSize = 16, 2, 32
	cfg.VocabSize, cfg.NSharedExperts, cfg.RoutedScalingFactor = 8, 1, 1.5
	cfg.RopeScaling, cfg.LongRope, cfg.RopeFactor, cfg.RopeOrigContext, cfg.RopeTheta = "", nil, 0, 0, 10000
	cfg.Window = []int{-1, -1}
	cfg.DeepSeekV41.CompressRatios = []int{0, 0}
	cfg.DeepSeekV41.KVSourceLayerIDs, cfg.DeepSeekV41.IndexSourceLayerIDs, cfg.DeepSeekV41.EngramLayerIDs = nil, nil, nil
	var tensors []NamedTensorF32
	seed := uint64(0x9E3779B97F4A7C15)
	add := func(name string, shape ...int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		data := make([]float32, n)
		for i := range data {
			switch {
			case name == "model.norm.weight" || strings.HasSuffix(name, "attn_norm.weight") || strings.HasSuffix(name, "ffn_norm.weight") || strings.HasSuffix(name, "mhc.scale"):
				data[i] = 1
			case strings.HasSuffix(name, "mhc.base"):
			default:
				seed = seed*6364136223846793005 + 1442695040888963407
				x := float32(seed>>40)/float32(1<<24)*2 - 1
				scale := float32(.1)
				if strings.Contains(name, "embed_tokens") {
					scale = .2
				}
				if strings.HasSuffix(name, "attn.sink") {
					scale = .25
				}
				data[i] = x * scale
			}
		}
		tensors = append(tensors, NamedTensorF32{Name: name, Shape: shape, Data: data})
	}
	add("model.embed_tokens.weight", 8, 64)
	add("lm_head.weight", 8, 64)
	add("model.norm.weight", 64)
	for l := 0; l < cfg.NumLayers; l++ {
		layer := func(name string, shape ...int) { add(fmt.Sprintf("model.layers.%d.%s", l, name), shape...) }
		layer("attn_norm.weight", 64)
		layer("ffn_norm.weight", 64)
		layer("mhc.mixes.weight", 24, 64)
		layer("mhc.base", 24)
		layer("mhc.scale", 3)
		layer("attn.wq_a.weight", 32, 64)
		layer("attn.wq_b.weight", 64, 32)
		layer("attn.wkv.weight", 32, 64)
		layer("attn.wo_a.weight", 16, 64)
		layer("attn.wo_b.weight", 64, 32)
		layer("attn.sink", 2)
		layer("ffn.gate.weight", 384, 64)
		layer("ffn.gate.e_score_correction_bias", 384)
		layer("ffn.shared_experts.w1.weight", 32, 64)
		layer("ffn.shared_experts.w3.weight", 32, 64)
		layer("ffn.shared_experts.w2.weight", 64, 32)
		for e := 0; e < 384; e++ {
			stem := fmt.Sprintf("ffn.experts.%d.", e)
			layer(stem+"w1.weight", 32, 64)
			layer(stem+"w3.weight", 32, 64)
			layer(stem+"w2.weight", 64, 32)
		}
	}
	m, err := NewFromF32Tensors(cfg, tensors)
	if err != nil {
		t.Fatalf("real F32 fixture construction: %v", err)
	}
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Errorf("close fixture weights: %v", err)
		}
	})
	return m
}

func prefixCoverageSession(t *testing.T) *Session {
	t.Helper()
	m := prefixCoverageModel(t)
	backend := compute.Pick("cpu-ref")
	if backend == nil {
		t.Fatal("qualified cpu-ref backend is required")
	}
	s, err := m.NewBackendSessionChecked(backend)
	if err != nil {
		t.Fatal(err)
	}
	if logits := s.Prefill([]int{1, 3, 5}); len(logits) == 0 {
		s.Close()
		t.Fatal("real prefix prefill returned no logits")
	}
	if s.v41Forward == nil || s.halKV == nil {
		s.Close()
		t.Fatal("real backend prefill did not produce snapshot continuation state")
	}
	return s
}
