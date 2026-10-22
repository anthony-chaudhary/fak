package model

import (
	"errors"
	"math"
	"runtime"
	"strings"
	"testing"
)

// v41NamedSites take their tensor names from the model's name table, so none
// of them may call layerName on a warm decode step. Their store lookups stay.
var v41NamedSites = []string{
	"(*Model).v41ForwardAdmitted", "(*Model).v41AdmitMHCNamed", "(*Model).v41AdmitGroupedWoA",
	"(*Model).v41MHCWeightLayoutNamed", "(*Model).v41RequirePlainLayerStepWeights",
	"(*Model).prepareV41RoleAttention", "(*Model).prepareV41PlainAttention",
	"(*Model).prepareV41LayerStepRoleFinishInto", "(*Model).v41FullFFNInput", "(*v41PlainLayerStep).attentionInput",
	"(*v41MHCResident).mixes", "(*v41MHCResident).splitParams", "(*Session).v41MHCMixWeight",
	"(*Session).v41RoutedLayerFunc", "(*Model).v41ExpertTripleInto", "(*Model).hostExpertDown",
	"(*Model).v41Names", "(*Model).v41LeafName", "(*Model).v41MHCName", "(*Model).v41MHCParamsKey",
	"(*Model).v41RoutedNames", "(*Model).v41ExpertStemNames",
}

type v41LookupCount struct {
	names, tensors, presence int
	named                    map[string]int
}

// v41CountLookups runs step with the lookup probe installed.
func v41CountLookups(t *testing.T, step func()) v41LookupCount {
	t.Helper()
	c := v41LookupCount{named: map[string]int{}}
	tensorLookupProbe = func(kind tensorLookupKind) {
		switch kind {
		case tensorLookupManifest:
			c.tensors++
			return
		case tensorLookupPresence:
			c.presence++
			return
		}
		c.names++
		pc, _, _, ok := runtime.Caller(2)
		if !ok {
			return
		}
		name := runtime.FuncForPC(pc).Name()
		for _, site := range v41NamedSites {
			if strings.Contains(name, site) {
				c.named[site]++
				return
			}
		}
	}
	defer func() { tensorLookupProbe = nil }()
	step()
	return c
}

// v41WarmNames fills every routed expert's names, as its first route would.
func v41WarmNames(m *Model) {
	for l := 0; l < m.Cfg.NumLayers; l++ {
		for e := 0; e < m.Cfg.NumExperts; e++ {
			m.v41RoutedNames(l, e)
		}
	}
}

// v41AssertWarmNamedStep steps the name-table session and the layerName
// reference in lockstep and checks the warm-step lookup contract.
func v41AssertWarmNamedStep(t *testing.T, step int, memoModel *Model, memo, ref func()) {
	t.Helper()
	v41WarmNames(memoModel)
	builds := v41NameTableBuilds.Load()
	got := v41CountLookups(t, memo)
	want := v41CountLookups(t, ref)
	if b := v41NameTableBuilds.Load() - builds; b != 0 {
		t.Fatalf("step %d: warm step rebuilt the name table %d times", step, b)
	}
	if len(got.named) != 0 {
		t.Fatalf("step %d: named sites called layerName on a warm step: %v", step, got.named)
	}
	if got.names >= want.names || got.tensors != want.tensors || got.presence != want.presence {
		t.Fatalf("step %d: lookups names %d tensors %d has %d, reference %d %d %d",
			step, got.names, got.tensors, got.presence, want.names, want.tensors, want.presence)
	}
	t.Logf("step %d layerName calls %d -> %d (tensor %d, has %d unchanged)", step, want.names, got.names, got.tensors, got.presence)
}

func v41AssertLogitsBitEqual(t *testing.T, step int, got, want []float32) {
	t.Helper()
	if len(got) != len(want) || len(got) == 0 {
		t.Fatalf("step %d: logits len %d, reference %d", step, len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("step %d: logit %d = %v, reference %v", step, i, got[i], want[i])
		}
	}
}

// v41AssertWarmStepAllocs measures each session's warm step after one
// discarded run per session, since the first measurement in a process reads
// high. The race detector skews allocation counts, so -race compares none.
func v41AssertWarmStepAllocs(t *testing.T, memo, ref *Session, start int, drop bool) {
	t.Helper()
	if raceEnabled() {
		t.Log("warm step allocs not compared under -race")
		return
	}
	measure := func(s *Session) float64 {
		next := start
		return testing.AllocsPerRun(3, func() { next = argmaxF32(s.Step(next)) })
	}
	measure(memo)
	measure(ref)
	memoAllocs := measure(memo)
	refAllocs := measure(ref)
	if memoAllocs > refAllocs || (drop && memoAllocs == refAllocs) {
		t.Fatalf("warm step allocs %v, layerName reference %v", memoAllocs, refAllocs)
	}
	t.Logf("warm step allocs: %v -> %v", refAllocs, memoAllocs)
}

type v41NamesPath struct {
	name string
	// allocsDrop: the device session built a heap bias name per routed pick
	// and an mHC split key per layer; the cpu reference path built neither.
	allocsDrop bool
	open       func(t *testing.T, off bool) (*Model, *Session)
}

func v41NamesPaths() []v41NamesPath {
	var paths []v41NamesPath
	for _, compressed := range []bool{false, true} {
		suffix := "plain"
		if compressed {
			suffix = "compressed"
		}
		paths = append(paths,
			v41NamesPath{"cpu_reference_" + suffix, false, func(t *testing.T, off bool) (*Model, *Session) {
				m, _ := v41IncrementalEngramFixture(t, compressed)
				m.v41NamesOff = off
				return m, &Session{M: m}
			}},
			v41NamesPath{"device_session_" + suffix, true, func(t *testing.T, off bool) (*Model, *Session) {
				m := v41PinnedSlateFixture(t, compressed)
				m.v41NamesOff = off
				s, err := m.NewBackendSessionChecked(newV41RoutedLayerTestBackend())
				if err != nil {
					t.Fatal(err)
				}
				return m, s
			}},
		)
	}
	return paths
}

// A warm decode step takes every named site's names from the table: it calls
// layerName fewer times than the layerName reference, keeps every tensor and
// presence lookup, allocates no more (less on the device session) and returns
// the same logits bit for bit.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime slow est=25s lane=default
func TestV41LayerNamesWarmStep(t *testing.T) {
	const steps = 4
	for _, path := range v41NamesPaths() {
		t.Run(path.name, func(t *testing.T) {
			memoModel, memo := path.open(t, false)
			defer memoModel.CloseWeights()
			defer memo.Close()
			refModel, ref := path.open(t, true)
			defer refModel.CloseWeights()
			defer ref.Close()

			prompt := v41DecodeSubmitsPrompt(4, memoModel.Cfg.VocabSize)
			got, want := memo.Prefill(prompt), ref.Prefill(prompt)
			for i := 0; i < steps; i++ {
				v41AssertLogitsBitEqual(t, i, got, want)
				next := argmaxF32(want)
				if i < 2 {
					got, want = memo.Step(next), ref.Step(next)
					continue
				}
				v41AssertWarmNamedStep(t, i, memoModel, func() { got = memo.Step(next) }, func() { want = ref.Step(next) })
			}
			v41AssertLogitsBitEqual(t, steps, got, want)
			v41AssertWarmStepAllocs(t, memo, ref, argmaxF32(want), path.allocsDrop)
		})
	}
}

// Every table name is the name layerName builds.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=100ms lane=default
func TestV41LayerNamesMatchLayerName(t *testing.T) {
	m, _ := v41IncrementalEngramFixture(t, true)
	defer m.CloseWeights()
	if m.v41Names() == nil {
		t.Fatal("V4.1 model has no name table")
	}
	builds := v41NameTableBuilds.Load()
	for l := 0; l < m.Cfg.NumLayers; l++ {
		for leaf, suffix := range v41LeafSuffix {
			if got, want := m.v41LeafName(l, v41Leaf(leaf)), layerName(l, suffix); got != want {
				t.Fatalf("layer %d %s: %q, want %q", l, suffix, got, want)
			}
			if mhc, ok := v41MHCLeafOf(suffix); ok && (mhc != v41Leaf(leaf) || m.v41MHCName(l, suffix) != layerName(l, suffix)) {
				t.Fatalf("layer %d %s: mHC leaf %d", l, suffix, mhc)
			}
		}
		for _, pair := range [][2]string{{"mhc.scale", "mhc.base"}, {"mhc.ffn_scale", "mhc.ffn_base"}} {
			want := "v41-mhc-split-params:" + layerName(l, pair[0]) + ":" + layerName(l, pair[1])
			if got := m.v41MHCParamsKey(l, pair[0], pair[1]); got != want {
				t.Fatalf("layer %d params key %q, want %q", l, got, want)
			}
		}
		for e := 0; e < m.Cfg.NumExperts; e++ {
			stem := "ffn.experts." + itoa(e)
			routed := m.v41RoutedNames(l, e)
			for i, proj := range []string{".w1.", ".w3.", ".w2."} {
				if routed.weights[i] != layerName(l, stem+proj+"weight") || routed.biases[i] != layerName(l, stem+proj+"bias") {
					t.Fatalf("layer %d expert %d%s: %q %q", l, e, proj, routed.weights[i], routed.biases[i])
				}
			}
			if m.v41RoutedNames(l, e) != routed || m.v41ExpertStemNames(l, stem) != routed.weights {
				t.Fatalf("layer %d expert %d: names were not resolved once", l, e)
			}
		}
	}
	if b := v41NameTableBuilds.Load() - builds; b != 0 {
		t.Fatalf("lookups rebuilt the name table %d times", b)
	}
	if m.v41RoutedNames(0, m.Cfg.NumExperts) != nil || m.v41RoutedNames(m.Cfg.NumLayers, 0) != nil || m.v41RoutedNames(0, -1) != nil {
		t.Fatal("name table answered outside its layers or experts")
	}
	m.Cfg.NumExperts++
	if m.v41Names() == nil || v41NameTableBuilds.Load()-builds != 1 {
		t.Fatal("changed expert count did not rebuild the name table once")
	}
	m.Cfg.NumExperts--
	m.v41NamesOff = true
	if m.v41Names() != nil || m.v41LeafName(1, v41LeafAttnSink) != layerName(1, "attn.sink") {
		t.Fatal("names off still served the table")
	}
	for e := 0; e < 1000; e++ {
		if got, ok := v41ExpertOfStem("ffn.experts." + itoa(e)); !ok || got != e {
			t.Fatalf("stem of expert %d parsed as %d, %v", e, got, ok)
		}
	}
	for _, stem := range []string{"ffn.experts.", "ffn.experts.01", "ffn.experts.-1", "ffn.experts.1x", "ffn.shared_experts", "experts.1"} {
		if _, ok := v41ExpertOfStem(stem); ok {
			t.Fatalf("stem %q parsed as a routed expert", stem)
		}
	}
}

// Names never stand in for store lookups: an in-place store change after a
// warm admission still refuses at the site that reads the store.
// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime fast est=500ms lane=default
func TestV41LayerNamesKeepStoreRefusals(t *testing.T) {
	m, _ := v41IncrementalEngramFixture(t, true)
	defer m.CloseWeights()
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatal(err)
	}
	reshape := func(t *testing.T, suffix string, shape []int) {
		name := layerName(1, suffix)
		meta, ok := m.manifest[name]
		if !ok {
			t.Fatalf("fixture has no %s", name)
		}
		changed := meta
		changed.Shape = shape
		m.manifest[name] = changed
		t.Cleanup(func() { m.manifest[name] = meta })
	}
	for _, tc := range []struct {
		suffix string
		shape  []int
	}{
		{"attn_norm.weight", []int{m.Cfg.HiddenSize + 1}},
		{"attn.sink", []int{m.Cfg.NumHeads + 1}},
		{"mhc.base", []int{v41MHCMixWidth + 1}},
		{"attn.wo_a.weight", []int{1, 1}},
	} {
		t.Run(tc.suffix, func(t *testing.T) {
			reshape(t, tc.suffix, tc.shape)
			if err := m.v41ForwardAdmitted(); !errors.Is(err, ErrV41ForwardStage) {
				t.Fatalf("in-place %s reshape admitted: %v", tc.suffix, err)
			}
		})
	}
	t.Run("mhc mix layout", func(t *testing.T) {
		reshape(t, "mhc.mixes.weight", []int{1, 1})
		if _, _, ok := m.v41MHCWeightLayout(1); ok {
			t.Fatal("in-place mix reshape kept its layout")
		}
		if err := m.v41AdmitMHC(1); !errors.Is(err, ErrV41ForwardStage) {
			t.Fatalf("in-place mix reshape admitted: %v", err)
		}
	})
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("restored model refused: %v", err)
	}
}
