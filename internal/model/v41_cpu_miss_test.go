package model

import (
	"bytes"
	"errors"
	"math"
	"reflect"
	"testing"
)

// v41CPUMissClose bounds a host-vs-reference row: the packed GEMV and the f32
// expansion sum in different orders, so a BF16 publication may round one ulp
// apart and carry into the down contraction. tol is relative to the row's
// largest magnitude.
func v41CPUMissClose(t *testing.T, what string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: width %d want %d", what, len(got), len(want))
	}
	scale := 0.0
	for _, v := range want {
		scale = math.Max(scale, math.Abs(float64(v)))
	}
	if scale == 0 {
		scale = 1
	}
	worst := 0.0
	for i := range got {
		if !finite32(got[i]) {
			t.Fatalf("%s: element %d is not finite", what, i)
		}
		worst = math.Max(worst, math.Abs(float64(got[i]-want[i]))/scale)
	}
	if worst > tol {
		t.Fatalf("%s: max relative deviation %.3g exceeds %.3g", what, worst, tol)
	}
}

func v41CPUMissWeights(t *testing.T, m *Model, l int, stem string) [3]v41CPUMissWeight {
	t.Helper()
	var w [3]v41CPUMissWeight
	for j, leaf := range [3]string{".w1.weight", ".w3.weight", ".w2.weight"} {
		var err error
		if w[j], err = m.v41CPUMissResolve(l, layerName(l, stem+leaf)); err != nil {
			t.Fatal(err)
		}
	}
	return w
}

// The packed host expert matches the host f32 triple arm on the pinned V4.1
// Flash slate (Q2_K gate/up, Q3_K down).
// fak-test:runtime fast est=1s lane=default
func TestV41CPUMissExpertParity(t *testing.T) {
	m := v41PinnedSlateFixture(t, false)
	picks, x := v41RoutedLayerFaultPicks(m)
	for _, pick := range picks {
		stem := "ffn.experts." + itoa(pick.expert)
		w := v41CPUMissWeights(t, m, 0, stem)
		if w[0].expert.kq == nil || w[2].expert.kq == nil {
			t.Fatal("pinned slate must resolve packed k-quant experts")
		}
		got, err := m.v41CPUMissExpert(0, stem, x, pick.weight, w)
		if err != nil {
			t.Fatal(err)
		}
		scratch := &v41ProjScratch{}
		want, err := m.v41FullRoutedExpertPublished(0, stem, x, pick.weight, m.Cfg, nil, nil, nil,
			func() ([]float32, []float32, []float32, error) { return m.v41ExpertTripleInto(0, stem, scratch, false) },
			func() ([]float32, error) { return m.hostExpertDown(0, stem, scratch) }, false, nil)
		if err != nil {
			t.Fatal(err)
		}
		v41CPUMissClose(t, stem, got, want, 1.0/64)
	}
}

// Splitting a layer's picks between the host arm and the batched device layer
// reproduces the device layer over every pick, hands the device only the
// picks the plan left it, and declines with the device.
// fak-test:runtime fast est=2s lane=default
func TestV41CPUMissLayerSplitMatchesDevice(t *testing.T) {
	m := v41PinnedSlateFixture(t, false)
	s, err := m.NewBackendSessionChecked(newV41RoutedLayerTestBackend())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	routed := s.v41State().routedLayer
	if routed == nil {
		t.Fatal("deferred-status backend must bind the batched routed layer")
	}
	picks, x := v41RoutedLayerFaultPicks(m)
	want, outcome, err := routed(0, picks, x)
	if err != nil || outcome != v41RoutedExpertHandled {
		t.Fatalf("device layer outcome=%v err=%v", outcome, err)
	}
	for _, c := range []struct {
		name    string
		host    []int
		decline bool
	}{
		{name: "split", host: []int{0, 2}},
		{name: "all host", host: []int{0, 1, 2, 3, 4, 5}[:len(picks)]},
		{name: "device declines", host: []int{1}, decline: true},
	} {
		t.Run(c.name, func(t *testing.T) {
			var seen [][]routePick
			scratch := &v41ProjScratch{
				routedLayer: func(l int, p []routePick, in []float32) ([]float32, v41RoutedExpertOutcome, error) {
					seen = append(seen, append([]routePick(nil), p...))
					if c.decline {
						return nil, v41RoutedExpertDeclined, nil
					}
					return routed(l, p, in)
				},
				cpuMiss: &v41CPUMissArm{plan: func(int, []routePick) ([]int, int, error) { return c.host, 0, nil }},
			}
			got, handled, err := m.v41RoutedLayerStep(0, picks, x, scratch)
			if err != nil {
				t.Fatal(err)
			}
			var device []routePick
			for slot, pick := range picks {
				onHost := false
				for _, h := range c.host {
					onHost = onHost || h == slot
				}
				if !onHost {
					device = append(device, pick)
				}
			}
			switch {
			case len(device) == 0 && len(seen) != 0:
				t.Fatalf("an all-host layer reached the device with %v", seen)
			case len(device) > 0 && (len(seen) != 1 || !reflect.DeepEqual(seen[0], device)):
				t.Fatalf("device saw %v, want %v", seen, device)
			}
			if c.decline {
				if handled || got != nil {
					t.Fatalf("device decline must decline the layer, handled=%t", handled)
				}
				return
			}
			if !handled {
				t.Fatal("split layer was not handled")
			}
			v41CPUMissClose(t, c.name, got, want, 1.0/64)
		})
	}
}

// Through the ordinary Session API: an unset flag binds nothing, the serve's
// device-only policy still binds an opted-in arm, an unparseable flag refuses the first decode step
// instead of replaying it on the whole-history fallback, and exact computes
// expert-ring page-in misses on the host with logits that track the device.
// fak-test:runtime slow est=40s lane=default
func TestV41CPUMissSessionDecode(t *testing.T) {
	newSession := func(t *testing.T, mode string) (*Model, *Session) {
		t.Helper()
		t.Setenv(ExpertCPUMissEnv, mode)
		m := v41PinnedSlateFixture(t, false)
		s, err := m.NewBackendSessionChecked(newV41RoutedLayerTestBackend())
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(s.Close)
		// A ring below the three layers' routed working set keeps decode
		// picks paging in, the regime the host arm exists for.
		s.ExpertRingBytes = 1 << 20
		return m, s
	}
	prompt := []int{1, 2, 3}

	if _, s := newSession(t, ""); s.v41State().cpuMiss != nil {
		t.Fatal("unset flag must not bind the host arm")
	}
	if _, s := newSession(t, "exact"); func() bool { s.SetExecutionPolicy(ExecutionPolicyDeviceOnly); return s.v41State().cpuMiss == nil }() {
		t.Fatal("fak serve's device-only policy must keep the opted-in host arm")
	}

	_, bad := newSession(t, "sometimes")
	last := bad.Prefill(prompt)
	func() {
		defer func() {
			err, _ := recover().(error)
			if !errors.Is(err, ErrExpertCPUMissFlag) {
				t.Fatalf("decode under an unparseable flag raised %v, want ErrExpertCPUMissFlag", err)
			}
		}()
		bad.Step(argmaxF32(last))
	}()

	const steps = 3
	_, device := newSession(t, "off")
	hostModel, host := newSession(t, "exact")
	want := v41RoutedLayerSteps(t, device, prompt, steps)
	got := v41RoutedLayerSteps(t, host, prompt, steps)
	for i := range want {
		v41CPUMissClose(t, "logits window "+itoa(i), got[i], want[i], 1.0/16)
	}
	dec := hostModel.V41ExpertFaultAttribution().Decode
	t.Logf("decode host misses=%d streamed=%d layers=%d host=%dns wait=%dns", dec.CPUMissExperts, dec.CPUMissStreamed, dec.CPUMissLayers, dec.CPUMissNanos, dec.CPUMissJoinWaitNanos)
	if dec.CPUMissExperts <= 0 || dec.CPUMissExperts > dec.CPUMissLayers*DefaultExpertCPUMissPerLayer {
		t.Fatalf("decode host misses=%d over %d layers, want 1..%d per layer", dec.CPUMissExperts, dec.CPUMissLayers, DefaultExpertCPUMissPerLayer)
	}
	if dec.CPUMissNanos <= 0 || dec.CPUMissExpertsPerToken <= 0 {
		t.Fatalf("host arm time=%d experts/token=%g", dec.CPUMissNanos, dec.CPUMissExpertsPerToken)
	}
}

// With FAK_V41_ROUTED_PREFETCH on beside the host arm, the batched layer's
// concurrent checkpoint read covers only the picks left on the device: a
// host-planned expert is read once, by the host arm, never prefetched for an
// upload that will not happen.
// fak-test:runtime fast est=1s lane=default
func TestV41CPUMissSkipsRoutedPrefetch(t *testing.T) {
	t.Setenv(v41LayerExpertBatchEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41RoutedPrefetchEnv, "1")
	m := v41IncrementalExpertFixture(t, true, false)
	tier := NewExpertCheckpointTier(0)
	names := func(e int) [3]string {
		stem := "ffn.experts." + itoa(e)
		return [3]string{layerName(0, stem+".w1.weight"), layerName(0, stem+".w3.weight"), layerName(0, stem+".w2.weight")}
	}
	var tripleBytes int64
	for p, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
		w := m.kqw[names(0)[p]]
		var blob []byte
		for e := 0; e < 3; e++ {
			blob = append(blob, m.kqw[names(e)[p]].raw...)
			delete(m.kqw, names(e)[p])
		}
		quant := ExpertCheckpointQ2K
		if p == 2 {
			quant = ExpertCheckpointQ3K
		}
		if err := tier.AddShard(bytes.NewReader(blob), int64(len(blob)), []FusedExpertTensor{{Name: proj, Layer: 0, Proj: proj, Arch: "deepseek41", Quant: quant, Experts: 3, Rows: w.out, Cols: w.in}}); err != nil {
			t.Fatal(err)
		}
		tripleBytes += int64(len(w.raw))
	}
	m.SetExpertCheckpoint(tier)
	s, err := m.NewBackendSessionChecked(newV41RoutedLayerTestBackend())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	s.ExpertRingBytes = 2 * tripleBytes
	layer := s.v41RoutedLayerFunc()
	if layer == nil {
		t.Fatal("layer did not bind")
	}
	picks := []routePick{{expert: 0, weight: .25}, {expert: 2, weight: .5}}
	_, x := v41RoutedLayerFaultPicks(m)
	scratch := &v41ProjScratch{
		routedLayer: layer,
		cpuMiss:     &v41CPUMissArm{plan: func(int, []routePick) ([]int, int, error) { return []int{1}, 0, nil }},
	}
	if _, handled, err := m.v41RoutedLayerStep(0, picks, x, scratch); err != nil || !handled {
		t.Fatalf("handled=%t err=%v, want a handled split layer", handled, err)
	}
	if st := tier.Stats(); st.Prefetched != 3 || st.Reads != 6 {
		t.Fatalf("prefetched=%d reads=%d, want the device pick's 3 projections prefetched and 6 reads in all", st.Prefetched, st.Reads)
	}
}
