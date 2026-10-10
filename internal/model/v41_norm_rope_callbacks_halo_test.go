//go:build vulkan && (windows || linux) && cgo

package model

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Refs #13768. The session keeps the native backend itself: callback wrappers
// only observe physical counters and returned rows, never emulate a backend.
type v41NormRoPEHaloObserver struct {
	t       *testing.T
	backend compute.Backend
	profile interface {
		VulkanDebugDispatchProfileSnapshot() compute.VulkanDispatchProfile
	}
	identity         compute.BackendRuntimeIdentity
	calls            map[string]int
	coldGains        int
	rotationsChanged int
	retained, copies [][]float32
}

func (w *v41NormRoPEHaloObserver) snapshot() compute.BackendExecutionSnapshot {
	w.t.Helper()
	x, ok, err := compute.CaptureBackendExecutionSnapshot(w.backend)
	if err != nil || !ok || x.Identity != w.identity || !x.TransferCountersObserved {
		w.t.Fatalf("physical snapshot unavailable or identity changed: available=%t err=%v", ok, err)
	}
	return x
}

func (w *v41NormRoPEHaloObserver) compare(label string, got, want []float32) {
	w.t.Helper()
	if len(got) == 0 || len(got) != len(want) {
		w.t.Fatalf("%s width=%d want=%d", label, len(got), len(want))
	}
	for i, v := range got {
		if !finite32(v) || !finite32(want[i]) || math.Abs(float64(v-want[i])) > 2e-4*math.Max(1, math.Abs(float64(want[i]))) {
			w.t.Fatalf("%s[%d]=%g scalar=%g", label, i, v, want[i])
		}
	}
	w.retained = append(w.retained, got)
	w.copies = append(w.copies, append([]float32(nil), got...))
}

func (w *v41NormRoPEHaloObserver) norm(s *Session, name, kind string, input []float32, run func() ([]float32, error)) ([]float32, error) {
	w.t.Helper()
	gain := append([]float32(nil), s.M.tensor(name)...)
	if len(input) == 0 || len(gain) != len(input) {
		w.t.Fatal("norm scalar operands invalid")
	}
	original := append([]float32(nil), input...)
	for _, v := range input {
		if math.Float32bits(v)&0xffff != 0 {
			w.t.Fatal("full norm owner omitted BF16 input publication")
		}
	}
	var sum float64
	for _, v := range input {
		sum += float64(v) * float64(v)
	}
	inv := 1 / math.Sqrt(sum/float64(len(input))+s.M.Cfg.RMSNormEps)
	want := make([]float32, len(input))
	for i, v := range input {
		want[i] = float32(float64(v) * inv * float64(gain[i]))
	}
	// This witness owns a fresh model and its first device session: an absent
	// memo also means no shared immutable gain was staged by another session.
	_, warm := s.halW[name]
	before, pb := w.snapshot(), w.profile.VulkanDebugDispatchProfileSnapshot()
	result, err := run()
	if err != nil {
		return result, err
	}
	after, pa := w.snapshot(), w.profile.VulkanDebugDispatchProfileSnapshot()
	obs, e := compute.BackendExecutionDelta(before, after)
	if e != nil {
		w.t.Fatal(e)
	}
	upload := uint64(4 * len(input))
	uploads := uint64(1)
	if !warm {
		upload += uint64(4 * len(gain))
		uploads++
		w.coldGains++
	}
	if pa.OtherNormDispatches-pb.OtherNormDispatches != 1 || pa.ComputeDispatches-pb.ComputeDispatches != 1 || obs.Counters.H2DBytes != upload || obs.Counters.H2DCount != uploads || obs.Counters.D2HBytes != uint64(4*len(input)) || obs.Counters.D2HCount != 1 {
		w.t.Fatalf("%s actual dispatch/transfer contract cold=%t delta=%+v profile=%+v/%+v", kind, !warm, obs.Counters, pb, pa)
	}
	if !reflect.DeepEqual(input, original) {
		w.t.Fatal("norm callback mutated borrowed input")
	}
	w.compare(kind, result, want)
	w.calls[kind]++
	data, _ := json.Marshal(obs)
	w.t.Logf("callback=%s cold_gain=%t width=%d physical=%s", kind, !warm, len(input), data)
	return result, nil
}

func (w *v41NormRoPEHaloObserver) rope(run v41TailRoPEFunc) v41TailRoPEFunc {
	return func(layer int, q, kv, cos, sin []float32, heads, dim, rd int) ([]float32, []float32, error) {
		originalQ, originalKV := append([]float32(nil), q...), append([]float32(nil), kv...)
		for _, row := range [][]float32{q, kv} {
			for _, v := range row {
				if math.Float32bits(v)&0xffff != 0 {
					w.t.Fatal("full rotary owner omitted BF16 input publication")
				}
			}
		}
		oracle := func(input []float32) []float32 {
			result := append([]float32(nil), input...)
			for row := 0; row < len(input)/dim; row++ {
				for j := 0; j < rd/2; j++ {
					at := row*dim + dim - rd + 2*j
					a, b := float64(input[at]), float64(input[at+1])
					c, s := float64(cos[j]), float64(sin[j])
					result[at], result[at+1] = float32(a*c-b*s), float32(b*c+a*s)
				}
			}
			return result
		}
		wantQ, wantKV := oracle(q), oracle(kv)
		for i := range q {
			if math.Abs(float64(wantQ[i]-q[i])) > 1e-5 {
				w.rotationsChanged++
				break
			}
		}
		before, pb := w.snapshot(), w.profile.VulkanDebugDispatchProfileSnapshot()
		qo, ko, err := run(layer, q, kv, cos, sin, heads, dim, rd)
		if err != nil {
			return qo, ko, err
		}
		after, pa := w.snapshot(), w.profile.VulkanDebugDispatchProfileSnapshot()
		obs, e := compute.BackendExecutionDelta(before, after)
		if e != nil {
			w.t.Fatal(e)
		}
		if pa.OtherRoPEDispatches-pb.OtherRoPEDispatches != 1 || pa.ComputeDispatches-pb.ComputeDispatches != 1 || obs.Counters.H2DCount != 3 || obs.Counters.D2HCount != 2 || obs.Counters.H2DBytes != uint64(4*(len(q)+len(kv)+rd)) || obs.Counters.D2HBytes != uint64(4*(len(q)+len(kv))) {
			w.t.Fatalf("tail RoPE actual dispatch/transfer contract=%+v profile=%+v/%+v", obs.Counters, pb, pa)
		}
		if !reflect.DeepEqual(q, originalQ) || !reflect.DeepEqual(kv, originalKV) {
			w.t.Fatal("rotary callback mutated inputs")
		}
		w.compare("tail_rope_q", qo, wantQ)
		w.compare("tail_rope_kv", ko, wantKV)
		w.calls["tail_rope"]++
		data, _ := json.Marshal(obs)
		w.t.Logf("callback=tail_rope layer=%d heads=%d width=%d rotary=%d physical=%s", layer, heads, dim, rd, data)
		return qo, ko, nil
	}
}

func v41NormRoPEHaloIsolate(st *v41ForwardState) {
	st.expertGateUp, st.expertDown = nil, nil
	st.denseProjection, st.groupedOutput, st.engramProjection = nil, nil, nil
	st.mhcProjection, st.mhcFFNProjection = nil, nil
	st.sharedActivation, st.sharedAttention = nil, nil
	st.compressorNorm, st.indexKeyNorm, st.indexerScore = nil, nil, nil
	st.indexScoreHealth = nil
}

// FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 with the Halo SPIRV
// bundle selects this serial physical witness. No checkpoint/performance claim.
// Runtime estimate only, unmeasured.
// fak-test:runtime slow est=30s lane=optin
func TestV41NormRoPECallbacksHalo(t *testing.T) {
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" {
		t.Skip("physical callback witness requires explicit device opt-in")
	}
	if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Fatal("physical dispatch profile must be enabled before process start")
	}
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		t.Fatal("required physical Vulkan backend unavailable")
	}
	profile, ok := be.(interface {
		VulkanDebugDispatchProfileSnapshot() compute.VulkanDispatchProfile
	})
	if !ok {
		t.Fatal("physical dispatch profile unavailable")
	}
	first, available, err := compute.CaptureBackendExecutionSnapshot(be)
	if err != nil || !available || !first.TransferCountersObserved {
		t.Fatal("physical identity/transfer counters unavailable")
	}
	identity := strings.ToLower(first.Identity.Device)
	if !strings.Contains(identity, "radeon") || !strings.Contains(identity, "8060s") || !strings.Contains(identity, "radv") {
		t.Fatalf("requires Radeon8060S RADV Halo, got %q", first.Identity.Device)
	}
	if expected := os.Getenv("FAK_VULKAN_EXPECT_DEVICE"); expected != "" && expected != first.Identity.Device {
		t.Fatalf("physical device=%q want=%q", first.Identity.Device, expected)
	}
	cpu, ok := compute.Lookup("cpu-ref")
	if !ok || cpu.Caps().DeviceMemory {
		t.Fatal("independent cpu-ref backend unavailable")
	}
	fixture := func() *Model {
		m := v41FullStepPatchedNorms(t)
		if full, e := v41ForwardGeometry(m.Cfg); e != nil || !full {
			t.Fatal("physical callback fixture is not full geometry")
		}
		for _, name := range []string{layerName(0, "ffn_norm.weight"), "model.norm.weight"} {
			gain := make([]float32, m.Cfg.HiddenSize)
			for i := range gain {
				gain[i] = 0.625 + float32(i%13)/16
			}
			v41WriteTensorF32(t, m, name, gain)
		}
		return m
	}
	m, hostModel := fixture(), fixture()
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	host, err := hostModel.NewBackendSessionChecked(cpu)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		s.Close()
		host.Close()
		if e := m.CloseWeights(); e != nil {
			t.Error(e)
		}
		if e := hostModel.CloseWeights(); e != nil {
			t.Error(e)
		}
	}()
	st, hs := s.v41State(), host.v41State()
	if st.compressorNorm != nil || st.indexKeyNorm != nil {
		t.Fatal("ordinary Vulkan unexpectedly acquired exact-publication norm qualification")
	}
	v41NormRoPEHaloIsolate(st)
	v41NormRoPEHaloIsolate(hs)
	hs.queryNorm, hs.kvNorm, hs.ffnNorm, hs.finalNorm, hs.tailRoPE = nil, nil, nil, nil, nil
	query, kv, ffn, final, rope := st.queryNorm, st.kvNorm, st.ffnNorm, st.finalNorm, st.tailRoPE
	if query == nil || kv == nil || ffn == nil || final == nil || rope == nil {
		t.Fatal("one of five required native callbacks is unbound")
	}
	w := &v41NormRoPEHaloObserver{t: t, backend: be, profile: profile, identity: first.Identity, calls: map[string]int{}}
	st.queryNorm = func(l int, x []float32) ([]float32, error) {
		return w.norm(s, layerName(l, "attn.wq_a_norm.weight"), "query_norm", x, func() ([]float32, error) { return query(l, x) })
	}
	st.kvNorm = func(l int, x []float32) ([]float32, error) {
		return w.norm(s, layerName(l, "attn.kv_norm.weight"), "kv_norm", x, func() ([]float32, error) { return kv(l, x) })
	}
	st.ffnNorm = func(l int, x []float32) ([]float32, error) {
		return w.norm(s, layerName(l, "ffn_norm.weight"), "ffn_norm", x, func() ([]float32, error) { return ffn(l, x) })
	}
	st.finalNorm = func(x []float32) ([]float32, error) {
		return w.norm(s, "model.norm.weight", "final_norm", x, func() ([]float32, error) { return final(x) })
	}
	st.tailRoPE = w.rope(rope)
	total := 0
	for phase, ids := range [][]int{{1, 2}, {3}, {4, 5}} {
		before := w.snapshot()
		var got, want []float32
		if phase == 1 {
			got = s.Step(ids[0])
		} else {
			got = s.Prefill(ids)
		}
		after := w.snapshot()
		hostBefore := w.snapshot()
		if phase == 1 {
			want = host.Step(ids[0])
		} else {
			want = host.Prefill(ids)
		}
		hostAfter := w.snapshot()
		hd, e := compute.BackendExecutionDelta(hostBefore, hostAfter)
		if e != nil {
			t.Fatal(e)
		}
		if hd.Counters.ComputeDispatches != 0 || hd.Counters.H2DBytes != 0 || hd.Counters.D2HBytes != 0 {
			t.Fatal("CPU control performed device work")
		}
		for _, value := range want {
			if !finite32(value) {
				t.Fatal("CPU control produced nonfinite logits")
			}
		}
		v41GroupedParity(t, got, want, 2e-3)
		if !s.v41IncrementalEligible() || st.callbackOwner != s {
			t.Fatal("physical callbacks lost continuation binding")
		}
		total += len(ids)
		for _, name := range []string{"query_norm", "kv_norm", "ffn_norm", "tail_rope"} {
			if w.calls[name] != total*m.Cfg.NumLayers {
				t.Fatalf("%s calls=%d want=%d", name, w.calls[name], total*m.Cfg.NumLayers)
			}
		}
		// Cold forward computes every head; incremental Step/suffix also computes one head per token.
		if w.calls["final_norm"] != total {
			t.Fatalf("final callbacks=%d want=%d", w.calls["final_norm"], total)
		}
		observation, e := compute.BackendExecutionDelta(before, after)
		if e != nil {
			t.Fatal(e)
		}
		if observation.Counters.ComputeDispatches != uint64(len(ids)*(4*m.Cfg.NumLayers+1)) {
			t.Fatal("phase contains missing or unexpected native dispatches")
		}
		data, _ := json.Marshal(observation)
		t.Logf("phase=%d tokens=%d callbacks=%v physical=%s", phase, len(ids), w.calls, data)
	}
	if w.rotationsChanged == 0 {
		t.Fatal("live rotary scalar discriminator was vacuous")
	}
	if w.coldGains != 3*m.Cfg.NumLayers+1 {
		t.Fatalf("cold immutable gains=%d want=%d", w.coldGains, 3*m.Cfg.NumLayers+1)
	}
	for i, row := range w.retained {
		if !reflect.DeepEqual(row, w.copies[i]) {
			t.Fatal("callback result did not retain owned readback bytes")
		}
	}
}
