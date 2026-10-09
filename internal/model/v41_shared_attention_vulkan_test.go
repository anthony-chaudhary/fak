//go:build vulkan && (windows || linux) && cgo

package model

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41AttentionPhysicalBackend interface {
	compute.V41SharedAttentionBackend
	VulkanPhysicalDeviceType() uint32
	VulkanDebugDispatchProfileSnapshot() compute.VulkanDispatchProfile
	BeginBatch()
	FlushBatch()
	Trim()
	UploadClass(compute.Tensor, compute.Dtype, compute.MemoryClass, string) compute.Tensor
}

// Deferred physical witness SOURCE. Use the same explicit opt-in and startup
// profiling environment as TestVulkanV41SharedAttentionPhysical, and run serially.
// The tiny synthetic models exercise public Prefill -> Step -> Prefill, with
// other optional device callbacks disabled only in this test. Whole-architecture
// strict refusal is unchanged: this is operation reachability, not device-only
// execution, full-checkpoint quality, or performance qualification.
// Bound: two two-layer models, four tokens each, H<=2, D<=512, <=40 contractions;
// child deadline 90s. The cost estimate is unmeasured. Fixed before device runs:
// contraction abs+rel=2e-5; normal-session logits abs+rel=1e-4.
// fak-test:runtime integration est=90s lane=optin
func TestV41SharedAttentionVulkan(t *testing.T) {
	if os.Getenv("FAK_V41_SHARED_ATTENTION_WITNESS") != "1" {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required-device run needs FAK_V41_SHARED_ATTENTION_WITNESS=1")
		}
		t.Skip("deferred real shared-attention session witness is opt-in")
	}
	for _, key := range []string{"FAK_VULKAN_REQUIRE_DEVICE", "FAK_VULKAN_DISPATCH_PROFILE", "FAK_VULKAN_TIMESTAMP_PROFILE", "FAK_VULKAN_TIMESTAMP_PROFILE_EVERY"} {
		if os.Getenv(key) != "1" {
			t.Fatalf("physical witness requires %s=1 before process startup", key)
		}
	}
	if os.Getenv("FAK_VULKAN_EXPECT_DEVICE") == "" {
		t.Fatal("physical witness needs the exact expected device name")
	}
	const childKey = "FAK_V41_ATTENTION_MODEL_CHILD"
	if os.Getenv(childKey) != "1" {
		executable, err := os.Executable()
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, executable, "-test.run=^TestV41SharedAttentionVulkan$", "-test.count=1", "-test.v", "-test.timeout=85s")
		cmd.Env = append(os.Environ(), childKey+"=1")
		output, err := cmd.CombinedOutput()
		t.Logf("isolated normal-session witness output:\n%s", output)
		if err != nil {
			t.Fatalf("physical child failed: %v (deadline=%v)", err, ctx.Err())
		}
		if strings.Contains(string(output), "--- SKIP:") || !strings.Contains(string(output), "V41_ATTENTION_SESSION_PHYSICAL_COMPLETE") {
			t.Fatal("child did not complete actual nonempty session operations")
		}
		pattern := regexp.MustCompile(`fak-vulkan ts-stage kernel=v41_shared_attention groups=\d+x\d+ calls/batch=([0-9.]+) ms/batch=([0-9.]+) us/call=([0-9.]+)`)
		measured := false
		for _, match := range pattern.FindAllStringSubmatch(string(output), -1) {
			calls, e1 := strconv.ParseFloat(match[1], 64)
			micros, e2 := strconv.ParseFloat(match[3], 64)
			measured = measured || (e1 == nil && e2 == nil && calls > 0 && micros > 0 && !math.IsNaN(micros) && !math.IsInf(micros, 0))
		}
		if !measured {
			t.Fatal("dedicated real device timestamp missing or rounded to zero")
		}
		return
	}
	backend, ok := compute.Lookup("vulkan")
	if !ok || backend == nil || backend.Name() != "vulkan" || !backend.Caps().DeviceMemory {
		t.Fatal("required actual Vulkan backend unavailable")
	}
	device, ok := backend.(v41AttentionPhysicalBackend)
	if !ok || !device.SupportsV41SharedAttention() {
		t.Fatal("required optional operation or observation interface unavailable")
	}
	kind := device.VulkanPhysicalDeviceType()
	if kind != 1 && kind != 2 {
		t.Fatalf("physical integrated/discrete GPU not established: type=%d", kind)
	}
	identity := v41AttentionPhysicalSnapshot(t, backend)
	if identity.Identity.Device != os.Getenv("FAK_VULKAN_EXPECT_DEVICE") {
		t.Fatalf("device=%q want exact %q", identity.Identity.Device, os.Getenv("FAK_VULKAN_EXPECT_DEVICE"))
	}
	if compute.Default().Name() != "cpu-ref" {
		t.Fatal("host control must be cpu-ref")
	}
	t.Logf("identity=%+v device_type=%d; cost unmeasured; synthetic operation reachability only", identity.Identity, kind)
	// Observe setup traffic explicitly. Pool draining below is a diagnostic
	// ownership check, so these timings must never be used as performance data.
	setup := device.UploadClass(compute.NewF32(compute.Default(), []int{2048}, make([]float32, 2048)), compute.F32, compute.MemoryActivation, "shared attention witness staging setup")
	_ = device.Read(setup)
	device.Free(setup)
	device.Trim()
	setupDelta, err := compute.BackendExecutionDelta(identity, v41AttentionPhysicalSnapshot(t, backend))
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("actual staging setup outside contraction windows: %+v", setupDelta)

	v41AttentionPhysicalMaskControls(t, device)
	allNonempty := 0
	for _, role := range []bool{false, true} {
		name := "plain-window"
		if role {
			name = "source-ratio2-reader-ratio0"
		}
		t.Run(name, func(t *testing.T) {
			fixture := func() *Model {
				if role {
					return v41ReaderWidthFixture(t)
				}
				m := v41IncrementalPlainModel(t, 2)
				m.Cfg.Window = []int{2}
				return m
			}
			m, hostModel := fixture(), fixture()
			if m.Cfg.NumLayers != 2 || m.Cfg.NumHeads > 2 || m.Cfg.HeadDim > 512 {
				t.Fatal("tiny physical fixture geometry expanded")
			}
			s := v41DenseTestSession(t, m, backend)
			host := hostModel.NewSession()
			defer host.Close()
			v41SharedAttentionOnly(s)
			st := s.v41State()
			selected := st.sharedAttention
			if selected == nil {
				t.Fatal("normal session failed to select the real operation")
			}
			calls, nonempty, readerCalls := 0, 0, 0
			var lastOutput, lastCopy []float32
			st.sharedAttention = func(layer int, request v41SharedAttentionRequest) ([]float32, error) {
				calls++
				if calls > 40 {
					t.Fatal("physical contraction bound exceeded")
				}
				if lastOutput != nil && !slices.Equal(lastOutput, lastCopy) {
					t.Fatal("a later operation changed previously owned output")
				}
				if role && layer == 1 {
					readerCalls++
					if request.mode != compute.V41SharedAttentionCompressed || request.compressed.Ratio != 2 {
						t.Fatal("reader used its private ratio instead of the source ratio")
					}
				}
				rows, gathered, heads, dim, scale := v41AttentionPhysicalSelection(t, request)
				if heads > 2 || dim > 512 || len(rows) > 4 {
					t.Fatal("physical callback geometry bound exceeded")
				}
				beforeRequest := v41AttentionPhysicalCopyRequest(request)
				device.Trim()
				before := v41AttentionPhysicalSnapshot(t, backend)
				profile := device.VulkanDebugDispatchProfileSnapshot()
				device.BeginBatch()
				out, err := selected(layer, request)
				device.FlushBatch()
				if err != nil {
					return out, err
				}
				device.Trim()
				after := v41AttentionPhysicalSnapshot(t, backend)
				observation, err := compute.BackendExecutionDelta(before, after)
				if err != nil {
					t.Fatal(err)
				}
				c := observation.Counters
				count, upload, readback := uint64(0), uint64(0), uint64(0)
				if len(rows) > 0 {
					count = 1
					nonempty++
					sinkElements := len(request.sink)
					if request.sink == nil {
						sinkElements = 1
					}
					upload = uint64(4 * (heads*dim + len(gathered) + sinkElements))
					readback = uint64(4*heads*dim + 16*heads)
				}
				if c.ComputeDispatches != count || c.OtherDispatches != count || device.VulkanDebugDispatchProfileSnapshot().OtherAttentionDispatches-profile.OtherAttentionDispatches != count || c.H2DCount != 3*count || c.H2DBytes != upload || c.D2HCount != 2*count || c.D2HBytes != readback || c.D2DCopies != 0 || c.D2DBytes != 0 || c.Fallbacks != 0 || after.DeviceAllocationLiveBytes != before.DeviceAllocationLiveBytes {
					t.Fatalf("actual adapter/status/transfer/lifetime observation=%+v nonempty=%d", observation, count)
				}
				if !reflect.DeepEqual(request, beforeRequest) {
					t.Fatal("selected operation mutated original query/KV/sink/index/publication inputs")
				}
				if lastOutput != nil && !slices.Equal(lastOutput, lastCopy) {
					t.Fatal("subsequent device work changed a previously returned host output")
				}
				want := v41AttentionPhysicalOracle(request.q, gathered, request.sink, heads, dim, scale, request.mode)
				v41AttentionPhysicalNear(t, "contraction", out, want, 2e-5)
				lastOutput, lastCopy = out, slices.Clone(out)
				t.Logf("layer=%d mode=%d selected_rows=%v calls=%d nonempty=%d actual=%+v", layer, request.mode, rows, calls, nonempty, observation)
				return out, nil
			}
			for phase, ids := range [][]int{{1, 2}, {3}, {4}} {
				openingCalls, openingNonempty := calls, nonempty
				attemptsBefore := m.V41ExpertFaultAttribution()
				var got, want []float32
				if phase == 1 {
					got, want = s.Step(ids[0]), host.Step(ids[0])
				} else {
					got, want = s.Prefill(ids), host.Prefill(ids)
				}
				if calls == openingCalls || nonempty == openingNonempty {
					t.Fatal("normal-session phase lacked a real nonempty operation")
				}
				attemptsAfter := m.V41ExpertFaultAttribution()
				attempts := attemptsAfter.Prefill.AttentionContractionCalls + attemptsAfter.Decode.AttentionContractionCalls - attemptsBefore.Prefill.AttentionContractionCalls - attemptsBefore.Decode.AttentionContractionCalls
				if attempts != calls-openingCalls {
					t.Fatalf("attempted contractions=%d actual adapter calls=%d; possible extra host replay", attempts, calls-openingCalls)
				}
				v41AttentionPhysicalNear(t, fmt.Sprintf("session-phase%d", phase), got, want, 1e-4)
				if !reflect.DeepEqual(s.v41State().history, host.v41State().history) {
					t.Fatal("device session prefix publication differs from the host control")
				}
				t.Logf("phase=%d cold=%t callback_calls=%d nonempty_dispatches=%d; sink traffic remains transient on both cold and warm calls", phase, phase == 0, calls-openingCalls, nonempty-openingNonempty)
			}
			if nonempty == 0 || (role && readerCalls == 0) {
				t.Fatal("normal-session reachability witness was vacuous")
			}
			allNonempty += nonempty
		})
	}
	if allNonempty == 0 || allNonempty > 40 {
		t.Fatalf("normal-session operation count=%d", allNonempty)
	}
	t.Logf("V41_ATTENTION_SESSION_PHYSICAL_COMPLETE nonempty_dispatches=%d identity=%+v", allNonempty, identity.Identity)
	// TODO: actual native allocation/submission/fence/D2H/DEVICE_LOST injection
	// requires a separately authorized driver-fault facility. Software fault
	// controls elsewhere do not qualify those physical failure paths.
}

func v41AttentionPhysicalSnapshot(t *testing.T, backend compute.Backend) compute.BackendExecutionSnapshot {
	t.Helper()
	snapshot, available, err := compute.CaptureBackendExecutionSnapshot(backend)
	if err != nil || !available || !snapshot.TransferCountersObserved || !snapshot.DeviceAllocationObserved {
		t.Fatalf("actual identity/transfer/allocation observation unavailable: %+v available=%t err=%v", snapshot, available, err)
	}
	return snapshot
}

// Independent slot filtering: do not call production gather, mask or BF16 code.
func v41AttentionPhysicalSelection(t *testing.T, r v41SharedAttentionRequest) (rows []int, values []float32, heads, dim int, scale float32) {
	t.Helper()
	if r.mode == compute.V41SharedAttentionPlain {
		heads, dim, scale = r.plain.Heads, r.plain.HeadDim, r.plain.Softmax
		limit := len(r.idx)
		if r.plain.TopKLength != nil {
			limit = int(r.plain.TopKLength[0])
		}
		for slot, index := range r.idx {
			if slot < limit && index >= 0 {
				rows = append(rows, int(index))
				values = append(values, r.kv[int(index)*dim:(int(index)+1)*dim]...)
			}
		}
	} else {
		heads, dim, scale = r.compressed.Heads, r.compressed.HeadDim, r.compressed.Softmax
		visible := func(group int) bool { return (group+1)*r.compressed.Ratio <= r.compressed.QueryOffset+1 }
		if r.idx == nil {
			for group := range r.values {
				if visible(group) {
					rows = append(rows, group)
				}
			}
		} else {
			for _, group := range r.idx {
				if group >= 0 && visible(int(group)) {
					rows = append(rows, int(group))
				}
			}
		}
		for _, group := range rows {
			values = append(values, r.values[group]...)
		}
	}
	return
}

func v41AttentionPhysicalCopyRequest(r v41SharedAttentionRequest) v41SharedAttentionRequest {
	r.q, r.kv, r.sink, r.idx = slices.Clone(r.q), slices.Clone(r.kv), slices.Clone(r.sink), slices.Clone(r.idx)
	r.plain.TopKLength = slices.Clone(r.plain.TopKLength)
	cloneRows := func(rows [][]float32) [][]float32 {
		result := slices.Clone(rows)
		for i := range result {
			result[i] = slices.Clone(result[i])
		}
		return result
	}
	r.values = cloneRows(r.values)
	r.compressed.Sink, r.compressed.Idx, r.compressed.SourceRows = slices.Clone(r.compressed.Sink), slices.Clone(r.compressed.Idx), cloneRows(r.compressed.SourceRows)
	return r
}

func v41AttentionPhysicalOracle(q, kv, sink []float32, heads, dim int, scale float32, mode compute.V41SharedAttentionMode) []float32 {
	result := make([]float32, heads*dim)
	if len(kv) == 0 {
		return result
	}
	f32 := func(x float64) float64 { return float64(float32(x)) }
	for head := 0; head < heads; head++ {
		seed := math.Inf(-1)
		if mode == compute.V41SharedAttentionCompressed {
			seed = -math.MaxFloat32
		}
		maximum := seed
		if sink != nil {
			maximum = float64(sink[head])
		}
		scores := make([]float64, len(kv)/dim)
		for row := range scores {
			for d := 0; d < dim; d++ {
				scores[row] = f32(scores[row] + f32(float64(q[head*dim+d])*float64(kv[row*dim+d])))
			}
			scores[row] = f32(scores[row] * float64(scale))
			maximum = math.Max(maximum, scores[row])
		}
		if maximum == seed {
			continue
		}
		denominator := float64(0)
		if sink != nil {
			denominator = f32(math.Exp(f32(float64(sink[head]) - maximum)))
		}
		for _, score := range scores {
			denominator = f32(denominator + f32(math.Exp(f32(score-maximum))))
		}
		for d := 0; d < dim; d++ {
			sum := float64(0)
			for row, score := range scores {
				weight := f32(f32(math.Exp(f32(score-maximum))) / denominator)
				sum = f32(sum + f32(weight*float64(kv[row*dim+d])))
			}
			result[head*dim+d] = float32(sum)
		}
	}
	return result
}

func v41AttentionPhysicalNear(t *testing.T, label string, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) == 0 || len(got) != len(want) {
		t.Fatalf("%s shape=%d/%d", label, len(got), len(want))
	}
	var maxAbs, maxRel float64
	for i := range want {
		a, b := float64(got[i]), float64(want[i])
		if math.IsNaN(a) || math.IsInf(a, 0) || math.IsNaN(b) || math.IsInf(b, 0) {
			t.Fatalf("%s nonfinite comparison at %d", label, i)
		}
		delta := math.Abs(a - b)
		maxAbs = math.Max(maxAbs, delta)
		maxRel = math.Max(maxRel, delta/math.Max(1e-30, math.Abs(b)))
		if delta > tolerance+tolerance*math.Abs(b) {
			t.Fatalf("%s[%d]=%g want=%g delta=%g", label, i, a, b, delta)
		}
	}
	t.Logf("%s max_abs=%g max_rel=%g", label, maxAbs, maxRel)
}

func v41AttentionPhysicalMaskControls(t *testing.T, device v41AttentionPhysicalBackend) {
	t.Helper()
	newSession := func() *Session { s := &Session{M: &Model{}, Backend: device}; t.Cleanup(s.Close); return s }
	s := newSession()
	callback := s.v41SharedAttentionFunc()
	if callback == nil {
		t.Fatal("physical adapter unavailable")
	}
	q, kv, sink, idx := []float32{0, 0}, []float32{4, -8, 12, 4}, []float32{0}, []int32{1, 1, 0, -1}
	options := V41SparseAttentionSinkOptions{B: 1, M: 1, Heads: 1, HeadDim: 2, N: 2, TopK: 4, Softmax: 1}
	for _, length := range []int32{4, 2, 0} {
		options.TopKLength = []int32{length}
		before := v41AttentionPhysicalSnapshot(t, device)
		profile := device.VulkanDebugDispatchProfileSnapshot()
		device.BeginBatch()
		got, err := v41SparseAttentionSinkWithDevice(0, q, kv, sink, idx, options, callback)
		device.FlushBatch()
		if err != nil {
			t.Fatal(err)
		}
		want := []float32{7, 0}
		if length == 2 {
			want = []float32{8, float32(8.0 / 3)}
		}
		if length == 0 {
			want = []float32{0, 0}
		}
		v41AttentionPhysicalNear(t, "independent-duplicate-length-mask", got, want, 2e-5)
		if length != 2 {
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatal("exact analytic/empty control failed")
				}
			}
		}
		after := v41AttentionPhysicalSnapshot(t, device)
		if length > 0 {
			rows := int(length)
			if length == 4 {
				rows = 3
			}
			delta, err := compute.BackendExecutionDelta(before, after)
			if err != nil || delta.Counters.ComputeDispatches != 1 || device.VulkanDebugDispatchProfileSnapshot().OtherAttentionDispatches-profile.OtherAttentionDispatches != 1 || delta.Counters.H2DCount != 3 || delta.Counters.H2DBytes != uint64(4*(2+rows*2+1)) || delta.Counters.D2HCount != 2 || delta.Counters.D2HBytes != 24 || delta.Counters.Fallbacks != 0 {
				t.Fatalf("mask control actual dispatch/transfers=%+v err=%v", delta, err)
			}
		}
		if length == 0 && (before.Counters != after.Counters || before.DeviceAllocationLiveBytes != after.DeviceAllocationLiveBytes) {
			t.Fatal("empty selection uploaded/dispatched/read/allocated")
		}
		if length == 0 {
			got[0] = 123
			fresh, err := v41SparseAttentionSinkWithDevice(0, q, kv, sink, idx, options, callback)
			if err != nil || fresh[0] != 0 {
				t.Fatal("empty output is not freshly owned")
			}
		}
	}
	for _, bad := range []string{"masked-index", "unused-nonfinite-row"} {
		broken := newSession()
		badKV, badIdx := slices.Clone(kv), slices.Clone(idx)
		if bad == "masked-index" {
			badIdx[3] = 2
		} else {
			badKV[0] = float32(math.NaN())
		}
		options.TopKLength = []int32{0}
		before := v41AttentionPhysicalSnapshot(t, device)
		out, err := v41SparseAttentionSinkWithDevice(3, q, badKV, sink, badIdx, options, broken.v41SharedAttentionFunc())
		after := v41AttentionPhysicalSnapshot(t, device)
		var selected *V41SharedAttentionOperationError
		var closed *BackendForwardOperationError
		if out != nil || !errors.As(err, &selected) || !errors.As(err, &closed) || closed.Path != "v41-shared-attention" || before.Counters != after.Counters {
			t.Fatalf("original input validation did not fail before device work: %s err=%v", bad, err)
		}
	}
	// A genuine finite-input arithmetic fault maps duplicate slot 0 to original
	// group 1. This is not an injected or fabricated driver failure.
	broken := newSession()
	beforeFault := v41AttentionPhysicalSnapshot(t, device)
	profileFault := device.VulkanDebugDispatchProfileSnapshot()
	device.BeginBatch()
	out, err := v41AttentionCompressedForwardWithDevice([]float32{math.MaxFloat32}, [][]float32{{1}, {2}}, V41AttentionSharedKVOptions{Layer: 7, Ratio: 1, QueryOffset: 1, Groups: 2, HeadDim: 1, Heads: 1, Softmax: 1, Idx: []int32{1, 1, 0}, IndexTopK: 3}, broken.v41SharedAttentionFunc())
	device.FlushBatch()
	var arithmetic *compute.V41SharedAttentionArithmeticError
	var closed *BackendForwardOperationError
	if out != nil || !errors.As(err, &arithmetic) || !errors.As(err, &closed) || closed.Layer != 7 || closed.Path != "v41-shared-attention" || arithmetic.Stage != compute.V41SharedAttentionStageScore || arithmetic.Head != 0 || arithmetic.SelectedSlot != 0 || arithmetic.Element != -1 || arithmetic.ValueBits != 0x7f800000 || !strings.Contains(err.Error(), "score accumulate") || !strings.Contains(err.Error(), "t=0 h=0 group=1") {
		t.Fatalf("physical fault attribution lost: %+v err=%v", arithmetic, err)
	}
	delta, deltaErr := compute.BackendExecutionDelta(beforeFault, v41AttentionPhysicalSnapshot(t, device))
	if deltaErr != nil || delta.Counters.ComputeDispatches != 1 || device.VulkanDebugDispatchProfileSnapshot().OtherAttentionDispatches-profileFault.OtherAttentionDispatches != 1 || delta.Counters.H2DCount != 3 || delta.Counters.H2DBytes != 20 || delta.Counters.D2HCount != 1 || delta.Counters.D2HBytes != 16 || delta.Counters.Fallbacks != 0 {
		t.Fatalf("actual selected error-status traffic=%+v err=%v", delta, deltaErr)
	}
}
