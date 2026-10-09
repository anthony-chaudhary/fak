//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"
)

// This is deferred witness SOURCE, not an execution receipt. Run serially with
// FAK_V41_SHARED_ATTENTION_WITNESS=1, FAK_VULKAN_REQUIRE_DEVICE=1,
// FAK_VULKAN_EXPECT_DEVICE=<exact reported device name>, and both
// FAK_VULKAN_DISPATCH_PROFILE=1 and FAK_VULKAN_TIMESTAMP_PROFILE=1,
// FAK_VULKAN_TIMESTAMP_PROFILE_EVERY=1 before process startup. The parent checks
// the child's actual native stderr timestamp report; a skip is never evidence.
// Bound: <=32 tiny dispatches, H<=3, R<=3, D<=512, child deadline 45 seconds.
// Runtime estimate is unmeasured. No checkpoint-quality or speedup claim follows.
// fak-test:runtime integration est=45s lane=optin
func TestVulkanV41SharedAttentionPhysical(t *testing.T) {
	if os.Getenv("FAK_V41_SHARED_ATTENTION_WITNESS") != "1" {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required-device run needs explicit FAK_V41_SHARED_ATTENTION_WITNESS=1")
		}
		t.Skip("deferred physical shared-attention witness is opt-in")
	}
	for _, key := range []string{"FAK_VULKAN_REQUIRE_DEVICE", "FAK_VULKAN_DISPATCH_PROFILE", "FAK_VULKAN_TIMESTAMP_PROFILE", "FAK_VULKAN_TIMESTAMP_PROFILE_EVERY"} {
		if os.Getenv(key) != "1" {
			t.Fatalf("physical witness requires %s=1 before process startup", key)
		}
	}
	if os.Getenv("FAK_VULKAN_EXPECT_DEVICE") == "" {
		t.Fatal("physical witness needs the exact expected device name")
	}
	const childKey = "FAK_V41_ATTENTION_COMPUTE_CHILD"
	if os.Getenv(childKey) != "1" {
		executable, err := os.Executable()
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, executable, "-test.run=^TestVulkanV41SharedAttentionPhysical$", "-test.count=1", "-test.v", "-test.timeout=40s")
		cmd.Env = append(os.Environ(), childKey+"=1")
		output, err := cmd.CombinedOutput()
		t.Logf("isolated physical witness output:\n%s", output)
		if err != nil {
			t.Fatalf("physical child failed: %v (deadline=%v)", err, ctx.Err())
		}
		if strings.Contains(string(output), "--- SKIP:") || !strings.Contains(string(output), "V41_ATTENTION_PHYSICAL_COMPLETE") {
			t.Fatal("child did not complete a nonempty physical operation")
		}
		timestamp := regexp.MustCompile(`fak-vulkan ts-stage kernel=v41_shared_attention groups=\d+x\d+ calls/batch=([0-9.]+) ms/batch=([0-9.]+) us/call=([0-9.]+)`)
		measured := false
		for _, match := range timestamp.FindAllStringSubmatch(string(output), -1) {
			calls, e1 := strconv.ParseFloat(match[1], 64)
			micros, e2 := strconv.ParseFloat(match[3], 64)
			measured = measured || (e1 == nil && e2 == nil && calls > 0 && micros > 0 && !math.IsInf(micros, 0) && !math.IsNaN(micros))
		}
		if !measured {
			t.Fatal("no positive dedicated device timestamp; unavailable/rounded-zero timing does not qualify")
		}
		return
	}

	v := vk(t)
	kind := v.VulkanPhysicalDeviceType()
	if kind != 1 && kind != 2 {
		t.Fatalf("required integrated/discrete GPU unavailable: device type=%d", kind)
	}
	if !v.SupportsV41SharedAttention() {
		t.Fatal("mandatory current Vulkan shared-attention pipeline unavailable")
	}
	identity, err := v.BackendExecutionSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	if identity.Identity.Backend != "vulkan" || identity.Identity.Device != os.Getenv("FAK_VULKAN_EXPECT_DEVICE") || !identity.TransferCountersObserved || !identity.DeviceAllocationObserved {
		t.Fatalf("physical identity/counter requirement failed: %+v", identity)
	}
	t.Logf("physical identity=%+v device_type=%d; fixed F32 abs/rel=2e-5, mathematical abs/rel=8e-5; cost unmeasured", identity.Identity, kind)
	if Default().Name() != "cpu-ref" {
		t.Fatal("host fixture constructor must remain cpu-ref")
	}
	// Account for persistent staging separately; native Free retains idle device
	// allocations in pools, so equality is checked only after an explicit Trim.
	setup := v.UploadClass(NewF32(Default(), []int{1536}, make([]float32, 1536)), F32, MemoryActivation, "shared attention witness staging setup")
	_ = v.Read(setup)
	v.Free(setup)
	v.Trim()
	setupAfter, err := v.BackendExecutionSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	setupDelta, err := BackendExecutionDelta(identity, setupAfter)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("actual staging setup (outside contraction windows): %+v", setupDelta)

	type fixture struct {
		name               string
		h, d               int
		q, kv, sink, exact []float32
	}
	corpus := []fixture{
		{"nil-sink", 2, 2, []float32{0, 0, 0, 0}, []float32{4, -8, 12, 4}, nil, []float32{8, -2, 8, -2}},
		{"unequal-sinks", 2, 2, []float32{0, 0, 0, 0}, []float32{4, -8, 12, 4}, []float32{0, float32(math.Ln2)}, nil},
		{"duplicate-slots", 1, 2, []float32{0, 0}, []float32{12, 4, 12, 4, 4, -8}, []float32{0}, []float32{7, 0}},
		{"huge-finite-score", 1, 1, []float32{1e30}, []float32{1}, nil, []float32{1}},
		{"huge-finite-zero-sink", 1, 1, []float32{1e30}, []float32{1}, []float32{0}, []float32{1}},
		{"finite-subtraction-negative-inf", 1, 1, []float32{math.MaxFloat32}, []float32{1, -1}, nil, []float32{1}},
		{"negative-nondefault-scale", 2, 2, []float32{0.25, -0.75, 0.13, 0.27}, []float32{4, -8, 12, 4}, []float32{0.5, -0.25}, nil},
	}
	wide := fixture{name: "bf16-literals-non-bf16-query", h: 2, d: 512, q: make([]float32, 1024), kv: make([]float32, 1536), sink: []float32{0, -0.75}}
	for i := range wide.q {
		wide.q[i] = float32((i%23)-11)*0.00371 + 0.000013
	}
	for i := range wide.kv {
		// Independently widened BF16 words; no production quantization helper.
		words := [...]uint32{0x3e80, 0xbe00, 0x3f00, 0xbf40, 0, 0x3d80, 0xbd80}
		wide.kv[i] = math.Float32frombits(words[i%len(words)] << 16)
	}
	corpus = append(corpus, wide)
	operations := 0
	for _, mode := range []V41SharedAttentionMode{V41SharedAttentionPlain, V41SharedAttentionCompressed} {
		cases := append([]fixture(nil), corpus...)
		for _, sink := range [][]float32{nil, {-math.MaxFloat32}} {
			want := float32(1)
			if sink != nil {
				want = 0.5
			}
			if mode == V41SharedAttentionCompressed {
				want = 0
			}
			cases = append(cases, fixture{fmt.Sprintf("sentinel-sink-%t", sink != nil), 1, 1, []float32{-math.MaxFloat32}, []float32{1}, sink, []float32{want}})
		}
		for _, tc := range cases {
			t.Run(fmt.Sprintf("mode%d/%s", mode, tc.name), func(t *testing.T) {
				v.Trim()
				before, err := v.BackendExecutionSnapshot()
				if err != nil {
					t.Fatal(err)
				}
				profile := v.VulkanDebugDispatchProfileSnapshot()
				sink := tc.sink
				if sink == nil {
					sink = []float32{0}
				}
				upload := func(shape []int, values []float32) Tensor {
					return v.UploadClass(NewF32(Default(), shape, values), F32, MemoryActivation, "shared attention witness input")
				}
				q := upload([]int{tc.h, tc.d}, tc.q)
				kv := upload([]int{len(tc.kv) / tc.d, tc.d}, tc.kv)
				st := upload([]int{len(sink)}, sink)
				defer v.Free(q)
				defer v.Free(kv)
				defer v.Free(st)
				v.BeginBatch()
				defer v.FlushBatch()
				scale := float32(1)
				if tc.name == "negative-nondefault-scale" {
					scale = -0.75
				}
				if tc.d == 512 {
					scale = 0.37
				}
				out, err := v.V41SharedAttention(q, kv, st, len(tc.kv)/tc.d, tc.h, tc.d, scale, mode, tc.sink != nil)
				operations++
				if err != nil {
					t.Fatal(err)
				}
				if out.Buf() == nil || out.Buf() == q.Buf() || out.Buf() == kv.Buf() || out.Buf() == st.Buf() || !out.buf.(*vulkanBuf).v41CheckedRead {
					t.Fatal("output ownership/checked-read marker invalid")
				}
				got := v.Read(out)
				v.Free(out)
				v.FlushBatch()
				after, err := v.BackendExecutionSnapshot()
				if err != nil {
					t.Fatal(err)
				}
				observation, err := BackendExecutionDelta(before, after)
				if err != nil {
					t.Fatal(err)
				}
				c := observation.Counters
				if c.ComputeDispatches != 1 || c.OtherDispatches != 1 || v.VulkanDebugDispatchProfileSnapshot().OtherAttentionDispatches-profile.OtherAttentionDispatches != 1 || c.H2DCount != 3 || c.H2DBytes != uint64(4*(len(tc.q)+len(tc.kv)+len(sink))) || c.D2HCount != 2 || c.D2HBytes != uint64(16*tc.h+4*len(tc.q)) || c.D2DCopies != 0 || c.Fallbacks != 0 {
					t.Fatalf("operation/status/transfer accounting: %+v", observation)
				}
				want32 := v41PhysicalAttentionOracle(tc.q, tc.kv, tc.sink, tc.h, tc.d, scale, mode, true)
				want64 := v41PhysicalAttentionOracle(tc.q, tc.kv, tc.sink, tc.h, tc.d, scale, mode, false)
				v41PhysicalAttentionCompare(t, got, want32, 2e-5, "rounded-F32")
				v41PhysicalAttentionCompare(t, got, want64, 8e-5, "independent-mathematical")
				if tc.exact != nil {
					v41PhysicalAttentionBits(t, got, tc.exact)
				}
				// Diagnostic reads are outside the isolated transfer window.
				v41PhysicalAttentionBits(t, v.Read(q), tc.q)
				v41PhysicalAttentionBits(t, v.Read(kv), tc.kv)
				v41PhysicalAttentionBits(t, v.Read(st), sink)
				v.Free(q)
				v.Free(kv)
				v.Free(st)
				v.Trim()
				closed, err := v.BackendExecutionSnapshot()
				if err != nil || closed.DeviceAllocationLiveBytes != before.DeviceAllocationLiveBytes {
					t.Fatalf("transient allocation leak: before=%d after=%d err=%v", before.DeviceAllocationLiveBytes, closed.DeviceAllocationLiveBytes, err)
				}
				t.Logf("actual operation observation=%+v", observation)
			})
		}
	}
	// Finite inputs still overflow. Both modes must retain exact first producer
	// coordinates/bits and publish no output, including the negative overflow.
	for _, mode := range []V41SharedAttentionMode{V41SharedAttentionPlain, V41SharedAttentionCompressed} {
		for _, sign := range []float32{1, -1} {
			v.Trim()
			q := v.UploadClass(NewF32(Default(), []int{3, 1}, []float32{1, sign * math.MaxFloat32, math.MaxFloat32}), F32, MemoryActivation, "shared attention overflow query")
			kv := v.UploadClass(NewF32(Default(), []int{3, 1}, []float32{0, 2, 2}), F32, MemoryActivation, "shared attention overflow KV")
			sink := v.UploadClass(NewF32(Default(), []int{1}, []float32{0}), F32, MemoryActivation, "shared attention overflow sink")
			before, err := v.BackendExecutionSnapshot()
			if err != nil {
				t.Fatal(err)
			}
			profile := v.VulkanDebugDispatchProfileSnapshot()
			v.BeginBatch()
			out, err := v.V41SharedAttention(q, kv, sink, 3, 3, 1, 1, mode, false)
			operations++
			v.FlushBatch()
			var arithmetic *V41SharedAttentionArithmeticError
			if out.Buf() != nil || !errors.As(err, &arithmetic) || arithmetic.Stage != V41SharedAttentionStageScore || arithmetic.Head != 1 || arithmetic.SelectedSlot != 1 || arithmetic.Element != -1 || arithmetic.ValueBits != math.Float32bits(float32(math.Inf(int(sign)))) {
				t.Fatalf("first arithmetic failure=%+v err=%v", arithmetic, err)
			}
			v.Trim() // idle output/status pooling is not a leaked live owner
			after, snapErr := v.BackendExecutionSnapshot()
			if snapErr != nil {
				t.Fatal(snapErr)
			}
			delta, deltaErr := BackendExecutionDelta(before, after)
			if deltaErr != nil || delta.Counters.ComputeDispatches != 1 || delta.Counters.D2HCount != 1 || delta.Counters.D2HBytes != 48 || delta.Counters.H2DCount != 0 || delta.Counters.Fallbacks != 0 || after.DeviceAllocationLiveBytes != before.DeviceAllocationLiveBytes || v.VulkanDebugDispatchProfileSnapshot().OtherAttentionDispatches-profile.OtherAttentionDispatches != 1 {
				t.Fatalf("failed operation did not fail closed: %+v err=%v", delta, deltaErr)
			}
			v.Free(q)
			v.Free(kv)
			v.Free(sink)
		}
	}
	v41PhysicalAttentionNativeControls(t, v)
	operations++ // the nonempty native status-initialization control below
	if operations == 0 || operations > 32 {
		t.Fatalf("operation bound invalid: %d", operations)
	}
	t.Logf("V41_ATTENTION_PHYSICAL_COMPLETE operations=%d identity=%+v", operations, identity.Identity)
	// TODO: actual descriptor/command/fence failure, DEVICE_LOST, malformed
	// device status, and failed D2H injection need a separately authorized native
	// driver-fault facility. This test neither fabricates failures nor claims them.
}

// Deliberately separate scalar mathematics and explicitly rounded binary32
// arithmetic. Neither reference calls production attention/selection helpers.
func v41PhysicalAttentionOracle(q, kv, sink []float32, heads, dim int, scale float32, mode V41SharedAttentionMode, rounded bool) []float32 {
	round := func(x float64) float64 {
		if rounded {
			return float64(float32(x))
		}
		return x
	}
	rows, out := len(kv)/dim, make([]float32, len(q))
	for h := 0; h < heads; h++ {
		seed := math.Inf(-1)
		if mode == V41SharedAttentionCompressed {
			seed = -math.MaxFloat32
		}
		maximum := seed
		if sink != nil {
			maximum = float64(sink[h])
		}
		scores := make([]float64, rows)
		for r := range scores {
			for d := 0; d < dim; d++ {
				scores[r] = round(scores[r] + round(float64(q[h*dim+d])*float64(kv[r*dim+d])))
			}
			scores[r] = round(scores[r] * float64(scale))
			maximum = math.Max(maximum, scores[r])
		}
		if maximum == seed {
			continue
		}
		denom := float64(0)
		if sink != nil {
			denom = round(math.Exp(round(float64(sink[h]) - maximum)))
		}
		for _, score := range scores {
			denom = round(denom + round(math.Exp(round(score-maximum))))
		}
		for d := 0; d < dim; d++ {
			value := float64(0)
			for r, score := range scores {
				weight := round(round(math.Exp(round(score-maximum))) / denom)
				value = round(value + round(weight*float64(kv[r*dim+d])))
			}
			out[h*dim+d] = float32(value)
		}
	}
	return out
}

func v41PhysicalAttentionCompare(t *testing.T, got, want []float32, tolerance float64, label string) {
	t.Helper()
	if len(got) == 0 || len(got) != len(want) {
		t.Fatalf("%s output shape=%d want=%d", label, len(got), len(want))
	}
	maxAbs, maxRel := float64(0), float64(0)
	for i, expected := range want {
		a, b := float64(got[i]), float64(expected)
		if math.IsNaN(a) || math.IsInf(a, 0) || math.IsNaN(b) || math.IsInf(b, 0) {
			t.Fatalf("%s nonfinite lane %d", label, i)
		}
		delta := math.Abs(a - b)
		maxAbs = math.Max(maxAbs, delta)
		maxRel = math.Max(maxRel, delta/math.Max(1e-30, math.Abs(b)))
		if delta > tolerance+tolerance*math.Abs(b) {
			t.Fatalf("%s lane %d got=%g want=%g delta=%g", label, i, a, b, delta)
		}
	}
	t.Logf("%s max_abs=%g max_rel=%g", label, maxAbs, maxRel)
}

func v41PhysicalAttentionBits(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("bit comparison shape=%d/%d", len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("bits[%d]=%08x want=%08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

func v41PhysicalAttentionNativeControls(t *testing.T, v *vulkanBackend) {
	t.Helper()
	upload := func(shape []int, values []float32) Tensor {
		x := v.UploadClass(NewF32(Default(), shape, values), F32, MemoryActivation, "shared attention native control")
		t.Cleanup(func() { v.Free(x) })
		return x
	}
	q := upload([]int{2, 2}, []float32{0, 0, 0, 0})
	kv := upload([]int{1, 2}, []float32{4, -8})
	sink := upload([]int{1}, []float32{0})
	out := upload([]int{2, 2}, []float32{99, 99, 99, 99})
	poison := make([]float32, 8)
	for i := range poison {
		poison[i] = math.Float32frombits(0xffffffff)
	}
	status := upload([]int{8}, poison)
	before, err := v.BackendExecutionSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	// All handles below came from genuine allocations. Never invent C pointers.
	for _, pair := range [][2]Tensor{{q, status}, {out, out}, {sink, status}, {out, sink}} {
		code := func() int {
			vulkanMu.Lock()
			defer vulkanMu.Unlock()
			return v41SharedAttentionStatusLocked(q.buf.(*vulkanBuf), kv.buf.(*vulkanBuf), sink.buf.(*vulkanBuf), pair[0].buf.(*vulkanBuf), pair[1].buf.(*vulkanBuf), 1, 2, 2, 1, V41SharedAttentionPlain, false)
		}()
		if code != 2 {
			t.Fatalf("native alias/extent refusal code=%d want=2", code)
		}
	}
	malformed := q
	malformed.Shape = []int{4}
	for _, bad := range []Tensor{malformed, NewF32(Default(), []int{2, 2}, make([]float32, 4))} {
		x, err := v.V41SharedAttention(bad, kv, sink, 1, 2, 2, 1, V41SharedAttentionPlain, false)
		if err == nil || x.Buf() != nil {
			t.Fatal("invalid Go operand published output")
		}
	}
	after, err := v.BackendExecutionSnapshot()
	if err != nil || after.Counters != before.Counters || after.DeviceAllocationLiveBytes != before.DeviceAllocationLiveBytes {
		t.Fatal("pre-dispatch refusal changed counters or allocations")
	}
	v.BeginBatch()
	code := func() int {
		vulkanMu.Lock()
		defer vulkanMu.Unlock()
		return v41SharedAttentionStatusLocked(q.buf.(*vulkanBuf), kv.buf.(*vulkanBuf), sink.buf.(*vulkanBuf), out.buf.(*vulkanBuf), status.buf.(*vulkanBuf), 1, 2, 2, 1, V41SharedAttentionPlain, false)
	}()
	if code != 0 {
		v.FlushBatch()
		t.Fatalf("native initialization control=%d", code)
	}
	words := v.Read(status)
	got := v.Read(out)
	v.FlushBatch()
	for i, word := range words {
		if math.Float32bits(word) != 0 {
			t.Fatalf("status word %d not reset: %08x", i, math.Float32bits(word))
		}
	}
	v41PhysicalAttentionBits(t, got, []float32{4, -8, 4, -8})
}
