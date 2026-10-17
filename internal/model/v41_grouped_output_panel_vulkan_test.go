//go:build vulkan && linux && cgo

package model

import (
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41GroupedPanelPhysicalBackend interface {
	compute.Backend
	compute.V41StepBatchBackend
	VulkanPhysicalDeviceType() uint32
	VulkanDebugDispatchProfileSnapshot() compute.VulkanDispatchProfile
	Trim()
}

// This is a synthetic composition witness for #12640/#13864, not a checkpoint,
// quality, or throughput qualification. It changes no production selector.
// Rebuild the Vulkan SPIR-V bundle from the same source as the tagged test binary.
// Before process startup set FAK_V41_GROUPED_PANEL_WITNESS=1,
// FAK_VULKAN_REQUIRE_DEVICE=1, FAK_VULKAN_EXPECT_DEVICE=<exact device identity>,
// FAK_VULKAN_DISPATCH_PROFILE=1, FAK_VULKAN_TIMESTAMP_PROFILE=1,
// FAK_VULKAN_TIMESTAMP_PROFILE_EVERY=1, FAK_VULKAN_Q2K_MATVEC=1,
// FAK_STRIX_WAVE32_GEMV_DECODE=0, FAK_V41_Q2K_TOKEN_PANEL=1,
// FAK_V41_Q3K_TOKEN_PANEL=1, FAK_V41_DEFERRED_BF16_PUBLISH=1, and
// FAK_VULKAN_SPIRV=<fresh bundle>. Run serially with:
// go test -tags vulkan ./internal/model -run '^TestV41GroupedOutputPanelVulkan$' -count=1 -timeout=5m
//
// Four isolated, serial children each own one packed fixture and two sequential
// Sessions. Eleven tokens exercise one admitted 11-row chunk; legacy 8+2+1
// refusal coverage remains in TestV41GroupedOutputPanelPackedRagged.
// Full A=[1024,4096] x8 and
// B=[5120,8192], or odd B=[65,256] with A=[128,512] x2. A is Q2_K; B
// covers Q2_K and Q3_K. Both graphs use full=true BF16 publication semantics.
// Total: 288 matmuls, 44 compared rows, 60s/child. Runtime is unmeasured.
// The four-row composition tile is not covered by this bounded witness.
//
// Largest packed payload is 29,032,448 bytes, shared once between the arms.
// Device planning also includes the 256 MiB weight-arena block, <=32 MiB
// transfer staging, 4,415,760 bytes retained by the largest tile, scalar-tile
// pools, scalar publication/readback, Q3 word padding and buffer alignment.
// Tile bytes = 4*n*(2*totalIn+3*joined+2*dim)+(2*groups+1)*16.
// Host planning includes packed bytes, input/snapshot, scalar and panel outputs,
// publication/readback copies, cache metadata and 8192 timestamp slots. There
// is no full F32 weight copy. External admission requires >=4 GiB host headroom;
// each child requires >=1 GiB observed free device memory and limits observed
// allocation growth to 512 MiB. Driver/query-pool/compiler memory is additional,
// not represented by the backend's allocation counter. No performance claim
// may use this instrumented, subprocess-isolated run.
// fak-test:runtime integration est=4m lane=optin
func TestV41GroupedOutputPanelVulkan(t *testing.T) {
	if os.Getenv("FAK_V41_GROUPED_PANEL_WITNESS") != "1" {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("UNQUALIFIED: required-device run needs FAK_V41_GROUPED_PANEL_WITNESS=1")
		}
		t.Skip("UNQUALIFIED: native grouped-panel composition witness is opt-in")
	}
	for _, key := range []string{
		"FAK_VULKAN_REQUIRE_DEVICE", "FAK_VULKAN_DISPATCH_PROFILE",
		"FAK_VULKAN_TIMESTAMP_PROFILE", "FAK_VULKAN_TIMESTAMP_PROFILE_EVERY",
		"FAK_VULKAN_Q2K_MATVEC", "FAK_V41_Q2K_TOKEN_PANEL",
		"FAK_V41_Q3K_TOKEN_PANEL", "FAK_V41_DEFERRED_BF16_PUBLISH",
	} {
		if os.Getenv(key) != "1" {
			t.Fatalf("UNQUALIFIED: requires %s=1 before process startup", key)
		}
	}
	if os.Getenv("FAK_STRIX_WAVE32_GEMV_DECODE") != "0" || os.Getenv("FAK_VULKAN_EXPECT_DEVICE") == "" || os.Getenv("FAK_VULKAN_SPIRV") == "" {
		t.Fatal("UNQUALIFIED: require wave32=0, exact expected device, and same-source SPIR-V bundle")
	}
	const childKey = "FAK_V41_GROUPED_PANEL_CHILD"
	cases := []string{"full-q2", "full-q3", "odd-q2", "odd-q3"}
	if child := os.Getenv(childKey); child != "" {
		for _, name := range cases {
			if child == name {
				v41GroupedPanelPhysicalCase(t, name)
				return
			}
		}
		t.Fatalf("invalid isolated grouped-panel case %q", child)
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range cases {
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, executable, "-test.run=^TestV41GroupedOutputPanelVulkan$", "-test.count=1", "-test.v", "-test.timeout=55s")
			cmd.Env = append(os.Environ(), childKey+"="+name)
			output, err := cmd.CombinedOutput()
			t.Logf("isolated %s output:\n%s", name, output)
			if err != nil {
				t.Fatalf("UNQUALIFIED: child failed: %v (deadline=%v)", err, ctx.Err())
			}
			if strings.Contains(string(output), "--- SKIP:") || !strings.Contains(string(output), "V41_GROUPED_PANEL_PHYSICAL_COMPLETE "+name) {
				t.Fatal("UNQUALIFIED: child did not complete the nonempty physical comparison")
			}
			v41GroupedPanelPhysicalTrace(t, string(output), strings.HasSuffix(name, "q3"))
		})
	}
}

// Native timestamp reports are cumulative and rounded, not exact call counts.
// Fresh children, scalar-before-panel order and independent dispatch deltas bind
// first-use route evidence. Reject incomplete top-40/query-slot reports before
// interpreting the absence of the Q2 serial fallback as evidence.
func v41GroupedPanelPhysicalTrace(t *testing.T, output string, q3 bool) {
	t.Helper()
	seen := map[string]map[string]bool{"scalar": {}, "panel": {}}
	begins, ends := map[string]int{}, map[string]int{}
	arm, reportArm, entries := "", "", 0
	finishReport := func() {
		if reportArm != "" && (entries == 0 || entries >= 40) {
			t.Fatal("UNQUALIFIED: empty or top-40-limited native timestamp report")
		}
		reportArm, entries = "", 0
	}
	for _, line := range strings.Split(output, "\n") {
		if strings.HasPrefix(line, "V41_GROUPED_PANEL_BEGIN ") {
			finishReport()
			if arm != "" {
				t.Fatal("nested physical trace arm")
			}
			arm = strings.TrimPrefix(line, "V41_GROUPED_PANEL_BEGIN ")
			if seen[arm] == nil {
				t.Fatal("unknown physical trace arm")
			}
			begins[arm]++
		} else if strings.HasPrefix(line, "V41_GROUPED_PANEL_END ") {
			finishReport()
			if arm == "" || line != "V41_GROUPED_PANEL_END "+arm {
				t.Fatal("mismatched physical trace arm")
			}
			ends[arm]++
			arm = ""
		} else if strings.HasPrefix(line, "fak-vulkan ts-profile ") {
			finishReport()
			if arm == "" {
				t.Fatal("timestamp report outside owned arm")
			}
			fields := strings.Fields(line)
			if fields[len(fields)-1] != "truncated=0" {
				t.Fatal("UNQUALIFIED: truncated native timestamp report")
			}
			reportArm = arm
		} else if strings.HasPrefix(line, "fak-vulkan ts-stage ") {
			if reportArm == "" || reportArm != arm {
				t.Fatal("unbound native kernel record")
			}
			kernel, calls := "", -1.0
			for _, field := range strings.Fields(line) {
				if strings.HasPrefix(field, "kernel=") {
					kernel = strings.TrimPrefix(field, "kernel=")
				}
				if strings.HasPrefix(field, "calls/batch=") {
					calls, _ = strconv.ParseFloat(strings.TrimPrefix(field, "calls/batch="), 64)
				}
			}
			if kernel == "q2k_matmul" {
				t.Fatal("UNQUALIFIED: actual Q2 scalar fallback kernel executed")
			}
			if kernel == "" || !(calls > 0) || math.IsInf(calls, 0) {
				t.Fatal("UNQUALIFIED: malformed native kernel observation")
			}
			entries++
			seen[arm][kernel] = true
		}
	}
	finishReport()
	for _, name := range []string{"scalar", "panel"} {
		if begins[name] != 1 || ends[name] != 1 {
			t.Fatalf("incomplete %s trace boundary", name)
		}
	}
	if arm != "" || !seen["scalar"]["q2k_matvec"] || !seen["panel"]["v41_q2k_token_panel"] ||
		seen["scalar"]["v41_q2k_token_panel"] || seen["scalar"]["v41_q3k_token_panel"] ||
		(q3 && (!seen["scalar"]["q3k_matmul"] || !seen["panel"]["v41_q3k_token_panel"])) {
		t.Fatal("UNQUALIFIED: actual scalar mode-0 and panel kernel routes were not established")
	}
}

func v41GroupedPanelPhysicalSnapshot(t *testing.T, be compute.Backend) compute.BackendExecutionSnapshot {
	t.Helper()
	s, available, err := compute.CaptureBackendExecutionSnapshot(be)
	if err != nil || !available || !s.TransferCountersObserved || !s.DeviceAllocationObserved {
		t.Fatalf("UNQUALIFIED: native identity/counters/allocation unavailable: available=%t error=%v", available, err)
	}
	return s
}

func v41GroupedPanelPhysicalOwned(t *testing.T, s *Session, device v41GroupedPanelPhysicalBackend, arm string, call func()) {
	t.Helper()
	scope := s.beginBackendExecution()
	complete := false
	defer scope.finishCompleted(&complete)
	if scope == nil || s.backendExecutionScope != scope {
		t.Fatal("actual Vulkan session owner was not acquired")
	}
	fmt.Fprintln(os.Stderr, "V41_GROUPED_PANEL_BEGIN "+arm)
	if !device.BeginV41StepBatch() {
		t.Fatal("UNQUALIFIED: native step batch unavailable")
	}
	defer func() {
		if err := device.EndV41StepBatch(); err != nil {
			t.Errorf("native step batch retirement: %v", err)
		}
		fmt.Fprintln(os.Stderr, "V41_GROUPED_PANEL_END "+arm)
	}()
	call()
	complete = true
}

func v41GroupedPanelPhysicalCase(t *testing.T, name string) {
	t.Helper()
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		t.Fatal("UNQUALIFIED: actual Vulkan backend unavailable")
	}
	device, ok := be.(v41GroupedPanelPhysicalBackend)
	if !ok {
		t.Fatal("UNQUALIFIED: native ownership/observation interfaces unavailable")
	}
	kind := device.VulkanPhysicalDeviceType()
	if kind != 1 && kind != 2 {
		t.Fatalf("UNQUALIFIED: physical GPU type=%d", kind)
	}
	opening := v41GroupedPanelPhysicalSnapshot(t, be)
	if opening.Identity.Device != os.Getenv("FAK_VULKAN_EXPECT_DEVICE") {
		t.Fatalf("physical device=%q differs from exact expected identity", opening.Identity.Device)
	}
	if !opening.DeviceMemoryObserved || opening.DeviceMemoryFreeBytes < 1<<30 {
		t.Fatal("UNQUALIFIED: less than 1 GiB observed free device memory")
	}
	window, available, err := compute.BeginBackendExecutionObservation(be)
	if err != nil || !available {
		t.Fatalf("UNQUALIFIED: allocation peak window unavailable: %v", err)
	}
	defer func() {
		observation, err := window.End()
		if err != nil {
			t.Error(err)
			return
		}
		if observation.Identity != opening.Identity || !observation.DeviceAllocationObserved || observation.DeviceAllocationPeakBytes > opening.DeviceAllocationLiveBytes+(512<<20) {
			t.Errorf("physical identity/allocation budget violated: %+v", observation)
		}
		t.Logf("instrumented physical window: %+v", observation)
	}()

	c := Config{ModelType: "deepseek41", NumLayers: 1, HiddenSize: 5120, NumHeads: 64, NumKVHeads: 1, HeadDim: 512, OGroups: 8, OLoraRank: 1024, RopeTheta: 10000}
	if strings.HasPrefix(name, "odd-") {
		c.HiddenSize, c.NumHeads, c.OGroups, c.OLoraRank = 65, 2, 2, 128
	}
	full, err := v41ForwardGeometry(c)
	if err != nil || !full {
		t.Fatal("both physical graphs must retain full BF16 publication semantics")
	}
	width, joined := c.NumHeads*c.HeadDim, c.OGroups*c.OLoraRank
	m := &Model{Cfg: c, manifest: map[string]tensorMeta{}, kqw: map[string]*kQuantTensor{}}
	defer func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
		device.Trim()
	}()
	aName, bName := layerName(0, "attn.wo_a.weight"), layerName(0, "attn.wo_b.weight")
	m.kqw[aName] = q2kFixtureTensor(joined, width/c.OGroups, 13864)
	q3 := strings.HasSuffix(name, "q3")
	if q3 {
		m.kqw[bName] = q3kFixtureTensor(c.HiddenSize, joined)
	} else {
		m.kqw[bName] = q2kFixtureTensor(c.HiddenSize, joined, 13865)
	}
	m.v41SetExpertFaultPhase(V41PhasePrefill)
	const rows = 11
	input := v41GroupedPanelInput(rows, width)
	original := append([]float32(nil), input...)
	scalar, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer scalar.Close()
	fn := scalar.v41GroupedOutputFunc()
	if fn == nil {
		t.Fatal("UNQUALIFIED: scalar device seam unavailable")
	}
	if _, ok := v41GroupedResidentCapabilities(be, c.OLoraRank, c.HiddenSize); !ok {
		t.Fatal("UNQUALIFIED: scalar resident BF16 composition unavailable")
	}
	want := make([][]float32, rows)
	before := device.VulkanDebugDispatchProfileSnapshot()
	v41GroupedPanelPhysicalOwned(t, scalar, device, "scalar", func() {
		for row := 0; row < rows; row++ {
			var outcome v41DenseProjectionOutcome
			want[row], outcome, err = fn(0, input[row*width:(row+1)*width], c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize)
			if err != nil || outcome != v41ProjectionHandled {
				t.Fatalf("UNQUALIFIED: scalar row %d outcome=%v error=%v", row, outcome, err)
			}
		}
	})
	v41GroupedPanelPhysicalDispatches(t, before, device.VulkanDebugDispatchProfileSnapshot(), rows, c.OGroups, q3)
	weights := map[string]compute.Buffer{}
	for key, w := range scalar.halW {
		if !strings.HasPrefix(key, "v41-grouped:") {
			t.Fatalf("unexpected fixture weight %q", key)
		}
		if _, borrowed := scalar.borrowedHALW[key]; !borrowed || w.Backend() != be || w.Buf() == nil {
			t.Fatal("scalar weight lacks immutable model ownership")
		}
		weights[key] = w.Buf()
	}
	if len(weights) != c.OGroups+1 {
		t.Fatalf("cached grouped weights=%d want=%d", len(weights), c.OGroups+1)
	}
	// A fresh same-source bundle must admit the exact 11-row contraction.
	// The measured dispatch and ledger deltas below independently require it.
	for key, w := range scalar.halW {
		if !v41GroupedPanelWeightSupports(be, w, rows) {
			t.Fatalf("UNQUALIFIED: weight %q does not admit the %d-row panel", key, rows)
		}
	}
	resident := m.immutableDeviceWeights(be)
	if resident == nil {
		t.Fatal("model weight owner missing after scalar staging")
	}
	scalar.Close()
	panel, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer panel.Close()
	ledgerBefore := v41GroupedPhase(t, m, "prefill")
	physicalBefore := v41GroupedPanelPhysicalSnapshot(t, be)
	before = device.VulkanDebugDispatchProfileSnapshot()
	var got [][]float32
	v41GroupedPanelPhysicalOwned(t, panel, device, "panel", func() {
		var outcome v41DenseProjectionOutcome
		got, outcome, err = panel.v41GroupedOutputPanel(0, input, rows, c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize)
		if err != nil || outcome != v41ProjectionHandled {
			t.Fatalf("UNQUALIFIED: actual panel declined/failed outcome=%v error=%v", outcome, err)
		}
	})
	v41GroupedPanelPhysicalDispatches(t, before, device.VulkanDebugDispatchProfileSnapshot(), 1, c.OGroups, q3)
	physicalDelta, err := compute.BackendExecutionDelta(physicalBefore, v41GroupedPanelPhysicalSnapshot(t, be))
	if err != nil || physicalDelta.Counters.H2DCount == 0 || physicalDelta.Counters.D2HCount == 0 {
		t.Fatalf("native panel transfer evidence: %+v error=%v", physicalDelta, err)
	}
	ledgerAfter := v41GroupedPhase(t, m, "prefill")
	for field, want := range map[string]float64{
		"grouped_output_device_calls": 1, "grouped_output_device_rows": rows,
		"grouped_output_host_calls": 0, "grouped_output_host_rows": 0,
		"grouped_output_matmul_calls":            float64(c.OGroups + 1),
		"grouped_output_activation_upload_bytes": float64(rows * width * 4),
		"grouped_output_readback_bytes":          float64(rows * c.HiddenSize * 4),
		"grouped_output_host_weight_f32_bytes":   0,
	} {
		if ledgerAfter[field]-ledgerBefore[field] != want {
			t.Fatalf("panel ledger %s delta=%v want=%v", field, ledgerAfter[field]-ledgerBefore[field], want)
		}
	}
	if len(panel.halW) != len(weights) {
		t.Fatal("panel cache cardinality changed")
	}
	for key, buf := range weights {
		if panel.halW[key].Buf() != buf {
			t.Fatalf("new session failed to borrow existing weight %q", key)
		}
		if _, borrowed := panel.borrowedHALW[key]; !borrowed {
			t.Fatalf("panel weight %q lost borrowed marker", key)
		}
	}
	if len(got) != rows {
		t.Fatalf("panel rows=%d want=%d", len(got), rows)
	}
	nonzero := false
	for row := range got {
		if len(got[row]) != c.HiddenSize || len(want[row]) != c.HiddenSize || !v41BF16Widened(got[row]) || !v41BF16Widened(want[row]) {
			t.Fatalf("row %d lost finite BF16 output geometry", row)
		}
		for col, v := range got[row] {
			if math.Float32bits(v) != math.Float32bits(want[row][col]) {
				t.Fatalf("bit mismatch row=%d col=%d panel=%08x scalar=%08x", row, col, math.Float32bits(v), math.Float32bits(want[row][col]))
			}
			nonzero = nonzero || v != 0
		}
	}
	if !nonzero {
		t.Fatal("vacuous all-zero grouped output")
	}
	for i, v := range original {
		if math.Float32bits(input[i]) != math.Float32bits(v) {
			t.Fatalf("input mutated at %d", i)
		}
	}
	// Mutating one returned row must not change another row or the reference.
	other, reference := math.Float32bits(got[1][0]), math.Float32bits(want[0][0])
	got[0][0] = math.Float32frombits(math.Float32bits(got[0][0]) ^ 0x80000000)
	if math.Float32bits(got[1][0]) != other || math.Float32bits(want[0][0]) != reference {
		t.Fatal("returned rows/reference alias")
	}
	panel.Close()
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	device.Trim()
	resident.mu.Lock()
	retired := resident.closed && len(resident.weights) == 0
	resident.mu.Unlock()
	if !retired {
		t.Fatal("model weight registry did not retire after both sessions closed")
	}
	closing := v41GroupedPanelPhysicalSnapshot(t, be)
	// The reusable transfer stage can persist until backend teardown. Its
	// maximum logical capacity here is 32 MiB; allow 1 MiB for alignment.
	if closing.Identity != opening.Identity || closing.DeviceAllocationLiveBytes > opening.DeviceAllocationLiveBytes+(33<<20) {
		t.Fatalf("post-retirement allocation exceeds bounded staging allowance: before=%d after=%d", opening.DeviceAllocationLiveBytes, closing.DeviceAllocationLiveBytes)
	}
	t.Logf("V41_GROUPED_PANEL_PHYSICAL_COMPLETE %s rows=%d width=%d groups=%d rank=%d dim=%d identity=%+v", name, rows, width, c.OGroups, c.OLoraRank, c.HiddenSize, opening.Identity)
}

func v41GroupedPanelPhysicalDispatches(t *testing.T, before, after compute.VulkanDispatchProfile, contractions, groups int, q3 bool) {
	t.Helper()
	q2, other := uint64(contractions*(groups+1)), uint64(0)
	if q3 {
		q2, other = uint64(contractions*groups), uint64(contractions)
	}
	if after.Q2KMatmulDispatches-before.Q2KMatmulDispatches != q2 || after.OtherMatmulDispatches-before.OtherMatmulDispatches != other || after.Q4KMatmulDispatches != before.Q4KMatmulDispatches {
		t.Fatalf("UNQUALIFIED: actual matmul dispatch delta differs from q2=%d other=%d: before=%+v after=%+v", q2, other, before, after)
	}
}
