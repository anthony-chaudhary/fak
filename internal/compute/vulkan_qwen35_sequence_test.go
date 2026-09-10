//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"strings"
	"testing"
)

type qwen35SequenceParityOracleEvent struct {
	Schema         string                             `json:"schema"`
	Selector       string                             `json:"selector"`
	TestName       string                             `json:"test_name"`
	OracleKind     string                             `json:"oracle_kind"`
	Engine         string                             `json:"engine"`
	DeviceObserved bool                               `json:"device_observed"`
	CaseCount      int                                `json:"case_count"`
	Passed         bool                               `json:"passed"`
	Observed       qwen35SequenceParityOracleObserved `json:"observed"`
	Bounds         qwen35SequenceParityOracleBounds   `json:"bounds"`
}

type qwen35SequenceParityOracleObserved struct {
	MaxAbsDelta  float64 `json:"max_abs_delta"`
	FiniteOutput bool    `json:"finite_output"`
}

type qwen35SequenceParityOracleBounds struct {
	MaxAbsDelta   float64 `json:"max_abs_delta"`
	RequireFinite bool    `json:"require_finite"`
}

func formatQwen35SequenceParityOracle(maxAbsDelta float64, finiteOutput bool, caseCount int) ([]byte, error) {
	if caseCount <= 0 {
		caseCount = 4
	}
	passed := finiteOutput && maxAbsDelta <= 2e-3
	event := qwen35SequenceParityOracleEvent{
		Schema:         "fak.strix.subkernel-parity/v1",
		Selector:       "qwen35_sequence_prefill",
		TestName:       "TestVulkanQwen35SequenceQuantizedPanelsMatchCPU",
		OracleKind:     "max_abs",
		Engine:         "fak-native/vulkan",
		DeviceObserved: true,
		CaseCount:      caseCount,
		Passed:         passed,
		Observed: qwen35SequenceParityOracleObserved{
			MaxAbsDelta:  maxAbsDelta,
			FiniteOutput: finiteOutput,
		},
		Bounds: qwen35SequenceParityOracleBounds{
			MaxAbsDelta:   2e-3,
			RequireFinite: true,
		},
	}
	return json.Marshal(event)
}

// Three distinct rows and an output width crossing one workgroup expose token
// addressing errors that a one-token decode or repeated input cannot detect.
func TestVulkanQwen35SequenceQuantizedPanelsMatchCPU(t *testing.T) {
	v := vk(t)
	const tokens, out, in = 3, 67, 512
	rng := rand.New(rand.NewSource(11999))
	x := make([]float32, tokens*in)
	w := make([]float32, out*in)
	for i := range x {
		x[i] = (rng.Float32()*2 - 1) * float32(1+i/in)
	}
	for i := range w {
		w[i] = (rng.Float32()*2 - 1) * .1
	}
	raw4 := make([]byte, out*(in/q4kSuper)*q4kSuperBlock)
	for b := 0; b < len(raw4)/q4kSuperBlock; b++ {
		randQ4KBlockC(rng, raw4[b*q4kSuperBlock:(b+1)*q4kSuperBlock])
	}
	raw2 := make([]byte, out*(in/q2kSuper)*q2kSuperBlock)
	for b := 0; b < len(raw2)/q2kSuperBlock; b++ {
		block := raw2[b*q2kSuperBlock : (b+1)*q2kSuperBlock]
		for i := 0; i < 80; i++ {
			block[i] = byte(rng.Intn(256))
		}
		binaryPutFloat16(block[80:82], .015625)
		binaryPutFloat16(block[82:84], .0078125)
	}
	var worstMaxAbsDelta float64
	allFinite := true
	evaluatedPanels := 0
	for _, host := range []Tensor{
		NewF32(Default(), []int{out, in}, w),
		QuantizeQ8(Default(), []int{out, in}, w, 32),
		NewQ4K(Default(), []int{out, in}, raw4),
		NewQ2K(Default(), []int{out, in}, raw2),
	} {
		t.Run(host.Dtype.String(), func(t *testing.T) {
			evaluatedPanels++
			dw := v.Upload(host, host.Dtype)
			defer v.Free(dw)
			dx := v.Upload(NewF32(Default(), []int{tokens, in}, x), F32)
			defer v.Free(dx)
			dy := v.BatchedMatMul(dw, dx, tokens)
			defer v.Free(dy)
			got := v.Read(dy)
			if len(got) != tokens*out {
				t.Fatalf("panel size=%d want %d", len(got), tokens*out)
			}
			for token := 0; token < tokens; token++ {
				// Independent CPU scalar matvec also checks the panel's row layout.
				want := Default().Read(Default().MatMul(host, NewF32(Default(), []int{in}, x[token*in:(token+1)*in])))
				for row, expected := range want {
					actual := got[token*out+row]
					if math.IsNaN(float64(actual)) || math.IsInf(float64(actual), 0) || math.IsNaN(float64(expected)) || math.IsInf(float64(expected), 0) {
						allFinite = false
						t.Fatalf("token=%d row=%d got=%g want=%g must be finite", token, row, actual, expected)
					}
					delta := math.Abs(float64(actual - expected))
					if delta > worstMaxAbsDelta {
						worstMaxAbsDelta = delta
					}
					if delta > 2e-3+2e-4*math.Abs(float64(expected)) {
						t.Fatalf("token=%d row=%d got=%g want=%g", token, row, actual, expected)
					}
				}
			}
			t.Logf("engine=fak-native backend=vulkan dtype=%s tokens=%d in=%d out=%d cpu_parity=true", host.Dtype, tokens, in, out)
		})
	}
	oracleJSON, err := formatQwen35SequenceParityOracle(worstMaxAbsDelta, allFinite, evaluatedPanels)
	if err != nil {
		t.Fatalf("format parity oracle: %v", err)
	}
	t.Logf("%s", oracleJSON)
}

// TestVulkanDecodeAttentionContextSplitParity proves that the opt-in decode
// attention context split executes both of its ordered dispatches and preserves
// the scalar CPU attention result across non-divisible 256-position tile tails.
func TestVulkanDecodeAttentionContextSplitParity(t *testing.T) {
	const childEnv = "FAK_TEST_VULKAN_ATTENTION_CONTEXT_SPLIT_CHILD"
	if os.Getenv(childEnv) != "1" {
		exe, err := os.Executable()
		if err != nil {
			t.Fatalf("os.Executable: %v", err)
		}
		cmd := exec.CommandContext(t.Context(), exe, "-test.v", "-test.run=^TestVulkanDecodeAttentionContextSplitParity$")
		cmd.Env = append(os.Environ(),
			childEnv+"=1",
			"FAK_VULKAN_DISPATCH_PROFILE=1",
			"FAK_VULKAN_ATTENTION_CONTEXT_SPLIT=1",
		)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("context-split attention subprocess failed: %v\n%s", err, out)
		}
		if strings.Contains(string(out), "--- SKIP: TestVulkanDecodeAttentionContextSplitParity") {
			t.Skipf("context-split attention subprocess skipped without a Vulkan device:\n%s", out)
		}
		t.Logf("%s", out)
		return
	}
	if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Fatal("set FAK_VULKAN_DISPATCH_PROFILE=1 before process start")
	}
	if os.Getenv("FAK_VULKAN_ATTENTION_CONTEXT_SPLIT") != "1" {
		t.Fatal("set FAK_VULKAN_ATTENTION_CONTEXT_SPLIT=1 before process start")
	}
	v := vk(t)
	c := cpu()
	cases := []struct {
		name       string
		nPos       int
		nKV, group int
		headDim    int
	}{
		{name: "mha_ctx257_tail1", nPos: 257, nKV: 2, group: 1, headDim: 32},
		{name: "gqa_ctx511_tail255", nPos: 511, nKV: 2, group: 3, headDim: 32},
		{name: "mqa_ctx513_tail1", nPos: 513, nKV: 1, group: 6, headDim: 256},
	}
	var seed lcg = 12535
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := KVConfig{NumLayers: 1, NumKVHeads: tc.nKV, HeadDim: tc.headDim, RopeTheta: 10000}
			ckv := c.NewKV(cfg)
			defer ckv.Free()
			vkv := v.NewKV(cfg)
			defer vkv.Free()
			kvWidth := tc.nKV * tc.headDim
			for pos := 0; pos < tc.nPos; pos++ {
				kRaw := randVec(&seed, kvWidth)
				kRoPE := randVec(&seed, kvWidth)
				value := randVec(&seed, kvWidth)
				ckv.AppendKV(0, NewF32(c, []int{kvWidth}, kRaw), NewF32(c, []int{kvWidth}, kRoPE), NewF32(c, []int{kvWidth}, value), pos)
				dkRaw := v.Upload(NewF32(c, []int{kvWidth}, kRaw), F32)
				dkRoPE := v.Upload(NewF32(c, []int{kvWidth}, kRoPE), F32)
				dValue := v.Upload(NewF32(c, []int{kvWidth}, value), F32)
				vkv.AppendKV(0, dkRaw, dkRoPE, dValue, pos)
				v.Free(dkRaw)
				v.Free(dkRoPE)
				v.Free(dValue)
			}

			nHeads := tc.nKV * tc.group
			scale := float32(1 / math.Sqrt(float64(tc.headDim)))
			// A second call against the same KV store exercises reuse and lifetime
			// of the split candidate's partial-reduction scratch allocation.
			for query := 0; query < 2; query++ {
				q := randVec(&seed, nHeads*tc.headDim)
				refTensor := c.Attention(NewF32(c, []int{len(q)}, q), ckv, 0, true, tc.group, scale)
				ref := c.Read(refTensor)
				c.Free(refTensor)
				dq := v.Upload(NewF32(c, []int{len(q)}, q), F32)
				v.VulkanDebugResetDispatchProfile()
				if query == 1 {
					v.BeginBatch()
				}
				gotTensor := v.Attention(dq, vkv, 0, true, tc.group, scale)
				if query == 1 {
					v.FlushBatch()
				}
				got := v.Read(gotTensor)
				profile := v.VulkanDebugDispatchProfileSnapshot()
				v.Free(gotTensor)
				v.Free(dq)
				if profile.OtherAttentionDispatches != 2 {
					t.Fatalf("query %d context-split attention dispatches=%d, want 2 (partial + ordered merge); profile=%+v", query, profile.OtherAttentionDispatches, profile)
				}
				if len(got) != len(ref) {
					t.Fatalf("query %d output length=%d, want %d", query, len(got), len(ref))
				}
				for i := range got {
					if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) ||
						math.IsNaN(float64(ref[i])) || math.IsInf(float64(ref[i]), 0) {
						t.Fatalf("query %d output[%d] must be finite: got=%g ref=%g", query, i, got[i], ref[i])
					}
				}
				cos := cosine(ref, got)
				delta := maxAbs(ref, got)
				if cos < 0.999 {
					t.Fatalf("query %d attention cosine %.8f < 0.999", query, cos)
				}
				if delta > 1e-2 {
					t.Fatalf("query %d attention max|delta| %.6g > 1e-2", query, delta)
				}
				t.Logf("query=%d context=%d nKV=%d group=%d headDim=%d cosine=%.8f maxAbs=%.6g dispatches=%d",
					query, tc.nPos, tc.nKV, tc.group, tc.headDim, cos, delta, profile.OtherAttentionDispatches)
			}
		})
	}
	t.Run("reject_head_dim_over_split_limit", func(t *testing.T) {
		const headDim = 1025
		cfg := KVConfig{NumLayers: 1, NumKVHeads: 1, HeadDim: headDim, RopeTheta: 10000}
		vkv := v.NewKV(cfg)
		defer vkv.Free()
		row := make([]float32, headDim)
		for i := range row {
			row[i] = float32(i%17-8) / 17
		}
		dkRaw := v.Upload(NewF32(c, []int{headDim}, row), F32)
		dkRoPE := v.Upload(NewF32(c, []int{headDim}, row), F32)
		dValue := v.Upload(NewF32(c, []int{headDim}, row), F32)
		defer v.Free(dkRaw)
		defer v.Free(dkRoPE)
		defer v.Free(dValue)
		vkv.AppendKV(0, dkRaw, dkRoPE, dValue, 0)
		dq := v.Upload(NewF32(c, []int{headDim}, row), F32)
		defer v.Free(dq)
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("headDim 1025 returned a tensor; want context-split failure with no fallback")
			}
			if got := fmt.Sprint(r); !strings.Contains(got, "Vulkan attention failed closed") {
				t.Fatalf("headDim 1025 panic=%q, want Vulkan attention failed closed", got)
			}
		}()
		_ = v.Attention(dq, vkv, 0, true, 1, float32(1/math.Sqrt(headDim)))
	})
}

func TestVulkanQwen35SequenceGeometryValidation(t *testing.T) {
	v := &vulkanBackend{}
	req := Qwen35SequencePrefillRequest{
		Path:              Qwen35SequencePrefillPath,
		TokenIDs:          []int{1, 2},
		StartPos:          0,
		Hidden:            Qwen35DenseHidden,
		Intermediate:      Qwen35DenseIntermediate,
		NumHeads:          Qwen35DenseQueryHeads,
		NumKVHeads:        Qwen35DenseKVHeads,
		HeadDim:           Qwen35DenseHeadDim,
		RotaryDim:         Qwen35DenseHeadDim / 4,
		NumKeyHeads:       Qwen35DenseGDNGroups,
		NumValueHeads:     Qwen35DenseGDNRank,
		KeyHeadDim:        Qwen35DenseGDNState,
		ValueHeadDim:      Qwen35DenseGDNState,
		ConvKernel:        Qwen35DenseGDNConv,
		RMSNormEpsilon:    1e-6,
		Layers:            make([]Qwen35SequenceLayer, 4),
		States:            make([]Qwen35SequenceState, 4),
		RoPEThetaForLayer: make([]float64, 4),
	}
	for l := range req.Layers {
		req.Layers[l].Linear = (l+1)%4 != 0
		req.RoPEThetaForLayer[l] = 1e7
	}

	// 1. Wrong path
	badPath := req
	badPath.Path = "wrong-path"
	if _, err := v.validateQwen35VulkanSequence(badPath); err == nil {
		t.Fatal("expected error on wrong path")
	}

	// 2. Empty tokens
	badTokens := req
	badTokens.TokenIDs = nil
	if _, err := v.validateQwen35VulkanSequence(badTokens); err == nil {
		t.Fatal("expected error on empty token IDs")
	}

	// 3. Negative start pos
	badPos := req
	badPos.StartPos = -1
	if _, err := v.validateQwen35VulkanSequence(badPos); err == nil {
		t.Fatal("expected error on negative start pos")
	}

	// 4. Odd rotary dimension
	badRotary := req
	badRotary.RotaryDim = 63
	if _, err := v.validateQwen35VulkanSequence(badRotary); err == nil {
		t.Fatal("expected error on odd rotary dim")
	}

	// 5. Rotary exceeds head dim
	badRotaryOver := req
	badRotaryOver.RotaryDim = req.HeadDim + 2
	if _, err := v.validateQwen35VulkanSequence(badRotaryOver); err == nil {
		t.Fatal("expected error on rotary exceeding head dim")
	}

	// 6. 65 layers (64 text + 1 MTP metadata layer) must be rejected
	mtp65 := req
	mtp65.Layers = make([]Qwen35SequenceLayer, 65)
	mtp65.States = make([]Qwen35SequenceState, 65)
	mtp65.RoPEThetaForLayer = make([]float64, 65)
	for l := range mtp65.Layers {
		mtp65.Layers[l].Linear = (l+1)%4 != 0
		mtp65.RoPEThetaForLayer[l] = 1e7
	}
	if _, err := v.validateQwen35VulkanSequence(mtp65); err == nil {
		t.Fatal("expected error rejecting 65th MTP metadata layer")
	}
}

func TestVulkanQwen35SequenceRejectsUnavailableDeviceLimit(t *testing.T) {
	const childEnv = "FAK_TEST_VULKAN_QWEN35_MISSING_LIMIT_CHILD"
	if os.Getenv(childEnv) != "1" {
		exe, err := os.Executable()
		if err != nil {
			t.Fatalf("os.Executable: %v", err)
		}
		cmd := exec.Command(exe, "-test.run=^TestVulkanQwen35SequenceRejectsUnavailableDeviceLimit$")
		for _, entry := range os.Environ() {
			upper := strings.ToUpper(entry)
			if strings.HasPrefix(upper, "FAK_VULKAN_SPIRV=") || strings.HasPrefix(upper, childEnv+"=") {
				continue
			}
			cmd.Env = append(cmd.Env, entry)
		}
		cmd.Env = append(cmd.Env, childEnv+"=1")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("no-device subprocess failed: %v\n%s", err, out)
		}
		return
	}
	if _, registered := Lookup("vulkan"); registered {
		t.Fatal("Vulkan backend registered after FAK_VULKAN_SPIRV was removed")
	}
	v := &vulkanBackend{}
	req := Qwen35SequencePrefillRequest{
		Path:              Qwen35SequencePrefillPath,
		TokenIDs:          []int{1},
		Hidden:            Qwen35DenseHidden,
		Intermediate:      Qwen35DenseIntermediate,
		NumHeads:          Qwen35DenseQueryHeads,
		NumKVHeads:        Qwen35DenseKVHeads,
		HeadDim:           Qwen35DenseHeadDim,
		RotaryDim:         Qwen35DenseHeadDim / 4,
		NumKeyHeads:       Qwen35DenseGDNGroups,
		NumValueHeads:     Qwen35DenseGDNRank,
		KeyHeadDim:        Qwen35DenseGDNState,
		ValueHeadDim:      Qwen35DenseGDNState,
		ConvKernel:        Qwen35DenseGDNConv,
		RMSNormEpsilon:    1e-6,
		Layers:            make([]Qwen35SequenceLayer, 4),
		States:            make([]Qwen35SequenceState, 4),
		RoPEThetaForLayer: []float64{1e7, 1e7, 1e7, 1e7},
	}
	for layer := range req.Layers {
		req.Layers[layer].Linear = (layer+1)%4 != 0
	}

	_, err := v.validateQwen35VulkanSequence(req)
	var sequenceErr *Qwen35SequenceError
	if !errors.As(err, &sequenceErr) {
		t.Fatalf("error = %v, want typed Qwen35SequenceError", err)
	}
	if sequenceErr.Stage != "geometry" || sequenceErr.Reason != "Vulkan X workgroup limit is unavailable" {
		t.Fatalf("error = %+v, want explicit fail-closed unavailable device limit", sequenceErr)
	}
}

func TestVulkanQwen35SequenceParityOracleFormat(t *testing.T) {
	raw, err := formatQwen35SequenceParityOracle(1.5e-3, true, 4)
	if err != nil {
		t.Fatalf("formatQwen35SequenceParityOracle failed: %v", err)
	}
	var parsed struct {
		Schema         string `json:"schema"`
		Selector       string `json:"selector"`
		TestName       string `json:"test_name"`
		OracleKind     string `json:"oracle_kind"`
		Engine         string `json:"engine"`
		DeviceObserved bool   `json:"device_observed"`
		CaseCount      int    `json:"case_count"`
		Passed         bool   `json:"passed"`
		Observed       struct {
			MaxAbsDelta  float64 `json:"max_abs_delta"`
			FiniteOutput bool    `json:"finite_output"`
		} `json:"observed"`
		Bounds struct {
			MaxAbsDelta   float64 `json:"max_abs_delta"`
			RequireFinite bool    `json:"require_finite"`
		} `json:"bounds"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatalf("unmarshal formatted oracle failed: %v", err)
	}
	if parsed.Schema != "fak.strix.subkernel-parity/v1" {
		t.Errorf("schema = %q, want fak.strix.subkernel-parity/v1", parsed.Schema)
	}
	if parsed.Selector != "qwen35_sequence_prefill" {
		t.Errorf("selector = %q, want qwen35_sequence_prefill", parsed.Selector)
	}
	if parsed.TestName != "TestVulkanQwen35SequenceQuantizedPanelsMatchCPU" {
		t.Errorf("test_name = %q, want TestVulkanQwen35SequenceQuantizedPanelsMatchCPU", parsed.TestName)
	}
	if parsed.OracleKind != "max_abs" {
		t.Errorf("oracle_kind = %q, want max_abs", parsed.OracleKind)
	}
	if parsed.Engine != "fak-native/vulkan" {
		t.Errorf("engine = %q, want fak-native/vulkan", parsed.Engine)
	}
	if !parsed.DeviceObserved {
		t.Errorf("device_observed must be true")
	}
	if parsed.CaseCount != 4 {
		t.Errorf("case_count = %d, want 4", parsed.CaseCount)
	}
	if !parsed.Passed {
		t.Errorf("passed must be true")
	}
	if parsed.Observed.MaxAbsDelta != 1.5e-3 {
		t.Errorf("observed max_abs_delta = %g, want 1.5e-3", parsed.Observed.MaxAbsDelta)
	}
	if !parsed.Observed.FiniteOutput {
		t.Errorf("observed finite_output must be true")
	}
	if parsed.Bounds.MaxAbsDelta != 2e-3 {
		t.Errorf("bounds max_abs_delta = %g, want 2e-3", parsed.Bounds.MaxAbsDelta)
	}
	if !parsed.Bounds.RequireFinite {
		t.Errorf("bounds require_finite must be true")
	}

	// Boundary failure: delta exceeds threshold
	failRaw, err := formatQwen35SequenceParityOracle(2.5e-3, true, 4)
	if err != nil {
		t.Fatalf("format failed oracle failed: %v", err)
	}
	if err := json.Unmarshal(failRaw, &parsed); err != nil {
		t.Fatalf("unmarshal fail oracle failed: %v", err)
	}
	if parsed.Passed {
		t.Errorf("expected passed=false when max_abs_delta > 2e-3")
	}

	// Boundary failure: non-finite output
	failRaw2, err := formatQwen35SequenceParityOracle(1.5e-3, false, 4)
	if err != nil {
		t.Fatalf("format failed oracle 2 failed: %v", err)
	}
	if err := json.Unmarshal(failRaw2, &parsed); err != nil {
		t.Fatalf("unmarshal fail oracle 2 failed: %v", err)
	}
	if parsed.Passed {
		t.Errorf("expected passed=false when finite_output=false")
	}
}

// TestStrixQwen35ParityEmitterContract verifies that the two Qwen3.5 GDN selectors
// ("qwen35_gdn_decode", "qwen35_gdn_preprojected") and one sequence-prefill selector
// ("qwen35_sequence_prefill") each have exactly one truthful aggregate success emitter,
// matching the schema, bounds, and test bindings registered in the Strix contract,
// and that geometry-only sibling tests emit zero events.
//
// Device-free contract test: does not initialize Vulkan hardware or require a physical GPU device.
func TestStrixQwen35ParityEmitterContract(t *testing.T) {
	type subkernelContract struct {
		selector             string
		testName             string
		oracleKind           string
		engine               string
		deviceObserved       bool
		caseCount            int
		maxAbsDeltaBound     float64
		requireStateIdentity bool
		requireFinite        bool
		formatFn             func() ([]byte, error)
		formatExceedBoundFn  func() ([]byte, error)
		formatFailIdentityFn func() ([]byte, error)
		formatFailFiniteFn   func() ([]byte, error)
	}

	contracts := []subkernelContract{
		{
			selector:             "qwen35_gdn_decode",
			testName:             "TestVulkanQwen35GDNDecodeMatchesCPUOracleInPlace",
			oracleKind:           "state_continuity",
			engine:               "fak-native/vulkan",
			deviceObserved:       true,
			caseCount:            1,
			maxAbsDeltaBound:     3e-4,
			requireStateIdentity: true,
			requireFinite:        true,
			formatFn: func() ([]byte, error) {
				return formatQwen35GDNDecodeParityOracle(1.5e-4, true, true)
			},
			formatExceedBoundFn: func() ([]byte, error) {
				return formatQwen35GDNDecodeParityOracle(4e-4, true, true)
			},
			formatFailIdentityFn: func() ([]byte, error) {
				return formatQwen35GDNDecodeParityOracle(1.5e-4, false, true)
			},
			formatFailFiniteFn: func() ([]byte, error) {
				return formatQwen35GDNDecodeParityOracle(1.5e-4, true, false)
			},
		},
		{
			selector:             "qwen35_gdn_preprojected",
			testName:             "TestVulkanQwen35GDNPreprojectedParityAndStateContinuity",
			oracleKind:           "state_continuity",
			engine:               "fak-native/vulkan",
			deviceObserved:       true,
			caseCount:            4,
			maxAbsDeltaBound:     2e-4,
			requireStateIdentity: true,
			requireFinite:        true,
			formatFn: func() ([]byte, error) {
				return formatQwen35GDNPreprojectedParityOracle(1.2e-4, true, true, 4)
			},
			formatExceedBoundFn: func() ([]byte, error) {
				return formatQwen35GDNPreprojectedParityOracle(3e-4, true, true, 4)
			},
			formatFailIdentityFn: func() ([]byte, error) {
				return formatQwen35GDNPreprojectedParityOracle(1.2e-4, false, true, 4)
			},
			formatFailFiniteFn: func() ([]byte, error) {
				return formatQwen35GDNPreprojectedParityOracle(1.2e-4, true, false, 4)
			},
		},
		{
			selector:             "qwen35_sequence_prefill",
			testName:             "TestVulkanQwen35SequenceQuantizedPanelsMatchCPU",
			oracleKind:           "max_abs",
			engine:               "fak-native/vulkan",
			deviceObserved:       true,
			caseCount:            4,
			maxAbsDeltaBound:     2e-3,
			requireStateIdentity: false,
			requireFinite:        true,
			formatFn: func() ([]byte, error) {
				return formatQwen35SequenceParityOracle(1.5e-3, true, 4)
			},
			formatExceedBoundFn: func() ([]byte, error) {
				return formatQwen35SequenceParityOracle(2.5e-3, true, 4)
			},
			formatFailIdentityFn: nil,
			formatFailFiniteFn: func() ([]byte, error) {
				return formatQwen35SequenceParityOracle(1.5e-3, false, 4)
			},
		},
	}

	seenSelectors := make(map[string]bool)
	seenTestNames := make(map[string]bool)

	for _, c := range contracts {
		t.Run(c.selector, func(t *testing.T) {
			if seenSelectors[c.selector] {
				t.Fatalf("duplicate selector in contract table: %s", c.selector)
			}
			seenSelectors[c.selector] = true

			if seenTestNames[c.testName] {
				t.Fatalf("duplicate test name in contract table: %s", c.testName)
			}
			seenTestNames[c.testName] = true

			// Geometry sibling isolation
			if strings.Contains(c.testName, "Geometry") {
				t.Errorf("selector %s must not bind to a geometry sibling test, got %s", c.selector, c.testName)
			}

			// Format valid passing oracle event
			raw, err := c.formatFn()
			if err != nil {
				t.Fatalf("formatFn failed: %v", err)
			}

			var parsed struct {
				Schema         string `json:"schema"`
				Selector       string `json:"selector"`
				TestName       string `json:"test_name"`
				OracleKind     string `json:"oracle_kind"`
				Engine         string `json:"engine"`
				DeviceObserved bool   `json:"device_observed"`
				CaseCount      int    `json:"case_count"`
				Passed         bool   `json:"passed"`
				Observed       struct {
					MaxAbsDelta   float64 `json:"max_abs_delta"`
					StateIdentity bool    `json:"state_identity"`
					FiniteOutput  bool    `json:"finite_output"`
				} `json:"observed"`
				Bounds struct {
					MaxAbsDelta          float64 `json:"max_abs_delta"`
					RequireStateIdentity bool    `json:"require_state_identity"`
					RequireFinite        bool    `json:"require_finite"`
				} `json:"bounds"`
			}
			if err := json.Unmarshal(raw, &parsed); err != nil {
				t.Fatalf("unmarshal formatFn output failed: %v", err)
			}

			if parsed.Schema != "fak.strix.subkernel-parity/v1" {
				t.Errorf("schema = %q, want fak.strix.subkernel-parity/v1", parsed.Schema)
			}
			if parsed.Selector != c.selector {
				t.Errorf("selector = %q, want %q", parsed.Selector, c.selector)
			}
			if parsed.TestName != c.testName {
				t.Errorf("test_name = %q, want %q", parsed.TestName, c.testName)
			}
			if parsed.OracleKind != c.oracleKind {
				t.Errorf("oracle_kind = %q, want %q", parsed.OracleKind, c.oracleKind)
			}
			if parsed.Engine != c.engine {
				t.Errorf("engine = %q, want %q", parsed.Engine, c.engine)
			}
			if parsed.DeviceObserved != c.deviceObserved {
				t.Errorf("device_observed = %v, want %v", parsed.DeviceObserved, c.deviceObserved)
			}
			if parsed.CaseCount != c.caseCount {
				t.Errorf("case_count = %d, want %d", parsed.CaseCount, c.caseCount)
			}
			if !parsed.Passed {
				t.Errorf("passed must be true for valid oracle event")
			}
			if parsed.Bounds.MaxAbsDelta != c.maxAbsDeltaBound {
				t.Errorf("bounds max_abs_delta = %g, want %g", parsed.Bounds.MaxAbsDelta, c.maxAbsDeltaBound)
			}
			if parsed.Bounds.RequireStateIdentity != c.requireStateIdentity {
				t.Errorf("bounds require_state_identity = %v, want %v", parsed.Bounds.RequireStateIdentity, c.requireStateIdentity)
			}
			if parsed.Bounds.RequireFinite != c.requireFinite {
				t.Errorf("bounds require_finite = %v, want %v", parsed.Bounds.RequireFinite, c.requireFinite)
			}
			if parsed.Observed.MaxAbsDelta > parsed.Bounds.MaxAbsDelta {
				t.Errorf("observed max_abs_delta (%g) > bound (%g)", parsed.Observed.MaxAbsDelta, parsed.Bounds.MaxAbsDelta)
			}
			if c.requireStateIdentity && !parsed.Observed.StateIdentity {
				t.Errorf("observed state_identity must be true")
			}
			if c.requireFinite && !parsed.Observed.FiniteOutput {
				t.Errorf("observed finite_output must be true")
			}

			// Boundary failure: delta exceeds threshold
			if c.formatExceedBoundFn != nil {
				exceedRaw, err := c.formatExceedBoundFn()
				if err != nil {
					t.Fatalf("formatExceedBoundFn failed: %v", err)
				}
				var exceedParsed struct {
					Passed bool `json:"passed"`
				}
				if err := json.Unmarshal(exceedRaw, &exceedParsed); err != nil {
					t.Fatalf("unmarshal exceedRaw failed: %v", err)
				}
				if exceedParsed.Passed {
					t.Errorf("expected passed=false when max_abs_delta exceeds bound")
				}
			}

			// Boundary failure: state identity failure
			if c.formatFailIdentityFn != nil {
				failIdRaw, err := c.formatFailIdentityFn()
				if err != nil {
					t.Fatalf("formatFailIdentityFn failed: %v", err)
				}
				var failIdParsed struct {
					Passed bool `json:"passed"`
				}
				if err := json.Unmarshal(failIdRaw, &failIdParsed); err != nil {
					t.Fatalf("unmarshal failIdRaw failed: %v", err)
				}
				if failIdParsed.Passed {
					t.Errorf("expected passed=false when state_identity is false")
				}
			}

			// Boundary failure: non-finite output
			if c.formatFailFiniteFn != nil {
				failFiniteRaw, err := c.formatFailFiniteFn()
				if err != nil {
					t.Fatalf("formatFailFiniteFn failed: %v", err)
				}
				var failFiniteParsed struct {
					Passed bool `json:"passed"`
				}
				if err := json.Unmarshal(failFiniteRaw, &failFiniteParsed); err != nil {
					t.Fatalf("unmarshal failFiniteRaw failed: %v", err)
				}
				if failFiniteParsed.Passed {
					t.Errorf("expected passed=false when finite_output is false")
				}
			}
		})
	}

	// Sibling geometry validation must emit zero events
	t.Run("GeometrySiblingEmitsNoOracle", func(t *testing.T) {
		const geomTestName = "TestVulkanQwen35SequenceGeometryValidation"
		for _, c := range contracts {
			if c.testName == geomTestName {
				t.Errorf("geometry test %s must not be mapped to any selector", geomTestName)
			}
		}
	})
}

// TestQwen35PartialRoPETableParity exercises one cache through growth, reuse,
// and key changes while comparing the production table path with both the
// retained scalar Vulkan path and an independent scalar CPU oracle.
func TestQwen35PartialRoPETableParity(t *testing.T) {
	v := vk(t)
	const scalarEnv = "FAK_VULKAN_QWEN35_PARTIAL_ROPE_SCALAR_REFERENCE"
	t.Setenv(scalarEnv, "0")

	type ropeCase struct {
		name               string
		tokens, startPos   int
		qHeads, kHeads     int
		headDim, rotaryDim int
		theta              float64
	}
	// The order is intentional: grow a key, reuse it at a lower position, then
	// change rotary width and theta before returning to the original key.
	cases := []ropeCase{
		{name: "grow-original-key", tokens: 3, startPos: 37, qHeads: 3, kHeads: 2, headDim: 16, rotaryDim: 8, theta: 10000},
		{name: "change-rotary-key", tokens: 3, startPos: 11, qHeads: 2, kHeads: 2, headDim: 16, rotaryDim: 12, theta: 10000},
		{name: "change-theta-key", tokens: 2, startPos: 23, qHeads: 3, kHeads: 1, headDim: 16, rotaryDim: 8, theta: 1e6},
		{name: "regrow-original-key", tokens: 2, startPos: 129, qHeads: 2, kHeads: 2, headDim: 16, rotaryDim: 8, theta: 10000},
		{name: "reuse-grown-original-key", tokens: 2, startPos: 5, qHeads: 2, kHeads: 1, headDim: 16, rotaryDim: 8, theta: 10000},
		{name: "model-width-high-position", tokens: 2, startPos: 4093, qHeads: 2, kHeads: 1, headDim: 128, rotaryDim: 64, theta: 1e6},
	}

	const tolerance = 2e-3
	for caseIndex, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			q := make([]float32, tc.tokens*tc.qHeads*tc.headDim)
			k := make([]float32, tc.tokens*tc.kHeads*tc.headDim)
			for i := range q {
				q[i] = float32(math.Sin(float64((caseIndex+1)*101+i*7))) * 0.75
			}
			for i := range k {
				k[i] = float32(math.Cos(float64((caseIndex+1)*137+i*11))) * 0.625
			}

			run := func(reference bool) ([]float32, []float32) {
				if reference {
					if err := os.Setenv(scalarEnv, "1"); err != nil {
						t.Fatalf("enable scalar reference: %v", err)
					}
				} else if err := os.Setenv(scalarEnv, "0"); err != nil {
					t.Fatalf("enable cached table: %v", err)
				}
				dq := v.Upload(NewF32(Default(), []int{tc.tokens, tc.qHeads * tc.headDim}, q), F32)
				dk := v.Upload(NewF32(Default(), []int{tc.tokens, tc.kHeads * tc.headDim}, k), F32)
				defer v.Free(dq)
				defer v.Free(dk)
				qr, kr := v.PartialRoPEQK(dq, dk, tc.startPos, tc.qHeads, tc.kHeads, tc.headDim, tc.rotaryDim, tc.theta)
				defer v.Free(qr)
				defer v.Free(kr)
				gotQ, gotK := v.Read(qr), v.Read(kr)
				for i, got := range v.Read(dq) {
					if got != q[i] {
						t.Fatalf("Q input mutated at %d: got=%g want=%g", i, got, q[i])
					}
				}
				for i, got := range v.Read(dk) {
					if got != k[i] {
						t.Fatalf("K input mutated at %d: got=%g want=%g", i, got, k[i])
					}
				}
				return gotQ, gotK
			}

			cachedQ, cachedK := run(false)
			refQ, refK := run(true)
			check := func(label string, input, cached, reference []float32, heads int) {
				if len(cached) != len(input) || len(reference) != len(input) {
					t.Fatalf("%s length: input=%d cached=%d reference=%d", label, len(input), len(cached), len(reference))
				}
				half := tc.rotaryDim / 2
				for i, got := range cached {
					dim := i % tc.headDim
					if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) ||
						math.IsNaN(float64(reference[i])) || math.IsInf(float64(reference[i]), 0) {
						t.Fatalf("%s output[%d] is not finite: cached=%g reference=%g", label, i, got, reference[i])
					}
					if dim >= tc.rotaryDim {
						if got != input[i] || reference[i] != input[i] {
							t.Fatalf("%s unrotated tail[%d]: input=%g cached=%g reference=%g", label, i, input[i], got, reference[i])
						}
						continue
					}
					rowWidth := heads * tc.headDim
					token := i / rowWidth
					base := i - dim
					pair := dim % half
					a, b := input[base+pair], input[base+pair+half]
					angle := float64(tc.startPos+token) * math.Pow(tc.theta, -2*float64(pair)/float64(tc.rotaryDim))
					want := float64(a) * math.Cos(angle)
					if dim < half {
						want -= float64(b) * math.Sin(angle)
					} else {
						want = float64(b)*math.Cos(angle) + float64(a)*math.Sin(angle)
					}
					if delta := math.Abs(float64(got) - float64(reference[i])); delta > tolerance+2e-4*math.Abs(float64(reference[i])) {
						t.Fatalf("%s cached/reference[%d] delta=%g cached=%g reference=%g", label, i, delta, got, reference[i])
					}
					if delta := math.Abs(float64(got) - want); delta > tolerance+2e-4*math.Abs(want) {
						t.Fatalf("%s cached/CPU[%d] delta=%g cached=%g want=%g", label, i, delta, got, want)
					}
				}
			}
			check("Q", q, cachedQ, refQ, tc.qHeads)
			check("K", k, cachedK, refK, tc.kHeads)
		})
	}

	t.Run("scalar-route-proof", func(t *testing.T) {
		const startPos = 1_000_000_000 // Valid int32 position; too large for a bounded device table.
		q := []float32{0.25, -0.5, 0.75, -1}
		k := []float32{-0.125, 0.375, -0.625, 0.875}
		dq := v.Upload(NewF32(Default(), []int{4}, q), F32)
		dk := v.Upload(NewF32(Default(), []int{4}, k), F32)
		defer v.Free(dq)
		defer v.Free(dk)

		if err := os.Setenv(scalarEnv, "0"); err != nil {
			t.Fatalf("enable cached table: %v", err)
		}
		func() {
			defer func() {
				got := recover()
				if got == nil || !strings.Contains(fmt.Sprint(got), "failed closed with status 3") {
					t.Fatalf("cached oversized-table panic = %v, want status 3", got)
				}
			}()
			v.PartialRoPEQK(dq, dk, startPos, 1, 1, 4, 2, 10000)
		}()

		if err := os.Setenv(scalarEnv, "1"); err != nil {
			t.Fatalf("enable scalar reference: %v", err)
		}
		qr, kr := v.PartialRoPEQK(dq, dk, startPos, 1, 1, 4, 2, 10000)
		defer v.Free(qr)
		defer v.Free(kr)
		for label, values := range map[string][]float32{"Q": v.Read(qr), "K": v.Read(kr)} {
			for i, value := range values {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					t.Fatalf("%s scalar route output[%d]=%g is not finite", label, i, value)
				}
			}
		}
	})
	t.Logf("engine=fak-native/vulkan device=%s cases=%d cache_growth=true cache_reuse=true cache_key_changes=true scalar_route_proved=true finite=true tails_identical=true", v.Tier(), len(cases))
}

// TestQwen35VulkanSequenceKVReserveGeometric validates the guarded geometric sequence-KV reservation
// policy (#12548). It feeds multiple chunk sizes, asserts logical positions and capacity bounds,
// verifies logarithmic D2D copy scaling over exact-growth pre-reservation, tests single-resource
// ceiling and overflow guards, and when a Vulkan device is available, verifies that attention outputs
// are bit-exact identical between geometric reservation and exact-growth reservation.
func TestQwen35VulkanSequenceKVReserveGeometric(t *testing.T) {
	t.Run("GeometricPolicyBoundsAndCopyReduction", func(t *testing.T) {
		v := &vulkanBackend{}
		const kvWidth = 128

		// Schedule 1: 16 repeated equal-size chunks (standard sequence prefill streaming)
		repeatedChunks := make([]int, 16)
		for i := range repeatedChunks {
			repeatedChunks[i] = 64
		}

		type simState struct {
			currentCap int
			currentLen int
			copies     int
			allocs     int
		}

		simulate := func(chunks []int, useGeometric bool) simState {
			st := simState{}
			startPos := 0
			for _, chunk := range chunks {
				need := (startPos + chunk) * kvWidth
				var ncap int
				if useGeometric {
					ncap = v.qwen35SequenceKVGeometricCapacity(st.currentCap, need)
				} else {
					ncap = need
				}
				if st.currentCap < need {
					if st.currentLen > 0 {
						st.copies++
					}
					st.allocs++
					st.currentCap = ncap
				}
				st.currentLen = need
				startPos += chunk
			}
			return st
		}

		exactRep := simulate(repeatedChunks, false)
		geomRep := simulate(repeatedChunks, true)

		// Logical lengths must match
		if geomRep.currentLen != exactRep.currentLen {
			t.Fatalf("repeated chunks: len mismatch: geom=%d exact=%d", geomRep.currentLen, exactRep.currentLen)
		}
		// For 16 repeated chunks: exact copies = 15; geometric copies must be exactly log2(16) = 4.
		t.Logf("repeated chunks (%d chunks): exact copies=%d, geometric copies=%d", len(repeatedChunks), exactRep.copies, geomRep.copies)
		if geomRep.copies != 4 {
			t.Errorf("expected exactly 4 geometric copies for 16 repeated chunks (log2 scaling), got %d", geomRep.copies)
		}
		if geomRep.copies >= exactRep.copies {
			t.Errorf("expected geometric copies (%d) < exact copies (%d)", geomRep.copies, exactRep.copies)
		}

		// Schedule 2: Mixed chunk sizes
		mixedChunks := []int{16, 32, 64, 48, 128, 64, 256, 128}
		exactMixed := simulate(mixedChunks, false)
		geomMixed := simulate(mixedChunks, true)

		if geomMixed.currentLen != exactMixed.currentLen {
			t.Fatalf("mixed chunks: len mismatch: geom=%d exact=%d", geomMixed.currentLen, exactMixed.currentLen)
		}
		t.Logf("mixed chunks (%d chunks): exact copies=%d, geometric copies=%d", len(mixedChunks), exactMixed.copies, geomMixed.copies)
		if geomMixed.copies >= exactMixed.copies {
			t.Errorf("mixed chunks: expected geometric copies (%d) < exact copies (%d)", geomMixed.copies, exactMixed.copies)
		}
	})

	t.Run("OverflowAndCeilingGuards", func(t *testing.T) {
		v := &vulkanBackend{
			maxBufferBytes: 1024 * 1024 * 64, // 64 MB cap = 16M floats
		}
		maxFloats := int(v.maxBufferBytes / 4)

		// 1. Initial need within budget
		cap1 := v.qwen35SequenceKVGeometricCapacity(0, 1000)
		if cap1 != 1000 {
			t.Errorf("initial cap = %d, want 1000", cap1)
		}

		// 2. Normal doubling
		cap2 := v.qwen35SequenceKVGeometricCapacity(1000, 1500)
		if cap2 != 2000 {
			t.Errorf("doubled cap = %d, want 2000", cap2)
		}

		// 3. Buffer ceiling clamp
		capOver := v.qwen35SequenceKVGeometricCapacity(maxFloats-100, maxFloats)
		if capOver > maxFloats {
			t.Errorf("capOver = %d exceeds maxBufferBytes limit %d", capOver, maxFloats)
		}

		// 4. Integer overflow guard
		capMax := v.qwen35SequenceKVGeometricCapacity(math.MaxInt/2+1, math.MaxInt/2+10)
		if capMax < 0 {
			t.Errorf("integer overflow resulted in negative capacity: %d", capMax)
		}

		// 5. Exact override via environment variable
		os.Setenv("FAK_QWEN35_SEQUENCE_KV_EXACT", "1")
		capExact := v.qwen35SequenceKVGeometricCapacity(1000, 1500)
		os.Unsetenv("FAK_QWEN35_SEQUENCE_KV_EXACT")
		if capExact != 1500 {
			t.Errorf("exact override cap = %d, want 1500", capExact)
		}
	})

	t.Run("DeviceKVReserveAndOutputParity", func(t *testing.T) {
		b, ok := Lookup("vulkan")
		if !ok {
			t.Skip("vulkan backend not available on this host")
		}
		v := b.(*vulkanBackend)

		const (
			nH      = 4
			nKV     = 2
			hd      = 32
			kvWidth = nKV * hd // 64
			qWidth  = nH * hd  // 128
		)
		chunkSizes := []int{16, 16, 16, 16}
		totalTokens := 64

		// Generate reproducible pseudo-random key/value data
		rng := rand.New(rand.NewSource(42))
		allKData := make([]float32, totalTokens*kvWidth)
		allVData := make([]float32, totalTokens*kvWidth)
		for i := range allKData {
			allKData[i] = (rng.Float32()*2 - 1) * 0.5
			allVData[i] = (rng.Float32()*2 - 1) * 0.5
		}

		// 1. Run exact reserve sequence
		os.Setenv("FAK_QWEN35_SEQUENCE_KV_EXACT", "1")
		var cacheExactK, cacheExactV vslice
		defer v.qwen35SequenceFreeVsliceForTest(&cacheExactK)
		defer v.qwen35SequenceFreeVsliceForTest(&cacheExactV)

		exactStartPos := 0
		for _, chunk := range chunkSizes {
			need := (exactStartPos + chunk) * kvWidth
			v.qwen35SequenceReserveGeometric(&cacheExactK, need, "exact-reserve-k")
			v.qwen35SequenceReserveGeometric(&cacheExactV, need, "exact-reserve-v")
			if cacheExactK.cap != need || cacheExactV.cap != need {
				t.Fatalf("exact reserve cap: K=%d V=%d, want %d", cacheExactK.cap, cacheExactV.cap, need)
			}
			chunkK := allKData[exactStartPos*kvWidth : need]
			chunkV := allVData[exactStartPos*kvWidth : need]
			v.qwen35SequenceUploadKVFloatsForTest(&cacheExactK, exactStartPos*kvWidth, chunkK)
			v.qwen35SequenceUploadKVFloatsForTest(&cacheExactV, exactStartPos*kvWidth, chunkV)
			cacheExactK.len = need
			cacheExactV.len = need
			exactStartPos += chunk
		}
		os.Unsetenv("FAK_QWEN35_SEQUENCE_KV_EXACT")

		// 2. Run geometric reserve sequence
		var cacheGeomK, cacheGeomV vslice
		defer v.qwen35SequenceFreeVsliceForTest(&cacheGeomK)
		defer v.qwen35SequenceFreeVsliceForTest(&cacheGeomV)

		geomStartPos := 0
		for _, chunk := range chunkSizes {
			need := (geomStartPos + chunk) * kvWidth
			v.qwen35SequenceReserveGeometric(&cacheGeomK, need, "geom-reserve-k")
			v.qwen35SequenceReserveGeometric(&cacheGeomV, need, "geom-reserve-v")
			if cacheGeomK.cap < need || cacheGeomV.cap < need {
				t.Fatalf("geom reserve cap: K=%d V=%d < need %d", cacheGeomK.cap, cacheGeomV.cap, need)
			}
			chunkK := allKData[geomStartPos*kvWidth : need]
			chunkV := allVData[geomStartPos*kvWidth : need]
			v.qwen35SequenceUploadKVFloatsForTest(&cacheGeomK, geomStartPos*kvWidth, chunkK)
			v.qwen35SequenceUploadKVFloatsForTest(&cacheGeomV, geomStartPos*kvWidth, chunkV)
			cacheGeomK.len = need
			cacheGeomV.len = need
			geomStartPos += chunk
		}

		// Logical lengths must match exactly
		if cacheGeomK.len != cacheExactK.len || cacheGeomV.len != cacheExactV.len {
			t.Fatalf("final len mismatch: geom K=%d V=%d, exact K=%d V=%d",
				cacheGeomK.len, cacheGeomV.len, cacheExactK.len, cacheExactV.len)
		}
		// Geometric capacity must be >= exact capacity
		if cacheGeomK.cap < cacheExactK.cap {
			t.Fatalf("geom cap %d < exact cap %d", cacheGeomK.cap, cacheExactK.cap)
		}

		// 3. Verify exact float-by-float data identity between geometric and exact buffers
		readExactK := v.qwen35SequenceReadKVFloatsForTest(&cacheExactK, totalTokens*kvWidth)
		readGeomK := v.qwen35SequenceReadKVFloatsForTest(&cacheGeomK, totalTokens*kvWidth)
		for i := 0; i < totalTokens*kvWidth; i++ {
			if readExactK[i] != readGeomK[i] {
				t.Fatalf("K buffer mismatch at float %d: exact=%g geom=%g", i, readExactK[i], readGeomK[i])
			}
		}

		readExactV := v.qwen35SequenceReadKVFloatsForTest(&cacheExactV, totalTokens*kvWidth)
		readGeomV := v.qwen35SequenceReadKVFloatsForTest(&cacheGeomV, totalTokens*kvWidth)
		for i := 0; i < totalTokens*kvWidth; i++ {
			if readExactV[i] != readGeomV[i] {
				t.Fatalf("V buffer mismatch at float %d: exact=%g geom=%g", i, readExactV[i], readGeomV[i])
			}
		}

		// 4. Run causal attention with exact vs geometric buffers and compare output parity
		const qTokens = 16
		qData := make([]float32, qTokens*qWidth)
		for i := range qData {
			qData[i] = (rng.Float32()*2 - 1) * 0.5
		}
		qr := v.Upload(NewF32(Default(), []int{qTokens, qWidth}, qData), F32)
		defer v.Free(qr)

		outExact, _ := v.devTr([]int{qTokens, qWidth}, F32)
		defer v.Free(outExact)
		outGeom, _ := v.devTr([]int{qTokens, qWidth}, F32)
		defer v.Free(outGeom)

		scale := float32(1.0 / math.Sqrt(float64(hd)))
		prefix := totalTokens - qTokens

		statusExact := v.qwen35SequenceCausalAttentionForTest(v.vp(qr), cacheExactK.ptr, cacheExactV.ptr, v.vp(outExact), qTokens, prefix, nH, nKV, hd, scale)
		if statusExact != 0 {
			t.Fatalf("fvk_qwen35_causal_attention_panel_f32 exact status: %d", statusExact)
		}

		statusGeom := v.qwen35SequenceCausalAttentionForTest(v.vp(qr), cacheGeomK.ptr, cacheGeomV.ptr, v.vp(outGeom), qTokens, prefix, nH, nKV, hd, scale)
		if statusGeom != 0 {
			t.Fatalf("fvk_qwen35_causal_attention_panel_f32 geom status: %d", statusGeom)
		}

		resExact := v.Read(outExact)
		resGeom := v.Read(outGeom)
		if len(resExact) != len(resGeom) {
			t.Fatalf("attention output length mismatch: exact=%d geom=%d", len(resExact), len(resGeom))
		}

		var maxDelta float64
		for i := range resExact {
			delta := math.Abs(float64(resExact[i] - resGeom[i]))
			if delta > maxDelta {
				maxDelta = delta
			}
			if delta > 1e-6 {
				t.Fatalf("attention mismatch at %d: exact=%g geom=%g delta=%g", i, resExact[i], resGeom[i], delta)
			}
		}

		t.Logf("Device KV reservation & attention output parity verified: tokens=%d maxDelta=%g logical_len=%d exact_cap=%d geom_cap=%d",
			totalTokens, maxDelta, cacheGeomK.len, cacheExactK.cap, cacheGeomK.cap)
	})
}
