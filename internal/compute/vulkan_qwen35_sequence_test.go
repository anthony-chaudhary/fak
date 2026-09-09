//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
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
