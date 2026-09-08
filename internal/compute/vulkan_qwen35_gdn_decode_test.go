//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"encoding/json"
	"math"
	"testing"
)

type qwen35GDNDecodeParityOracleEvent struct {
	Schema         string                              `json:"schema"`
	Selector       string                              `json:"selector"`
	TestName       string                              `json:"test_name"`
	OracleKind     string                              `json:"oracle_kind"`
	Engine         string                              `json:"engine"`
	DeviceObserved bool                                `json:"device_observed"`
	CaseCount      int                                 `json:"case_count"`
	Passed         bool                                `json:"passed"`
	Observed       qwen35GDNDecodeParityOracleObserved `json:"observed"`
	Bounds         qwen35GDNDecodeParityOracleBounds   `json:"bounds"`
}

type qwen35GDNDecodeParityOracleObserved struct {
	MaxAbsDelta   float64 `json:"max_abs_delta"`
	StateIdentity bool    `json:"state_identity"`
	FiniteOutput  bool    `json:"finite_output"`
}

type qwen35GDNDecodeParityOracleBounds struct {
	MaxAbsDelta          float64 `json:"max_abs_delta"`
	RequireStateIdentity bool    `json:"require_state_identity"`
	RequireFinite        bool    `json:"require_finite"`
}

func formatQwen35GDNDecodeParityOracle(maxAbsDelta float64, stateIdentity, finiteOutput bool) ([]byte, error) {
	passed := stateIdentity && finiteOutput && maxAbsDelta <= 3e-4
	event := qwen35GDNDecodeParityOracleEvent{
		Schema:         "fak.strix.subkernel-parity/v1",
		Selector:       "qwen35_gdn_decode",
		TestName:       "TestVulkanQwen35GDNDecodeMatchesCPUOracleInPlace",
		OracleKind:     "state_continuity",
		Engine:         "fak-native/vulkan",
		DeviceObserved: true,
		CaseCount:      1,
		Passed:         passed,
		Observed: qwen35GDNDecodeParityOracleObserved{
			MaxAbsDelta:   maxAbsDelta,
			StateIdentity: stateIdentity,
			FiniteOutput:  finiteOutput,
		},
		Bounds: qwen35GDNDecodeParityOracleBounds{
			MaxAbsDelta:          3e-4,
			RequireStateIdentity: true,
			RequireFinite:        true,
		},
	}
	return json.Marshal(event)
}

func TestVulkanQwen35GDNDecodeMatchesCPUOracleInPlace(t *testing.T) {
	be := vk(t)
	hidden, nK, nV, kHd, vHd, kernel := 7, 1, 2, 3, 65, 3
	keyDim, valueDim := nK*kHd, nV*vHd
	convDim := 2*keyDim + valueDim
	seq := uint64(0x9680)
	vec := func(n int, scale float32) []float32 {
		out := make([]float32, n)
		for i := range out {
			seq = seq*6364136223846793005 + 1442695040888963407
			out[i] = (float32(uint32(seq>>32))/float32(uint64(1)<<32) - .5) * scale
		}
		return out
	}
	upload := func(shape []int, data []float32, class MemoryClass, name string) Tensor {
		return be.UploadClass(NewF32(Default(), shape, data), F32, class, name)
	}
	xData := vec(hidden, .4)
	qkvData, zData := vec(convDim*hidden, .2), vec(valueDim*hidden, .2)
	bData, aData := vec(nV*hidden, .2), vec(nV*hidden, .2)
	convData, aLogData := vec(convDim*kernel, .2), vec(nV, .2)
	for i := range aLogData {
		aLogData[i] -= .5
	}
	dtData, normData, outData := vec(nV, .2), vec(vHd, .1), vec(hidden*valueDim, .2)
	for i := range normData {
		normData[i] += 1
	}
	convStateData := make([]float32, (kernel-1)*convDim)
	recStateData := make([]float32, nV*kHd*vHd)
	x := upload([]int{hidden}, xData, MemoryActivation, "gdn input")
	qkv := upload([]int{convDim, hidden}, qkvData, MemoryWeights, "gdn qkv")
	z := upload([]int{valueDim, hidden}, zData, MemoryWeights, "gdn z")
	b := upload([]int{nV, hidden}, bData, MemoryWeights, "gdn b")
	a := upload([]int{nV, hidden}, aData, MemoryWeights, "gdn a")
	conv := upload([]int{convDim, kernel}, convData, MemoryWeights, "gdn conv")
	aLog := upload([]int{nV}, aLogData, MemoryWeights, "gdn alog")
	dt := upload([]int{nV}, dtData, MemoryWeights, "gdn dt")
	norm := upload([]int{vHd}, normData, MemoryWeights, "gdn norm")
	outW := upload([]int{hidden, valueDim}, outData, MemoryWeights, "gdn out")
	convState := upload([]int{kernel - 1, convDim}, convStateData, MemoryKVCache, "gdn conv state")
	recState := upload([]int{nV, kHd, vHd}, recStateData, MemoryKVCache, "gdn recurrent state")
	oldC, oldR := convState.Buf(), recState.Buf()

	got, nextC, nextR, err := be.Qwen35GDNDecode(x, qkv, z, b, a, conv, aLog, dt, norm, outW, convState, recState, nK, nV, kHd, vHd, kernel, 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	stateIdentity := nextC.Buf() == oldC && nextR.Buf() == oldR
	if !stateIdentity {
		t.Fatal("persistent state identity changed")
	}
	matvec := func(w []float32, rows int, in []float32) []float32 {
		y := make([]float32, rows)
		for r := 0; r < rows; r++ {
			for c, xv := range in {
				y[r] += w[r*len(in)+c] * xv
			}
		}
		return y
	}
	mixed, zh := matvec(qkvData, convDim, xData), matvec(zData, valueDim, xData)
	bh, ah := matvec(bData, nV, xData), matvec(aData, nV, xData)
	core, wantC, wantR := qwen35GDNPreprojectedOracle(mixed, zh, bh, ah, convData, aLogData, dtData, normData, convStateData, recStateData, 1, nK, nV, kHd, vHd, kernel, 1e-5)
	want := matvec(outData, hidden, core)
	maxObservedDelta := 0.0
	finiteOutput := true
	closeVec := func(name string, got, want []float32) {
		if len(got) != len(want) {
			t.Fatalf("%s length=%d want %d", name, len(got), len(want))
		}
		for i := range got {
			if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) || math.IsNaN(float64(want[i])) || math.IsInf(float64(want[i]), 0) {
				finiteOutput = false
				t.Fatalf("%s[%d] must be finite: got=%g want=%g", name, i, got[i], want[i])
			}
			delta := math.Abs(float64(got[i] - want[i]))
			if delta > maxObservedDelta {
				maxObservedDelta = delta
			}
			if delta > 3e-4 {
				t.Fatalf("%s[%d]=%g want %g", name, i, got[i], want[i])
			}
		}
	}
	closeVec("output", be.Read(got), want)
	closeVec("conv state", be.Read(nextC), wantC)
	closeVec("recurrent state", be.Read(nextR), wantR)

	oracleJSON, err := formatQwen35GDNDecodeParityOracle(maxObservedDelta, stateIdentity, finiteOutput)
	if err != nil {
		t.Fatalf("format parity oracle: %v", err)
	}
	t.Logf("%s", oracleJSON)
}

func TestVulkanQwen35GDNDecodeParityOracleFormat(t *testing.T) {
	raw, err := formatQwen35GDNDecodeParityOracle(1.5e-4, true, true)
	if err != nil {
		t.Fatalf("formatQwen35GDNDecodeParityOracle failed: %v", err)
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
		t.Fatalf("unmarshal formatted oracle failed: %v", err)
	}
	if parsed.Schema != "fak.strix.subkernel-parity/v1" {
		t.Errorf("schema = %q, want fak.strix.subkernel-parity/v1", parsed.Schema)
	}
	if parsed.Selector != "qwen35_gdn_decode" {
		t.Errorf("selector = %q, want qwen35_gdn_decode", parsed.Selector)
	}
	if parsed.TestName != "TestVulkanQwen35GDNDecodeMatchesCPUOracleInPlace" {
		t.Errorf("test_name = %q, want TestVulkanQwen35GDNDecodeMatchesCPUOracleInPlace", parsed.TestName)
	}
	if parsed.OracleKind != "state_continuity" {
		t.Errorf("oracle_kind = %q, want state_continuity", parsed.OracleKind)
	}
	if parsed.Engine != "fak-native/vulkan" {
		t.Errorf("engine = %q, want fak-native/vulkan", parsed.Engine)
	}
	if !parsed.DeviceObserved {
		t.Errorf("device_observed must be true")
	}
	if parsed.CaseCount != 1 {
		t.Errorf("case_count = %d, want 1", parsed.CaseCount)
	}
	if !parsed.Passed {
		t.Errorf("passed must be true")
	}
	if parsed.Observed.MaxAbsDelta != 1.5e-4 {
		t.Errorf("observed max_abs_delta = %g, want 1.5e-4", parsed.Observed.MaxAbsDelta)
	}
	if !parsed.Observed.StateIdentity {
		t.Errorf("observed state_identity must be true")
	}
	if !parsed.Observed.FiniteOutput {
		t.Errorf("observed finite_output must be true")
	}
	if parsed.Bounds.MaxAbsDelta != 3e-4 {
		t.Errorf("bounds max_abs_delta = %g, want 3e-4", parsed.Bounds.MaxAbsDelta)
	}
	if !parsed.Bounds.RequireStateIdentity {
		t.Errorf("bounds require_state_identity must be true")
	}
	if !parsed.Bounds.RequireFinite {
		t.Errorf("bounds require_finite must be true")
	}

	// Boundary failure: delta exceeds threshold
	failRaw, err := formatQwen35GDNDecodeParityOracle(4e-4, true, true)
	if err != nil {
		t.Fatalf("format failed oracle failed: %v", err)
	}
	if err := json.Unmarshal(failRaw, &parsed); err != nil {
		t.Fatalf("unmarshal fail oracle failed: %v", err)
	}
	if parsed.Passed {
		t.Errorf("expected passed=false when max_abs_delta > 3e-4")
	}

	// Boundary failure: state identity broken
	failRaw2, err := formatQwen35GDNDecodeParityOracle(1.5e-4, false, true)
	if err != nil {
		t.Fatalf("format failed oracle 2 failed: %v", err)
	}
	if err := json.Unmarshal(failRaw2, &parsed); err != nil {
		t.Fatalf("unmarshal fail oracle 2 failed: %v", err)
	}
	if parsed.Passed {
		t.Errorf("expected passed=false when state_identity=false")
	}

	// Boundary failure: non-finite output
	failRaw3, err := formatQwen35GDNDecodeParityOracle(1.5e-4, true, false)
	if err != nil {
		t.Fatalf("format failed oracle 3 failed: %v", err)
	}
	if err := json.Unmarshal(failRaw3, &parsed); err != nil {
		t.Fatalf("unmarshal fail oracle 3 failed: %v", err)
	}
	if parsed.Passed {
		t.Errorf("expected passed=false when finite_output=false")
	}
}
