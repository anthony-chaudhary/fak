package model

import (
	"encoding/json"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

var v41GroupedFields = []string{"grouped_output_device_calls", "grouped_output_host_calls", "grouped_output_device_rows", "grouped_output_host_rows", "grouped_output_matmul_calls", "grouped_output_activation_upload_bytes", "grouped_output_readback_bytes", "grouped_output_nanos", "grouped_output_host_weight_f32_bytes"}

func v41GroupedPhase(t *testing.T, m *Model, phase string) map[string]float64 {
	t.Helper()
	raw, err := json.Marshal(m.V41ExpertFaultAttribution())
	if err != nil {
		t.Fatal(err)
	}
	var phases map[string]map[string]float64
	// Other attribution fields contain backend names, so decode the selected fields individually.
	var objects map[string]map[string]json.RawMessage
	if err := json.Unmarshal(raw, &objects); err != nil {
		t.Fatal(err)
	}
	phases = map[string]map[string]float64{phase: {}}
	for _, key := range v41GroupedFields {
		value, ok := objects[phase][key]
		if !ok {
			t.Errorf("default %s missing %s", phase, key)
			continue
		}
		var n float64
		if err := json.Unmarshal(value, &n); err != nil {
			t.Fatal(err)
		}
		phases[phase][key] = n
	}
	return phases[phase]
}

func v41GroupedFixture(t *testing.T, dtype string, role, flat bool) *Model {
	t.Helper()
	m := v41IncrementalPlainModel(t, 1)
	if role {
		m = v41IncrementalExpertFixture(t, true, false)
	}
	if dtype == "Q2_K" || dtype == "Q4_K" {
		m.Cfg.NumHeads, m.Cfg.OGroups, m.Cfg.OLoraRank = 16, 2, 128
	}
	c := m.Cfg
	groupWidth := c.NumHeads / c.OGroups * c.HeadDim
	aShape := []int{c.OGroups * c.OLoraRank, groupWidth}
	if flat {
		aShape = []int{c.OLoraRank, c.NumHeads * c.HeadDim}
	}
	tensors := []synthTensor{
		{layerName(0, "attn.wq_b.weight"), []int{c.NumHeads * c.HeadDim, c.QLoraRank}},
		{layerName(0, "attn.wo_a.weight"), aShape},
		{layerName(0, "attn.wo_b.weight"), []int{c.HiddenSize, c.OGroups * c.OLoraRank}},
		{layerName(0, "attn.sink"), []int{c.NumHeads}},
	}
	manifest, raw := synthBuildRaw(tensors, synthMatmulFill)
	for name, meta := range manifest {
		meta.Offset += len(m.raw)
		m.manifest[name] = meta
	}
	m.raw = append(m.raw, raw...)
	for _, leaf := range []string{"attn.wo_a.weight", "attn.wo_b.weight"} {
		name := layerName(0, leaf)
		out, in := c.OGroups*c.OLoraRank, groupWidth
		if leaf == "attn.wo_b.weight" {
			out, in = c.HiddenSize, c.OGroups*c.OLoraRank
		}
		w, ok := m.residentF32Mat(name)
		if !ok {
			t.Fatal("grouped fixture weight absent")
		}
		switch dtype {
		case "Q8_0":
			if m.q8w == nil {
				m.q8w = map[string]*q8Tensor{}
			}
			m.q8w[name] = quantizeQ8(w, out, in)
		case "Q4_K":
			if m.q4kw == nil {
				m.q4kw = map[string]*q4kTensor{}
			}
			m.q4kw[name] = quantizeQ4KFromRaw(v41DenseTestQ4Raw(out, in), out, in)
		case "Q2_K":
			if m.kqw == nil {
				m.kqw = map[string]*kQuantTensor{}
			}
			seed := uint64(13668)
			if leaf == "attn.wo_b.weight" {
				seed++
			}
			m.kqw[name] = q2kFixtureTensor(out, in, seed)
		}
		if dtype != "F32" {
			delete(m.manifest, name)
		}
	}
	return m
}

type v41GroupedOp struct {
	leaf               string
	rows, upload, read int
}
type v41GroupedBackend struct {
	*v41DenseTestBackend
	a              [][]float32
	b              []float32
	grouped        []v41GroupedOp
	outputs        map[compute.Buffer]int
	weights        map[compute.Buffer]bool
	weightFrees    map[compute.Buffer]int
	decline        compute.Dtype
	deny           bool
	fault          string
	cause          any
	failing        bool
	matmulAttempts int
}

func newV41GroupedBackend(t *testing.T, m *Model) *v41GroupedBackend {
	t.Helper()
	a, ok := m.residentF32Mat(layerName(0, "attn.wo_a.weight"))
	if !ok {
		t.Fatal("missing grouped A oracle")
	}
	b, ok := m.residentF32Mat(layerName(0, "attn.wo_b.weight"))
	if !ok {
		t.Fatal("missing grouped B oracle")
	}
	groups := make([][]float32, m.Cfg.OGroups)
	for g := range groups {
		groups[g] = append([]float32(nil), a[g*len(a)/len(groups):(g+1)*len(a)/len(groups)]...)
	}
	return &v41GroupedBackend{v41DenseTestBackend: newV41DenseTestBackend(), a: groups, b: b, outputs: map[compute.Buffer]int{}, weights: map[compute.Buffer]bool{}, weightFrees: map[compute.Buffer]int{}, decline: compute.Dtype(255)}
}
func (b *v41GroupedBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return !b.deny && dt != b.decline && b.v41DenseTestBackend.SupportsDeviceWeightDtype(dt)
}
func (b *v41GroupedBackend) weightLeaf(w compute.Tensor) string {
	values := b.Backend.Read(w)
	if w.Dtype.Quantized() {
		host, ok := w.Buf().(compute.HostBuffer)
		if !ok {
			return ""
		}
		codes := host.I8()
		values = make([]float32, w.Shape[0]*w.Shape[1])
		if w.Dtype == compute.Q8_0 {
			for i, code := range codes {
				values[i] = float32(code) * w.Quant.Scale[i/qBlk]
			}
		} else {
			raw := make([]byte, len(codes))
			for i, code := range codes {
				raw[i] = byte(code)
			}
			stride := q4kBlockBytes
			if w.Dtype == compute.Q2_K {
				stride = q2kBlockBytes
			}
			for offset := 0; offset < len(raw); offset += stride {
				dst := values[offset/stride*qkK:]
				if w.Dtype == compute.Q4_K {
					q4kDequantSuperBlock(dst, raw[offset:offset+stride])
				} else if w.Dtype == compute.Q2_K {
					q2kDequantSuperBlock(dst, raw[offset:offset+stride])
				} else {
					return ""
				}
			}
		}
	}
	if reflect.DeepEqual(values, b.b) {
		return "b"
	}
	for _, group := range b.a {
		if reflect.DeepEqual(values, group) {
			return "a"
		}
	}
	return ""
}
func (b *v41GroupedBackend) multiply(w, x compute.Tensor, rows int, batch bool) compute.Tensor {
	leaf := b.weightLeaf(w)
	if leaf != "" {
		b.weights[w.Buf()] = true
		b.matmulAttempts++
		if b.fault == "matmul" && leaf == "b" {
			b.failing = true
			panic(b.cause)
		}
	}
	var out compute.Tensor
	if batch {
		out = b.v41DenseTestBackend.BatchedMatMul(w, x, rows)
	} else {
		out = b.v41DenseTestBackend.MatMul(w, x)
	}
	if leaf != "" {
		b.grouped = append(b.grouped, v41GroupedOp{leaf: leaf, rows: rows, upload: b.uploads[x.Buf()]})
		b.outputs[out.Buf()] = len(b.grouped) - 1
	}
	return out
}
func (b *v41GroupedBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	return b.multiply(w, x, 1, false)
}
func (b *v41GroupedBackend) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	return b.multiply(w, x, rows, true)
}
func (b *v41GroupedBackend) Free(x compute.Tensor) {
	if b.weights[x.Buf()] {
		b.weightFrees[x.Buf()]++
	}
	b.v41DenseTestBackend.Free(x)
}
func (b *v41GroupedBackend) Read(x compute.Tensor) []float32 {
	op, grouped := b.outputs[x.Buf()]
	if grouped && b.grouped[op].leaf == "b" && b.fault == "read" {
		b.failing = true
		panic(b.cause)
	}
	values := b.v41DenseTestBackend.Read(x)
	if grouped && b.grouped[op].leaf == "b" {
		if b.fault == "length" {
			b.failing = true
			values = append(values, 0)
		}
		if b.fault == "finite" {
			b.failing = true
			values[0] = float32(math.NaN())
		}
	}
	if grouped {
		b.grouped[op].read += 4 * len(values)
	}
	return values
}

func v41GroupedParity(t *testing.T, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) || len(got) == 0 {
		t.Fatalf("grouped logits width got=%d want=%d", len(got), len(want))
	}
	var signal, delta float64
	for i, value := range got {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatal("nonfinite grouped output")
		}
		signal = max(signal, math.Abs(float64(want[i])))
		delta = max(delta, math.Abs(float64(value-want[i])))
	}
	if signal == 0 {
		t.Fatal("grouped logits oracle has no signal")
	}
	if delta > tolerance*max(1, signal) {
		t.Fatalf("grouped parity maximum_diff=%g scale=%g tolerance=%g", delta, signal, tolerance)
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=2ms lane=default
func TestV41GroupedOutputIndependentAlgebraAndIsolation(t *testing.T) {
	t.Parallel()
	const groups, rank, width, rows, dim = 2, 3, 4, 2, 5
	a, b, x := make([]float32, groups*rank*width), make([]float32, dim*groups*rank), make([]float32, rows*groups*width)
	for i := range a {
		a[i] = float32((i*7)%17-8) / 11
	}
	for i := range b {
		b[i] = float32((i*11)%19-9) / 13
	}
	for i := range x {
		x[i] = float32(i*i%23-11) / 7
	}
	want := make([]float32, rows*dim)
	for row := 0; row < rows; row++ {
		for d := 0; d < dim; d++ {
			for g := 0; g < groups; g++ {
				for r := 0; r < rank; r++ {
					for k := 0; k < width; k++ {
						want[row*dim+d] += x[row*groups*width+g*width+k] * a[(g*rank+r)*width+k] * b[d*groups*rank+g*rank+r]
					}
				}
			}
		}
	}
	got, err := V41GroupedOutputProjection(x, a, b, 1, rows, groups, width, groups, rank, dim)
	if err != nil {
		t.Fatal(err)
	}
	v41GroupedParity(t, got, want, 1e-6)
	flat := make([]float32, rows*dim)
	for row := 0; row < rows; row++ {
		for d := 0; d < dim; d++ {
			for r := 0; r < rank; r++ {
				for k := 0; k < groups*width; k++ {
					flat[row*dim+d] += x[row*groups*width+k] * a[r*groups*width+k] * b[d*groups*rank+r]
				}
			}
		}
	}
	if loraMaxAbsDiff(flat, want) < 0.1 {
		t.Fatal("ordinary flat negative control does not distinguish grouping")
	}
	for g := 0; g < groups; g++ {
		masked := append([]float32(nil), b...)
		for d := 0; d < dim; d++ {
			for r := 0; r < rank; r++ {
				masked[d*groups*rank+(1-g)*rank+r] = 0
			}
		}
		first, err := V41GroupedOutputProjection(x, a, masked, 1, rows, groups, width, groups, rank, dim)
		if err != nil {
			t.Fatal(err)
		}
		changed := append([]float32(nil), x...)
		for row := 0; row < rows; row++ {
			for k := 0; k < width; k++ {
				changed[row*groups*width+(1-g)*width+k] += 100
			}
		}
		second, err := V41GroupedOutputProjection(changed, a, masked, 1, rows, groups, width, groups, rank, dim)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(first, second) {
			t.Error("one group's projection reads another group's activation")
		}
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=1ms lane=default
func TestV41GroupedOutputDefaultExportedLedger(t *testing.T) {
	t.Parallel()
	phase := reflect.TypeOf(V41ExpertFaultAttribution{}.Prefill)
	fields := []string{"GroupedOutputDeviceCalls", "GroupedOutputHostCalls", "GroupedOutputDeviceRows", "GroupedOutputHostRows", "GroupedOutputMatMulCalls", "GroupedOutputActivationUploadBytes", "GroupedOutputReadbackBytes", "GroupedOutputNanos", "GroupedOutputHostWeightF32Bytes"}
	for i, name := range fields {
		field, ok := phase.FieldByName(name)
		if !ok || field.PkgPath != "" {
			t.Errorf("missing exported phase field %s", name)
			continue
		}
		if field.Tag.Get("json") != v41GroupedFields[i] {
			t.Errorf("field %s does not expose default JSON key %s", name, v41GroupedFields[i])
		}
	}
	m := &Model{}
	for _, phase := range []string{"prefill", "decode"} {
		for key, n := range v41GroupedPhase(t, m, phase) {
			if n != 0 {
				t.Errorf("inert %s=%g", key, n)
			}
		}
	}
}
