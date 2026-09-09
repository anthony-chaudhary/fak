package model

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const flashDenseQName = "layers.0.attn.wq_a.weight"

func TestDeepSeekV4FlashDenseOfficialPairsLoadScaleFirstIntoQ8(t *testing.T) {
	dir, weights, scales := writeFlashDenseQuantDir(t)
	shardPath := filepath.Join(dir, "model-00001-of-00001.safetensors")
	sf, err := openSafetensorsFile(shardPath)
	if err != nil {
		t.Fatal(err)
	}
	names := safetensorsTensorNames(sf.hdr)
	sf.Close()
	scalePos, weightPos := -1, -1
	for i, name := range names {
		switch name {
		case "layers.0.attn.wq_a.scale":
			scalePos = i
		case flashDenseQName:
			weightPos = i
		}
	}
	if scalePos < 0 || weightPos < 0 || scalePos >= weightPos {
		t.Fatalf("fixture does not exercise scale-first ordering: scale=%d weight=%d names=%v", scalePos, weightPos, names)
	}
	m, err := LoadSafetensorsQuantConfigDir(dir)
	if err != nil {
		t.Fatal(err)
	}

	if m.q8w[flashDenseQName] == nil {
		t.Errorf("official dense tensor %s missing from native Q8 store", flashDenseQName)
	}
	if _, ok := m.manifest["layers.0.attn.wq_a.scale"]; ok {
		t.Errorf("scale sibling for %s leaked into the ordinary manifest", flashDenseQName)
	}
	for _, name := range []string{"layers.0.ffn.gate.weight", "head.weight"} {
		if m.q8w[name] != nil {
			t.Errorf("official BF16 tensor %s was mistaken for an FP8 pair", name)
		}
		if _, ok := m.manifest[name]; !ok {
			t.Errorf("official BF16 tensor %s did not follow the ordinary loader path", name)
		}
	}

	q := m.q8w[flashDenseQName]
	if q != nil {
		wantQ, wantD := flashDenseQ8Oracle(weights, 1024, 4096, scales, 32)
		if q.out != 1024 || q.in != 4096 || q.nblk != 128 {
			t.Fatalf("Q8 shape = out %d in %d blocks %d, want 1024/4096/128", q.out, q.in, q.nblk)
		}
		if len(q.q) != len(wantQ) || len(q.d) != len(wantD) {
			t.Fatalf("Q8 payload lengths = %d/%d, want %d/%d", len(q.q), len(q.d), len(wantQ), len(wantD))
		}
		for i := range wantQ {
			if q.q[i] != wantQ[i] {
				t.Fatalf("Q8 code[%d] = %d, want %d from independent E4M3*E8M0 oracle", i, q.q[i], wantQ[i])
			}
		}
		for i := range wantD {
			if math.Float32bits(q.d[i]) != math.Float32bits(wantD[i]) {
				t.Fatalf("Q8 scale[%d] bits = %08x, want %08x", i, math.Float32bits(q.d[i]), math.Float32bits(wantD[i]))
			}
		}
	}

	routed := "layers.0.ffn.experts.0.w1.weight"
	if m.q8w[routed] != nil {
		t.Fatalf("lazy routed FP4 tensor %s was eagerly placed in Q8", routed)
	}
	if _, ok := m.manifest[routed]; ok {
		t.Fatalf("lazy routed FP4 tensor %s was eagerly placed in manifest", routed)
	}
	hashTable := "layers.0.ffn.gate.tid2eid"
	if m.q8w[hashTable] != nil {
		t.Fatalf("lazy I64 hash table %s was eagerly placed in Q8", hashTable)
	}
	if _, ok := m.manifest[hashTable]; ok {
		t.Fatalf("lazy I64 hash table %s was eagerly placed in manifest", hashTable)
	}
	if m.sourceDir != dir {
		t.Fatalf("lazy routed sourceDir = %q, want %q", m.sourceDir, dir)
	}
}

func TestDeepSeekV4FlashDenseSkipsLazyHashTablesBeforePayloadRead(t *testing.T) {
	_, cfg := readDeepSeekV4FlashConfig(t)
	hdr := map[string]json.RawMessage{}
	names := make([]string, 0, 3)
	for layer := 0; layer < 3; layer++ {
		name := "layers." + itoa(layer) + ".ffn.gate.tid2eid"
		names = append(names, name)
		hdr[name] = flashDenseRawEntry(t, stEntry{Dtype: "I64", Shape: []int{129280, 6}})
	}
	reads := 0
	m := &Model{Cfg: cfg, manifest: map[string]tensorMeta{}, q8w: map[string]*q8Tensor{}}
	var raw []byte
	off := 0
	err := quantizeNamedTensorsInto(names, hdr, func(stEntry) ([]byte, error) {
		reads++
		return nil, errors.New("unexpected lazy hash payload read")
	}, nil, m, false, &raw, &off)
	if err != nil || reads != 0 || len(m.manifest) != 0 || len(m.q8w) != 0 {
		t.Fatalf("lazy hash skip error=%v reads=%d manifest=%d q8=%d, want metadata-only handoff", err, reads, len(m.manifest), len(m.q8w))
	}
}

func TestDeepSeekV4FlashDenseOfficialRoleShapesValidateBeforePayloadRead(t *testing.T) {
	_, cfg := readDeepSeekV4FlashConfig(t)
	for _, role := range flashDenseOfficialRoles() {
		t.Run(role.name, func(t *testing.T) {
			scaleName := strings.TrimSuffix(role.name, ".weight") + ".scale"
			hdr := map[string]json.RawMessage{
				role.name: flashDenseRawEntry(t, stEntry{Dtype: "F8_E4M3", Shape: role.weightShape}),
				scaleName: flashDenseRawEntry(t, stEntry{Dtype: "F8_E8M0", Shape: role.scaleShape}),
			}
			readerSentinel := errors.New("tensor reader reached")
			reads := 0
			m := &Model{Cfg: cfg, manifest: map[string]tensorMeta{}, q8w: map[string]*q8Tensor{}}
			var raw []byte
			off := 0
			err := quantizeNamedTensorsInto([]string{scaleName, role.name}, hdr, func(stEntry) ([]byte, error) {
				reads++
				return nil, readerSentinel
			}, nil, m, false, &raw, &off)
			if !errors.Is(err, readerSentinel) || reads != 1 {
				t.Fatalf("valid official metadata error=%v reads=%d, want one sentinel read", err, reads)
			}

			for _, malformed := range []struct {
				name        string
				weightShape []int
				scaleShape  []int
			}{
				{name: "weight", weightShape: append([]int{role.weightShape[0] - 1}, role.weightShape[1:]...), scaleShape: role.scaleShape},
				{name: "scale", weightShape: role.weightShape, scaleShape: append([]int{role.scaleShape[0] - 1}, role.scaleShape[1:]...)},
			} {
				t.Run("wrong_"+malformed.name+"_shape", func(t *testing.T) {
					hdr[role.name] = flashDenseRawEntry(t, stEntry{Dtype: "F8_E4M3", Shape: malformed.weightShape})
					hdr[scaleName] = flashDenseRawEntry(t, stEntry{Dtype: "F8_E8M0", Shape: malformed.scaleShape})
					reads = 0
					err := quantizeNamedTensorsInto([]string{scaleName, role.name}, hdr, func(stEntry) ([]byte, error) {
						reads++
						return nil, readerSentinel
					}, nil, m, false, &raw, &off)
					if err == nil || !strings.Contains(err.Error(), "shape") || reads != 0 {
						t.Fatalf("malformed metadata error=%v reads=%d, want shape refusal before payload read", err, reads)
					}
				})
			}
		})
	}
}

func TestDeepSeekV4FlashDenseHandlerPreservesProOldScaleFormat(t *testing.T) {
	name := "model.layers.0.self_attn.q_a_proj.weight"
	path := writeTinySafetensors(t, map[string]tinySTTensor{
		name:                {dtype: "F8_E4M3", shape: []int{1, 32}, data: flashFilledBytes(32, 0x38)},
		name + "_scale_inv": {dtype: "F32", shape: []int{1, 1}, data: f32TestBytes([]float32{2})},
	})
	m, err := LoadSafetensorsQuant(path, pinnedV4Config())
	if err != nil {
		t.Fatalf("Pro old-format FP8 scale pair regressed: %v", err)
	}
	if m.q8w[name] == nil {
		t.Fatalf("Pro old-format tensor %s missing from Q8 store", name)
	}
}

func TestDeepSeekV4FlashDenseSkipsMTPPairsBeforePayloadRead(t *testing.T) {
	_, cfg := readDeepSeekV4FlashConfig(t)
	weightName := "mtp.layers.0.attn.wq_a.weight"
	scaleName := "mtp.layers.0.attn.wq_a.scale"
	hdr := map[string]json.RawMessage{
		weightName: flashDenseRawEntry(t, stEntry{Dtype: "F8_E4M3", Shape: []int{1024, 4096}}),
		scaleName:  flashDenseRawEntry(t, stEntry{Dtype: "F8_E8M0", Shape: []int{8, 32}}),
	}
	original := RetainMTP
	t.Cleanup(func() { RetainMTP = original })
	run := func() (int, error) {
		reads := 0
		m := &Model{Cfg: cfg, manifest: map[string]tensorMeta{}, q8w: map[string]*q8Tensor{}}
		var raw []byte
		off := 0
		err := quantizeNamedTensorsInto([]string{scaleName, weightName}, hdr, func(stEntry) ([]byte, error) {
			reads++
			return nil, errors.New("unexpected MTP payload read")
		}, nil, m, false, &raw, &off)
		return reads, err
	}

	RetainMTP = false
	if reads, err := run(); err != nil || reads != 0 {
		t.Fatalf("default-unretained MTP pair error=%v reads=%d, want silent metadata skip", err, reads)
	}
	RetainMTP = true
	reads, err := run()
	if err == nil || reads != 0 || !strings.Contains(err.Error(), "RetainMTP") || !strings.Contains(err.Error(), "unsupported") || !strings.Contains(err.Error(), "mtp.layers.0.attn.wq_a") {
		t.Fatalf("retained MTP pair error=%v reads=%d, want explicit pre-read unsupported refusal naming the pair", err, reads)
	}
}

func TestDeepSeekV4FlashDenseRejectsMalformedOfficialPairs(t *testing.T) {
	_, cfg := readDeepSeekV4FlashConfig(t)
	baseWeights := flashFilledBytes(1024*4096, 0x38)
	baseScales := flashFilledBytes(8*32, 127)
	nanWeights := append([]byte(nil), baseWeights...)
	nanWeights[0] = 0x7f
	overflowWeights := append([]byte(nil), baseWeights...)
	overflowWeights[0] = 0x7e
	nanScales := append([]byte(nil), baseScales...)
	nanScales[0] = 0xff
	overflowScales := append([]byte(nil), baseScales...)
	overflowScales[0] = 254
	cases := []struct {
		name    string
		tensors map[string]tinySTTensor
		want    string
	}{
		{
			name: "missing scale",
			tensors: map[string]tinySTTensor{
				flashDenseQName: {dtype: "F8_E4M3", shape: []int{1024, 4096}, data: baseWeights},
			},
			want: "missing",
		},
		{
			name: "wrong weight dtype",
			tensors: map[string]tinySTTensor{
				flashDenseQName:            {dtype: "BF16", shape: []int{1024, 4096}, data: make([]byte, 1024*4096*2)},
				"layers.0.attn.wq_a.scale": {dtype: "F8_E8M0", shape: []int{8, 32}, data: baseScales},
			},
			want: "dtype",
		},
		{
			name: "wrong scale dtype",
			tensors: map[string]tinySTTensor{
				flashDenseQName:            {dtype: "F8_E4M3", shape: []int{1024, 4096}, data: baseWeights},
				"layers.0.attn.wq_a.scale": {dtype: "F32", shape: []int{8, 32}, data: f32TestBytes(make([]float32, 8*32))},
			},
			want: "dtype",
		},
		{
			name: "wrong scale shape",
			tensors: map[string]tinySTTensor{
				flashDenseQName:            {dtype: "F8_E4M3", shape: []int{1024, 4096}, data: baseWeights},
				"layers.0.attn.wq_a.scale": {dtype: "F8_E8M0", shape: []int{8, 31}, data: flashFilledBytes(8*31, 127)},
			},
			want: "shape",
		},
		{
			name: "non-finite scale",
			tensors: map[string]tinySTTensor{
				flashDenseQName:            {dtype: "F8_E4M3", shape: []int{1024, 4096}, data: baseWeights},
				"layers.0.attn.wq_a.scale": {dtype: "F8_E8M0", shape: []int{8, 32}, data: nanScales},
			},
			want: "NaN",
		},
		{
			name: "non-finite E4M3 weight",
			tensors: map[string]tinySTTensor{
				flashDenseQName:            {dtype: "F8_E4M3", shape: []int{1024, 4096}, data: nanWeights},
				"layers.0.attn.wq_a.scale": {dtype: "F8_E8M0", shape: []int{8, 32}, data: baseScales},
			},
			want: "NaN",
		},
		{
			name: "finite operands overflow float32",
			tensors: map[string]tinySTTensor{
				flashDenseQName:            {dtype: "F8_E4M3", shape: []int{1024, 4096}, data: overflowWeights},
				"layers.0.attn.wq_a.scale": {dtype: "F8_E8M0", shape: []int{8, 32}, data: overflowScales},
			},
			want: "non-finite",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := writeTinySafetensors(t, tc.tensors)
			if _, err := LoadSafetensorsQuant(path, cfg); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("malformed pair error = %v, want actionable %q refusal", err, tc.want)
			}
		})
	}
}

func writeFlashDenseQuantDir(t *testing.T) (string, []byte, []byte) {
	t.Helper()
	dir := t.TempDir()
	rawConfig, _ := readDeepSeekV4FlashConfig(t)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), rawConfig, 0o600); err != nil {
		t.Fatal(err)
	}

	weights := make([]byte, 1024*4096)
	codes := []byte{0x38, 0x40, 0x30, 0xb8, 0xc0, 0x34, 0x3c, 0x28}
	for i := range weights {
		weights[i] = codes[i%len(codes)]
	}
	scales := make([]byte, 8*32)
	for i := range scales {
		scales[i] = []byte{126, 127, 128, 129}[i%4]
	}
	tensors := map[string]tinySTTensor{}
	tensors[flashDenseQName] = tinySTTensor{dtype: "F8_E4M3", shape: []int{1024, 4096}, data: weights}
	tensors["layers.0.attn.wq_a.scale"] = tinySTTensor{dtype: "F8_E8M0", shape: []int{8, 32}, data: scales}
	for _, name := range []string{"layers.0.ffn.gate.weight", "head.weight"} {
		tensors[name] = tinySTTensor{dtype: "BF16", shape: []int{256}, data: make([]byte, 256*2)}
	}
	tensors["layers.0.ffn.experts.0.w1.weight"] = tinySTTensor{dtype: "I8", shape: []int{1, 16}, data: make([]byte, 16)}
	tensors["layers.0.ffn.gate.tid2eid"] = tinySTTensor{dtype: "I64", shape: []int{129280, 6}, data: make([]byte, 129280*6*8)}

	shard := "model-00001-of-00001.safetensors"
	writeV4RawShard(t, filepath.Join(dir, shard), tensors)
	weightMap := make(map[string]string, len(tensors))
	for name := range tensors {
		weightMap[name] = shard
	}
	index, err := json.Marshal(map[string]any{"weight_map": weightMap})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), index, 0o600); err != nil {
		t.Fatal(err)
	}
	return dir, weights, scales
}

type flashDenseRole struct {
	name        string
	weightShape []int
	scaleShape  []int
}

func flashDenseOfficialRoles() []flashDenseRole {
	return []flashDenseRole{
		{name: "layers.0.attn.wq_a.weight", weightShape: []int{1024, 4096}, scaleShape: []int{8, 32}},
		{name: "layers.0.attn.wq_b.weight", weightShape: []int{32768, 1024}, scaleShape: []int{256, 8}},
		{name: "layers.0.attn.wkv.weight", weightShape: []int{512, 4096}, scaleShape: []int{4, 32}},
		{name: "layers.0.attn.wo_a.weight", weightShape: []int{8192, 4096}, scaleShape: []int{64, 32}},
		{name: "layers.0.attn.wo_b.weight", weightShape: []int{4096, 8192}, scaleShape: []int{32, 64}},
		{name: "layers.2.attn.indexer.wq_b.weight", weightShape: []int{8192, 1024}, scaleShape: []int{64, 8}},
		{name: "layers.0.ffn.shared_experts.w1.weight", weightShape: []int{2048, 4096}, scaleShape: []int{16, 32}},
		{name: "layers.0.ffn.shared_experts.w2.weight", weightShape: []int{4096, 2048}, scaleShape: []int{32, 16}},
		{name: "layers.0.ffn.shared_experts.w3.weight", weightShape: []int{2048, 4096}, scaleShape: []int{16, 32}},
	}
}

func flashDenseRawEntry(t *testing.T, entry stEntry) json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(entry)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func flashFilledBytes(n int, value byte) []byte {
	out := make([]byte, n)
	for i := range out {
		out[i] = value
	}
	return out
}

// flashDenseQ8Oracle independently decodes published E4M3 values, applies the
// sibling E8M0 128x128 block scale, then computes the Q8_0 representation.
func flashDenseQ8Oracle(weights []byte, rows, cols int, scales []byte, scaleCols int) ([]int8, []float32) {
	decoded := make([]float32, len(weights))
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			scale := float32(math.Ldexp(1, int(scales[(row/128)*scaleCols+col/128])-127))
			decoded[row*cols+col] = flashE4M3Oracle(weights[row*cols+col]) * scale
		}
	}
	nblk := cols / 32
	q := make([]int8, len(decoded))
	d := make([]float32, rows*nblk)
	for row := 0; row < rows; row++ {
		for block := 0; block < nblk; block++ {
			base := row*cols + block*32
			var amax float32
			for _, value := range decoded[base : base+32] {
				if a := float32(math.Abs(float64(value))); a > amax {
					amax = a
				}
			}
			delta := amax / 127
			d[row*nblk+block] = delta
			for i, value := range decoded[base : base+32] {
				q[base+i] = flashQ8RoundOracle(value / delta)
			}
		}
	}
	return q, d
}

func flashE4M3Oracle(bits byte) float32 {
	sign := float32(1)
	if bits&0x80 != 0 {
		sign = -1
	}
	exponent := int((bits >> 3) & 0x0f)
	mantissa := int(bits & 0x07)
	if exponent == 0 {
		return sign * float32(math.Ldexp(float64(mantissa)/8, -6))
	}
	return sign * float32(math.Ldexp(1+float64(mantissa)/8, exponent-7))
}

func flashQ8RoundOracle(value float32) int8 {
	if value >= 0 {
		return int8(math.Floor(float64(value) + 0.5))
	}
	return int8(math.Ceil(float64(value) - 0.5))
}
