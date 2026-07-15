package model

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func pinnedV4RuntimeConfig() Config {
	return Config{ModelType: "deepseek_v4", NumLayers: 61, HiddenSize: 7168, NumExperts: 384, NumExpertsPerTok: 6, MoEIntermediateSize: 3072, NSharedExperts: 1, ExpertDtype: "fp4", NormTopKProb: true, RoutedScalingFactor: 2.5, ScoringFunc: "sqrtsoftplus", TopKMethod: "noaux_tc", SwigluLimit: 10}
}

func writeV4RuntimeFixture(t *testing.T) (string, map[string][]float32) {
	t.Helper()
	dir := t.TempDir()
	weightMap := map[string]string{}
	decoded := map[string][]float32{}
	for layer := 0; layer < v4HashLayers; layer++ {
		name := v4HashTensorName(layer)
		data := make([]byte, v4HashVocab*v4HashTopK*8)
		for token := 0; token < v4HashVocab; token++ {
			for slot := 0; slot < v4HashTopK; slot++ {
				binary.LittleEndian.PutUint64(data[(token*v4HashTopK+slot)*8:], uint64((token+slot+layer*17)%v4HashExperts))
			}
		}
		shard := "hash" + string(rune('0'+layer)) + ".safetensors"
		writeV4RawShard(t, filepath.Join(dir, shard), map[string]tinySTTensor{name: {dtype: "I64", shape: []int{v4HashVocab, v4HashTopK}, data: data}})
		weightMap[name] = shard
	}
	for _, layer := range []int{0, 3} {
		for expert := 0; expert < 12; expert++ {
			for _, proj := range []string{"w1", "w2", "w3"} {
				name := v4RuntimeTensorName(layer, expert, proj)
				rows, cols := 32, 32
				vals := runtimeProjectionValues(layer, expert, proj)
				codes, scales := encodeV4RuntimeFP4(vals, rows, cols)
				weightShard := name + ".safetensors"
				scaleName := strings.TrimSuffix(name, ".weight") + ".scale"
				scaleShard := scaleName + ".safetensors"
				writeV4RawShard(t, filepath.Join(dir, weightShard), map[string]tinySTTensor{name: {dtype: "I8", shape: []int{rows, cols / 2}, data: codes}})
				writeV4RawShard(t, filepath.Join(dir, scaleShard), map[string]tinySTTensor{scaleName: {dtype: "F8_E8M0", shape: []int{rows, 1}, data: scales}})
				weightMap[name], weightMap[scaleName] = weightShard, scaleShard
				decoded[name] = vals
			}
		}
	}
	ib, err := json.Marshal(map[string]any{"weight_map": weightMap})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), ib, 0o600); err != nil {
		t.Fatal(err)
	}
	return dir, decoded
}

func v4RuntimeTensorName(layer, expert int, proj string) string {
	return "layers." + itoa(layer) + ".ffn.experts." + itoa(expert) + "." + proj + ".weight"
}

func runtimeProjectionValues(layer, expert int, proj string) []float32 {
	const dim = 32
	vals := make([]float32, dim*dim)
	base := float32((layer+expert)%3 + 1)
	for row := 0; row < dim; row++ {
		for col := 0; col < dim; col++ {
			if row != col {
				continue
			}
			switch proj {
			case "w1":
				vals[row*dim+col] = base
			case "w3":
				vals[row*dim+col] = base / 2
			default:
				vals[row*dim+col] = base / 2
			}
		}
	}
	return vals
}

func encodeV4RuntimeFP4(vals []float32, rows, cols int) ([]byte, []byte) {
	codes := make([]byte, rows*cols/2)
	scales := make([]byte, rows)
	for r := 0; r < rows; r++ {
		scales[r] = 127
		for c := 0; c < cols; c += 2 {
			lo := v4RuntimeFP4Code(vals[r*cols+c])
			hi := v4RuntimeFP4Code(vals[r*cols+c+1])
			codes[(r*cols+c)/2] = lo | hi<<4
		}
	}
	return codes, scales
}

func v4RuntimeFP4Code(v float32) byte {
	if v == 0 {
		return 0
	}
	sign := byte(0)
	if v < 0 {
		sign, v = 8, -v
	}
	levels := []float32{0, .5, 1, 1.5, 2, 3, 4, 6}
	best := 0
	for i := range levels {
		if math.Abs(float64(levels[i]-v)) < math.Abs(float64(levels[best]-v)) {
			best = i
		}
	}
	return sign | byte(best)
}

func TestV4ExpertRuntimeScoredAndHashEndToEnd(t *testing.T) {
	restoreSpecs := useTinyV4RuntimeQuantSpecs()
	defer restoreSpecs()
	dir, weights := writeV4RuntimeFixture(t)
	be := compute.Default()
	r, err := newV4ExpertRuntime(dir, pinnedV4RuntimeConfig(), be, 16384, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if got := r.Stats(); got.ExpertOpenCount != 0 || got.ExpertReadCount != 0 || got.HashOpenCount != 0 || got.HashReadCount != 0 || got.ResidentBytes != 0 {
		t.Fatalf("constructor performed payload work: %+v", got)
	}
	xHost := make([]float32, 32)
	for i := range xHost {
		xHost[i] = float32((i%5)-2) / 2
	}
	x := be.Upload(compute.NewF32(be, []int{32}, xHost), compute.F32)
	defer be.Free(x)

	logits := make([]float32, 384)
	for i := range logits {
		logits[i] = -10
	}
	for i := 0; i < 6; i++ {
		logits[i] = float32(6 - i)
	}
	scored, err := r.forwardScored(3, logits, nil, x)
	if err != nil {
		t.Fatal(err)
	}
	picks, _ := v4ScoredRoute(logits, nil, 6, 2.5)
	if want := runtimeResidentOracle(3, picks, xHost, weights, 10); !closeSlice(scored, want, 2e-5) {
		t.Fatalf("scored=%v want=%v", scored, want)
	}

	hashLogits := make([]float32, 384)
	for i := range hashLogits {
		hashLogits[i] = float32(i%9) / 2
	}
	hashed, err := r.forwardHash(0, 0, hashLogits, x)
	if err != nil {
		t.Fatal(err)
	}
	ids := []int{0, 1, 2, 3, 4, 5}
	hashPicks, _ := v4HashRoute(hashLogits, ids, 2.5)
	if want := runtimeResidentOracle(0, hashPicks, xHost, weights, 10); !closeSlice(hashed, want, 2e-5) {
		t.Fatalf("hash=%v want=%v", hashed, want)
	}
	stats := r.Stats()
	if stats.HashReadCount != 1 || stats.HashReadBytes != 48 || stats.SourceReads == 0 || stats.PageIns == 0 || stats.Evictions == 0 || stats.PeakResident > stats.RingBudget || stats.ResidentBytes > stats.RingBudget {
		t.Fatalf("runtime stats do not prove bounded selective execution: %+v", stats)
	}
}

func runtimeResidentOracle(layer int, picks []routePick, x []float32, weights map[string][]float32, limit float32) []float32 {
	out := make([]float32, len(x))
	for _, pick := range picks {
		w1 := weights[v4RuntimeTensorName(layer, pick.expert, "w1")]
		w2 := weights[v4RuntimeTensorName(layer, pick.expert, "w2")]
		w3 := weights[v4RuntimeTensorName(layer, pick.expert, "w3")]
		for hidden := range x {
			var gate, up float32
			for input := range x {
				gate += w1[hidden*len(x)+input] * x[input]
				up += w3[hidden*len(x)+input] * x[input]
			}
			gate = min(gate, limit)
			up = max(-limit, min(up, limit))
			act := pick.weight * float32(float64(gate)/(1+math.Exp(float64(-gate)))) * up
			for output := range out {
				out[output] += w2[output*len(x)+hidden] * act
			}
		}
	}
	return out
}

func useTinyV4RuntimeQuantSpecs() func() {
	old := v4ExpertQuantSpecs
	v4ExpertQuantSpecs = map[string]v4ExpertQuantSpec{
		"w1": {weightRows: 32, weightCols: 16, scaleRows: 32, scaleCols: 1},
		"w2": {weightRows: 32, weightCols: 16, scaleRows: 32, scaleCols: 1},
		"w3": {weightRows: 32, weightCols: 16, scaleRows: 32, scaleCols: 1},
	}
	return func() { v4ExpertQuantSpecs = old }
}

func TestV4ExpertRuntimeFailsClosed(t *testing.T) {
	restoreSpecs := useTinyV4RuntimeQuantSpecs()
	defer restoreSpecs()
	dir, _ := writeV4RuntimeFixture(t)
	cfg := pinnedV4RuntimeConfig()
	bad := cfg
	bad.NumExperts = 383
	if _, err := newV4ExpertRuntime(dir, bad, compute.Default(), 16384, 1); !errors.Is(err, ErrV4ExpertRuntime) {
		t.Fatalf("bad config error=%v", err)
	}
	if _, err := newV4ExpertRuntime(dir, cfg, compute.Default(), 0, 1); !errors.Is(err, ErrV4ExpertRuntime) {
		t.Fatalf("bad cap error=%v", err)
	}
	r, err := newV4ExpertRuntime(dir, cfg, compute.Default(), 16384, 1)
	if err != nil {
		t.Fatal(err)
	}
	x := compute.Default().Upload(compute.NewF32(compute.Default(), []int{32}, make([]float32, 32)), compute.F32)
	if _, err := r.forwardScored(2, make([]float32, 384), nil, x); !errors.Is(err, ErrV4ExpertRuntime) {
		t.Fatalf("wrong scored layer error=%v", err)
	}
	if _, err := r.forwardHash(3, 0, make([]float32, 384), x); !errors.Is(err, ErrV4ExpertRuntime) {
		t.Fatalf("wrong hash layer error=%v", err)
	}
	if _, err := r.forwardHash(0, 0, make([]float32, 383), x); !errors.Is(err, ErrV4ExpertRuntime) {
		t.Fatalf("bad logits error=%v", err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := r.forwardHash(0, 0, make([]float32, 384), x); !errors.Is(err, ErrV4ExpertRuntime) {
		t.Fatalf("use after close error=%v", err)
	}
}

func closeSlice(a, b []float32, eps float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > eps {
			return false
		}
	}
	return true
}
