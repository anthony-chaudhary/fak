package model

// cohere_loader_routing_test.go — the ROUTING witness for the HF-source checkpoint
// constructor. Companion to cohere_rotary.go, which implements the load-time Cohere
// rotary re-layout, and to family_cohere_cpu_oracle_test.go, which proves that
// constructor's numerics against an independent HuggingFace transcription.
//
// The gap this closes. The CPU oracle calls newHFCheckpointModel DIRECTLY, so it
// witnesses the constructor but says nothing about whether production ever REACHES it.
// Reverting any of the three HF-source call sites —
//
//	safetensors.go loadSafetensorsDir  (public LoadSafetensorsDir, sharded snapshot)
//	safetensors.go loadSafetensorsFile (public LoadSafetensors, single .safetensors)
//	weights.go     Load                (public Load, the export_oracle.py dir)
//
// back to newModel would silently restore the wrong rotary pairing on every real Cohere
// download while the entire package stayed green: map those two files back to their
// pre-change bytes and not one existing test reds. So this file drives each PUBLIC entry
// point over an on-disk Cohere fixture and asserts the one thing only the
// newHFCheckpointModel path produces — q/k (and the per-head q_norm/k_norm) rows in fak's
// HALF-SPLIT component order instead of the INTERLEAVED order HuggingFace wrote.
//
// Two independent assertions per entry point, because either alone is escapable:
//
//   - LAYOUT. Every rotary-axis tensor of the loaded model must equal sigma(checkpoint
//     bytes), where sigma is transcribed HERE from the HF convention (element 2j -> j,
//     2j+1 -> j+head_dim/2) rather than obtained by calling interleavedToHalfSplitRows —
//     so the expectation does not come out of the code under test. A reverted call site
//     leaves the tensor byte-identical to the checkpoint, which the failure names.
//   - LOGITS. The publicly loaded model's Forward must agree with a model built straight
//     through newHFCheckpointModel on the same bytes. That is what makes the routing
//     observable in production OUTPUT rather than only in the manifest; the rotary
//     divergence it pins is worth ~1e-3 on this fixture, a thousand times the tolerance.
//
// The fixture is the oracle's own Cohere geometry (cohereOracleCfg / cohereOracleTensors)
// with use_qk_norm ON, so the per-head q_norm/k_norm rows are on the component axis too
// and head_dim 8 keeps the interleaved pairing (0,1),(2,3),... distinguishable from the
// half-split pairing (0,4),(1,5),....
//
// NOT covered, deliberately: the GPTQ loader. gptq.go still calls newModel, exactly as
// cohere_rotary.go's KNOWN RESIDUAL paragraph records — a quantized checkpoint keeps
// q_proj/k_proj in its own decoded store rather than in the f32 blob this pass rewrites,
// so routing it through newHFCheckpointModel would not fix it. When that is fixed it
// needs its own witness, which is not this one.

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// cohereRoutingRow names one fixture tensor on the head-local component axis together
// with how many f32 values one row on that axis spans: the input width for a [out,in]
// projection weight, 1 for a per-head (heads, head_dim) norm weight.
type cohereRoutingRow struct {
	name     string
	rowElems int
}

// cohereRoutingFixture builds the Cohere fixture and returns the derived config, the
// manifest, and the PRISTINE checkpoint bytes in HuggingFace order. Nothing here has been
// through a loader, so these bytes are what a real download presents.
func cohereRoutingFixture(t *testing.T) (Config, map[string]tensorMeta, []byte) {
	t.Helper()
	cfg := cohereOracleCfg(true) // use_qk_norm on: q_norm/k_norm join the rotary axis
	if err := cfg.deriveConfigAxes(configJSONHints{}); err != nil {
		t.Fatalf("deriveConfigAxes: %v", err)
	}
	man, raw := synthBuildRaw(cohereOracleTensors(cfg), func(name string, next func() float32) float32 {
		if isCPUOracleNormWeight(name) {
			return 1 + 0.25*next()
		}
		return synthMatmulFill(name, next)
	})
	return cfg, man, raw
}

// cohereRoutingValues decodes one named f32 tensor out of a (manifest, raw) pair.
func cohereRoutingValues(t *testing.T, man map[string]tensorMeta, raw []byte, name string) []float32 {
	t.Helper()
	meta, ok := man[name]
	if !ok {
		t.Fatalf("fixture tensor %q missing from manifest", name)
	}
	n := meta.Nbytes / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[meta.Offset+i*4:]))
	}
	return out
}

// cohereRoutingHalfSplit is an INDEPENDENT transcription of the interleaved -> half-split
// row permutation, written from Cohere's published convention and NOT by calling
// interleavedToHalfSplitRows: within each head_dim-sized group of rows, source row 2j
// becomes destination row j and source row 2j+1 becomes destination row j+head_dim/2.
// rowElems is the width of one row.
func cohereRoutingHalfSplit(src []float32, hd, rowElems int) []float32 {
	out := make([]float32, len(src))
	half := hd / 2
	groups := len(src) / rowElems / hd
	for g := 0; g < groups; g++ {
		base := g * hd
		for j := 0; j < half; j++ {
			lo := (base + j) * rowElems
			hi := (base + half + j) * rowElems
			copy(out[lo:lo+rowElems], src[(base+2*j)*rowElems:(base+2*j+1)*rowElems])
			copy(out[hi:hi+rowElems], src[(base+2*j+1)*rowElems:(base+2*j+2)*rowElems])
		}
	}
	return out
}

// cohereRoutingRotaryRoster is every fixture tensor the re-layout must touch.
func cohereRoutingRotaryRoster(cfg Config) []cohereRoutingRow {
	var rows []cohereRoutingRow
	for l := 0; l < cfg.NumLayers; l++ {
		rows = append(rows,
			cohereRoutingRow{layerName(l, "self_attn.q_proj.weight"), cfg.HiddenSize},
			cohereRoutingRow{layerName(l, "self_attn.k_proj.weight"), cfg.HiddenSize},
			cohereRoutingRow{layerName(l, "self_attn.q_norm.weight"), 1},
			cohereRoutingRow{layerName(l, "self_attn.k_norm.weight"), 1},
		)
	}
	return rows
}

// cohereRoutingSTTensors renders the fixture as an ordered safetensors tensor list in the
// checkpoint's own HF order.
func cohereRoutingSTTensors(t *testing.T, cfg Config, man map[string]tensorMeta, hf []byte) []stTensor {
	t.Helper()
	var out []stTensor
	for _, st := range cohereOracleTensors(cfg) {
		out = append(out, stTensor{st.name, st.shape, cohereRoutingValues(t, man, hf, st.name)})
	}
	return out
}

// assertCohereRotaryRelayout is the LAYOUT half of the witness: every rotary-axis tensor
// of the loaded model must be in half-split order. If it is still byte-identical to the
// checkpoint, the loader called newModel and skipped the re-layout entirely — which is
// exactly the regression this file exists to catch, so it is reported as such.
func assertCohereRotaryRelayout(t *testing.T, got *Model, cfg Config, man map[string]tensorMeta, hf []byte) {
	t.Helper()
	hd := cfg.HeadDim
	moved := 0
	for _, row := range cohereRoutingRotaryRoster(cfg) {
		if _, ok := man[row.name]; !ok {
			continue
		}
		src := cohereRoutingValues(t, man, hf, row.name)
		want := cohereRoutingHalfSplit(src, hd, row.rowElems)
		if !cohereRoutingSameF32(want, src) {
			moved++
		}
		if !got.has(row.name) {
			t.Fatalf("%s: loaded model has no %s", t.Name(), row.name)
		}
		have := got.tensor(row.name)
		if cohereRoutingSameF32(have, want) {
			continue
		}
		if cohereRoutingSameF32(have, src) {
			t.Fatalf("%s is byte-identical to the HuggingFace checkpoint: the loader did NOT apply the "+
				"Cohere rotary re-layout, so this entry point is calling newModel instead of "+
				"newHFCheckpointModel (cohere_rotary.go). Every position past 0 would rotate q/k with "+
				"the wrong component pairs.", row.name)
		}
		i, a, b := cohereRoutingFirstDiff(have, want)
		t.Fatalf("%s is in neither the half-split nor the checkpoint layout: first difference vs the "+
			"expected half-split order at element %d (have %v, want %v)", row.name, i, a, b)
	}
	if moved == 0 {
		t.Fatal("the re-layout is a no-op on every fixture tensor — this witness cannot distinguish " +
			"newHFCheckpointModel from newModel and is vacuous")
	}
}

// assertCohereRoutingLogits is the LOGITS half: the publicly loaded model must produce the
// same forward as one built straight through newHFCheckpointModel on the same bytes.
func assertCohereRoutingLogits(t *testing.T, got *Model, cfg Config, man map[string]tensorMeta, hf []byte) {
	t.Helper()
	wantMan := make(map[string]tensorMeta, len(man))
	for k, v := range man {
		wantMan[k] = v
	}
	want, err := newHFCheckpointModel(cfg, wantMan, append([]byte(nil), hf...))
	if err != nil {
		t.Fatalf("newHFCheckpointModel(reference): %v", err)
	}
	ids := cohereOracleIDs
	wantLogits := want.Forward(ids)
	haveLogits := got.Forward(ids)
	for pos := range ids {
		// The two models hold identical tensor VALUES and run identical production code,
		// so this should agree to the bit; the rotary divergence a reverted call site
		// reintroduces is ~1e-3 on this fixture, a thousand times the bound below.
		if d := cpuOracleMaxAbsDiff(haveLogits.Logits[pos], wantLogits.Logits[pos]); d > 1e-6 {
			t.Fatalf("logits at position %d differ from the newHFCheckpointModel reference by %.3e — "+
				"this entry point is not routed through the HF-checkpoint constructor", pos, d)
		}
	}
}

// cohereRoutingSameF32 reports exact equality; the re-layout only permutes rows, so there
// is no arithmetic and nothing to round.
func cohereRoutingSameF32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// cohereRoutingFirstDiff locates the first differing element so a failure is diagnosable.
func cohereRoutingFirstDiff(a, b []float32) (int, float32, float32) {
	for i := range a {
		if i >= len(b) {
			return i, a[i], 0
		}
		if a[i] != b[i] {
			return i, a[i], b[i]
		}
	}
	return len(a), 0, 0
}

// cohereRoutingWriteExportDir writes the fixture in the export_oracle.py layout that
// weights.go Load reads: config.json (through Config's own HF json tags, so decoding it
// re-runs the real deriveConfigAxes), manifest.json, weights.f32.
func cohereRoutingWriteExportDir(t *testing.T, cfg Config, man map[string]tensorMeta, hf []byte) string {
	t.Helper()
	dir := t.TempDir()
	cb, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal config: %v", err)
	}
	mb, err := json.Marshal(man)
	if err != nil {
		t.Fatalf("marshal manifest: %v", err)
	}
	for name, body := range map[string][]byte{
		"config.json":   cb,
		"manifest.json": mb,
		"weights.f32":   hf,
	} {
		if err := os.WriteFile(filepath.Join(dir, name), body, 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}
	return dir
}

// cohereRoutingWriteShardedDir writes the fixture as a two-shard HuggingFace snapshot with
// a model.safetensors.index.json weight_map, so loadSafetensorsDir takes its SHARDED branch
// rather than falling back to the single-file path.
func cohereRoutingWriteShardedDir(t *testing.T, tensors []stTensor) string {
	t.Helper()
	dir := t.TempDir()
	const shard1, shard2 = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
	cut := len(tensors) / 2
	if cut == 0 || cut == len(tensors) {
		t.Fatalf("fixture has %d tensors, too few to shard", len(tensors))
	}
	weightMap := map[string]string{}
	for _, part := range []struct {
		file string
		ts   []stTensor
	}{{shard1, tensors[:cut]}, {shard2, tensors[cut:]}} {
		writeSafetensorsShard(t, filepath.Join(dir, part.file), part.ts)
		for _, ts := range part.ts {
			weightMap[ts.name] = part.file
		}
	}
	ib, err := json.Marshal(map[string]any{
		"metadata":   map[string]any{"total_size": 0},
		"weight_map": weightMap,
	})
	if err != nil {
		t.Fatalf("marshal index: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), ib, 0o644); err != nil {
		t.Fatalf("write index: %v", err)
	}
	return dir
}

// TestCohereHFLoadersRouteThroughHFCheckpointModel pins all three HF-source loader call
// sites to newHFCheckpointModel by loading a Cohere fixture through the PUBLIC API. It
// fails the moment any of them is reverted to newModel — the case no other test in this
// package can see, because the oracle constructs its model directly.
func TestCohereHFLoadersRouteThroughHFCheckpointModel(t *testing.T) {
	cfg, man, hf := cohereRoutingFixture(t)
	if !cfg.usesCohereInterleavedRope() {
		t.Fatal("fixture config is not on the Cohere interleaved-rope lineage; the witness would be vacuous")
	}
	tensors := cohereRoutingSTTensors(t, cfg, man, hf)

	for _, tc := range []struct {
		name string
		load func(t *testing.T) *Model
	}{
		{
			// safetensors.go loadSafetensorsFile
			name: "LoadSafetensors",
			load: func(t *testing.T) *Model {
				path := filepath.Join(t.TempDir(), "model.safetensors")
				writeSafetensorsShard(t, path, tensors)
				m, err := LoadSafetensors(path, cfg)
				if err != nil {
					t.Fatalf("LoadSafetensors: %v", err)
				}
				return m
			},
		},
		{
			// safetensors.go loadSafetensorsDir, sharded branch
			name: "LoadSafetensorsDir",
			load: func(t *testing.T) *Model {
				m, err := LoadSafetensorsDir(cohereRoutingWriteShardedDir(t, tensors), cfg)
				if err != nil {
					t.Fatalf("LoadSafetensorsDir: %v", err)
				}
				return m
			},
		},
		{
			// weights.go Load
			name: "Load",
			load: func(t *testing.T) *Model {
				m, err := Load(cohereRoutingWriteExportDir(t, cfg, man, hf))
				if err != nil {
					t.Fatalf("Load: %v", err)
				}
				// Load decodes its own config, so guard the axes the forward reads: a
				// config-decode drift must report as itself, not as a rotary failure.
				if m.Cfg.HeadDim != cfg.HeadDim || m.Cfg.NumHeads != cfg.NumHeads ||
					m.Cfg.NumKVHeads != cfg.NumKVHeads || m.Cfg.QKNorm != cfg.QKNorm ||
					m.Cfg.QKNormPerHeadWeight != cfg.QKNormPerHeadWeight ||
					m.Cfg.LayerNorm != cfg.LayerNorm || m.Cfg.LogitScale != cfg.LogitScale ||
					m.Cfg.BlockTopology != cfg.BlockTopology {
					t.Fatalf("Load decoded a different config than the fixture: got %+v", m.Cfg)
				}
				return m
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := tc.load(t)
			assertCohereRotaryRelayout(t, m, cfg, man, hf)
			assertCohereRoutingLogits(t, m, cfg, man, hf)
		})
	}
}
