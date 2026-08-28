package model

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// Ornith 1.0 (DeepReinforce) is a Qwen3.5 architecture (#1026): the published config
// declares family `qwen3_5` / `qwen3_5_moe`, so fak serves it through the existing
// qwen35.go hybrid path plus the generic MoE FFN — no new arch port. This is the
// per-size HF-oracle parity SCAFFOLD for that claim (#1031): a runnable witness that
// the pure-Go Ornith forward reproduces HF transformers to f32 tolerance, AND that the
// loader derived the three documented Ornith specifics:
//
//   - M-RoPE collapses to text-only — Ornith inherits Qwen's multimodal rope_dim_per_layer
//     but, with the vision tower dropped, rotary must reduce to the plain text rotary
//     denominator (no mrope section interleave on the text path).
//   - attn_output_gate is applied on the full-attention layers (the sigmoid readout gate
//     that distinguishes Qwen3.5 full attention from a plain GQA layer).
//   - vision/MTP heads are skipped at load (model.visual.* and mtp.*), exactly like the
//     hybrid qwen35 materialization already does.
//
// HONEST SCAFFOLD: the real Ornith oracle fixture (HF reference vectors + weights) is NOT
// present on a CI box, so this test SKIPS with a clear regeneration hint when the fixture
// directory (or the FAK_ORNITH_ORACLE_DIR override) is absent — it never falsely passes.
// It turns green only against a real fixture. To regenerate, point the exporter at a tiny
// qwen3_5 / qwen3_5_moe Ornith checkpoint the way ornithOracleExportHint describes.

// ornithOracleDirEnv overrides the default in-tree fixture dir with an out-of-tree path
// (e.g. a fixture exported on an artifact node and mounted elsewhere on a verify box).
const ornithOracleDirEnv = "FAK_ORNITH_ORACLE_DIR"

// ornithOracleDir is the gitignored default fixture export location, sibling of the other
// .cache/oracle-* fixtures resolveOracleDir already understands.
const ornithOracleDir = ".cache/oracle-ornith"

const ornithOracleModel = ".cache/ornith-tiny"
const ornithOraclePromptIDsJSON = `[[785,6722,315,9621,374],[16,11,220,17,11,220,18,11,220,19,11],[750,912,2877,11,293,982,262,470]]`

// ornithOracleExportHint builds the dense Ornith-9B text surrogate with the exact
// M-RoPE/partial-RoPE/gated-attention axes, then exports HF reference traces through
// the shared qwen35 exporter. The 35B MoE builder and both committed parity fixtures
// remain acceptance work for #1031; this command repairs only the dense regeneration spine.
const ornithOracleExportHint = "from fak/: python3 internal/model/make_qwen35_tiny.py " + ornithOracleModel +
	" --ornith && python3 internal/model/export_oracle.py --online --model " + ornithOracleModel +
	" --out internal/model/" + ornithOracleDir + " --prompt-ids-json '" + ornithOraclePromptIDsJSON + "'"

func TestOrnithOracleExportHintUsesRealBuilderContract(t *testing.T) {
	if strings.Contains(ornithOracleExportHint, "python ") ||
		!strings.Contains(ornithOracleExportHint, "python3 internal/model/make_qwen35_tiny.py") ||
		!strings.Contains(ornithOracleExportHint, " --ornith") {
		t.Fatalf("Ornith regeneration hint does not use the python3 --ornith builder contract: %s", ornithOracleExportHint)
	}
	builder, err := os.ReadFile("make_qwen35_tiny.py")
	if err != nil {
		t.Fatal(err)
	}
	for _, option := range []string{`add_argument("--ornith"`, `add_argument("--describe"`} {
		if !strings.Contains(string(builder), option) {
			t.Fatalf("Ornith regeneration hint names an unavailable builder option %q", option)
		}
	}
	if !strings.Contains(string(builder), `Qwen3_5TextConfig(**spec["config_kwargs"])`) {
		t.Fatal("Ornith Qwen3_5TextConfig construction is not fed by the described shared spec")
	}
}

func TestOrnithOracleBuilderContract(t *testing.T) {
	python := "python3"
	if runtime.GOOS == "windows" {
		python = "python"
	}
	out, err := exec.Command(python, "-S", "make_qwen35_tiny.py", "--ornith", "--describe").CombinedOutput()
	if err != nil {
		t.Fatalf("describe Ornith builder contract: %v\n%s", err, out)
	}
	var got struct {
		Schema               string `json:"schema"`
		Scope                string `json:"scope"`
		PublishedWrapperType string `json:"published_wrapper_model_type"`
		BuilderConfigType    string `json:"builder_config_model_type"`
		TextOnly             bool   `json:"text_only"`
		RotaryDim            int    `json:"rotary_dim"`
		ConfigKwargs         struct {
			HeadDim             int  `json:"head_dim"`
			AttentionOutputGate bool `json:"attn_output_gate"`
			RopeParameters      struct {
				MROPEInterleaved    bool    `json:"mrope_interleaved"`
				MROPESection        []int   `json:"mrope_section"`
				PartialRotaryFactor float64 `json:"partial_rotary_factor"`
			} `json:"rope_parameters"`
		} `json:"config_kwargs"`
	}
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("decode Ornith builder contract: %v\n%s", err, out)
	}
	if got.Schema != "fak.model.ornith-tiny-builder/1" ||
		got.Scope != "dense-9b-text-oracle-builder-not-parity-evidence" ||
		got.PublishedWrapperType != "qwen3_5" || got.BuilderConfigType != "qwen3_5_text" ||
		!got.TextOnly || got.ConfigKwargs.HeadDim != 256 ||
		got.ConfigKwargs.RopeParameters.PartialRotaryFactor != 0.25 || got.RotaryDim != 64 ||
		!got.ConfigKwargs.RopeParameters.MROPEInterleaved || !got.ConfigKwargs.AttentionOutputGate {
		t.Fatalf("Ornith builder contract drifted: %+v", got)
	}
	wantSection := []int{11, 11, 10}
	section := got.ConfigKwargs.RopeParameters.MROPESection
	if len(section) != len(wantSection) {
		t.Fatalf("mrope_section = %v, want %v", section, wantSection)
	}
	for i := range wantSection {
		if section[i] != wantSection[i] {
			t.Fatalf("mrope_section = %v, want %v", section, wantSection)
		}
	}
}

// ornithFixtureDir resolves the fixture directory: the FAK_ORNITH_ORACLE_DIR override wins
// when set, otherwise the in-tree default. The actual present/absent decision (and the
// clean t.Skip when no weights are exported) is left to loadFixtureDir/resolveOracleDir so
// this test shares the exact skip-guard idiom of every other oracle-gated test here.
func ornithFixtureDir() string {
	return ornithFixtureDirFrom(os.Getenv(ornithOracleDirEnv), ornithOracleDir)
}

func ornithFixtureDirFrom(override, defaultDir string) string {
	if override != "" {
		return override
	}
	return defaultDir
}

func TestOrnithFixtureDirResolution(t *testing.T) {
	t.Run("absent override uses missing default", func(t *testing.T) {
		defaultDir := filepath.Join(t.TempDir(), "missing")
		got := ornithFixtureDirFrom("", defaultDir)
		if got != defaultDir {
			t.Fatalf("ornithFixtureDirFrom() = %q, want default %q", got, defaultDir)
		}
		if resolved, ok := resolveOracleDir(got); ok {
			t.Fatalf("resolveOracleDir(%q) = %q, true; want absent fixture", got, resolved)
		}
	})

	t.Run("override wins", func(t *testing.T) {
		override := filepath.Join(t.TempDir(), "override")
		t.Setenv(ornithOracleDirEnv, override)
		if got := ornithFixtureDir(); got != override {
			t.Fatalf("ornithFixtureDir() = %q, want override %q", got, override)
		}
	})

	t.Run("present default fixture resolves", func(t *testing.T) {
		defaultDir := t.TempDir()
		if err := os.WriteFile(filepath.Join(defaultDir, "weights.f32"), nil, 0o600); err != nil {
			t.Fatalf("write fixture marker: %v", err)
		}
		got := ornithFixtureDirFrom("", defaultDir)
		resolved, ok := resolveOracleDir(got)
		if !ok {
			t.Fatalf("resolveOracleDir(%q) did not find present fixture", got)
		}
		if resolved != defaultDir {
			t.Fatalf("resolveOracleDir(%q) = %q, want %q", got, resolved, defaultDir)
		}
	})
}

// TestOptionalOrnithOracleForwardMatchesHF is the per-size Ornith HF-oracle parity witness.
// It is OPTIONAL by construction: loadFixtureDir t.Skips cleanly when the gitignored fixture
// is absent (and under -short), so a CI box with no weights stays green instead of falsely
// passing. With a real fixture present it asserts the Ornith specifics, then per-layer
// hidden-state cosine >= 0.9999 and argmax parity at every position via the shared oracle
// asserter.
func TestOptionalOrnithOracleForwardMatchesHF(t *testing.T) {
	dir := ornithFixtureDir()
	m, doc := loadFixtureDir(t, dir, true) // missingIsSkip=true: no fixture -> clean t.Skip, never a pass
	cfg := m.Cfg

	// Ornith is a qwen3_5 / qwen3_5_moe arch (#1026); the hybrid linear-attention axis must
	// have loaded, or the fixture is not the model this test witnesses.
	if !cfg.IsQwen35Hybrid() {
		t.Fatalf("%s did not load as a qwen3_5 Ornith hybrid (layer_types=%v); regenerate: %s",
			dir, cfg.LayerTypes, ornithOracleExportHint)
	}

	// --- Ornith specific #1: attn_output_gate applied on full-attention layers ----------
	// The sigmoid readout gate is architectural for Qwen3.5 full attention; a missing knob
	// would silently change the math and make the parity check below meaningless.
	if !cfg.AttnOutputGate {
		t.Fatalf("%s attn_output_gate not derived (full-attention readout gate); regenerate: %s",
			dir, ornithOracleExportHint)
	}
	// There must be at least one full-attention layer for that gate to apply to — otherwise
	// the assertion is vacuous (an all-linear stack would never exercise the gated path).
	nFull, nLinear := 0, 0
	for l := 0; l < cfg.NumLayers; l++ {
		if cfg.isLinearAttnLayer(l) {
			nLinear++
		} else {
			nFull++
		}
	}
	if nFull == 0 || nLinear == 0 {
		t.Fatalf("%s layer mix is vacuous for the Ornith witness: %d full / %d linear of %d (need both)",
			dir, nFull, nLinear, cfg.NumLayers)
	}

	// --- Ornith specific #2: M-RoPE collapses to text-only ------------------------------
	// Ornith inherits Qwen's multimodal rope_dim_per_layer, but with the vision tower
	// dropped the rotary must reduce to the plain text rotary. When the loader carries a
	// per-layer rope-dim section vector it must be uniform across the decoder (no
	// position/section interleave on the text path); a non-uniform vector would mean the
	// mrope section split survived into the text forward, which is the bug this row guards.
	if rd := cfg.RopeDimPerLayer; len(rd) > 0 {
		for l := 1; l < len(rd); l++ {
			if rd[l] != rd[0] {
				t.Fatalf("%s M-RoPE did not collapse to text-only: rope_dim_per_layer non-uniform %v "+
					"(mrope section interleave leaked into the text path)", dir, rd)
			}
		}
	}

	// --- Ornith specific #3: vision tower + MTP head skipped at load --------------------
	// Ornith ships a multimodal vision encoder (model.visual.*) and a multi-token-prediction
	// head (mtp.*) the text forward never reads; the qwen35 load path drops them. Assert the
	// family gate is on AND no such tensor survived into the loaded manifest.
	if !cfg.dropsMtpAndVisualAtLoad() {
		t.Fatalf("%s dropsMtpAndVisualAtLoad=false; Ornith vision/MTP skip not gated", dir)
	}
	for name := range m.manifest {
		if isQwen35DroppedTensor(name) {
			t.Fatalf("%s vision/MTP tensor %q survived into the loaded manifest (should be skipped)", dir, name)
		}
	}

	// --- parity witness -----------------------------------------------------------------
	// Real fixtures only: per-layer hidden-state cosine >= 0.9999 and argmax parity at every
	// position, plus the cached-prefill cross-check, via the shared oracle asserter.
	resolved, _ := resolveOracleDir(dir)
	assertForwardMatchesHFOracle(t, resolved, m, doc)
}

// isQwen35DroppedTensor reports whether a tensor name belongs to the vision tower or the
// MTP head that the Ornith/qwen35 load path drops. Kept local to this scaffold so the
// assertion reads the same family-agnostic prefixes the loader's skip uses.
func isQwen35DroppedTensor(name string) bool {
	return strings.HasPrefix(name, "model.visual.") || strings.HasPrefix(name, "mtp.")
}
