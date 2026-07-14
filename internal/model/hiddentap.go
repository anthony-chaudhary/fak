package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
)

// hiddenTap is the env-gated per-layer / per-op decode-time hidden-state dump that feeds the
// Qwen3.6-27B token-3 correctness-divergence probe
// (experiments/qwen36/token3-divergence-probe; design in
// experiments/qwen36/token3-drift-investigation-2026-06-28.md §3b/§5 step 3). It turns the
// abstract "fak and llama.cpp agree for two tokens then diverge on the third" into capturable
// evidence: when armed for the single decode forward at absolute position `pos`, it writes the
// residual-stream hidden after EVERY decoder layer (`layer_NN.f32`) a sibling `<dir>.logits.tsv` containing the top-10 token ids/logits at that position, plus, inside each
// linear_attention (Gated-DeltaNet) layer, the named GDN per-op intermediates
// (`layer_NN_op_<name>.f32`), and a `meta.json` describing the dump — exactly the on-disk format
// the probe's comparator consumes. The Mac/27B + llama.cpp side of the capture is the gated
// follow-on; this is the host-independent producer of fak's half.
//
// Cost model: a *hiddenTap is nil in every normal session, so the only hot-path cost is one
// pointer load + nil check per layer per token (blockStep.activeTap). It is constructed only when
// FAK_HIDDEN_TAP=<dir> is set (or a test sets Session.tap directly).
//
// Concurrency: the tap is intended for a single-stream (batch=1) diagnostic capture — the parity
// and perf captures it feeds are all single-stream. The transient "which tap is armed for this
// forward" lives on the Session (s.tapActive), not on the shared tap, so it is per-session; but
// two concurrent sessions sharing the SAME env tap dir would write the same filenames. Run one
// model at a time.
type hiddenTap struct {
	dir        string       // output directory (created on first write)
	logitsPath string       // sibling top-logit TSV for the selected position
	pos        int          // first absolute position whose forward to dump
	positions  map[int]bool // optional absolute-position set; nil preserves the single-pos contract
	ops        bool         // also dump the per-op GDN intermediates inside linear_attention layers
	promptIDs  []int        // optional provenance carried into meta.json

	mu       sync.Mutex
	dirReady bool
	err      error // first write error, surfaced once to stderr; the forward never fails on it
}

// hiddenTapFromEnv builds a hiddenTap from FAK_HIDDEN_TAP (output dir), FAK_HIDDEN_TAP_POS (the
// absolute position whose decode forward to dump — set it to the step that PREDICTS the divergent
// token; default 0), and FAK_HIDDEN_TAP_OPS (set to "0" to suppress the finer GDN per-op taps;
// default on). It returns nil when FAK_HIDDEN_TAP is unset — the universal case.
func hiddenTapFromEnv() *hiddenTap {
	dir := os.Getenv("FAK_HIDDEN_TAP")
	if dir == "" {
		return nil
	}
	pos := 0
	positions := map[int]bool{}
	if v := os.Getenv("FAK_HIDDEN_TAP_POS"); v != "" {
		for _, raw := range strings.Split(v, ",") {
			if n, err := strconv.Atoi(strings.TrimSpace(raw)); err == nil {
				positions[n] = true
				if len(positions) == 1 {
					pos = n
				}
			}
		}
	}
	if len(positions) < 2 {
		positions = nil
	}
	ops := os.Getenv("FAK_HIDDEN_TAP_OPS") != "0"
	return &hiddenTap{dir: dir, logitsPath: dir + ".logits.tsv", pos: pos, positions: positions, ops: ops}
}

var (
	envHiddenTapOnce sync.Once
	envHiddenTap     *hiddenTap
)

// envTap returns the process-wide env-configured tap (FAK_HIDDEN_TAP), resolved exactly once so
// the env read never lands on the hot path more than once. It returns nil when the env is unset.
func envTap() *hiddenTap {
	envHiddenTapOnce.Do(func() { envHiddenTap = hiddenTapFromEnv() })
	return envHiddenTap
}

// activeTap resolves the tap for this session: an explicit Session.tap override (tests) wins,
// otherwise the env-configured global. Nil means "no tap" — the universal case.
func (s *Session) activeTap() *hiddenTap {
	if s.tap != nil {
		return s.tap
	}
	return envTap()
}

func (t *hiddenTap) wants(pos int) bool {
	if t == nil {
		return false
	}
	if t.positions != nil {
		return t.positions[pos]
	}
	return pos == t.pos
}

// layerKindLabel names layer l for the probe's per-layer metadata. The qwen35 hybrid
// distinguishes its Gated-DeltaNet linear_attention layers from its periodic full_attention
// layers; any other arch is just "attention".
func layerKindLabel(cfg Config, l int) string {
	if cfg.isLinearAttnLayer(l) {
		return "linear_attention"
	}
	if cfg.IsQwen35Hybrid() {
		return "full_attention"
	}
	return "attention"
}

func (t *hiddenTap) ensureDir() {
	if t.dirReady {
		return
	}
	if err := os.MkdirAll(t.dir, 0o755); err != nil {
		t.reportErr(err)
	}
	t.dirReady = true
}

// reportErr records the first write error and prints it once; a tap failure must never abort the
// forward (the tap is a diagnostic, not a correctness gate).
func (t *hiddenTap) reportErr(err error) {
	if t.err == nil {
		t.err = err
		fmt.Fprintf(os.Stderr, "FAK_HIDDEN_TAP: %v\n", err)
	}
}

// writeF32 serializes v as little-endian float32 to <dir>/name. The bytes are fully written before
// it returns, so callers may pass a slice that is mutated immediately afterward (no copy needed).
func (t *hiddenTap) writeF32(name string, v []float32) {
	t.ensureDir()
	b := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(f))
	}
	if err := os.WriteFile(filepath.Join(t.dir, name), b, 0o644); err != nil {
		t.reportErr(err)
	}
}

// dumpLayer writes the residual-stream hidden after decoder layer l. kind is unused on disk (the
// kinds live in meta.json) but kept in the signature so the call site reads self-documenting.
func (t *hiddenTap) dumpLayer(l int, kind string, x []float32) {
	t.writeF32(fmt.Sprintf("layer_%02d.f32", l), x)
}

// dumpLogits appends a compact top-logit fingerprint for the selected absolute position.
// A sibling TSV preserves the binary hidden-vector contract while giving #4273 a direct
// token-id/logit comparison seam against llama.cpp or Hugging Face.
func (t *hiddenTap) dumpLogits(pos int, logits []float32) {
	if t == nil || t.logitsPath == "" || len(logits) == 0 {
		return
	}
	type ranked struct {
		id    int
		logit float32
	}
	const topN = 10
	top := make([]ranked, 0, topN)
	for id, value := range logits {
		at := len(top)
		for i := range top {
			if value > top[i].logit || value == top[i].logit && id < top[i].id {
				at = i
				break
			}
		}
		if at >= topN {
			continue
		}
		top = append(top, ranked{})
		copy(top[at+1:], top[at:])
		top[at] = ranked{id: id, logit: value}
		if len(top) > topN {
			top = top[:topN]
		}
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	f, err := os.OpenFile(t.logitsPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		t.reportErr(err)
		return
	}
	defer f.Close()
	for rank, item := range top {
		if _, err := fmt.Fprintf(f, "%d\t%d\t%d\t%.9g\n", pos, rank+1, item.id, item.logit); err != nil {
			t.reportErr(err)
			return
		}
	}
}

// dumpOp writes one GDN per-op intermediate inside layer l. op is one of the canonical names the
// probe localizes over: convOut, qk_norm, recurrent, gated_norm, out.
func (t *hiddenTap) dumpOp(l int, op string, v []float32) {
	t.writeF32(fmt.Sprintf("layer_%02d_op_%s.f32", l, op), v)
}

type hiddenTapLayerMeta struct {
	Index int    `json:"index"`
	Kind  string `json:"kind"`
}

type hiddenTapMeta struct {
	Hidden     int                  `json:"hidden"`
	DecodeStep int                  `json:"decode_step"`
	PromptIDs  []int                `json:"prompt_ids,omitempty"`
	Layers     []hiddenTapLayerMeta `json:"layers"`
}

// writeMeta writes meta.json describing the dump: the hidden width, the absolute position dumped,
// optional prompt-id provenance, and the [{index,kind}] layer list (computed from cfg, so it is
// correct regardless of which layers were reached). The schema matches the probe's `meta` reader.
func (t *hiddenTap) writeMeta(cfg Config, hidden, step int) {
	layers := make([]hiddenTapLayerMeta, 0, cfg.NumLayers)
	for l := 0; l < cfg.NumLayers; l++ {
		layers = append(layers, hiddenTapLayerMeta{Index: l, Kind: layerKindLabel(cfg, l)})
	}
	m := hiddenTapMeta{Hidden: hidden, DecodeStep: step, PromptIDs: t.promptIDs, Layers: layers}
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		t.reportErr(err)
		return
	}
	t.ensureDir()
	if err := os.WriteFile(filepath.Join(t.dir, "meta.json"), b, 0o644); err != nil {
		t.reportErr(err)
	}
}
