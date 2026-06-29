// Command q8bench is an INDEPENDENT verifier for the int8-quantized in-kernel forward path
// (internal/model: Model.Quantize + Session.Quant). It drives ONLY the package's PUBLIC API,
// so it is decoupled from the quant kernel internals (pure-Go, AVX2, or AVX-512 — whatever
// the package dispatches) and survives kernel refactors. Its job is the fleet's distrust
// discipline applied to the speedup claim: confirm, with a witness that did not author the
// kernels, that quantized fak (1) stays argmax-EXACT vs the HF-authored oracle and (2) beats
// the same-rung HF int8 peer at decode — the "beat HF on the int8 rung" goal.
//
// It does, on the same box:
//  1. CORRECTNESS gate — teacher-force the quant KV-session over each oracle prompt and
//     require per-position argmax == the oracle's argmax_per_pos (the SAME bar the f32
//     oracle test enforces, now at int8). Reports greedy agreement + last-pos logit max|Δ|.
//  2. SPEED — decode ms/tok for f32 vs int8, measured INTERLEAVED per rep (so both sample
//     the same time-varying load on this shared box) with a MIN headline (least-contended
//     estimate); prefill P=16/64/256 per path. Same LCG id protocol as cmd/modelbench.
//  3. VERDICT — reads experiments/model-baseline/hf.json (f32) and hf-int8.json (dynamic
//     int8) and prints whether quant-fak's decode beats each. The int8 row is the goal.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"time"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

// lcgIDs — bit-for-bit the recurrence in cmd/modelbench/main.go and bench_hf.py.
func lcgIDs(n, vocab int) []int {
	ids := make([]int, n)
	state := uint64(2463534242)
	for i := 0; i < n; i++ {
		state = (state*1103515245 + 12345) & 0x7fffffff
		ids[i] = int(state % uint64(vocab))
	}
	return ids
}

func medianMS(ds []time.Duration) float64 {
	cp := append([]time.Duration(nil), ds...)
	sort.Slice(cp, func(i, j int) bool { return cp[i] < cp[j] })
	return float64(cp[len(cp)/2].Nanoseconds()) / 1e6
}

func minMS(ds []time.Duration) float64 {
	mn := ds[0]
	for _, d := range ds {
		if d < mn {
			mn = d
		}
	}
	return float64(mn.Nanoseconds()) / 1e6
}

func argmax(v []float32) int {
	bi, bv := 0, v[0]
	for i, x := range v {
		if x > bv {
			bv, bi = x, i
		}
	}
	return bi
}

type prefillResult struct {
	Tokens    int     `json:"tokens"`
	Reps      int     `json:"reps"`
	MedianMS  float64 `json:"median_ms"`
	TokPerSec float64 `json:"tok_per_sec"`
}

type decodeResult struct {
	PromptTokens  int     `json:"prompt_tokens"`
	DecodeSteps   int     `json:"decode_steps"`
	Reps          int     `json:"reps"`
	PerTokenMedMS float64 `json:"per_token_median_ms"` // carries the MIN (contention-robust headline)
	TokPerSec     float64 `json:"tok_per_sec"`
}

type oraclePrompt struct {
	Index     int   `json:"index"`
	Ids       []int `json:"ids"`
	ArgmaxPos []int `json:"argmax_per_pos"`
	GreedyIds []int `json:"greedy_ids"`
}
type oracleDoc struct {
	Prompts []oraclePrompt `json:"prompts"`
}

type promptCheck struct {
	Index          int     `json:"index"`
	Positions      int     `json:"positions"`
	ArgmaxMatches  int     `json:"argmax_matches"`
	GreedyAgree    int     `json:"greedy_agreement_len"`
	GreedyTotal    int     `json:"greedy_total"`
	LastMaxAbsDiff float64 `json:"last_pos_logit_max_abs_diff_vs_oracle"`
	LastArgmaxOK   bool    `json:"last_argmax_matches_oracle"`
}

func readOracleLastLogits(dir string, idx, seq, vocab int) []float32 {
	b, err := os.ReadFile(filepath.Join(dir, "oracle", fmt.Sprintf("%d.logits.f32", idx)))
	if err != nil {
		return nil
	}
	all := unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
	if len(all) < seq*vocab {
		return nil
	}
	return all[(seq-1)*vocab : seq*vocab]
}

// checkPromptQuant teacher-forces the int8 KV-session over an oracle prompt and reports
// per-position argmax agreement, greedy agreement, and last-pos fidelity vs the HF oracle.
func checkPromptQuant(m *model.Model, dir string, p oraclePrompt, vocab int) promptCheck {
	pc := promptCheck{Index: p.Index, Positions: len(p.Ids), GreedyTotal: len(p.GreedyIds)}
	s := m.NewSession()
	s.Quant = true
	logits := s.Prefill(p.Ids[:1]) // logits at position 0
	var lastLogits []float32
	check := func(pos int, lg []float32) {
		if pos < len(p.ArgmaxPos) && argmax(lg) == p.ArgmaxPos[pos] {
			pc.ArgmaxMatches++
		}
		lastLogits = lg
	}
	check(0, logits)
	for i := 1; i < len(p.Ids); i++ {
		logits = s.Step(p.Ids[i]) // logits at position i
		check(i, logits)
	}
	if ref := readOracleLastLogits(dir, p.Index, len(p.Ids), vocab); ref != nil && lastLogits != nil {
		mx := 0.0
		for i := range ref {
			d := math.Abs(float64(lastLogits[i] - ref[i]))
			if d > mx {
				mx = d
			}
		}
		pc.LastMaxAbsDiff = mx
		pc.LastArgmaxOK = argmax(lastLogits) == argmax(ref)
	}
	g := m.NewSession()
	g.Quant = true
	gl := g.Prefill(p.Ids)
	for k := 0; k < len(p.GreedyIds); k++ {
		nid := argmax(gl)
		if nid != p.GreedyIds[k] {
			break
		}
		pc.GreedyAgree++
		if m.Cfg.IsEOS(nid) {
			break
		}
		gl = g.Step(nid)
	}
	return pc
}

// interleavedDecode times f32 and int8 decode INTERLEAVED per rep so both sample the same
// machine load (this box runs other fleet sessions concurrently). Returns [min, median]
// ms/tok per path; MIN is the least-contended estimate.
func interleavedDecode(m *model.Model, promptLen, steps, reps, vocab int) (f32, q8 [2]float64) {
	prompt := lcgIDs(promptLen, vocab)
	f32d := make([]time.Duration, 0, reps)
	q8d := make([]time.Duration, 0, reps)
	timeOne := func(quant bool, r int) time.Duration {
		s := m.NewSession()
		s.Quant = quant
		s.Prefill(prompt)
		id := int(uint64(r*131+7) % uint64(vocab))
		t := time.Now()
		for i := 0; i < steps; i++ {
			_ = s.Step(id)
			id = (id*48271 + 1) % vocab
		}
		return time.Since(t) / time.Duration(steps)
	}
	for r := 0; r < reps; r++ {
		f32d = append(f32d, timeOne(false, r))
		q8d = append(q8d, timeOne(true, r))
	}
	return [2]float64{minMS(f32d), medianMS(f32d)}, [2]float64{minMS(q8d), medianMS(q8d)}
}

func benchPrefill(m *model.Model, quant bool, P, reps, vocab int) float64 {
	ids := lcgIDs(P, vocab)
	ds := make([]time.Duration, reps)
	for r := 0; r < reps; r++ {
		s := m.NewSession()
		s.Quant = quant
		t := time.Now()
		s.Prefill(ids)
		ds[r] = time.Since(t)
	}
	return medianMS(ds)
}

// hfBestDecode reads an HF bench JSON and returns the best (smallest) decode ms/tok across
// its configs, or 0 if unreadable.
func hfBestDecode(path string) float64 {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	var doc struct {
		Configs []struct {
			Decode struct {
				PerTokenMedianMS float64 `json:"per_token_median_ms"`
			} `json:"decode"`
		} `json:"configs"`
	}
	if json.Unmarshal(b, &doc) != nil {
		return 0
	}
	best := 0.0
	for _, c := range doc.Configs {
		if d := c.Decode.PerTokenMedianMS; d > 0 && (best == 0 || d < best) {
			best = d
		}
	}
	return best
}

func main() {
	dir := flag.String("dir", "internal/model/.cache/smollm2-135m", "model export dir")
	out := flag.String("out", "", "write JSON result here (default stdout)")
	prefillReps := flag.Int("prefill-reps", 4, "reps per prefill size (median)")
	decodeReps := flag.Int("decode-reps", 25, "interleaved decode reps (min headline)")
	decodeSteps := flag.Int("decode-steps", 32, "tokens to decode")
	decodePrompt := flag.Int("decode-prompt", 16, "prompt length before decode")
	expDir := flag.String("exp", "experiments/model-baseline", "dir holding hf.json / hf-int8.json for the verdict")
	flag.Parse()
	// Expand a leading ~ in path flags (Go/PowerShell don't), so ~/... opens as intended.
	*dir = pathutil.ExpandTilde(*dir)
	prefillSizes := []int{16, 64, 256}

	t0 := time.Now()
	m, err := model.Load(*dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	loadMS := float64(time.Since(t0).Nanoseconds()) / 1e6
	vocab := m.Cfg.VocabSize

	tq := time.Now()
	m.Quantize()
	quantizeMS := float64(time.Since(tq).Nanoseconds()) / 1e6

	// warm both paths (page in weights + the int8 store; JIT allocs) — not timed.
	{
		s := m.NewSession()
		s.Prefill(lcgIDs(8, vocab))
		s.Step(s.Cache.Len() % vocab)
		q := m.NewSession()
		q.Quant = true
		q.Prefill(lcgIDs(8, vocab))
		q.Step(q.Cache.Len() % vocab)
	}

	// ---- correctness gate (deterministic, contention-independent) --------
	var doc oracleDoc
	b, rerr := os.ReadFile(filepath.Join(*dir, "oracle.json"))
	correctnessOK := true
	var checks []promptCheck
	totalPos, totalMatch := 0, 0
	if rerr == nil {
		_ = json.Unmarshal(b, &doc)
		for _, p := range doc.Prompts {
			pc := checkPromptQuant(m, *dir, p, vocab)
			checks = append(checks, pc)
			totalPos += pc.Positions
			totalMatch += pc.ArgmaxMatches
			if pc.ArgmaxMatches != pc.Positions {
				correctnessOK = false
			}
			fmt.Fprintf(os.Stderr, "[int8 correctness] prompt %d: argmax %d/%d  greedy %d/%d  last|Δ|=%.4f argmaxOK=%v\n",
				pc.Index, pc.ArgmaxMatches, pc.Positions, pc.GreedyAgree, pc.GreedyTotal, pc.LastMaxAbsDiff, pc.LastArgmaxOK)
		}
		fmt.Fprintf(os.Stderr, "[int8 correctness] TOTAL argmax %d/%d -> %s\n", totalMatch, totalPos,
			map[bool]string{true: "ARGMAX-EXACT vs HF oracle", false: "ARGMAX DRIFT"}[correctnessOK])
	} else {
		fmt.Fprintln(os.Stderr, "[int8 correctness] no oracle.json; skipping correctness gate")
	}

	// ---- speed --------------------------------------------------------------
	f32D, q8D := interleavedDecode(m, *decodePrompt, *decodeSteps, *decodeReps, vocab)
	mkDec := func(msMin float64) decodeResult {
		return decodeResult{PromptTokens: *decodePrompt, DecodeSteps: *decodeSteps, Reps: *decodeReps,
			PerTokenMedMS: msMin, TokPerSec: 1.0 / (msMin / 1e3)}
	}
	f32Dec, q8Dec := mkDec(f32D[0]), mkDec(q8D[0])
	fmt.Fprintf(os.Stderr, "[decode min ms/tok] f32=%.2f  int8=%.2f   (median: %.2f / %.2f)\n", f32D[0], q8D[0], f32D[1], q8D[1])
	prefAll := func(quant bool, tag string) []prefillResult {
		var prefs []prefillResult
		for _, p := range prefillSizes {
			ms := benchPrefill(m, quant, p, *prefillReps, vocab)
			prefs = append(prefs, prefillResult{Tokens: p, Reps: *prefillReps, MedianMS: ms, TokPerSec: float64(p) / (ms / 1e3)})
			fmt.Fprintf(os.Stderr, "[fak %s] prefill P=%d: %.1f ms (%.1f tok/s)\n", tag, p, ms, float64(p)/(ms/1e3))
		}
		return prefs
	}
	f32Pre := prefAll(false, "f32")
	q8Pre := prefAll(true, "int8")

	// ---- verdict vs the HF peers -------------------------------------------
	hfF32 := hfBestDecode(filepath.Join(*expDir, "hf.json"))
	hfInt8 := hfBestDecode(filepath.Join(*expDir, "hf-int8.json"))
	beatsHFInt8 := hfInt8 > 0 && q8Dec.PerTokenMedMS < hfInt8
	beatsHFf32 := hfF32 > 0 && q8Dec.PerTokenMedMS < hfF32

	report := map[string]any{
		"app_version": appversion.Current(),
		"engine":      "fak-in-kernel int8 (Model.Quantize + Session.Quant; pure-Go/AVX dispatch)",
		"model":       "SmolLM2-135M (int8)",
		"load_ms":     loadMS,
		"quantize_ms": quantizeMS,
		"decode":      q8Dec, // headline = int8 (folds into compare.py beside fak-par.json)
		"prefill":     q8Pre,
		"f32_control": map[string]any{"decode": f32Dec, "prefill": f32Pre},
		"decode_interleaved_min_median_ms": map[string]any{
			"note": "interleaved per-rep on a shared box; [min, median]; min = least-contended.",
			"f32":  f32D,
			"int8": q8D,
		},
		"decode_speedup_min_f32_over_int8": f32Dec.PerTokenMedMS / q8Dec.PerTokenMedMS,
		"correctness": map[string]any{
			"gate_argmax_exact_vs_hf_oracle": correctnessOK,
			"total_positions":                totalPos,
			"total_argmax_matches":           totalMatch,
			"per_prompt":                     checks,
		},
		"verdict": map[string]any{
			"hf_f32_best_decode_ms":  hfF32,
			"hf_int8_best_decode_ms": hfInt8,
			"fak_int8_decode_ms_min": q8Dec.PerTokenMedMS,
			"beats_hf_int8":          beatsHFInt8,
			"beats_hf_f32":           beatsHFf32,
		},
	}
	jb, _ := benchcli.MarshalReport(report)
	if *out != "" {
		if err := os.WriteFile(*out, jb, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write:", err)
			os.Exit(1)
		}
		fmt.Fprintln(os.Stderr, "wrote", *out)
	} else {
		fmt.Println(string(jb))
	}
	fmt.Fprintf(os.Stderr, "\nVERDICT  fak int8 decode(min)=%.2f ms/tok | HF int8=%.2f (beat=%v) | HF f32=%.2f (beat=%v) | argmax-exact=%v\n",
		q8Dec.PerTokenMedMS, hfInt8, beatsHFInt8, hfF32, beatsHFf32, correctnessOK)
	if !correctnessOK {
		os.Exit(2) // argmax drift in the shipped int8 path is a hard failure
	}
}
