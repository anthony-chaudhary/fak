// Command fakchat runs an end-to-end chat completion with fak's OWN in-kernel
// engine — no llama-server, no external proxy. It wires the modular pieces:
//
//	internal/tokenizer (BPE text<->ids)  ->  internal/model (Prefill/Step logits)
//	  ->  greedy/temperature sampling  ->  internal/tokenizer (detok)  ->  stream
//
// On Apple Silicon it runs the documented hybrid split: PREFILL on the Metal GPU
// (-metal, requires `-tags fakmetal`) and DECODE on the CPU Q8 (NEON) path, since
// prefill is compute-bound and decode is bandwidth-bound (see QWEN36-PARITY-RESULTS.md).
//
// Example:
//
//	go build -tags fakmetal -o fakchat ./cmd/fakchat
//	./fakchat -hf ~/.cache/fak-models/qwen2.5-1.5b-instruct \
//	          -tok ~/.cache/fak-models/tokenizers/qwen2.5 -metal \
//	          -p "Explain unified memory in one sentence."
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/demoui"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func main() {
	hf := flag.String("hf", "", "HuggingFace model dir (config.json + model.safetensors[.index.json])")
	gguf := flag.String("gguf", "", "GGUF checkpoint path; loads through the memory-lean quant path")
	tokDir := flag.String("tok", "", "tokenizer dir containing tokenizer.json (default: -hf dir/cache, or GGUF sidecar tokenizer.json)")
	sys := flag.String("sys", "You are a helpful assistant.", "system prompt")
	prompt := flag.String("p", "", "user prompt — REQUIRED")
	maxNew := flag.Int("n", 256, "max new tokens to generate")
	metal := flag.Bool("metal", false, "run prefill on the Metal GPU (requires -tags fakmetal; decode stays CPU Q8)")
	temp := flag.Float64("temp", 0, "sampling temperature (0 = greedy/argmax)")
	seed := flag.Int64("seed", 1, "RNG seed for temperature sampling")
	quiet := flag.Bool("quiet", false, "suppress the hardware line and the load/prefill spinners (the model=… banner + token stream are unaffected)")
	flag.Parse()

	// Expand a leading ~ so `-hf ~/...`, `-gguf ~/...`, `-tok ~/...` open as intended
	// (Go and most shells/PowerShell don't expand it for us).
	*hf = pathutil.ExpandTilde(*hf)
	*gguf = pathutil.ExpandTilde(*gguf)
	*tokDir = pathutil.ExpandTilde(*tokDir)

	if (*hf == "") == (*gguf == "") || *prompt == "" {
		fmt.Fprintln(os.Stderr, "usage: fakchat (-hf <model-dir> | -gguf <model.gguf>) -p <prompt> [-tok <dir>] [-metal] [-n N] [-temp T]")
		os.Exit(2)
	}

	// Show the real compute surface this build runs on (cores / matmul workers /
	// accelerator) so the run is never silent about its hardware. On a CPU build the
	// summary says so plainly rather than implying a GPU. -quiet suppresses it.
	if !*quiet {
		fmt.Fprintln(os.Stderr, "hardware:", demoui.Probe().Summary)
	}

	// spin starts a stderr spinner for a long blocking phase (model load, prefill) so
	// the terminal shows a live "⠙ <label>… 12.3s" instead of freezing. It is a no-op
	// under -quiet, and the returned stop func is always safe to defer.
	spin := func(label string) func() {
		if *quiet {
			return func() {}
		}
		return demoui.Spinner(os.Stderr, label)
	}

	cfg, err := readModelConfig(*hf, *gguf)
	if err != nil {
		fmt.Fprintln(os.Stderr, "config:", err)
		os.Exit(1)
	}

	// Qwen3.5/3.6 hybrid (Gated-DeltaNet): the recurrent linear-attention state is now
	// kernel-owned session state, so this family runs the f32 cached Prefill/Step path
	// and ignores -metal/Q8. The Llama family keeps the lean-Q8 Prefill/Step fast path.
	hybrid := cfg.IsQwen35Hybrid()

	// Metal prefill is heavy on unified memory (CPU model + f16 GPU copy); take a
	// machine-wide lease so concurrent runs queue instead of stacking (see gpulease).
	useMetal := *metal && !hybrid && metalgemm.Available()
	if *metal && !useMetal {
		if metalgemm.Compiled() {
			fmt.Fprintln(os.Stderr, "metal: no usable device; falling back to CPU Q8 prefill")
		} else {
			fmt.Fprintln(os.Stderr, "metal: not compiled in (rebuild with -tags fakmetal); falling back to CPU Q8 prefill")
		}
	}
	if useMetal {
		lease, lerr := gpulease.Acquire(gpulease.Options{NoWait: os.Getenv("FAK_GPU_LEASE_NOWAIT") != ""})
		if lerr != nil {
			fmt.Fprintln(os.Stderr, "metal lease:", lerr)
			os.Exit(1)
		}
		defer lease.Release()
	}

	// Load weights. Llama family: quantize-at-load (Q8) so 1.5B–14B fit on 36 GB.
	// GGUF streams through the quantized loader; HF hybrid still uses the validated f32 GDN path.
	t0 := time.Now()
	// Loading + quantizing the weights is the longest silent phase (tens of seconds on
	// the bigger rungs); spin the terminal so it shows a live elapsed counter instead of
	// freezing. Stopped before the model=… banner prints below.
	stopLoad := spin("Loading model")
	var m *model.Model
	quantLoaded := false
	q4kLoad := os.Getenv("FAK_Q4K") != ""
	if *gguf != "" {
		if q4kLoad {
			// Direct-resident-Q4_K loader (plan P1): name-eligible Q4_K matmul tensors are
			// held raw (no Q4→f32→Q8 round-trip), Q6_K minority + small tensors via Q8/f32.
			m, err = ggufload.LoadModelQ4K(*gguf)
		} else {
			m, err = ggufload.LoadModelQuant(*gguf)
		}
		quantLoaded = true
	} else if hybrid {
		m, err = model.LoadSafetensorsDir(*hf, cfg)
	} else {
		m, err = loadLean(*hf, cfg)
		quantLoaded = true
	}
	if err != nil {
		stopLoad()
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	if !hybrid && !quantLoaded {
		m.Quantize()
	}
	stopLoad()
	loadMS := time.Since(t0).Seconds() * 1e3

	// Tokenizer.
	td := *tokDir
	if td == "" {
		if *hf != "" {
			if _, err := os.Stat(filepath.Join(*hf, "tokenizer.json")); err == nil {
				td = *hf
			} else if home, herr := os.UserHomeDir(); herr == nil {
				td = filepath.Join(home, ".cache", "fak-models", "tokenizers", "qwen2.5")
			}
		} else if _, err := os.Stat(filepath.Join(filepath.Dir(*gguf), "tokenizer.json")); err == nil {
			td = filepath.Dir(*gguf)
		} else {
			fmt.Fprintln(os.Stderr, "tokenizer: -tok is required with -gguf unless tokenizer.json is next to the GGUF")
			os.Exit(2)
		}
	}
	tok, err := tokenizer.LoadJSON(filepath.Join(td, "tokenizer.json"))
	if err != nil {
		fmt.Fprintln(os.Stderr, "tokenizer:", err)
		os.Exit(1)
	}
	stops := stopIDs(tok, cfg)

	// ChatML prompt (Qwen / SmolLM2 family). Special tokens are parsed literally.
	chat := "<|im_start|>system\n" + *sys + "<|im_end|>\n" +
		"<|im_start|>user\n" + *prompt + "<|im_end|>\n" +
		"<|im_start|>assistant\n"
	ids, err := tok.Encode(chat)
	if err != nil {
		fmt.Fprintln(os.Stderr, "encode:", err)
		os.Exit(1)
	}

	// Hybrid qwen3_5: cached decode. Linear-attention recurrent state and full-attention
	// KV both live in the Session, so generation no longer reruns whole-sequence Forward.
	if hybrid {
		q4 := os.Getenv("FAK_Q4") != ""
		q4k := q4kLoad
		// The current FAK_Q4 path builds q4w from the resident Q8_0 copy, so the Q8_0 27B
		// (~26 GB) is materialized at load before q4 frees it → ~28 GB peak. On a shared
		// 36 GB fleet box that pressures peers, so it is gated behind FAK_Q4_FORCE. The
		// memory-lean route (a direct GGUF→int4 loader that never builds Q8_0, ~15–20 GB
		// peak) is the scoped next step — see QWEN36-NATIVE-PERF-PLAN-2026-06-19.md P1b.
		if q4 && os.Getenv("FAK_Q4_FORCE") == "" {
			fmt.Fprintln(os.Stderr, "FAK_Q4: the q8-intermediate path peaks ~28 GB; set FAK_Q4_FORCE=1 to acknowledge and run on this box, or wait for the lean GGUF→int4 loader (P1b).")
			os.Exit(0)
		}
		if q4 {
			m.QuantizeQ4() // build resident int4 weights (from q8w); decode streams ~1.8× fewer bytes
		}
		backend := "fak in-kernel Gated-DeltaNet (f32, cached)"
		if quantLoaded {
			backend = "fak in-kernel Gated-DeltaNet (GGUF->Q8, cached)"
		}
		if q4 {
			backend = "fak in-kernel Gated-DeltaNet (Q8 prefill + int4 decode, cached)"
		}
		if q4k {
			backend = "fak in-kernel Gated-DeltaNet (resident Q4_K decode + Q8 fallback, cached)"
			if os.Getenv("FAK_METAL") != "" {
				backend += " [Metal q4_k prefill]"
			}
		}
		fmt.Fprintf(os.Stderr, "model=%s  load=%.0fms  prompt_tokens=%d  backend=%s\n",
			cfg.ModelType, loadMS, len(ids), backend)
		out := bufio.NewWriter(os.Stdout)
		rng := rand.New(rand.NewSource(*seed))
		s := m.NewSession()
		s.Quant = quantLoaded
		s.Q4 = q4
		s.Q4K = q4k
		// FAK_METAL routes the resident-Q4_K hybrid PREFILL's q4_k-majority GEMMs to the Metal
		// q4_k dequant-GEMM (needs -tags fakmetal; a no-op flag on the pure-Go build). Decode
		// stays CPU. See QWEN36-NATIVE-PERF-PLAN P5.
		s.MetalQ4K = q4k && os.Getenv("FAK_METAL") != ""
		var pp *model.PhaseProfiler
		if os.Getenv("FAK_QPROFILE") != "" {
			pp = model.NewPhaseProfiler()
			s.PhaseProfiler = pp
		}
		// Prefill is a silent compute block (the model is "thinking" before the first
		// token); spin so the terminal isn't frozen. Stopped before the decode stream.
		stopPrefill := spin("Thinking")
		tp := time.Now()
		logits := s.Prefill(ids)
		prefillS := time.Since(tp).Seconds()
		stopPrefill()
		if pp != nil {
			printPhaseProfile("prefill", pp.Snapshot("prefill", len(ids), 0, time.Since(tp).Nanoseconds()))
			pp.Reset() // separate the decode phase split from prefill's
		}
		tg := time.Now()
		gen := 0
		for ; gen < *maxNew; gen++ {
			next := sample(logits, *temp, rng)
			if stops[next] {
				break
			}
			if piece, derr := tok.Decode([]int{next}); derr == nil {
				out.WriteString(piece)
				out.Flush()
			}
			logits = s.Step(next)
		}
		out.Flush()
		decodeNanos := time.Since(tg).Nanoseconds()
		decodeS := float64(decodeNanos) / 1e9
		if pp != nil {
			printPhaseProfile("decode", pp.Snapshot("decode", 0, gen, decodeNanos))
		}
		prefTPS := 0.0
		if prefillS > 0 {
			prefTPS = float64(len(ids)) / prefillS
		}
		decTPS := 0.0
		if decodeS > 0 {
			decTPS = float64(gen) / decodeS
		}
		fmt.Fprintf(os.Stderr, "\n---\nprefill: %d tok in %.2fs (%.1f tok/s)  |  cached qwen3_5 decode: %d tok in %.2fs (%.1f tok/s)\n",
			len(ids), prefillS, prefTPS, gen, decodeS, decTPS)
		return
	}

	s := m.NewSession()
	s.Quant = true
	s.Metal = useMetal

	backend := "CPU Q8 (NEON) prefill + decode"
	if useMetal {
		backend = "Metal GPU prefill + CPU Q8 (NEON) decode [hybrid]"
	}
	fmt.Fprintf(os.Stderr, "model=%s  load=%.0fms  prompt_tokens=%d  backend=%s\n",
		cfg.ModelType, loadMS, len(ids), backend)

	// Prefill (timed). Silent compute block before the first token — spin so the
	// terminal shows progress, stopped right before the decode stream starts.
	stopPrefill := spin("Thinking")
	tp := time.Now()
	logits := s.Prefill(ids)
	prefillS := time.Since(tp).Seconds()
	stopPrefill()

	// Decode loop (timed, streamed).
	out := bufio.NewWriter(os.Stdout)
	rng := rand.New(rand.NewSource(*seed))
	td0 := time.Now()
	gen := 0
	for ; gen < *maxNew; gen++ {
		next := sample(logits, *temp, rng)
		if stops[next] {
			break
		}
		piece, derr := tok.Decode([]int{next})
		if derr == nil {
			out.WriteString(piece)
			out.Flush()
		}
		logits = s.Step(next)
	}
	decodeS := time.Since(td0).Seconds()
	out.Flush()

	prefTPS := 0.0
	if prefillS > 0 {
		prefTPS = float64(len(ids)) / prefillS
	}
	decTPS := 0.0
	if decodeS > 0 {
		decTPS = float64(gen) / decodeS
	}
	fmt.Fprintf(os.Stderr, "\n---\nprefill: %d tok in %.2fs (%.1f tok/s)  |  decode: %d tok in %.2fs (%.1f tok/s)\n",
		len(ids), prefillS, prefTPS, gen, decodeS, decTPS)
}

func readModelConfig(hf, gguf string) (model.Config, error) {
	if gguf != "" {
		f, err := ggufload.Open(gguf)
		if err != nil {
			return model.Config{}, fmt.Errorf("gguf: %w", err)
		}
		return f.Config()
	}
	return readHFConfig(hf)
}

// readHFConfig mirrors cmd/modelbench: read config.json into model.Config and fill
// head_dim if implicit. Family-specific defaults live in model.Config derivation.
func readHFConfig(dir string) (model.Config, error) {
	var cfg model.Config
	cb, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return cfg, fmt.Errorf("config.json: %w", err)
	}
	if err := json.Unmarshal(cb, &cfg); err != nil {
		return cfg, fmt.Errorf("config.json parse: %w", err)
	}
	if cfg.HeadDim == 0 && cfg.NumHeads != 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumHeads
	}
	return cfg, nil
}

// loadLean quantize-at-load: sharded dirs via the index, else the single file.
func loadLean(dir string, cfg model.Config) (*model.Model, error) {
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors.index.json")); err == nil {
		return model.LoadSafetensorsQuantDir(dir, cfg)
	}
	return model.LoadSafetensorsQuant(filepath.Join(dir, "model.safetensors"), cfg)
}

// stopIDs collects the generation stop tokens: <|im_end|>, <|endoftext|>, and any
// EOS ids the config declares.
func stopIDs(tok *tokenizer.Tokenizer, cfg model.Config) map[int]bool {
	stops := map[int]bool{}
	for id, content := range tok.SpecialTokens() {
		if content == "<|im_end|>" || content == "<|endoftext|>" {
			stops[id] = true
		}
	}
	if cfg.EOSTokenID > 0 {
		stops[cfg.EOSTokenID] = true
	}
	for _, e := range cfg.EOSTokenIDs {
		if e > 0 {
			stops[e] = true
		}
	}
	return stops
}

// printPhaseProfile dumps the FAK_QPROFILE=1 coarse phase split (which real phase ate the
// run — recurrence vs projections vs attention) to stderr, top phases first. Opt-in only;
// the default path attaches no profiler and pays zero instrumentation cost.
func printPhaseProfile(mode string, pr *model.PhaseProfile) {
	if pr == nil {
		return
	}
	fmt.Fprintf(os.Stderr, "[phaseprof %s] total=%.1fms bottleneck=%s\n", mode, pr.TotalMS, pr.Bottleneck)
	for i, ph := range pr.Phases {
		if i >= 12 {
			break
		}
		fmt.Fprintf(os.Stderr, "  %-26s %8.1fms  %5.1f%%  (%d calls)\n", ph.Phase, ph.MS, ph.TimePct, ph.Calls)
	}
}

// sample returns the next token id: argmax when temp<=0, else a temperature-scaled
// softmax draw.
func sample(logits []float32, temp float64, rng *rand.Rand) int {
	if temp <= 0 {
		best, bi := float32(-math.MaxFloat32), 0
		for i, x := range logits {
			if x > best {
				best, bi = x, i
			}
		}
		return bi
	}
	maxL := float32(-math.MaxFloat32)
	for _, x := range logits {
		if x > maxL {
			maxL = x
		}
	}
	var sum float64
	probs := make([]float64, len(logits))
	for i, x := range logits {
		p := math.Exp(float64(x-maxL) / temp)
		probs[i] = p
		sum += p
	}
	r := rng.Float64() * sum
	for i, p := range probs {
		r -= p
		if r <= 0 {
			return i
		}
	}
	return len(logits) - 1
}
