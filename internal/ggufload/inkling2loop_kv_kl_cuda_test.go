//go:build cuda && cgo

package ggufload

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type inkling2KVKLPromptSummary struct {
	Name         string  `json:"name"`
	PromptTokens int     `json:"prompt_tokens"`
	Positions    int     `json:"positions"`
	MeanKL       float64 `json:"mean_kl"`
	MaxKL        float64 `json:"max_kl"`
	GreedyMatch  int     `json:"greedy_match"`
}

type inkling2KVKLSummary struct {
	Model              string                      `json:"model"`
	RefKV              string                      `json:"ref_kv"`
	CandKV             string                      `json:"cand_kv"`
	Ctx                int                         `json:"ctx"`
	Prompts            int                         `json:"prompts"`
	Positions          int                         `json:"positions"`
	MeanKL             float64                     `json:"mean_kl"`
	MaxKL              float64                     `json:"max_kl"`
	CI95Lo             float64                     `json:"ci95_lo"`
	CI95Hi             float64                     `json:"ci95_hi"`
	BootstrapResamples int                         `json:"bootstrap_resamples"`
	BootstrapSeed      int64                       `json:"bootstrap_seed"`
	GreedyMatch        int                         `json:"greedy_match"`
	GreedyMatchRate    float64                     `json:"greedy_match_rate"`
	Threshold          float64                     `json:"threshold"`
	Pass               bool                        `json:"pass"`
	PerPrompt          []inkling2KVKLPromptSummary `json:"per_prompt"`
}

type inkling2KVKLConfig struct {
	candKV     string
	wantDType  Inkling2KVDType
	jsonPrefix string
	defaultMax float64
	checkReady func(t *testing.T, be compute.Backend)
}

// inkling2PairedBootstrapCI computes the two-sided 95% percentile bootstrap confidence
// interval [lo, hi] for the mean of paired per-token KL observations using a fixed seed.
func inkling2PairedBootstrapCI(samples []float64, resamples int, seed int64) (lo, hi float64) {
	n := len(samples)
	if n == 0 || resamples <= 0 {
		return 0, 0
	}
	rng := rand.New(rand.NewSource(seed))
	means := make([]float64, resamples)
	invN := 1.0 / float64(n)
	for b := 0; b < resamples; b++ {
		var sum float64
		for i := 0; i < n; i++ {
			sum += samples[rng.Intn(n)]
		}
		means[b] = sum * invN
	}
	sort.Float64s(means)
	loIdx := int(math.Floor(0.025 * float64(resamples)))
	hiIdx := int(math.Ceil(0.975*float64(resamples))) - 1
	if loIdx < 0 {
		loIdx = 0
	}
	if hiIdx >= resamples {
		hiIdx = resamples - 1
	}
	return means[loIdx], means[hiIdx]
}

// fak-test:runtime integration est=120s lane=optin
//
// TestInkling2CUDAKVFP8KLGate verifies full-vocab KL(P_fp32 || P_fp8) on teacher-forced
// continuations for Inkling-2-Mini global FP8 E4M3 KV caches against the FP32 reference.
//
// Environment controls:
//   - FAK_INKLING2_KV_KL_TEST=1  (required to run)
//   - FAK_INKLING2_GGUF=<path>   (required GGUF model path)
//   - FAK_INKLING2_KL_TOKENS=N   (optional measured positions per prompt, default 64)
//   - FAK_INKLING2_KL_CTX=C      (optional session ctx & deep prompt depth, e.g. 4096/8192/32768)
//   - FAK_INKLING2_KL_MAX=T      (optional 95% CI upper bound threshold in nats, default 0.0015)
func TestInkling2CUDAKVFP8KLGate(t *testing.T) {
	runInkling2CUDAKVKLGate(t, inkling2KVKLConfig{
		candKV:     "fp8",
		wantDType:  Inkling2KVFP8,
		jsonPrefix: "inkling2_kv_fp8_kl_json",
		defaultMax: 0.0015,
		checkReady: func(t *testing.T, be compute.Backend) {
			t.Helper()
			fp8Dev, ok := be.(compute.Inkling2FP8KVDevice)
			if !ok {
				t.Fatalf("cuda backend %s does not implement compute.Inkling2FP8KVDevice", be.Name())
			}
			if err := fp8Dev.Ink2PFP8KVReady(); err != nil {
				t.Fatalf("cuda FP8 KV ops not ready: %v", err)
			}
		},
	})
}

// fak-test:runtime integration est=120s lane=optin
//
// TestInkling2CUDAKVNVFP4KLGate verifies full-vocab KL(P_fp32 || P_nvfp4) on teacher-forced
// continuations for Inkling-2-Mini global block-16 NVFP4 E4M3-scaled KV caches against the
// FP32 reference.
//
// Environment controls:
//   - FAK_INKLING2_KV_KL_TEST=1  (required to run)
//   - FAK_INKLING2_GGUF=<path>   (required GGUF model path)
//   - FAK_INKLING2_KL_TOKENS=N   (optional measured positions per prompt, default 64)
//   - FAK_INKLING2_KL_CTX=C      (optional session ctx & deep prompt depth, e.g. 4096/8192/32768)
//   - FAK_INKLING2_KL_MAX=T      (optional 95% CI upper bound threshold in nats, default 0.0015)
func TestInkling2CUDAKVNVFP4KLGate(t *testing.T) {
	runInkling2CUDAKVKLGate(t, inkling2KVKLConfig{
		candKV:     "nvfp4",
		wantDType:  Inkling2KVNVFP4,
		jsonPrefix: "inkling2_kv_nvfp4_kl_json",
		defaultMax: 0.0015,
		checkReady: func(t *testing.T, be compute.Backend) {
			t.Helper()
			nvfp4Dev, ok := be.(compute.Inkling2NVFP4KVDevice)
			if !ok {
				t.Fatalf("cuda backend %s does not implement compute.Inkling2NVFP4KVDevice", be.Name())
			}
			if err := nvfp4Dev.Ink2PNVFP4KVReady(); err != nil {
				t.Fatalf("cuda NVFP4 KV ops not ready: %v", err)
			}
		},
	})
}

func runInkling2CUDAKVKLGate(t *testing.T, cfg inkling2KVKLConfig) {
	t.Helper()
	if strings.TrimSpace(os.Getenv("FAK_INKLING2_KV_KL_TEST")) != "1" {
		t.Skip("FAK_INKLING2_KV_KL_TEST!=1")
	}
	m, be := openInkling2CUDAForTest(t, "FAK_INKLING2_KV_KL_TEST")
	defer m.Close()

	cfg.checkReady(t, be)

	tokens := 64
	if raw := strings.TrimSpace(os.Getenv("FAK_INKLING2_KL_TOKENS")); raw != "" {
		v, err := strconv.Atoi(raw)
		if err != nil || v < 1 {
			t.Fatalf("FAK_INKLING2_KL_TOKENS=%q: want positive integer", raw)
		}
		tokens = v
	}

	klCtx := 0
	if raw := strings.TrimSpace(os.Getenv("FAK_INKLING2_KL_CTX")); raw != "" {
		v, err := strconv.Atoi(raw)
		if err != nil || v < Inkling2ResidentCtxMin {
			t.Fatalf("FAK_INKLING2_KL_CTX=%q: want integer >= %d", raw, Inkling2ResidentCtxMin)
		}
		klCtx = v
	}

	maxKL := cfg.defaultMax
	if raw := strings.TrimSpace(os.Getenv("FAK_INKLING2_KL_MAX")); raw != "" {
		v, err := strconv.ParseFloat(raw, 64)
		if err != nil || v <= 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("FAK_INKLING2_KL_MAX=%q: want positive float", raw)
		}
		maxKL = v
	}

	type klPrompt struct {
		name   string
		tokens []int
	}
	var prompts []klPrompt
	if klCtx > 0 {
		deepLen := klCtx - tokens - 1
		if deepLen < 1 {
			t.Fatalf("FAK_INKLING2_KL_CTX=%d too small for FAK_INKLING2_KL_TOKENS=%d", klCtx, tokens)
		}
		prompts = append(prompts,
			klPrompt{name: fmt.Sprintf("pp%d", deepLen), tokens: inkling2PrefillPrompt(deepLen)},
			klPrompt{name: inkling2OracleCases[2].name, tokens: inkling2OracleCases[2].prompt},
			klPrompt{name: inkling2OracleCases[0].name, tokens: inkling2OracleCases[0].prompt},
		)
	} else {
		for _, c := range inkling2OracleCases {
			prompts = append(prompts, klPrompt{name: c.name, tokens: c.prompt})
		}
	}

	maxNeeded := 0
	for _, p := range prompts {
		if need := len(p.tokens) + tokens + 1; need > maxNeeded {
			maxNeeded = need
		}
	}
	sessionCtx := klCtx
	if sessionCtx == 0 && maxNeeded > Inkling2ResidentCtxDefault {
		sessionCtx = maxNeeded
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	openArm := func(kv string) *inkling2Resident {
		t.Setenv("FAK_INKLING2_KV", kv)
		var (
			s   *inkling2Resident
			err error
		)
		if sessionCtx > 0 {
			s, err = newInkling2ResidentSized(m, be, sessionCtx, inkling2ArenaFloats)
		} else {
			s, err = newInkling2Resident(m, be)
		}
		if err != nil {
			t.Fatalf("open resident session (kv=%s ctx=%d): %v", kv, sessionCtx, err)
		}
		return s
	}

	feedPrompt := func(s *inkling2Resident, prompt []int) int {
		if s.batched {
			next, err := s.prefill(context.Background(), prompt)
			if err != nil {
				t.Fatalf("prefill (%d tokens): %v", len(prompt), err)
			}
			return next
		}
		var next int
		var err error
		for _, tok := range prompt {
			if next, err = s.step(tok); err != nil {
				t.Fatalf("step prompt: %v", err)
			}
		}
		return next
	}

	readLogProbs := func(s *inkling2Resident) []float64 {
		raw, err := s.lastLogits()
		if err != nil {
			t.Fatalf("lastLogits: %v", err)
		}
		return inkling2LogProbs(raw, m.HP.VocabUnpadded)
	}

	var allKL []float64
	var totalAgree int
	var totalMaxKL float64
	var perPrompt []inkling2KVKLPromptSummary
	reportedCtx := sessionCtx

	for _, p := range prompts {
		// 1. Reference pass: FP32 global KV, capture greedy continuation + full-vocab log-probs.
		ref := openArm("f32")
		if got := ref.kvDType(); got != Inkling2KVF32 {
			ref.close()
			t.Fatalf("reference session did not activate FP32 global KV (got %v)", got)
		}
		if reportedCtx == 0 {
			for slot, c := range ref.caps {
				if !inkling2SlotIsLocal(m.HP, slot) && c > reportedCtx {
					reportedCtx = c
				}
			}
		}
		refBatched := ref.batched
		next := feedPrompt(ref, p.tokens)
		refLogProbs := make([][]float64, 0, tokens)
		refLogProbs = append(refLogProbs, readLogProbs(ref))
		forced := make([]int, 0, tokens-1)
		for i := 1; i < tokens; i++ {
			forced = append(forced, next)
			var err error
			next, err = ref.step(next)
			if err != nil {
				ref.close()
				t.Fatalf("ref f32 step %d (%s): %v", i, p.name, err)
			}
			refLogProbs = append(refLogProbs, readLogProbs(ref))
		}
		ref.close()
		inkling2TrimIdle(be)

		// 2. Candidate pass on exact same teacher-forced trajectory.
		cand := openArm(cfg.candKV)
		if got := cand.kvDType(); got != cfg.wantDType {
			cand.close()
			t.Fatalf("candidate session did not activate %s global KV (got %v)", cfg.candKV, got)
		}
		cand.batched = refBatched
		_ = feedPrompt(cand, p.tokens)

		var pSum, pMax float64
		pAgree := 0
		recordPos := func(idx int) {
			candLP := readLogProbs(cand)
			kl, same := inkling2KL(refLogProbs[idx], candLP)
			if math.IsNaN(kl) || math.IsInf(kl, 0) {
				cand.close()
				t.Fatalf("%s pos %d: non-finite KL=%v", p.name, idx, kl)
			}
			if kl < 0 {
				kl = 0
			}
			allKL = append(allKL, kl)
			pSum += kl
			if kl > pMax {
				pMax = kl
			}
			if kl > totalMaxKL {
				totalMaxKL = kl
			}
			if same {
				pAgree++
				totalAgree++
			}
		}

		recordPos(0)
		for i, tok := range forced {
			if _, err := cand.step(tok); err != nil {
				cand.close()
				t.Fatalf("cand %s step %d (%s): %v", cfg.candKV, i+1, p.name, err)
			}
			recordPos(i + 1)
		}
		if got := cand.kvDType(); got != cfg.wantDType {
			cand.close()
			t.Fatalf("candidate session fell back out of %s global KV during %s (got %v)", cfg.candKV, p.name, got)
		}
		cand.close()
		inkling2TrimIdle(be)

		pMean := pSum / float64(len(refLogProbs))
		perPrompt = append(perPrompt, inkling2KVKLPromptSummary{
			Name:         p.name,
			PromptTokens: len(p.tokens),
			Positions:    len(refLogProbs),
			MeanKL:       pMean,
			MaxKL:        pMax,
			GreedyMatch:  pAgree,
		})
		t.Logf("inkling2 kv %s KL: %s pp%d pos=%d mean=%.6f max=%.6f greedy_match=%d/%d (%.2f%%)",
			cfg.candKV, p.name, len(p.tokens), len(refLogProbs), pMean, pMax, pAgree, len(refLogProbs),
			100*float64(pAgree)/float64(len(refLogProbs)))
	}

	const (
		bootstrapResamples = 1000
		bootstrapSeed      = int64(20261126)
	)
	var sumKL float64
	for _, v := range allKL {
		sumKL += v
	}
	meanKL := sumKL / float64(len(allKL))
	ciLo, ciHi := inkling2PairedBootstrapCI(allKL, bootstrapResamples, bootstrapSeed)
	pass := ciHi < maxKL

	summary := inkling2KVKLSummary{
		Model:              filepath.Base(os.Getenv("FAK_INKLING2_GGUF")),
		RefKV:              "f32",
		CandKV:             cfg.candKV,
		Ctx:                reportedCtx,
		Prompts:            len(prompts),
		Positions:          len(allKL),
		MeanKL:             meanKL,
		MaxKL:              totalMaxKL,
		CI95Lo:             ciLo,
		CI95Hi:             ciHi,
		BootstrapResamples: bootstrapResamples,
		BootstrapSeed:      bootstrapSeed,
		GreedyMatch:        totalAgree,
		GreedyMatchRate:    float64(totalAgree) / float64(len(allKL)),
		Threshold:          maxKL,
		Pass:               pass,
		PerPrompt:          perPrompt,
	}
	jb, err := json.Marshal(summary)
	if err != nil {
		t.Fatalf("marshal KL summary: %v", err)
	}
	t.Logf("%s=%s", cfg.jsonPrefix, string(jb))

	if !pass {
		t.Fatalf("%s global KV KL gate FAILED: 95%% bootstrap CI upper bound %.6f >= threshold %.6f (mean=%.6f max=%.6f greedy_match=%d/%d)",
			strings.ToUpper(cfg.candKV), ciHi, maxKL, meanKL, totalMaxKL, totalAgree, len(allKL))
	}
}
