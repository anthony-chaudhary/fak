package agent

// inkernel_planner.go — the in-kernel chat Planner. When fak serve boots with a
// preloaded GGUF model (modelengine.Preload / PreloadQ4K) and a tokenizer, and no
// upstream --base-url, this planner drives BOTH /v1/chat/completions and
// /v1/messages (they share s.planner.Complete) from the model fused into the
// kernel — real ChatML chat through internal/tokenizer, not the byte-tokenized
// dispatch demo in modelengine.Complete.
//
// The decode recipe is the proven cmd/fakchat hybrid path: render ChatML → Encode
// → Session.Prefill → argmax/temperature sample → Session.Step → Decode, stopping
// on <|im_end|>/<|endoftext|>. fakchat's end-to-end coherent chat (Qwen2.5-1.5B/7B,
// FAK-NATIVE-CHAT-RESULTS.md) is the witness that this recipe produces real text;
// this file factors it into a Planner so the gateway can serve it on both wires.

import (
	"context"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// InKernelPlanner is an agent.Planner backed by the in-kernel model. One Complete
// call renders the transcript as ChatML, runs a real Prefill + decode over the
// kernel-owned session cache, and returns the assistant's text. It does not itself
// emit structured tool calls — the gateway's adjudication layer still runs on
// whatever the caller proposed.
type InKernelPlanner struct {
	m       *model.Model
	tok     *tokenizer.Tokenizer
	modelID string
	q4k     bool            // resident-Q4_K load: decode runs Session.Q4K (SDOT int8 GEMV)
	backend compute.Backend // non-nil → decode runs through the device HAL (e.g. CUDA) instead of the CPU session
	maxNew  int
	temp    float64
	seed    int64
}

// NewInKernelPlanner builds a planner over an already-loaded model + tokenizer.
// q4k flags a resident-Q4_K load so the decode engages Session.Q4K. Generation
// depth/sampling default to a greedy 256-token turn but are overridable via
// FAK_INKERNEL_MAX_TOKENS / FAK_INKERNEL_TEMP / FAK_INKERNEL_SEED.
func NewInKernelPlanner(m *model.Model, tok *tokenizer.Tokenizer, modelID string, q4k bool, backend compute.Backend) *InKernelPlanner {
	return &InKernelPlanner{
		m:       m,
		tok:     tok,
		modelID: modelID,
		q4k:     q4k,
		backend: backend,
		maxNew:  envInt("FAK_INKERNEL_MAX_TOKENS", 256),
		temp:    envFloat("FAK_INKERNEL_TEMP", 0),
		seed:    int64(envInt("FAK_INKERNEL_SEED", 0)),
	}
}

// Model reports the model id (for /v1/models provenance + the planner seam).
func (p *InKernelPlanner) Model() string { return p.modelID }

// Complete renders the transcript as ChatML and runs one in-kernel decode turn,
// returning the generated assistant text. Mirrors cmd/fakchat's hybrid path. The
// per-request SampleOpts override this planner's configured decode length,
// temperature, TopP (nucleus cutoff), and TopK (top-k cutoff) for THIS turn, and a
// per-request Stop sequence ends the turn early (string-suffix stop, orthogonal to
// the token-ID <|im_end|>/EOS stops). All five per-request sampling controls the
// HTTP wires forward are now honored on the in-kernel path too.
func (p *InKernelPlanner) Complete(_ context.Context, messages []Message, _ []ToolDef, opts ...SampleOpt) (*Completion, error) {
	sp := applySampleOpts(opts...)
	maxNew := p.maxNew
	if sp.MaxTokens != nil && *sp.MaxTokens > 0 {
		maxNew = *sp.MaxTokens
	}
	temp := p.temp
	if sp.Temperature != nil {
		temp = *sp.Temperature
	}
	// Per-request nucleus cutoff; 0 (the zero value) disables truncation so an omitted
	// top_p keeps the full softmax draw, identical to the pre-seam path.
	topP := 0.0
	if sp.TopP != nil {
		topP = *sp.TopP
	}
	// Per-request top-k; 0 (the zero value, and any value <=0) disables truncation so
	// an omitted top_k keeps the full distribution, identical to the pre-seam path.
	topK := 0
	if sp.TopK != nil {
		topK = *sp.TopK
	}

	chat := renderChatML(messages)
	ids, err := p.tok.Encode(chat)
	if err != nil {
		return nil, err
	}
	stops := inKernelStopIDs(p.tok, p.m.Cfg)

	// Device path: when a backend (e.g. CUDA) is wired, run prefill+decode through the
	// compute HAL (Session.Prefill/Step dispatch on s.Backend). The session owns
	// device-resident KV+weights, so Close() frees them when the turn ends. Otherwise
	// the legacy CPU session. NOTE: a fresh device session per turn re-uploads weights;
	// correct and proves the GPU serves, but a persistent resident session is the perf
	// follow-up (tracked).
	var s *model.Session
	if p.backend != nil {
		s = p.m.NewBackendSession(p.backend)
		defer s.Close()
	} else {
		s = p.m.NewSession()
	}
	s.Quant = true
	s.Q4K = p.q4k && p.backend == nil // resident-Q4_K is a CPU-only decode path; the device HAL uses Q8/F32
	tp := time.Now()
	logits := s.Prefill(ids)
	prefillS := time.Since(tp).Seconds()

	rng := rand.New(rand.NewSource(p.seed))
	var sb strings.Builder
	td := time.Now()
	gen := 0
	// finishReason is honest about WHY decode ended: "stop" when a token-ID stop or a
	// per-request Stop sequence fired, "length" when maxNew was the only limit hit.
	finishReason := "length"
	for ; gen < maxNew; gen++ {
		next := sampleLogits(logits, temp, topP, topK, rng)
		if next < 0 || stops[next] {
			finishReason = "stop"
			break
		}
		if piece, derr := p.tok.Decode([]int{next}); derr == nil {
			sb.WriteString(piece)
		}
		// String-suffix stop: a per-request Stop sequence ends the turn and its text
		// is trimmed (the stop string is not echoed back, matching the HTTP wires).
		if trimmed, hit := checkStop(sb.String(), sp.Stop); hit {
			sb.Reset()
			sb.WriteString(trimmed)
			gen++ // this token WAS generated; count it before exiting the loop
			finishReason = "stop"
			break
		}
		logits = s.Step(next)
	}
	decodeS := time.Since(td).Seconds()
	// Witness line (mirrors cmd/fakchat): real per-turn prefill/decode tok/s through
	// the in-kernel model, so a served chat turn self-reports its perf (vs the 7.29
	// q4_k_m bar) without a separate bench harness.
	prefTPS, decTPS := 0.0, 0.0
	if prefillS > 0 {
		prefTPS = float64(len(ids)) / prefillS
	}
	if decodeS > 0 {
		decTPS = float64(gen) / decodeS
	}
	log.Printf("inkernel_chat model=%s q4k=%v prompt=%dtok prefill=%.2fs/%.1ftok/s decode=%dtok/%.2fs/%.1ftok/s",
		p.modelID, p.q4k, len(ids), prefillS, prefTPS, gen, decodeS, decTPS)

	return &Completion{
		Message:      Message{Role: "assistant", Content: sb.String()},
		FinishReason: finishReason,
		Usage:        Usage{PromptTokens: len(ids), CompletionTokens: gen, TotalTokens: len(ids) + gen},
	}, nil
}

// checkStop reports whether the accumulated decode text ends with any of the
// per-request stop sequences, returning the text with the matched stop suffix
// trimmed. It mirrors the HTTP wires' contract: the stop string ends generation and
// is NOT echoed in the returned content. The LONGEST matching stop wins so the trim
// is maximal, and an empty stop string is ignored (it would otherwise match every
// text and truncate every turn to nothing). An empty stop set never fires, so the
// default in-kernel path is byte-for-byte the pre-seam behavior.
func checkStop(text string, stop []string) (string, bool) {
	best := ""
	for _, s := range stop {
		if s == "" {
			continue
		}
		if strings.HasSuffix(text, s) && len(s) > len(best) {
			best = s
		}
	}
	if best == "" {
		return text, false
	}
	return text[:len(text)-len(best)], true
}

// renderChatML renders the transcript as Qwen/SmolLM2 ChatML, terminating with an
// open assistant turn for generation. System messages fold into one leading system
// block. tool / tool-call messages render as plain-text turns so the model sees
// prior tool I/O as context (no structured tool-call emission from this planner).
func renderChatML(messages []Message) string {
	var b strings.Builder
	var sys []string
	for _, m := range messages {
		if m.Role == "system" && strings.TrimSpace(m.Content) != "" {
			sys = append(sys, m.Content)
		}
	}
	if len(sys) > 0 {
		b.WriteString("<|im_start|>system\n")
		b.WriteString(strings.Join(sys, "\n"))
		b.WriteString("<|im_end|>\n")
	}
	for _, m := range messages {
		role, content := m.Role, m.Content
		switch role {
		case "system":
			continue
		case "tool":
			// A tool result reads as user-supplied context to the model.
			role = "user"
			if m.Name != "" {
				content = m.Name + ": " + content
			}
		case "assistant":
			for _, tc := range m.ToolCalls {
				content += "\n[tool_call " + tc.Function.Name + " " + tc.Function.Arguments + "]"
			}
		}
		b.WriteString("<|im_start|>")
		b.WriteString(role)
		b.WriteString("\n")
		b.WriteString(content)
		b.WriteString("<|im_end|>\n")
	}
	b.WriteString("<|im_start|>assistant\n")
	return b.String()
}

// inKernelStopIDs mirrors cmd/fakchat.stopIDs: <|im_end|>, <|endoftext|>, and any
// EOS ids the model config declares.
func inKernelStopIDs(tok *tokenizer.Tokenizer, cfg model.Config) map[int]bool {
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

// sampleLogits mirrors cmd/fakchat.sample: argmax when temp<=0, else a
// temperature-scaled softmax draw. topK then topP truncate the stochastic path, in
// that order (the standard top-k → top-p pipeline): top-k keeps only the k
// highest-probability tokens, then nucleus (top-p) keeps the smallest set whose
// cumulative mass reaches topP. The tail each step excludes is zeroed before the
// draw. A topK<=0 or topK>=len(logits) disables top-k; a topP<=0 or topP>=1 disables
// nucleus — with both off the draw is the full softmax, byte-for-byte the pre-seam
// behavior. The single most-probable token is always kept so neither cutoff can
// empty the candidate set. Both shape only the stochastic path: temp<=0 stays pure
// argmax (top-k/top-p never change the argmax winner).
func sampleLogits(logits []float32, temp, topP float64, topK int, rng *rand.Rand) int {
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
	if topK > 0 && topK < len(probs) {
		sum = topKTruncate(probs, sum, topK)
	}
	if topP > 0 && topP < 1 {
		sum = nucleusTruncate(probs, sum, topP)
	}
	r := rng.Float64() * sum
	for i, p := range probs {
		r -= p
		if r <= 0 {
			return i
		}
	}
	// Fall back to the last token with nonzero mass (nucleus zeroed the tail).
	for i := len(probs) - 1; i >= 0; i-- {
		if probs[i] > 0 {
			return i
		}
	}
	return len(logits) - 1
}

// nucleusTruncate zeroes every probability outside the top-p nucleus in place and
// returns the surviving mass (the new normalization sum). The nucleus is the
// smallest set of highest-probability tokens whose cumulative mass reaches topP;
// the single most-probable token is always kept so the nucleus is never empty.
// probs is unsorted on entry and stays index-aligned to the caller's logits.
func nucleusTruncate(probs []float64, sum, topP float64) float64 {
	order := make([]int, len(probs))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return probs[order[a]] > probs[order[b]] })
	target := topP * sum
	var cum float64
	kept := make(map[int]bool, len(order))
	for rank, idx := range order {
		// Stop BEFORE adding this token once the nucleus already reached the target —
		// the kept set is the minimal prefix whose mass >= target. Rank 0 is always
		// kept (the head token) so the nucleus is never empty.
		if rank > 0 && cum >= target {
			break
		}
		kept[idx] = true
		cum += probs[idx]
	}
	var newSum float64
	for i := range probs {
		if kept[i] {
			newSum += probs[i]
		} else {
			probs[i] = 0
		}
	}
	return newSum
}

// topKTruncate zeroes every probability outside the top-k highest-probability
// tokens in place and returns the surviving mass (the new normalization sum). Ties
// at the k-th rank are broken by index order (the sort is stable on equal probs via
// the index comparator), so the kept set is deterministic. probs is unsorted on
// entry and stays index-aligned to the caller's logits. The caller guarantees
// 0 < k < len(probs); k>=len(probs) is a no-op handled before the call so the full
// distribution stays byte-for-byte the pre-seam draw.
func topKTruncate(probs []float64, sum float64, k int) float64 {
	order := make([]int, len(probs))
	for i := range order {
		order[i] = i
	}
	// Highest probability first; ties resolve to the lower index so the kept set is
	// stable and reproducible across runs.
	sort.Slice(order, func(a, b int) bool {
		if probs[order[a]] != probs[order[b]] {
			return probs[order[a]] > probs[order[b]]
		}
		return order[a] < order[b]
	})
	kept := make(map[int]bool, k)
	for rank := 0; rank < k; rank++ {
		kept[order[rank]] = true
	}
	var newSum float64
	for i := range probs {
		if kept[i] {
			newSum += probs[i]
		} else {
			probs[i] = 0
		}
	}
	return newSum
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envFloat(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}
