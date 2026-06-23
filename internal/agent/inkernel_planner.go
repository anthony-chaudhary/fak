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
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
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
	quant   bool            // Q8_0 decode/prefill path (the served default); tests flip it to exercise the proven f32 reuse path
	backend compute.Backend // non-nil → decode runs through the device HAL (e.g. CUDA) instead of the CPU session
	maxNew  int
	temp    float64
	seed    int64

	// tree is the process-scoped RadixAttention prefix cache (internal/radixkv): the
	// multi-thousand-token static system+tool-schema prefix is prefilled once and the
	// next turn REUSES its KV, prefilling only the divergent suffix — the candidate-#13
	// win, bit-identical to a full recompute (proven in internal/model's KV-prefix-reuse
	// rung). nil disables reuse (every turn full-prefills, the pre-#13 behavior); the
	// device-HAL path (backend != nil) never reuses (the reuse clone is a CPU session).
	// mu guards every tree access — the gateway can drive Complete concurrently, and the
	// tree is shared mutable state (radixkv itself is deliberately lock-free).
	//
	// MEMORY NOTE: radixkv stores the FULL-prefix KV per node, so a long single growing
	// conversation accumulates nested KV clones (see radixkv's Tokens-vs-PrefixTokens
	// note). FAK_INKERNEL_RADIX_BUDGET sets the LRU edge-token budget (0 = unbounded, the
	// default — the maximal-reuse regime the witnesses measure). Operators serving long
	// sessions should set a budget; bounding the deep-chain footprint is tracked.
	mu   sync.Mutex
	tree *radixkv.Tree
}

// NewInKernelPlanner builds a planner over an already-loaded model + tokenizer.
// q4k flags a resident-Q4_K load so the decode engages Session.Q4K. Generation
// depth/sampling default to a greedy 256-token turn but are overridable via
// FAK_INKERNEL_MAX_TOKENS / FAK_INKERNEL_TEMP / FAK_INKERNEL_SEED.
func NewInKernelPlanner(m *model.Model, tok *tokenizer.Tokenizer, modelID string, q4k bool, backend compute.Backend) *InKernelPlanner {
	p := &InKernelPlanner{
		m:       m,
		tok:     tok,
		modelID: modelID,
		q4k:     q4k,
		quant:   true, // the served in-kernel path runs the Q8_0 forward (a quantized model)
		backend: backend,
		maxNew:  envInt("FAK_INKERNEL_MAX_TOKENS", 256),
		temp:    envFloat("FAK_INKERNEL_TEMP", 0),
		seed:    int64(envInt("FAK_INKERNEL_SEED", 0)),
	}
	// RadixAttention KV-prefix reuse is ON by default; FAK_INKERNEL_RADIX=off disables it
	// (the A/B "tree OFF" arm). The reuse clone is a CPU session, so it only engages when
	// no device backend is wired — the device path keeps its current full-prefill behavior.
	if os.Getenv("FAK_INKERNEL_RADIX") != "off" && backend == nil {
		p.tree = radixkv.New(envInt("FAK_INKERNEL_RADIX_BUDGET", 0))
	}
	return p
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

	// emit runs per generated token: decode the piece, accumulate the text, and apply the
	// per-request string-suffix Stop (orthogonal to the token-ID stops). Returning true
	// ends the turn with the token counted and its text trimmed (the stop string is not
	// echoed back, matching the HTTP wires). Factoring decode into this closure keeps the
	// token-level reuse/decode core (generateReused) tokenizer-free, so the candidate-#13
	// reuse and #14 eviction are witnessable on a synthetic model with no tokenizer fixture.
	var sb strings.Builder
	emit := func(next int) bool {
		if piece, derr := p.tok.Decode([]int{next}); derr == nil {
			sb.WriteString(piece)
		}
		if trimmed, hit := checkStop(sb.String(), sp.Stop); hit {
			sb.Reset()
			sb.WriteString(trimmed)
			return true
		}
		return false
	}

	gen, promptTok, matched, prefillS, decodeS, stopped := p.generateReused(ids, maxNew, temp, topP, topK, stops, emit)
	// finishReason is honest about WHY decode ended: "stop" when a token-ID stop or a
	// per-request Stop sequence fired, "length" when maxNew was the only limit hit.
	finishReason := "length"
	if stopped {
		finishReason = "stop"
	}

	// Witness line (mirrors cmd/fakchat): real per-turn prefill/decode tok/s through the
	// in-kernel model, now also reporting the RadixAttention prefix reuse (reused vs
	// prompt) so a served chat turn self-reports the candidate-#13 win. prefill tok/s is
	// over the COMPUTED suffix (prompt minus the reused prefix) — the work actually done.
	computed := promptTok - matched
	prefTPS, decTPS := 0.0, 0.0
	if prefillS > 0 {
		prefTPS = float64(computed) / prefillS
	}
	if decodeS > 0 {
		decTPS = float64(gen) / decodeS
	}
	log.Printf("inkernel_chat model=%s q4k=%v prompt=%dtok reused=%dtok prefill=%dtok/%.2fs/%.1ftok/s decode=%dtok/%.2fs/%.1ftok/s",
		p.modelID, p.q4k, promptTok, matched, computed, prefillS, prefTPS, gen, decodeS, decTPS)

	return &Completion{
		Message:      Message{Role: "assistant", Content: sb.String()},
		FinishReason: finishReason,
		Usage:        Usage{PromptTokens: promptTok, CompletionTokens: gen, TotalTokens: promptTok + gen},
	}, nil
}

// generateReused runs prefill + decode for an already-encoded prompt, REUSING the longest
// cached KV prefix (the radix tree) when enabled and FAILING OPEN to a full prefill on a
// miss — the candidate-#13 core, factored out of Complete so the reuse/decode path is
// exercisable on a synthetic model with no tokenizer.
//
// emit is invoked with each generated token id AFTER sampling and BEFORE the next Step;
// returning true stops decode with that token counted (Complete's string-suffix stop
// closes over the tokenizer there). A token-id stop (stops[next]) or next<0 ends decode
// WITHOUT emitting — the served contract that a stop token is not echoed.
//
// SNAPSHOT/LEASE discipline: the full-prompt KV is snapshotted (Cloned) right after
// Prefill — BEFORE the decode loop mutates s.Cache by appending generated positions — and
// inserted under a FRESH Lookup so radixkv's lease handoff (Lookup→Insert→Done) is honored
// entirely inside the lock, with no unexported *node escaping this scope. The reuse clone
// (SessionFromPrefix) is also taken under the lock, so a concurrent eviction of the tree
// node can never race our read of its KV. Returns the generated-token count, the prompt
// length, the reused-prefix length, prefill/decode seconds, and whether a stop (not maxNew)
// ended the turn.
func (p *InKernelPlanner) generateReused(ids []int, maxNew int, temp, topP float64, topK int, stops map[int]bool, emit func(int) bool) (gen, promptTok, matched int, prefillS, decodeS float64, stopped bool) {
	promptTok = len(ids)
	if len(ids) == 0 {
		return
	}
	reuse := p.tree != nil && p.backend == nil

	// 1) Acquire a session, reusing the longest cached KV prefix when enabled. The clone
	// (SessionFromPrefix) happens under the lock, so once we unlock our session owns an
	// independent copy and a concurrent tree eviction cannot affect this turn's decode.
	var s *model.Session
	if reuse {
		p.mu.Lock()
		b, m := p.tree.Lookup(ids)
		if k := b.KV(); k != nil {
			s = p.m.SessionFromPrefix(k) // an independent clone; cache.Len() == m
			matched = m
		}
		p.tree.Done(b) // release the lease — we have our clone (or matched nothing)
		p.mu.Unlock()
		// Fully cached (an exact-duplicate transcript): the cached KV has the prefix but
		// not the last-token logits decode must start from. Fail OPEN to a fresh full
		// prefill rather than truncate (some recurrent architectures refuse Evict); the
		// exact-replay case is not the reuse hot path.
		if s != nil && matched >= len(ids) {
			s, matched = nil, 0
		}
	}
	if s == nil {
		matched = 0
		if p.backend != nil {
			s = p.m.NewBackendSession(p.backend)
			defer s.Close()
		} else {
			s = p.m.NewSession()
		}
	}
	s.Quant = p.quant
	s.Q4K = p.q4k && p.backend == nil // resident-Q4_K is a CPU-only decode path; the device HAL uses Q8/F32

	// 2) Prefill ONLY the divergent suffix (the whole prompt on a miss).
	tp := time.Now()
	logits := s.Prefill(ids[matched:])
	prefillS = time.Since(tp).Seconds()

	// 3) Snapshot the full-prompt KV (before decode mutates s.Cache) and cache it under a
	// fresh Lookup→Insert→Done. The snapshot covers the FULL ids prefix, so it is a valid
	// leaf kv no matter how much a concurrent turn may have inserted since step 1.
	if reuse {
		snap := s.Cache.Clone()
		p.mu.Lock()
		b, m := p.tree.Lookup(ids)
		leaf := p.tree.Insert(b, ids[m:], snap)
		p.tree.Done(leaf)
		p.mu.Unlock()
	}

	// 4) Decode.
	rng := rand.New(rand.NewSource(p.seed))
	td := time.Now()
	for gen = 0; gen < maxNew; gen++ {
		next := sampleLogits(logits, temp, topP, topK, rng)
		if next < 0 || stops[next] {
			stopped = true
			break
		}
		if emit != nil && emit(next) {
			gen++ // this token WAS generated; count it before exiting the loop
			stopped = true
			break
		}
		logits = s.Step(next)
	}
	decodeS = time.Since(td).Seconds()
	return
}

// PoisonEvictor is the narrow seam the gateway drives on a tool-result QUARANTINE: the
// in-kernel KV cache must drop any cached prefix that attended to the now-poisoned result
// (candidate #14), so a later turn re-prefills instead of replaying the poisoned KV. It is
// implemented by InKernelPlanner; the gateway type-asserts its planner to it, so a proxy/
// mock planner — or an in-kernel planner with reuse disabled — simply does not engage it.
type PoisonEvictor interface {
	// EvictPoisoned drops the cached KV prefix along the transcript THROUGH and including
	// messages[throughIdx] (the quarantined tool result, rendered with its ORIGINAL
	// content). Returns the freed token count (0 if nothing was cached / reuse is off).
	EvictPoisoned(messages []Message, throughIdx int) int
}

// EvictPoisoned renders the transcript up to and including the poisoned message — WITHOUT
// the trailing assistant-open marker, so the token path ends exactly on the poison's
// <|im_end|> turn boundary — encodes it, and evicts the cached branch along that path.
// Because each turn ends on the atomic <|im_end|> special token, the encoded partial
// transcript is a genuine token-prefix of any cached turn that contained these leading
// messages, so the walk lands on (and EvictNode drops) the node whose KV saw the poison
// while sparing benign siblings. It is the gateway-facing wrapper over evictPoisonedIDs.
func (p *InKernelPlanner) EvictPoisoned(messages []Message, throughIdx int) int {
	if p.tree == nil || throughIdx < 0 || throughIdx >= len(messages) {
		return 0
	}
	ids, err := p.tok.Encode(renderTranscript(messages[:throughIdx+1]))
	if err != nil || len(ids) == 0 {
		return 0
	}
	freed := p.evictPoisonedIDs(ids)
	if freed > 0 {
		log.Printf("inkernel_chat poison-evict model=%s through_msg=%d freed=%dtok", p.modelID, throughIdx, freed)
	}
	return freed
}

// evictPoisonedIDs drops the cached prefix lying along `ids` (a poisoned transcript token
// path) — the token-level #14 seam EvictPoisoned wraps. Guarded by mu; no-op when reuse
// is disabled.
func (p *InKernelPlanner) evictPoisonedIDs(ids []int) int {
	if p.tree == nil {
		return 0
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.tree.EvictPrefix(ids)
}

// cachedPrefixLen reports how many leading tokens of `ids` are already resident in the
// prefix cache (read-only). It is the reuse-state probe the witnesses assert on; 0 when
// reuse is disabled.
func (p *InKernelPlanner) cachedPrefixLen(ids []int) int {
	if p.tree == nil {
		return 0
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.tree.MatchLen(ids)
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
	return renderTranscript(messages) + "<|im_start|>assistant\n"
}

// renderTranscript renders the messages as complete ChatML turns (system messages folded
// into one leading block) WITHOUT the trailing open assistant turn. renderChatML appends
// that open turn for generation; the poison-eviction path uses the bare transcript so its
// token path ends exactly on a turn boundary (the atomic <|im_end|> special token),
// keeping it a clean token-prefix of any cached turn that began with these messages.
func renderTranscript(messages []Message) string {
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
