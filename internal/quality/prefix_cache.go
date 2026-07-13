package quality

import (
	"fmt"
	"strings"
)

// prefix_cache.go is the prefix-cache parity child of the quality spine (#4534):
// serving a prompt through the prefix cache must be a pure latency optimization —
// cache-ON output for a prompt must equal cache-OFF output token for token. The
// reference path decodes with the cache off; the engine path decodes with the
// cache on, reusing a warmed entry. The defect class this gate exists for is the
// STALE/MISMATCHED cached prefix: a cache key that ignores a differing prompt
// suffix serves output computed for ANOTHER prompt, so the decode is faithful over
// the shared prefix and silently wrong from the first suffix-influenced token —
// exactly the fluent-but-wrong shape a downstream text metric averages away.

// pcVocab is the small fixed vocabulary the model decode draws from. Eight entries
// keep the token space tiny while making an accidental full-stream collision
// between two different prompts' decodes vanishingly unlikely over the step budget.
var pcVocab = []string{"amber", "birch", "cobalt", "dune", "echo", "fjord", "garnet", "haze"}

const (
	// pcDecodeSteps is the fixed generation budget the demo case is pinned to, so
	// the cached and fresh paths emit the same number of tokens.
	pcDecodeSteps = 8
	// pcSharedPrefixWords is how many leading prompt words the case and stale
	// prompts share — and, in the defective engine, how many words the broken
	// cache key hashes (every word after them is ignored).
	pcSharedPrefixWords = 4
	// pcCasePrompt is the prompt the case decodes; pcStalePrompt is the earlier
	// prompt the defective engine's cache was warmed with. They share exactly the
	// first pcSharedPrefixWords words and differ at word index 4 ("july" vs
	// "june"), so a suffix-blind cache key cannot tell them apart.
	pcCasePrompt  = "summarize throughput for the july window"
	pcStalePrompt = "summarize throughput for the june window"
	// pcStaleDivergeStep is the first token influenced by the differing suffix
	// word: token i is drawn from the state after words[0..i], so the stale trace
	// matches the case decode through token 3 and departs at token 4.
	pcStaleDivergeStep = 4
)

// pcMix is the splitmix64 finalizer — a bijection on uint64, so two different
// fold states can never mix to the same draw at the same step.
func pcMix(z uint64) uint64 {
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// pcHashWord is FNV-1a 64 over the word's bytes: the per-word input to the
// prompt-processing fold.
func pcHashWord(w string) uint64 {
	h := uint64(0xcbf29ce484222325)
	for i := 0; i < len(w); i++ {
		h ^= uint64(w[i])
		h *= 0x100000001b3
	}
	return h
}

// pcStates returns the cumulative prompt-processing state after each word:
// states[i] is the fold of words[0..i]. This is the quantity a real prefix cache
// stores — the state after processing a prompt prefix — and the quantity a stale
// entry silently substitutes with another prompt's.
func pcStates(words []string) []uint64 {
	states := make([]uint64, len(words))
	state := uint64(0x9e3779b97f4a7c15)
	for i, w := range words {
		state = pcMix(state ^ pcHashWord(w))
		states[i] = state
	}
	return states
}

// pcDecode is the cache-OFF golden decode: fold the prompt word by word, then emit
// steps tokens where token i is drawn from the state after the first i+1 words
// (clamped to the full prompt). Early tokens therefore depend only on the prompt
// PREFIX and later tokens on the full prompt — the structure a prefix cache
// exploits, and the structure a mismatched cached prefix silently corrupts. It is
// a pure function of (prompt, steps): same inputs, byte-identical Trace.
func pcDecode(prompt string, steps int) Trace {
	words := strings.Fields(prompt)
	states := pcStates(words)
	if len(states) == 0 {
		states = []uint64{0x9e3779b97f4a7c15}
	}
	toks := make([]string, 0, steps)
	for i := 0; i < steps; i++ {
		idx := i
		if idx >= len(states) {
			idx = len(states) - 1
		}
		draw := pcMix(states[idx] + uint64(i+1)*0x9e3779b97f4a7c15)
		toks = append(toks, pcVocab[draw%uint64(len(pcVocab))])
	}
	return Trace{Tokens: toks, Text: strings.Join(toks, " ")}
}

// pcCacheKey derives the cache key for a prompt. A strict key covers the FULL
// prompt, so only a byte-identical prompt can hit a warmed entry. The defective
// key hashes only the first pcSharedPrefixWords words — it ignores a differing
// suffix, which is precisely the injected defect: two different prompts collide
// onto one cache slot and one prompt is served the other's output.
func pcCacheKey(prompt string, strict bool) string {
	words := strings.Fields(prompt)
	if !strict && len(words) > pcSharedPrefixWords {
		words = words[:pcSharedPrefixWords]
	}
	return strings.Join(words, " ")
}

// pcCachedRunner is the cache-ON engine path: it consults its prefix cache before
// decoding, serves a hit verbatim, and decodes fresh on a miss. hits counts served
// cache hits so a test can prove a green run actually exercised reuse (a pass on a
// cache miss would not witness the cache at all).
type pcCachedRunner struct {
	label  string
	strict bool
	cache  map[string]Trace
	hits   *int
}

func (r pcCachedRunner) Name() string {
	if r.label != "" {
		return r.label
	}
	return "engine-prefix-cache"
}

func (r pcCachedRunner) Run(c QualityCase) (Trace, error) {
	if hit, ok := r.cache[pcCacheKey(c.Prompt, r.strict)]; ok {
		if r.hits != nil {
			*r.hits++
		}
		hit.Runner = r.Name()
		return hit, nil
	}
	t := pcDecode(c.Prompt, c.Params.MaxTokens)
	t.Runner = r.Name()
	return t, nil
}

// pcEngine returns a cache-on engine runner with an optional injected defect:
// "" is the faithful path — the cache was warmed by an earlier decode of the SAME
// prompt under a strict full-prompt key, so the run is a true cache hit that must
// still match cache-off exactly; "cold" is a faithful engine with an empty cache
// (a miss decodes fresh); "stale-prefix" is the defect — the cache was warmed by
// a DIFFERENT prompt sharing only the first pcSharedPrefixWords words, and the
// suffix-blind key collides the two prompts, so the warm entry is served as if it
// were a full-prompt hit. This is the deterministic mutant source the tests use
// to prove the parity gate trips.
func pcEngine(defect string) pcCachedRunner {
	hits := new(int)
	switch defect {
	case "stale-prefix":
		warm := pcDecode(pcStalePrompt, pcDecodeSteps)
		return pcCachedRunner{
			label:  "engine-cache-stale-prefix",
			strict: false,
			cache:  map[string]Trace{pcCacheKey(pcStalePrompt, false): warm},
			hits:   hits,
		}
	case "cold":
		return pcCachedRunner{label: "engine-cache-cold", strict: true, cache: map[string]Trace{}, hits: hits}
	default:
		warm := pcDecode(pcCasePrompt, pcDecodeSteps)
		return pcCachedRunner{
			label:  "engine-cache-hit",
			strict: true,
			cache:  map[string]Trace{pcCacheKey(pcCasePrompt, true): warm},
			hits:   hits,
		}
	}
}

// pcCase builds the prefix-cache parity case: a temperature-zero decode of
// pcCasePrompt whose reference trace IS the cache-off decode, judged by the
// prefix-cache-parity oracle. Params pin the step budget so the cached and fresh
// paths generate the same number of tokens.
func pcCase() QualityCase {
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "prefix-cache-parity-demo",
		Version:   1,
		Prompt:    pcCasePrompt,
		Params:    SamplingParams{Temperature: 0, MaxTokens: pcDecodeSteps},
		Reference: pcDecode(pcCasePrompt, pcDecodeSteps),
		Oracles:   []string{"prefix-cache-parity"},
	}
}

// pcParity is the differential oracle for prefix-cache parity (#4534): the
// cache-on token stream must equal the cache-off reference stream exactly,
// because the cache is a latency optimization with no license to change output.
// Any mismatch — a stale cached prefix, a key collision, a truncated cached
// entry — is reported as the FIRST divergence, so "the cache changed the answer"
// localizes to "token 4 was 'echo' where cache-off emitted 'dune'".
type pcParity struct{}

func (pcParity) Name() string { return "prefix-cache-parity" }
func (pcParity) Kind() string { return "differential" }

func (pcParity) Judge(ref, eng Trace, _ QualityCase) Verdict {
	v := Verdict{Oracle: "prefix-cache-parity", Kind: "differential", Pass: true}
	n := len(ref.Tokens)
	if len(eng.Tokens) < n {
		n = len(eng.Tokens)
	}
	for i := 0; i < n; i++ {
		if ref.Tokens[i] != eng.Tokens[i] {
			v.Pass = false
			v.FirstDivergence = &Divergence{Index: i, Reference: ref.Tokens[i], Engine: eng.Tokens[i]}
			v.Detail = fmt.Sprintf("cache-on decode diverged from cache-off at token %d: reference %q, engine %q — the cached prefix did not match this prompt",
				i, ref.Tokens[i], eng.Tokens[i])
			return v
		}
	}
	if len(ref.Tokens) != len(eng.Tokens) {
		v.Pass = false
		v.FirstDivergence = &Divergence{Index: n, Reference: tokenAt(ref.Tokens, n), Engine: tokenAt(eng.Tokens, n)}
		v.Detail = fmt.Sprintf("cache-on decode length diverged from cache-off at %d: reference has %d tokens, engine has %d",
			n, len(ref.Tokens), len(eng.Tokens))
		return v
	}
	v.Detail = fmt.Sprintf("cache-on output matched cache-off exactly over %d tokens", len(ref.Tokens))
	return v
}

func init() {
	Register(pcParity{})
}
