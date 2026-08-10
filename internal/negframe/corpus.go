package negframe

// corpus.go is the WHOLE-CORPUS equivalence half of the emit-time stick (#4421). reframe.go
// proves ONE string is safe to reframe (token-superset, idempotent). This file names the
// surface those reframes ride -- the guard-runtime prose fak assembles and injects into a
// session -- and gives the gate that proves a reframe of a WHOLE such string is simultaneously
// lossless, polarity-safe, and semantically close to the original. Positive-state framing only
// earns its place if it is lossless: a guard string may lead with the affordance, but it must
// still carry every reason code, flag, and path a downstream matcher depends on, and it must
// never read MORE negative than the prose it replaced.
//
// Two things live here that the rest of the epic builds on:
//
//   - BroadcastTier tags each hot-path surface by how often the workspace re-reads it (a
//     per-turn guard directive is re-injected every turn; a per-session hint once per session;
//     a cold skill doc only when its verb fires). The negation tax is paid per broadcast, so
//     the tier is the weight paydown should sort by (#4408) and the axis the hot-path debt fold
//     leads with (#4419).
//   - The equivalence gate (Equivalent / equivalenceOf) combines the three properties a
//     reframe of a broadcast string must hold, so one table-driven test can witness the whole
//     corpus at once and a newly added guard string that ships un-reframed or lossily reframed
//     reds CI instead of sliding in green.

import (
	"fmt"
	"math"
	"regexp"
	"strings"
)

// BroadcastTier is how often the workspace re-reads a hot-path surface -- the multiplier on the
// negation tax a reframable negative in that surface costs. Ordered cold < per-session < per-turn
// so a larger tier is a hotter (more re-broadcast) surface.
type BroadcastTier int

const (
	// TierCold: prose read only when a specific verb fires (a skill doc). Paid rarely.
	TierCold BroadcastTier = iota
	// TierPerSession: prose injected once per session (a SessionStart hint, a resume-recovery
	// prompt handed to a resumed session). Paid once per session.
	TierPerSession
	// TierPerTurn: guard-runtime prose re-injected on every turn (a step-advice directive, a
	// refusal note emitted on each offending call). Paid on every turn -- the hottest surface.
	TierPerTurn
)

// Weight is the broadcast multiplier used to order paydown (#4408): a per-turn negative outranks
// an equal per-session one, which outranks an equal cold one. The exact integers are ordinal, not
// physical -- only their ordering is contractual.
func (t BroadcastTier) Weight() int {
	switch t {
	case TierPerTurn:
		return 100
	case TierPerSession:
		return 10
	default:
		return 1
	}
}

// String renders the tier for the card's per-finding provenance.
func (t BroadcastTier) String() string {
	switch t {
	case TierPerTurn:
		return "per-turn"
	case TierPerSession:
		return "per-session"
	default:
		return "cold"
	}
}

// HotPathString is one enumerated guard-runtime surface: a stable Name, the prose fak injects
// (in its pre-Reframe source form), and the broadcast tier that surface is read at.
type HotPathString struct {
	Name string
	Text string
	Tier BroadcastTier
}

// SimilarityFloor is the coarse lexical-cosine floor a reframe must clear against its original
// (property (1) of the gate). It is intentionally lenient: the token-superset and polarity checks
// do the precise work, while this floor is the backstop that catches a GROSS drift -- a future
// rule that rewrites a whole sentence into something unrelated, or truncates it -- that the other
// two checks would miss. It is a bag-of-words cosine, a deterministic lexical proxy for semantic
// closeness, not an embedding: negframe stays stdlib + pkg/scorecard, so the "semantic" floor is
// honestly a lexical one.
const SimilarityFloor = 0.45

// EquivalenceVerdict is the outcome of gating one reframe against its original. OK is the closed
// admit bit; Reason names the first failing property (or the pass condition) for a legible test
// failure and a card cell.
type EquivalenceVerdict struct {
	Name         string  `json:"name"`
	Original     string  `json:"original"`
	Reframed     string  `json:"reframed"`
	TokenSafe    bool    `json:"token_safe"`    // (2) every must-keep token of the original survives
	PolaritySafe bool    `json:"polarity_safe"` // (3) the reframe reads no MORE negative than the original
	Similarity   float64 `json:"similarity"`    // (1) lexical cosine between original and reframe
	SimilarityOK bool    `json:"similarity_ok"`
	OK           bool    `json:"ok"`
	Reason       string  `json:"reason"`
}

// Equivalent runs the emit-time Reframe on text and gates the result: it is the single-string
// entry point a call site (or a binding test over the live guard symbols) uses to assert its
// injected prose is safe to reframe. A string with nothing to reframe returns OK with Reframed
// == text (the identity is trivially equivalent to itself).
func Equivalent(text string) EquivalenceVerdict {
	return equivalenceOf("", text, Reframe(text))
}

// equivalenceOf is the pure three-property comparator behind Equivalent, split out so the tests
// can drive it with a HAND-TAMPERED candidate (a token dropped, a negative injected, the text
// truncated) and prove each property independently has teeth -- Reframe itself is token-safe by
// construction, so a lossy candidate cannot be produced by Reframe and must be synthesized.
//
// The three properties, checked in refusal-first order so Reason names the most contract-breaking
// failure:
//
//	(2) TOKEN-SUPERSET  -- candidate keeps every must-keep token of orig at full multiplicity.
//	(3) POLARITY-SAFE   -- candidate carries no MORE negative findings than orig (never flips
//	                       toward the negative pole; a reframe may only hold or reduce negativity).
//	(1) SIMILARITY      -- lexical cosine(orig, candidate) >= SimilarityFloor (gross-drift backstop).
func equivalenceOf(name, orig, candidate string) EquivalenceVerdict {
	v := EquivalenceVerdict{Name: name, Original: orig, Reframed: candidate}
	v.TokenSafe = tokenSuperset(mustKeepSet(orig), mustKeepSet(candidate))
	v.PolaritySafe = negativity(candidate) <= negativity(orig)
	v.Similarity = lexicalCosine(orig, candidate)
	v.SimilarityOK = v.Similarity >= SimilarityFloor
	switch {
	case !v.TokenSafe:
		v.Reason = "dropped a must-keep contract token"
	case !v.PolaritySafe:
		v.Reason = "reframe reads more negative than the original"
	case !v.SimilarityOK:
		v.Reason = fmt.Sprintf("lexical similarity %.2f below floor %.2f", v.Similarity, SimilarityFloor)
	default:
		v.OK = true
		v.Reason = "reframed, token-preserving, polarity-safe"
	}
	return v
}

// CorpusEquivalence gates the Reframe of every surface in corpus, preserving order, so one call
// yields the whole-corpus witness the harness asserts over.
func CorpusEquivalence(corpus []HotPathString) []EquivalenceVerdict {
	out := make([]EquivalenceVerdict, 0, len(corpus))
	for _, hp := range corpus {
		v := equivalenceOf(hp.Name, hp.Text, Reframe(hp.Text))
		out = append(out, v)
	}
	return out
}

// negativity is the polarity metric: the count of negatively-framed findings (both tiers) in s.
// A reframe that removes a mechanical negative lowers it; one that leaves judgement-tier prose in
// place holds it; a tampered candidate that injects a fresh "do not" raises it (and is refused).
func negativity(s string) int { return len(Classify("", s)) }

// wordRE tokenizes prose into word runs for the lexical-cosine similarity (letters and digits;
// punctuation and whitespace are separators). ALL-CAPS reason tokens survive as their own terms,
// so a candidate that drops one also loses similarity -- but the token-superset check is the
// authoritative guard for those; similarity is the gross-drift backstop.
var wordRE = regexp.MustCompile(`[\p{L}\p{N}]+`)

// termFreq is the lowercased bag-of-words term-frequency vector of s.
func termFreq(s string) map[string]int {
	m := map[string]int{}
	for _, w := range wordRE.FindAllString(strings.ToLower(s), -1) {
		m[w]++
	}
	return m
}

// lexicalCosine is the cosine similarity of the two texts' term-frequency vectors, in [0,1]. Two
// empty texts are identical (1); one empty and one not share no terms (0). A small-span reframe
// ("do not forget to X" -> "remember to X") keeps most terms, so it scores high; a wholesale
// rewrite or a truncation scores low.
func lexicalCosine(a, b string) float64 {
	va, vb := termFreq(a), termFreq(b)
	if len(va) == 0 && len(vb) == 0 {
		return 1
	}
	if len(va) == 0 || len(vb) == 0 {
		return 0
	}
	var dot, na, nb float64
	for t, ca := range va {
		na += float64(ca * ca)
		if cb, ok := vb[t]; ok {
			dot += float64(ca * cb)
		}
	}
	for _, cb := range vb {
		nb += float64(cb * cb)
	}
	if na == 0 || nb == 0 {
		return 0
	}
	r := dot / (math.Sqrt(na) * math.Sqrt(nb))
	if r > 1 { // float rounding can nudge an identical-vector cosine just past 1
		r = 1
	}
	return r
}

// GuardRuntimeCorpus enumerates the guard-runtime injected-prose surface as reframe-equivalence
// fixtures -- the SHAPES fak assembles and pushes to the model, each tagged with the tier it is
// broadcast at. These mirror the live emit sites in cmd/fak (guard_sessionstart.go's
// guardSessionStartHint, guard_carryforward.go's guardRecoveryPrompt); the LIVE bytes are bound
// to this gate by the equivalence test in cmd/fak, which reruns Equivalent over the actual symbols
// so an emit-site edit that this fixture list has not caught up to still reds there. Kept here so
// the negframe package -- which owns the reframe contract -- can witness the whole surface with no
// dependency on package main.
func GuardRuntimeCorpus() []HotPathString {
	return []HotPathString{
		{
			Name: "sessionstart-affordance",
			Tier: TierPerSession,
			Text: "fak substrate available (MCP server `fak`): before working as a generic coder, reach for the fak verbs. Call `mcp__fak__fak_capabilities` to discover the task-scoped toolbelt; `mcp__fak__fak_admit` / `mcp__fak__fak_adjudicate` to gate/execute a tool call through the kernel; `mcp__fak__fak_memory_run` for durable memory; `mcp__fak__fak_tools_search` to page in the rest. These are deferred tools — you must invoke them explicitly, they will not auto-load.",
		},
		{
			// A resume-recovery prompt whose interpolated per-reason fix carries a mechanical idiom
			// ("do not forget to stamp the commit"): the gate must both flip it to the positive voice
			// AND keep the OFF_TRUNK reason token, so this fixture exercises a real reframe, not a
			// no-op. TierPerSession -- injected once into the resumed session.
			Name: "resume-recovery-prompt",
			Tier: TierPerSession,
			Text: "[fak] resume recovery: the previous guarded run recorded capability-floor refusal(s). Treat this resumed turn as recovery/debugging, not a blind retry. Clear the blocker or choose an allowed alternative before continuing. Prior refusal(s): OFF_TRUNK x1 (fix: do not forget to stamp the commit). Keep fak guard wrapped after the blocker is cleared.",
		},
		{
			// A per-turn refusal note lead: emitted on every offending tool call, the hottest surface.
			// Leads with the permitted path, keeps the POLICY_BLOCK reason token.
			Name: "refusal-note-lead",
			Tier: TierPerTurn,
			Text: "fak guard held this call (POLICY_BLOCK): route the write through `fak commit` on a wrapped branch to proceed.",
		},
	}
}
