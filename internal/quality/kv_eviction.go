package quality

import (
	"fmt"
	"strings"
)

// kv_eviction.go is the KV-eviction parity child of the quality spine (#4535):
// evicting KV-cache entries and recomputing them on demand must be output-invariant
// — the decode must equal a never-evicted decode token for token. This file models
// a tiny deterministic attention decode where every emitted token is a digest of
// ALL prior positions' KV cells, runs it once with a full (never-evicted) cache as
// the reference and once through a sliding-window cache that evicts old positions
// and recomputes them on demand as the engine, and registers a differential oracle
// that pins the FIRST token whose value depended on a lost or miscomputed evicted
// position.

// kvEvictionWindow is the sliding-window capacity of the evicting engine's KV
// cache: only the most recent kvEvictionWindow positions stay resident; anything
// older is evicted and must be recomputed when a later step attends to it.
const kvEvictionWindow = 3

// kvEvictionFirstMissStep is the first decode step that attends to an evicted
// position: at step i the engine attends to positions 0..i-1 but only the last
// kvEvictionWindow of them are resident, so the first cache miss happens at step
// kvEvictionWindow+1 (position 0 has just been evicted). It is mid-sequence by
// construction, so the passing prefix proves the oracle's localization is doing
// work — a recompute defect fails HERE, not at token 0.
const kvEvictionFirstMissStep = kvEvictionWindow + 1

// kvEvictionFoldSeed is the initial attention-fold state (FNV-1a offset basis).
const kvEvictionFoldSeed uint64 = 0xcbf29ce484222325

// kvEvictionCell is the KV content of one position: a pure splitmix64-style mix of
// (seed, position). Because a cell is a pure function of its inputs — no carried
// state, no ambient entropy — an on-demand recompute of an evicted position can
// reproduce it EXACTLY, which is precisely the parity contract the oracle enforces.
func kvEvictionCell(seed int64, pos int) uint64 {
	z := uint64(seed)*0x9e3779b97f4a7c15 + (uint64(pos)+1)*0xd1b54a32d192ed03
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// kvEvictionFoldStep folds one attended cell into the running attention state
// (FNV-1a style: xor then multiply, order-sensitive).
func kvEvictionFoldStep(fold, cell uint64) uint64 {
	return (fold ^ cell) * 0x100000001b3
}

// kvEvictionToken renders step i's emitted token as a digest of the full attention
// fold over positions 0..i-1. Carrying all 64 fold bits in the token means any
// change to any attended cell's value provably changes the token — a recompute
// defect cannot alias back to the reference stream at the step it corrupts.
func kvEvictionToken(i int, fold uint64) string {
	return fmt.Sprintf("p%d:%016x", i, fold)
}

// kvEvictionDecodeFull is the never-evicted reference decode: every position's KV
// cell stays resident, and token i digests the fold over cells 0..i-1.
func kvEvictionDecodeFull(seed int64, steps int) Trace {
	cells := make([]uint64, 0, steps)
	toks := make([]string, 0, steps)
	for i := 0; i < steps; i++ {
		fold := kvEvictionFoldSeed
		for _, c := range cells {
			fold = kvEvictionFoldStep(fold, c)
		}
		toks = append(toks, kvEvictionToken(i, fold))
		cells = append(cells, kvEvictionCell(seed, i))
	}
	return Trace{Tokens: toks, Text: strings.Join(toks, " ")}
}

// kvEvictionDecodeEvicting is the evicting decode path: a sliding-window cache
// keeps the last kvEvictionWindow positions; an attended position that has been
// evicted is recomputed on demand via recompute (and NOT re-admitted — recompute,
// use, discard). It returns the trace plus the number of on-demand recomputes so a
// test can prove eviction actually fired (a parity pass over a cache that never
// evicted would be vacuous). With recompute = kvEvictionCell this is
// output-identical to kvEvictionDecodeFull by construction; the mutants below
// differ ONLY in recompute.
func kvEvictionDecodeEvicting(seed int64, steps int, recompute func(int64, int) uint64) (Trace, int) {
	cache := map[int]uint64{}
	recomputes := 0
	toks := make([]string, 0, steps)
	for i := 0; i < steps; i++ {
		fold := kvEvictionFoldSeed
		for p := 0; p < i; p++ {
			cell, ok := cache[p]
			if !ok {
				cell = recompute(seed, p)
				recomputes++
			}
			fold = kvEvictionFoldStep(fold, cell)
		}
		toks = append(toks, kvEvictionToken(i, fold))
		cache[i] = kvEvictionCell(seed, i)
		for p := range cache {
			if p <= i-kvEvictionWindow {
				delete(cache, p)
			}
		}
	}
	return Trace{Tokens: toks, Text: strings.Join(toks, " ")}, recomputes
}

// kvEvictionRunner is the evicting-engine adapter: it decodes the case under the
// sliding-window cache, recomputing evicted positions with a faithful or defective
// recompute. It is the ScriptedRunner-style seam a real paged/evicting KV engine
// wires in behind — judged only on its emitted tokens, never its self-description.
type kvEvictionRunner struct {
	Label  string
	defect string
}

func (r kvEvictionRunner) Name() string {
	if r.Label != "" {
		return r.Label
	}
	return "kv-evicting-engine"
}

func (r kvEvictionRunner) Run(c QualityCase) (Trace, error) {
	recompute := kvEvictionCell
	switch r.defect {
	case "lost-position":
		// The content-loss defect: eviction dropped the position and the
		// "recompute" has nothing to rebuild it from — it returns an empty cell.
		recompute = func(int64, int) uint64 { return 0 }
	case "stale-recompute":
		// The miscompute defect: the recompute reconstructs the position from the
		// wrong offset (off by one), a paging/indexing bug shape.
		recompute = func(seed int64, pos int) uint64 { return kvEvictionCell(seed, pos+1) }
	}
	t, _ := kvEvictionDecodeEvicting(c.Params.Seed, c.Params.MaxTokens, recompute)
	t.Runner = r.Name()
	return t, nil
}

// KVEvictionEngine returns an evicting engine runner with an optional injected
// recompute defect: "" recomputes evicted positions faithfully (parity holds);
// "lost-position" loses an evicted position's content; "stale-recompute"
// reconstructs evicted positions from the wrong offset. Both defects first corrupt
// the fold at the first step that attends to an evicted position, so the oracle
// fails at exactly kvEvictionFirstMissStep — the token that depended on it.
func KVEvictionEngine(defect string) kvEvictionRunner {
	switch defect {
	case "lost-position":
		return kvEvictionRunner{Label: "engine-kv-lost-position", defect: defect}
	case "stale-recompute":
		return kvEvictionRunner{Label: "engine-kv-stale-recompute", defect: defect}
	default:
		return kvEvictionRunner{Label: "engine-kv-evicting-clean"}
	}
}

// KVEvictionCase builds the eviction-parity case: a greedy (temperature-zero)
// decode long enough that the engine's sliding window MUST evict and recompute
// (MaxTokens > kvEvictionWindow+1), with the reference trace produced by the
// never-evicted full-KV decode under the same seed.
func KVEvictionCase(seed int64) QualityCase {
	params := SamplingParams{Temperature: 0, MaxTokens: 8, Seed: seed}
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "kv-eviction-parity-demo",
		Version:   1,
		Prompt:    "Decode with KV eviction and on-demand recomputation; output must match the never-evicted decode.",
		Params:    params,
		Reference: kvEvictionDecodeFull(seed, params.MaxTokens),
		Oracles:   []string{"kv-eviction-parity"},
	}
}

// KVEvictionParity is the differential oracle for KV eviction+recomputation
// (#4535): the evicting engine's token stream must equal the never-evicted
// reference stream exactly, because a faithful recompute rebuilds each evicted
// position bit-for-bit. Any mismatch — a lost position, a miscomputed recompute, a
// truncated decode — is reported as the FIRST divergence, so "eviction corrupted
// the decode" localizes to the first token that depended on the bad position.
type KVEvictionParity struct{}

func (KVEvictionParity) Name() string { return "kv-eviction-parity" }
func (KVEvictionParity) Kind() string { return "differential" }

func init() { Register(KVEvictionParity{}) }

func (KVEvictionParity) Judge(ref, eng Trace, _ QualityCase) Verdict {
	v := Verdict{Oracle: "kv-eviction-parity", Kind: "differential", Pass: true}
	n := len(ref.Tokens)
	if len(eng.Tokens) < n {
		n = len(eng.Tokens)
	}
	for i := 0; i < n; i++ {
		if ref.Tokens[i] != eng.Tokens[i] {
			v.Pass = false
			v.FirstDivergence = &Divergence{Index: i, Reference: ref.Tokens[i], Engine: eng.Tokens[i]}
			v.Detail = fmt.Sprintf("evicted-KV decode diverged at token %d: reference (full KV) %q, engine (evict+recompute) %q — a recomputed position feeding this step lost or miscomputed its content",
				i, ref.Tokens[i], eng.Tokens[i])
			return v
		}
	}
	if len(ref.Tokens) != len(eng.Tokens) {
		v.Pass = false
		v.FirstDivergence = &Divergence{Index: n, Reference: tokenAt(ref.Tokens, n), Engine: tokenAt(eng.Tokens, n)}
		v.Detail = fmt.Sprintf("evicted-KV decode length diverged at %d: reference has %d tokens, engine has %d",
			n, len(ref.Tokens), len(eng.Tokens))
		return v
	}
	v.Detail = fmt.Sprintf("evict+recompute decode matched the never-evicted reference for all %d tokens", len(ref.Tokens))
	return v
}
