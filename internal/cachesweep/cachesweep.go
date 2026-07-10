// Package cachesweep turns fak's radixkv prefix-cache engine into a budget→reuse
// SWEEP: replay ONE recorded prefix-access trace at each of N cached-token budgets
// PLUS one unbounded pass, and report the reuse-vs-budget curve, the infinite-cache
// theoretical ceiling, and the smallest budget that reaches 99% of it (the ROI knee).
//
// This is the tair-kvcache borrow filed as #3952 (see
// docs/notes/tair-kvcache-borrow-study-2026-07-10.md). The load-bearing upstream trick
// is the infinite-capacity warmup pass used as the hit-rate ceiling, with the finite
// sweep early-stopping at 99% of it — so `--compact-history-budget` and the radixkv LRU
// budget can be sized from EVIDENCE (a curve + a knee) instead of intuition.
//
// The heavy lifting — longest-prefix matching, LRU-leaf eviction, edge splits — is NOT
// reinvented here: every pass runs the proven internal/radixkv engine in pure-accounting
// mode (kv=nil). This package only drives that engine across budgets and folds the
// realized reuse into a curve. radixkv already exposes the exact measurement seam this
// needs: Lookup returns the matched-prefix length against the pre-admission tree state.
//
// Purity: no clock, no network, no filesystem. The only notion of time — used solely by
// the optional write-delay overlay — is carried IN the trace (Access.TimeNs), never read
// from a real clock, so a given (trace, options) pair always yields byte-identical output.
package cachesweep

import (
	"sort"

	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// DefaultKneeFraction is the ROI-knee threshold as a fraction of the infinite-cache
// ceiling: the smallest budget whose realized reuse reaches this fraction of the ceiling
// is the point past which more cache buys almost nothing. 0.99 matches the upstream
// early-stop.
const DefaultKneeFraction = 0.99

// Access is one prefix-access in a replay trace: the token-id sequence the request
// presented to the cache, plus a logical timestamp used ONLY by the write-delay overlay.
// TimeNs is named for the upstream write_delay_ns knob but is unitless here — it just has
// to be consistent with Options.WriteDelayNs (the CLI defaults it to the access ordinal
// when a trace carries no timestamps, so a delay is then measured in access-steps).
type Access struct {
	Tokens []int `json:"tokens"`
	TimeNs int64 `json:"t_ns,omitempty"`
}

// Trace is an ordered sequence of prefix-accesses — the replay workload. The order is
// significant: it is the arrival order the cache would have seen.
type Trace struct {
	Accesses []Access `json:"accesses"`
}

// TotalTokens is the sum of requested tokens across the trace (the reuse-ratio denominator).
func (tr Trace) TotalTokens() int64 {
	var n int64
	for _, a := range tr.Accesses {
		n += int64(len(a.Tokens))
	}
	return n
}

// Point is one (budget → realized reuse) sample on the sweep curve. Budget 0 denotes the
// unbounded pass (the infinite-cache ceiling).
type Point struct {
	Budget       int     `json:"budget"`        // cached-token LRU budget (0 = unbounded)
	ReuseRatio   float64 `json:"reuse_ratio"`   // ReusedTokens / TotalTokens
	ReusedTokens int64   `json:"reused_tokens"` // tokens served from cache (Σ matched prefix len)
	TotalTokens  int64   `json:"total_tokens"`  // tokens requested (Σ access len)
	Evictions    int     `json:"evictions"`     // LRU leaf evictions performed at this budget
}

// Result is the full sweep report: the finite-budget curve, the infinite-cache ceiling,
// and the ROI knee. It is JSON-serializable (the CLI emits it verbatim under --json).
type Result struct {
	Curve        []Point `json:"curve"`          // one Point per finite budget, ascending
	Ceiling      Point   `json:"ceiling"`        // the unbounded (budget=0) pass — the hit-rate ceiling
	Knee         Point   `json:"knee"`           // smallest budget reaching KneeFraction*ceiling (zero value if none)
	KneeReached  bool    `json:"knee_reached"`   // false → no finite budget hit the threshold (or ceiling is 0)
	KneeFraction float64 `json:"knee_fraction"`  // the threshold applied (e.g. 0.99)
	WriteDelayNs int64   `json:"write_delay_ns"` // the visibility-latency overlay applied (0 = disabled)
	Accesses     int     `json:"accesses"`       // trace length
	TotalTokens  int64   `json:"total_tokens"`   // Σ access len (constant across budgets)
}

// Options configure a sweep.
type Options struct {
	// Budgets are the finite cached-token budgets to replay. They are deduped, sorted
	// ascending, and non-positive values are dropped (budget 0 means "unbounded" to
	// radixkv and is already covered by the ceiling pass).
	Budgets []int
	// KneeFraction is the knee threshold as a fraction of the ceiling; <=0 uses
	// DefaultKneeFraction (0.99).
	KneeFraction float64
	// WriteDelayNs optionally models a KV write-delay window: a re-request that arrives
	// before its prefix's cache became visible (first-write time + this delay) counts the
	// not-yet-ready portion as a MISS even though it is structurally resident. 0 disables
	// the overlay (the default — pure structural reuse).
	WriteDelayNs int64
}

// Sweep replays the trace at each finite budget plus one unbounded pass and folds the
// realized reuse into a curve + ceiling + knee. It is pure and deterministic: identical
// (trace, options) inputs always produce an identical Result.
func Sweep(tr Trace, opt Options) Result {
	kneeFrac := opt.KneeFraction
	if kneeFrac <= 0 {
		kneeFrac = DefaultKneeFraction
	}
	res := Result{
		KneeFraction: kneeFrac,
		WriteDelayNs: opt.WriteDelayNs,
		Accesses:     len(tr.Accesses),
		TotalTokens:  tr.TotalTokens(),
	}

	// The infinite-cache ceiling: one unbounded pass (budget 0 disables eviction), so the
	// only misses are cold prefixes and — when enabled — the write-delay window.
	res.Ceiling = replay(tr, 0, opt.WriteDelayNs)

	budgets := normalizeBudgets(opt.Budgets)
	res.Curve = make([]Point, 0, len(budgets))
	threshold := kneeFrac * res.Ceiling.ReuseRatio
	for _, b := range budgets {
		p := replay(tr, b, opt.WriteDelayNs)
		res.Curve = append(res.Curve, p)
		// Smallest budget crossing the knee wins: budgets ascend, so the first crossing
		// is the smallest. A zero ceiling has no ROI curve, so there is no knee.
		if !res.KneeReached && res.Ceiling.ReuseRatio > 0 && p.ReuseRatio >= threshold {
			res.Knee = p
			res.KneeReached = true
		}
	}
	return res
}

// replay runs a single pass of the trace through a fresh radixkv tree at the given budget
// (0 = unbounded) and returns the realized-reuse Point. It measures reuse from the matched
// length Lookup returns against the PRE-admission tree — exactly SGLang's hit-rate seam —
// then admits the full request (Insert of the suffix) so later accesses can reuse it.
func replay(tr Trace, budget int, writeDelayNs int64) Point {
	tree := radixkv.New(budget)
	var reused, total int64
	var fw firstWrite
	if writeDelayNs > 0 {
		fw = firstWrite{}
	}
	for _, acc := range tr.Accesses {
		toks := acc.Tokens
		// Lookup leases the matched boundary and returns the structural match against the
		// tree as it stood before this request — the realized reuse this request would get.
		b, m := tree.Lookup(toks)
		match := m
		if writeDelayNs > 0 {
			// The overlay can only SHRINK reuse (a resident-but-not-yet-visible prefix),
			// never grow it — residency is still owned by radixkv (min of the two).
			match = fw.readyMatch(toks, m, acc.TimeNs, writeDelayNs)
		}
		reused += int64(match)
		total += int64(len(toks))
		// Admit the full request: attach the suffix beyond the STRUCTURAL match, hand the
		// lease to the new leaf, then release it. Insert enforces the LRU budget.
		leaf := tree.Insert(b, toks[m:], nil)
		tree.Done(leaf)
		if writeDelayNs > 0 {
			fw.record(toks, acc.TimeNs)
		}
	}
	return Point{
		Budget:       budget,
		ReuseRatio:   ratio(reused, total),
		ReusedTokens: reused,
		TotalTokens:  total,
		Evictions:    tree.Stats().Evictions,
	}
}

// normalizeBudgets dedupes, drops non-positive entries, and sorts ascending.
func normalizeBudgets(in []int) []int {
	seen := make(map[int]bool, len(in))
	out := make([]int, 0, len(in))
	for _, b := range in {
		if b <= 0 || seen[b] {
			continue
		}
		seen[b] = true
		out = append(out, b)
	}
	sort.Ints(out)
	return out
}

func ratio(reused, total int64) float64 {
	if total <= 0 {
		return 0
	}
	return float64(reused) / float64(total)
}

// --- write-delay overlay -----------------------------------------------------------
//
// firstWrite maps a token-prefix (by rolling FNV-1a hash) to the EARLIEST logical time
// any request wrote a path covering it. A prefix becomes reusable writeDelayNs after that
// first write; a re-request arriving inside that window counts the covered portion as a
// miss. This is a pure VISIBILITY overlay layered on top of radixkv's residency — it never
// changes what the tree caches or evicts, only how much of a structural match is counted.

type firstWrite map[uint64]int64

const (
	fnvOffset64 = 1469598103934665603
	fnvPrime64  = 1099511628211
)

// prefixHashes returns the rolling FNV-1a hash of toks[:k] for every k in 1..len(toks),
// so hs[k-1] identifies the k-token prefix. Equal prefixes across different accesses hash
// identically, which is what lets record/readyMatch agree on "the same prefix".
func prefixHashes(toks []int) []uint64 {
	hs := make([]uint64, len(toks))
	var h uint64 = fnvOffset64
	for i, t := range toks {
		u := uint64(int64(t)) // token ids are small non-negative ints; fold all 8 bytes
		for b := 0; b < 8; b++ {
			h ^= (u >> (8 * b)) & 0xff
			h *= fnvPrime64
		}
		hs[i] = h
	}
	return hs
}

// record stamps the first-write time for every prefix length of an admitted request
// (min-wins, so out-of-order timestamps still resolve to the earliest write).
func (fw firstWrite) record(toks []int, t int64) {
	for _, h := range prefixHashes(toks) {
		if prev, ok := fw[h]; !ok || t < prev {
			fw[h] = t
		}
	}
}

// readyMatch returns the longest prefix length L (≤ structuralMatch) whose cache had
// become visible by time t — i.e. firstWrite(L) + delay ≤ t. firstWrite is non-decreasing
// in L along a real path (a deeper prefix is written no earlier than its ancestors), so
// readiness is a downward-closed threshold and the first ready L found scanning from the
// top is the longest one.
func (fw firstWrite) readyMatch(toks []int, structuralMatch int, t, delay int64) int {
	if structuralMatch <= 0 {
		return 0
	}
	hs := prefixHashes(toks)
	for L := structuralMatch; L >= 1; L-- {
		if wt, ok := fw[hs[L-1]]; ok && wt+delay <= t {
			return L
		}
	}
	return 0
}
