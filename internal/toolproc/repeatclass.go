package toolproc

// repeatclass.go — the DECISION spine for #4764: classify repeated tool calls
// (native Codex rollout `shell_command`s, skill reads, status polls) and decide,
// per class, whether a repeat is safe to serve locally. It is the analytics +
// policy half of the issue; the live cache/coalescing enforcement wires onto the
// gateway seam and is tracked as a labeled follow-on (see the doc block below).
//
// WHY. A 2026-07-14 audit of the 100 largest Codex rollout files found 143,432
// tool-result records / 501.8 MB of output, dominated by high-frequency repeats:
// `git status --short --branch` ran 3,707 times, one immutable `super-loop/SKILL.md`
// was read 640+204 times through two equivalent command spellings, and
// `git push origin main` ran 474 times. Long sessions pay provider-input,
// process-spawn, page-fault, and transcript-storage cost for these repeats. Blind
// caching is unsafe: git/dispatch status is MUTABLE, a version-addressed skill file
// is IMMUTABLE until its identity changes, and a write must NEVER be reused. So the
// kernel needs a TYPED workload inventory first, then a per-class reuse policy with
// explicit freshness and invalidation — never a blanket cache.
//
// THE FOLD. ClassifyRepeats(records, cfg) is a pure function from a stream of
// normalized tool-call records to a typed repeat report: same input + same config
// ⇒ byte-identical output, exactly like Fold above. It NEVER retains a secret or a
// raw output body — Normalize redacts secret-bearing arguments at the boundary and
// the record carries only OutputBytes (the size), never the payload.
//
// WHAT THIS LEAF IS NOT. It does not itself cache, coalesce, or invalidate on the
// live wire, and it owns no competing invalidation model — the immutable-read cache
// keyed on (resolved path + content digest) joins #2917's mutation-invalidation
// seam, and the freshness-window coalescing of registered mutable status commands
// arms at the gateway. Those are the labeled next steps; this leaf ships the typed
// inventory + the per-class reuse DECISION and its failure-class proofs.

import (
	"sort"
)

// CommandClass is the CLOSED registry-level classification of a normalized
// command by its MUTABILITY — the property that decides whether a repeat is ever
// safe to serve locally. It is assigned by the registered-command matcher; an
// unmatched command is CmdUnknown (fail-closed: never reused).
type CommandClass string

const (
	// CmdImmutableRead: a read of content that is version/identity-addressed and
	// immutable until its identity changes (a skill file, a pinned config). Safe
	// to serve from a digest-keyed cache until a mutation invalidates it.
	CmdImmutableRead CommandClass = "immutable_read"
	// CmdMutableQuery: a read of MUTABLE state (git/dispatch status). Never a
	// stable cache; reuse only within a bounded freshness window, with stale-age
	// exposed on every reuse.
	CmdMutableQuery CommandClass = "mutable_query"
	// CmdIdempotentWrite: a write attempt (a push, a commit). Never reused, never
	// coalesced — a write's effect is not a value to serve.
	CmdIdempotentWrite CommandClass = "idempotent_write"
	// CmdUnknown: an unrecognized command. Fail-closed — never reused.
	CmdUnknown CommandClass = "unknown"
)

// RepeatClass is the CLOSED per-group classification the report renders — the
// five classes #4764 names. It is CommandClass promoted by repeat SHAPE: a
// mutable query whose repeats are frequent and regular enough becomes a
// POLL_STORM (the 3,707× `git status` tax), which the report highlights as the
// avoidable poll storm freshness coalescing collapses.
type RepeatClass string

const (
	ClassImmutableRead   RepeatClass = "IMMUTABLE_READ"
	ClassMutableQuery    RepeatClass = "MUTABLE_QUERY"
	ClassIdempotentWrite RepeatClass = "IDEMPOTENT_WRITE"
	ClassPollStorm       RepeatClass = "POLL_STORM"
	ClassUnknown         RepeatClass = "UNKNOWN"
)

// ReuseMode is the CLOSED per-group reuse DECISION — the one-bit-plus-mode gate
// a live cache/coalescer keys on. A finding never carries free text here.
type ReuseMode string

const (
	// ReuseKeyed: serve a repeat from a cache keyed on (resolved path + content
	// digest); valid until a mutation changes the identity (#2917's seam).
	ReuseKeyed ReuseMode = "KEYED"
	// ReuseFreshnessBounded: serve a repeat only within FreshMS of the last real
	// fetch; expose stale-age and source on every reuse.
	ReuseFreshnessBounded ReuseMode = "FRESHNESS_BOUNDED"
	// ReuseNever: never serve a repeat locally (writes, unknowns).
	ReuseNever ReuseMode = "NEVER"
)

// CallRecord is one normalized tool-call observation streamed from a native Codex
// rollout log (or fak's own hook journal). It carries the tool, the RAW argument
// line, an observation time, and the tool result's SIZE — never the result body.
// Normalize redacts secrets out of Raw at the boundary before anything is
// retained, so the analytics surface never holds a secret or an output payload.
type CallRecord struct {
	Tool        string `json:"tool"`
	Raw         string `json:"raw"`                    // command/arg line (secrets redacted by Normalize)
	AtMS        int64  `json:"at_unix_ms"`             // observation time
	OutputBytes int64  `json:"output_bytes,omitempty"` // SIZE of the tool result, never its content
	// Digest is the content/identity digest of an immutable-read target observed
	// at read time (a file hash or version tag) — never the content, only its
	// address. When present it is folded into the read Identity so a re-read after
	// a MUTATION (new digest) forms a NEW identity and is NOT served from the stale
	// entry: the invalidation-after-mutation contract, realized as a key that joins
	// #2917's model rather than a competing invalidation engine. Empty => the
	// conservative path-only fold the 640+204 skill-read audit line uses.
	Digest string `json:"digest,omitempty"`
}

// NormalCall is the redacted, canonicalized form of a CallRecord: tool + canonical
// arguments, a resolved read Path (immutable-read identity, else ""), the registry
// CommandClass, the freshness window for a mutable query, and the reuse Identity
// key. Two records that differ only by equivalent spelling (path separators,
// quoting, flag order/alias) share one Identity — the near-duplicate fold.
type NormalCall struct {
	Tool      string       `json:"tool"`
	Canonical string       `json:"canonical"`
	Path      string       `json:"path,omitempty"`
	Digest    string       `json:"digest,omitempty"` // folded into an immutable-read Identity when observed
	Class     CommandClass `json:"class"`
	FreshMS   int64        `json:"fresh_ms,omitempty"`
	Identity  string       `json:"identity"`
}

// RepeatConfig tunes the classifier. Zero values are safe: the defaults below.
type RepeatConfig struct {
	// PollingMinRepeats: a mutable query observed at least this many times is a
	// candidate to be promoted to POLL_STORM. 0 => DefaultPollingMinRepeats.
	PollingMinRepeats int `json:"polling_min_repeats,omitempty"`
	// PollingMaxMedianSpacingMS: the promotion also requires median inter-call
	// spacing at or below this — regular, tight repetition, not occasional polls.
	// 0 => DefaultPollingMaxMedianSpacingMS.
	PollingMaxMedianSpacingMS int64 `json:"polling_max_median_spacing_ms,omitempty"`
	// DefaultFreshnessMS is the freshness window applied to a registered mutable
	// query that declares none. 0 => DefaultFreshnessWindowMS.
	DefaultFreshnessMS int64 `json:"default_freshness_ms,omitempty"`
}

// Classifier defaults — a mutable query repeated >= 5 times at <= 60s median
// spacing is a poll storm; a registered status command with no declared window
// coalesces within 2s.
const (
	DefaultPollingMinRepeats         = 5
	DefaultPollingMaxMedianSpacingMS = 60_000
	DefaultFreshnessWindowMS         = 2_000
)

// RepeatGroup is one identity's folded repeat record — the row the report renders.
type RepeatGroup struct {
	Identity  string      `json:"identity"`
	Tool      string      `json:"tool"`
	Canonical string      `json:"canonical"`
	Path      string      `json:"path,omitempty"`
	Digest    string      `json:"digest,omitempty"` // the read identity this group reuses on (immutable reads)
	Class     RepeatClass `json:"class"`
	Reuse     ReuseMode   `json:"reuse"`

	Count     int `json:"count"`      // total observations of this identity
	ExactDups int `json:"exact_dups"` // repeats whose raw spelling exactly matched a prior one
	NearDups  int `json:"near_dups"`  // repeats that matched by identity but via a NEW equivalent spelling

	OutputBytes int64 `json:"output_bytes"` // total result bytes across the group (sizes, not content)

	FirstMS int64 `json:"first_unix_ms"`
	LastMS  int64 `json:"last_unix_ms"`

	MinSpacingMS    int64 `json:"min_spacing_ms,omitempty"`
	MaxSpacingMS    int64 `json:"max_spacing_ms,omitempty"`
	MedianSpacingMS int64 `json:"median_spacing_ms,omitempty"`

	FreshMS int64 `json:"fresh_ms,omitempty"` // freshness window (mutable/polling)

	// AvoidableSpawns is the count of repeats a SAFE per-class policy could have
	// served locally: for an immutable read, every repeat after the first; for a
	// mutable query/poll loop, every repeat inside the freshness window of the
	// last real fetch; for a write or unknown, zero. This is the net-true saving
	// against the TUNED (already-deduped) baseline, not a naive one.
	AvoidableSpawns int `json:"avoidable_spawns"`
	// AvoidableInputBytes is the result bytes those avoidable repeats would NOT
	// re-inject into context — the transcript/provider-input saving.
	AvoidableInputBytes int64 `json:"avoidable_input_bytes"`
}

// RepeatTotals is the report's headline roll-up.
type RepeatTotals struct {
	Records             int   `json:"records"`
	Groups              int   `json:"groups"`
	OutputBytes         int64 `json:"output_bytes"`
	AvoidableSpawns     int   `json:"avoidable_spawns"`
	AvoidableInputBytes int64 `json:"avoidable_input_bytes"`
	// PerClass counts groups by their final RepeatClass — the typed inventory.
	PerClass map[RepeatClass]int `json:"per_class"`
}

// RepeatReport is the deterministic classifier output.
type RepeatReport struct {
	Schema string        `json:"schema"`
	Config RepeatConfig  `json:"config"`
	Groups []RepeatGroup `json:"groups"`
	Totals RepeatTotals  `json:"totals"`
}

// RepeatReportSchema stamps the classifier output.
const RepeatReportSchema = "fak.toolproc-repeat.v1"

// ClassifyRepeats folds a stream of CallRecords into a typed repeat report. It is
// pure: same records + same config ⇒ byte-identical report. Records are grouped
// by reuse Identity (equivalent spellings collapse into one group); each group is
// classified into one of the five closed RepeatClasses, assigned a per-class
// ReuseMode, and scored for the net-true avoidable spawns/input-bytes a safe
// policy would save against the tuned (already-deduped) baseline.
func ClassifyRepeats(records []CallRecord, cfg RepeatConfig) RepeatReport {
	if cfg.PollingMinRepeats == 0 {
		cfg.PollingMinRepeats = DefaultPollingMinRepeats
	}
	if cfg.PollingMaxMedianSpacingMS == 0 {
		cfg.PollingMaxMedianSpacingMS = DefaultPollingMaxMedianSpacingMS
	}
	if cfg.DefaultFreshnessMS == 0 {
		cfg.DefaultFreshnessMS = DefaultFreshnessWindowMS
	}

	type acc struct {
		g       RepeatGroup
		nc      NormalCall
		times   []int64 // observation times, in stream order
		bytes   []int64 // per-observation output sizes, aligned to times
		seenRaw map[string]bool
	}
	groups := map[string]*acc{}
	var order []string

	for _, rec := range records {
		nc := Normalize(rec, cfg)
		a, ok := groups[nc.Identity]
		if !ok {
			a = &acc{
				g: RepeatGroup{
					Identity: nc.Identity, Tool: nc.Tool, Canonical: nc.Canonical,
					Path: nc.Path, Digest: nc.Digest, FreshMS: nc.FreshMS,
					FirstMS: rec.AtMS, LastMS: rec.AtMS,
				},
				nc:      nc,
				seenRaw: map[string]bool{},
			}
			groups[nc.Identity] = a
			order = append(order, nc.Identity)
		}
		a.g.Count++
		a.g.OutputBytes += rec.OutputBytes
		if rec.AtMS < a.g.FirstMS {
			a.g.FirstMS = rec.AtMS
		}
		if rec.AtMS > a.g.LastMS {
			a.g.LastMS = rec.AtMS
		}
		if a.g.Count > 1 { // a repeat: exact spelling, or a new equivalent one?
			if a.seenRaw[rec.Raw] {
				a.g.ExactDups++
			} else {
				a.g.NearDups++
			}
		}
		a.seenRaw[rec.Raw] = true
		a.times = append(a.times, rec.AtMS)
		a.bytes = append(a.bytes, rec.OutputBytes)
	}

	report := RepeatReport{Schema: RepeatReportSchema, Config: cfg}
	report.Totals.Records = len(records)
	report.Totals.PerClass = map[RepeatClass]int{}
	for _, id := range order {
		a := groups[id]
		spacings := interSpacings(a.times)
		a.g.MinSpacingMS, a.g.MaxSpacingMS, a.g.MedianSpacingMS = spacingStats(spacings)
		a.g.Class = classify(a.nc.Class, a.g.Count, a.g.MedianSpacingMS, cfg)
		a.g.Reuse = reuseFor(a.g.Class)
		a.g.AvoidableSpawns, a.g.AvoidableInputBytes = avoidable(a.g, a.times, a.bytes)
		report.Groups = append(report.Groups, a.g)
		report.Totals.Groups++
		report.Totals.OutputBytes += a.g.OutputBytes
		report.Totals.AvoidableSpawns += a.g.AvoidableSpawns
		report.Totals.AvoidableInputBytes += a.g.AvoidableInputBytes
		report.Totals.PerClass[a.g.Class]++
	}
	// Deterministic order: biggest avoidable saving first, then identity.
	sort.SliceStable(report.Groups, func(i, j int) bool {
		if report.Groups[i].AvoidableInputBytes != report.Groups[j].AvoidableInputBytes {
			return report.Groups[i].AvoidableInputBytes > report.Groups[j].AvoidableInputBytes
		}
		if report.Groups[i].AvoidableSpawns != report.Groups[j].AvoidableSpawns {
			return report.Groups[i].AvoidableSpawns > report.Groups[j].AvoidableSpawns
		}
		return report.Groups[i].Identity < report.Groups[j].Identity
	})
	return report
}

// classify promotes a mutable query to a POLL_STORM when its repeats are both
// frequent (>= PollingMinRepeats) and tightly, regularly spaced (median spacing
// <= PollingMaxMedianSpacingMS). Every other class maps straight through.
func classify(cc CommandClass, count int, medianSpacingMS int64, cfg RepeatConfig) RepeatClass {
	switch cc {
	case CmdImmutableRead:
		return ClassImmutableRead
	case CmdIdempotentWrite:
		return ClassIdempotentWrite
	case CmdMutableQuery:
		if count >= cfg.PollingMinRepeats && medianSpacingMS > 0 && medianSpacingMS <= cfg.PollingMaxMedianSpacingMS {
			return ClassPollStorm
		}
		return ClassMutableQuery
	default:
		return ClassUnknown
	}
}

// reuseFor is the per-class reuse decision — the whole safety contract in one
// closed map: reads are digest-keyed, mutable state is freshness-bounded, writes
// and unknowns are never reused.
func reuseFor(rc RepeatClass) ReuseMode {
	switch rc {
	case ClassImmutableRead:
		return ReuseKeyed
	case ClassMutableQuery, ClassPollStorm:
		return ReuseFreshnessBounded
	default:
		return ReuseNever
	}
}

// avoidable scores the net-true saving a SAFE policy yields for a group, against
// the tuned baseline (the first real fetch is always paid). times is the group's
// observation times in stream order; bytes is the per-observation output size in
// the same order.
func avoidable(g RepeatGroup, times []int64, bytes []int64) (int, int64) {
	switch g.Reuse {
	case ReuseKeyed:
		// Immutable: every repeat after the first serves from the digest cache.
		spawns := 0
		var ib int64
		for i := 1; i < len(times); i++ {
			spawns++
			ib += bytes[i]
		}
		return spawns, ib
	case ReuseFreshnessBounded:
		// Mutable: replay a bounded freshness cache in time order — a repeat
		// inside FreshMS of the last REAL fetch is coalesced; otherwise it is a
		// fresh fetch and the window resets. Requires sorted times.
		ordered := append([]int64(nil), times...)
		idx := indexSort(ordered)
		fresh := g.FreshMS
		spawns := 0
		var ib int64
		lastFetch := int64(-1 << 62)
		first := true
		for _, k := range idx {
			t := ordered[k]
			if first {
				lastFetch = t
				first = false
				continue
			}
			if t-lastFetch <= fresh {
				spawns++
				ib += bytes[k]
			} else {
				lastFetch = t
			}
		}
		return spawns, ib
	default:
		return 0, 0
	}
}

// The per-observation output sizes `avoidable` needs are accumulated inline with
// the group's observation times (see the acc.bytes append above), so the fold
// stays a single pass. It used to be a per-group re-scan of the whole record
// stream that re-ran Normalize on every record, which made ClassifyRepeats
// O(groups x records) — on the real top-100 rollout corpus that is ~10^9+ Normalize
// calls and the replay never finishes (#5120, and the reason #5410 deleted the
// replay witness instead of running it). Accumulating is exactly equivalent: the
// re-scan selected the same records, in the same stream order, as the loop that
// appends to acc.times.

// interSpacings returns the inter-arrival gaps of a group's observations, sorted
// ascending so the stats are order-independent (a group's times may arrive out of
// order across interleaved sessions).
func interSpacings(times []int64) []int64 {
	if len(times) < 2 {
		return nil
	}
	ordered := append([]int64(nil), times...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i] < ordered[j] })
	gaps := make([]int64, 0, len(ordered)-1)
	for i := 1; i < len(ordered); i++ {
		gaps = append(gaps, ordered[i]-ordered[i-1])
	}
	return gaps
}

// spacingStats returns (min, max, median) of ascending gaps. Median uses the
// lower-middle element for even counts so the result is deterministic.
func spacingStats(gaps []int64) (int64, int64, int64) {
	if len(gaps) == 0 {
		return 0, 0, 0
	}
	sorted := append([]int64(nil), gaps...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	return sorted[0], sorted[len(sorted)-1], sorted[(len(sorted)-1)/2]
}

// indexSort returns the indices of xs ordered by ascending value (stable on ties),
// so a caller can walk observations in time order while keeping the parallel
// bytes slice aligned to the original stream position.
func indexSort(xs []int64) []int {
	idx := make([]int, len(xs))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(i, j int) bool { return xs[idx[i]] < xs[idx[j]] })
	return idx
}
