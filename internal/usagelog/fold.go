package usagelog

import (
	"sort"
)

// VerbStat is the per-verb roll-up: how many times a verb ran, how many of those
// runs exited non-zero, and the median wall-clock duration. It is the row of the
// `fak usage --by-verb` table.
type VerbStat struct {
	Verb   string `json:"verb"`
	Count  int    `json:"count"`
	Errors int    `json:"errors"` // runs with exit_code != 0
	P50MS  int64  `json:"p50_ms"` // median duration_ms over this verb's runs
}

// Fold is the read-side roll-up of a usage journal — the answer to "how is fak
// being used?" that `fak usage` prints. Every number here is an OBSERVED aggregate
// of the process's own self-reported rows, never a witness of downstream effect.
type Fold struct {
	Total     int         `json:"total"`      // rows folded (after the --since cutoff)
	Errors    int         `json:"errors"`     // rows with exit_code != 0
	P50MS     int64       `json:"p50_ms"`     // median duration over all folded rows
	ByVerb    []VerbStat  `json:"by_verb"`    // sorted: most-used first, ties broken by verb name
	ExitCodes map[int]int `json:"exit_codes"` // exit-code distribution
	Recent    []Row       `json:"recent"`     // the last TopN rows, oldest-first
}

// FoldOptions parameterizes a fold. The zero value folds every row and returns a
// default-sized Recent tail.
type FoldOptions struct {
	// SinceUnixNano drops rows older than this wall-clock cutoff (0 = fold all).
	SinceUnixNano int64
	// TopN bounds the Recent tail (<=0 selects defaultRecent).
	TopN int
}

const defaultRecent = 20

// FoldRows rolls a slice of usage rows (as returned by ReadRows) into a Fold. It is
// pure: no I/O, deterministic order, so the `fak usage` command is a thin formatter
// over this and the same fold is trivially unit-testable. Rows whose schema is not
// SchemaV1 are skipped (a foreign JSONL line never pollutes the usage roll-up).
func FoldRows(rows []Row, opt FoldOptions) Fold {
	topN := opt.TopN
	if topN <= 0 {
		topN = defaultRecent
	}
	out := Fold{ExitCodes: map[int]int{}}

	type acc struct {
		count  int
		errors int
		durs   []int64
	}
	byVerb := map[string]*acc{}
	var allDurs []int64
	var kept []Row

	for _, r := range rows {
		if r.Schema != SchemaV1 {
			continue
		}
		if opt.SinceUnixNano > 0 && r.TSUnixNano < opt.SinceUnixNano {
			continue
		}
		out.Total++
		if r.ExitCode != 0 {
			out.Errors++
		}
		out.ExitCodes[r.ExitCode]++
		allDurs = append(allDurs, r.DurationMS)

		a := byVerb[r.Verb]
		if a == nil {
			a = &acc{}
			byVerb[r.Verb] = a
		}
		a.count++
		if r.ExitCode != 0 {
			a.errors++
		}
		a.durs = append(a.durs, r.DurationMS)

		kept = append(kept, r)
	}

	out.P50MS = medianInt64(allDurs)

	out.ByVerb = make([]VerbStat, 0, len(byVerb))
	for verb, a := range byVerb {
		out.ByVerb = append(out.ByVerb, VerbStat{
			Verb:   verb,
			Count:  a.count,
			Errors: a.errors,
			P50MS:  medianInt64(a.durs),
		})
	}
	// Most-used first; ties broken by verb name so the order is deterministic.
	sort.Slice(out.ByVerb, func(i, j int) bool {
		if out.ByVerb[i].Count != out.ByVerb[j].Count {
			return out.ByVerb[i].Count > out.ByVerb[j].Count
		}
		return out.ByVerb[i].Verb < out.ByVerb[j].Verb
	})

	// Recent tail: the last topN kept rows, oldest-first (kept is already in file
	// order, which is chronological).
	if len(kept) > topN {
		kept = kept[len(kept)-topN:]
	}
	out.Recent = kept
	return out
}

// medianInt64 returns the median of a set of durations (the p50). It copies before
// sorting so the caller's slice is left untouched, and returns 0 for an empty set.
// For an even count it takes the lower-middle element (a deterministic p50 that
// avoids inventing a fractional millisecond between two real observations).
func medianInt64(v []int64) int64 {
	if len(v) == 0 {
		return 0
	}
	cp := make([]int64, len(v))
	copy(cp, v)
	sort.Slice(cp, func(i, j int) bool { return cp[i] < cp[j] })
	return cp[(len(cp)-1)/2]
}
