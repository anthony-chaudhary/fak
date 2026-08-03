package issuefanout

// turn.go is the loop-turn seam (#2523).
//
// The planner, the durable ledger (#2515) and the outcome fold (#2519) all existed
// before this file, and none of them had a caller in the dispatch path: fanning out at
// spine-ship time was a step written down in a skill and remembered by an agent. A
// default that lives in an agent's memory is not a default. Turn is the ONE fold a loop
// tick calls so the default lives in the pipeline instead.
//
// What it decides, and what it deliberately does not:
//
//   - It PLANS. Filing stays the verb's --live mode, which needs the tracker, a bounded
//     dedupe scan and an operator's judgement — none of which belong inside a tick.
//     "Invoke the verb at the right moment" is satisfied by the verb's OWN default
//     mode; the plan and its ledger row are the durable record that it fired.
//   - One row per LEAF per turn. A turn that ships three commits inside one leaf fans
//     out once, because the marker key is fanout-<leaf>-<slug>: a second pass would plan
//     the same keys twice, and filed they would double-file.
//   - It never fails the turn. Every fault becomes a row (carrying the refusal reason)
//     or a counter, because a tick that stops dispatching over a fan-out it could not
//     plan has traded the spine for the follow-on.
//   - Nothing is dropped silently. A ship whose diff named no leaf is counted, and a
//     leaf past the cap is listed by name, so a turn that fanned out less than it looks
//     like says so on the same artifact an operator is already reading.
//
// Purity is unchanged: the caller owns the clock (at) and the file (ledger), exactly as
// BuildLogged and AppendRow already required.

import (
	"io"
	"strings"
	"time"
)

// TurnSchema identifies one loop-turn fan-out record — the artifact that shows the
// invocation happened without anyone asking for it.
const TurnSchema = "fak.issue-fanout-turn.v1"

// TurnLeafCap bounds how many leaves ONE turn fans out. A tick that swept an unusual
// number of finished workers (a fleet-wide restart, a long-idle runs dir) would
// otherwise expand the whole taxonomy for each of them inside the dispatch hot path.
// The cap is generous against a real turn — a turn ships one or two leaves — and every
// leaf it refuses is named in Dropped rather than swallowed.
const TurnLeafCap = 8

// Ship is one shipped spine the turn witnessed: the commit that landed (SpineRef), the
// issue it resolved, and the paths that commit changed. Paths is what the leaf is
// derived from, so the caller hands over the diff's file list and this leaf owns the
// reduction — no lane string is guessed from an issue's prose.
type Ship struct {
	Issue    int      `json:"issue,omitempty"`
	SpineRef string   `json:"spine_ref"`
	Paths    []string `json:"paths,omitempty"`
}

// TurnRow is one leaf's fan-out on this turn: what was planned, from which spine, and
// how it came out. A refused or failed row carries the reason verbatim, so the artifact
// explains itself without a second lookup.
type TurnRow struct {
	Leaf       string  `json:"leaf"`
	SpineRef   string  `json:"spine_ref"`
	Issue      int     `json:"issue,omitempty"`
	Candidates int     `json:"candidates"`
	Outcome    Outcome `json:"outcome"`
	Reason     string  `json:"reason,omitempty"`
}

// TurnResult is the captured loop-turn artifact: how many ships the turn saw, what the
// fan-out did per leaf, and what it could not do. NoLeaf and Dropped are the honesty
// half — without them a turn that planned nothing and a turn that had nothing to plan
// read identically.
type TurnResult struct {
	Schema  string        `json:"schema"`
	Ships   int           `json:"ships"`
	Counts  OutcomeCounts `json:"counts"`
	Rows    []TurnRow     `json:"rows,omitempty"`
	NoLeaf  int           `json:"no_leaf,omitempty"`
	Dropped []string      `json:"dropped,omitempty"`
}

// Turn fans out every leaf the turn's ships touched, appending one durable ledger row
// per invocation through BuildLogged — the seam that leaf already documented as "the
// one-call seam a verb or wave uses". A nil ledger is a no-op append, so a caller that
// cannot open its file still gets the plan and the counts.
//
// It returns no error on purpose: the result IS the report, and a loop tick reads it
// rather than branching on it.
func Turn(ships []Ship, at time.Time, ledger io.Writer) TurnResult {
	res := TurnResult{Schema: TurnSchema, Ships: len(ships)}
	seen := map[string]bool{}
	for _, s := range ships {
		spine := strings.TrimSpace(s.SpineRef)
		leaves := NewLeavesFromPaths(s.Paths)
		if len(leaves) == 0 {
			// A commit entirely outside internal/<leaf>/ — docs, tools, a cmd-only
			// change. There is no lane to key a marker on, so this ships nothing to fan
			// out; counted rather than skipped.
			res.NoLeaf++
			continue
		}
		for _, leaf := range leaves {
			if seen[leaf] {
				continue
			}
			seen[leaf] = true
			if len(res.Rows) >= TurnLeafCap {
				res.Dropped = append(res.Dropped, leaf)
				continue
			}
			// Title is the leaf itself: the loop has no human name for what shipped that
			// is not free text it would have to invent, and the leaf is exactly the
			// scope the generated follow-ons are bound to.
			plan, err := BuildLogged(Input{Title: leaf, Leaf: leaf, SpineRef: spine}, at, ledger)
			row := TurnRow{
				Leaf:       leaf,
				SpineRef:   spine,
				Issue:      s.Issue,
				Candidates: len(plan.Candidates),
				Outcome:    res.Counts.Observe(err),
			}
			if err != nil {
				row.Reason = err.Error()
			}
			res.Rows = append(res.Rows, row)
		}
	}
	return res
}
