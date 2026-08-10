package flowmetrics

import "fmt"

// TreeWIP is the census of work that exists ONLY in the working tree: started,
// not landed, and therefore invisible to every issue-based dashboard.
//
// WHY THIS IS A FLOW METRIC. Counting open issues measures declared WIP. It
// misses the work that is furthest along and most at risk — a file written but
// never committed. On a shared checkout driven by a fleet, this is where WIP
// actually accumulates: uncommitted files carry no issue number, appear in no
// query, and are lost outright if the tree is reset. A backlog dashboard can
// read green while the tree holds a hundred unlanded files.
//
// The zero value means NOT MEASURED, and Build reports that as unmeasured rather
// than clean, so a skipped gather can never be mistaken for an empty tree.
type TreeWIP struct {
	Measured bool `json:"measured"`
	// Rev is the HEAD sha the census was taken at. Without it the numbers
	// are not reproducible, because the tip moves under a live fleet.
	Rev string `json:"rev,omitempty"`

	UntrackedGo int `json:"untracked_go"`
	ModifiedGo  int `json:"modified_go"`
	// ScratchLitter counts throwaway probe files (zz_*-style names and
	// dot-prefixed sources). They are separated from real WIP because the
	// remedy differs: litter is deleted, unlanded work is committed.
	ScratchLitter int `json:"scratch_litter"`
	// HiddenGo counts dot-prefixed .go files, which the go tool ignores and
	// most greps miss, so they are WIP nothing can even see.
	HiddenGo int `json:"hidden_go"`

	AddedLines   int `json:"added_lines"`
	DeletedLines int `json:"deleted_lines"`

	// OldestUntrackedHours is the age of the longest-unlanded untracked
	// source file.
	OldestUntrackedHours float64 `json:"oldest_untracked_hours"`

	// RecentWriters counts dirty source files written within
	// RecentWriterWindowMinutes. On a single-operator tree this is ~0; a
	// double-digit value means several sessions are writing one checkout
	// concurrently, which is the collision regime.
	RecentWriters int `json:"recent_writers"`

	// StatFailures counts dirty files whose mtime could not be read. It is
	// surfaced rather than swallowed because the two mtime-derived fields above
	// both read 0 when a stat fails, and a zero there is indistinguishable from
	// a pristine tree — so a broken path join would otherwise present itself as
	// good news. This is the same "unmeasured must not look clean" rule
	// Measured enforces for the census as a whole.
	StatFailures int `json:"stat_failures,omitempty"`

	// BuildProbed records whether buildability was actually tested;
	// Buildable is meaningless without it.
	BuildProbed bool   `json:"build_probed"`
	Buildable   bool   `json:"buildable"`
	BuildError  string `json:"build_error,omitempty"`
}

const (
	// RecentWriterWindowMinutes is the window for the concurrency probe.
	// Ten minutes is long enough to catch an active peer session between its
	// edits and short enough that a finished session drops out quickly.
	RecentWriterWindowMinutes = 10

	// UntrackedGoCeiling is how many unlanded source files are tolerable.
	// AGENTS.md requires one issue to land as one commit, so a healthy tree
	// holds at most the handful of files belonging to the change in hand;
	// 20 allows a large single change plus normal scratch.
	UntrackedGoCeiling = 20

	// ScratchLitterCeiling is how many throwaway probe files may accumulate
	// before the tree is being used as a scratchpad rather than a checkout.
	ScratchLitterCeiling = 10

	// RecentWritersCeiling is the concurrent-writer count above which
	// sessions are provably interleaving edits in one tree. Set at 4: a
	// single session touching a few files in ten minutes is normal, while
	// five or more distinct dirty files churning at once is not one actor.
	RecentWritersCeiling = 4
)

// kpiLocalWIP grades the uncommitted-work census. It is the only KPI in this
// package whose facts come from the filesystem rather than from issue and commit
// records, and it is deliberately the loudest, because unlanded work is the
// single most perishable form of WIP.
func kpiLocalWIP(t TreeWIP) KPI {
	k := KPI{KPI: "local_wip", Group: "wip", Defects: []string{}, Soft: []string{}}
	if !t.Measured {
		k.Score, k.Value = 0, -1
		k.Detail = "working-tree census not gathered"
		k.Soft = append(k.Soft, "local_wip: unmeasured — pass a gathered TreeWIP to grade unlanded work")
		return k
	}
	k.Value = float64(t.UntrackedGo)
	k.Detail = fmt.Sprintf(
		"%d untracked and %d modified source files uncommitted (+%d/-%d lines); %d scratch probes, %d hidden; oldest unlanded %.1fd",
		t.UntrackedGo, t.ModifiedGo, t.AddedLines, t.DeletedLines,
		t.ScratchLitter, t.HiddenGo, t.OldestUntrackedHours/24)

	if t.UntrackedGo > UntrackedGoCeiling {
		k.Defects = append(k.Defects, fmt.Sprintf(
			"local_wip: %d untracked source files exceed the ceiling of %d — this is started work no issue query can see and a tree reset destroys; land or delete it",
			t.UntrackedGo, UntrackedGoCeiling))
	}
	if t.ScratchLitter > ScratchLitterCeiling {
		k.Defects = append(k.Defects, fmt.Sprintf(
			"local_wip: %d scratch probe files exceed the ceiling of %d — throwaway probes are being left in the shared checkout; delete them at the end of the run that created them",
			t.ScratchLitter, ScratchLitterCeiling))
	}
	if t.BuildProbed && !t.Buildable {
		k.Defects = append(k.Defects, fmt.Sprintf(
			"local_wip: the shared tree does not build (%s) — AGENTS.md requires leaving it buildable and gating incomplete work behind a //go:build wip_<feature> tag; every peer session is blocked until this is resolved",
			firstLine(t.BuildError)))
	}
	if t.StatFailures > 0 {
		k.Soft = append(k.Soft, fmt.Sprintf(
			"local_wip: %d dirty files could not be stat'd, so the oldest-unlanded and concurrent-writer ages below are understated — treat them as a floor, not a reading",
			t.StatFailures))
	}
	if t.RecentWriters > RecentWritersCeiling {
		k.Defects = append(k.Defects, fmt.Sprintf(
			"local_wip: %d source files were written in the last %dm, above the %d expected of one session — multiple sessions are editing one checkout concurrently, so any build result is a snapshot of a moving tree; serialise through per-worker worktrees",
			t.RecentWriters, RecentWriterWindowMinutes, RecentWritersCeiling))
	}

	// The score degrades on the dominant term rather than averaging, so one
	// severe axis cannot be hidden by three clean ones.
	k.Score = 100
	if t.UntrackedGo > 0 {
		k.Score = score01(1 - float64(t.UntrackedGo)/float64(UntrackedGoCeiling*3))
	}
	if t.BuildProbed && !t.Buildable {
		k.Score = 0
	}
	return k
}

// firstLine trims a multi-line compiler error to its first line so a KPI detail
// stays one readable sentence.
func firstLine(s string) string {
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' || s[i] == '\r' {
			return s[:i]
		}
	}
	if s == "" {
		return "no error captured"
	}
	return s
}
