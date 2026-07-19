package steerpr

// `fak steer pause` / `fak steer resume` (#5031): stop the fleet spending on an
// intent while the operator decides, without touching anything that landed.
//
// Redirect (#5030) re-aims an intent's next tick, but the fleet keeps moving.
// Some units need to stop NOW and be decided later. The pause is that verb: it
// holds the unit's BOUND ISSUE out of the dispatch loop's future work by riding
// the dispatcher's existing BLOCKED_BY_HUMAN backpressure token — the same
// closed reason the guard's HUMAN_RESIDUAL escalations already project. Three
// rules keep it honest:
//
//   - A pause targets the intent's FUTURE dispatch, never its present or past.
//     It is NOT a kill: an in-flight worker finishes and lands cleanly; the
//     hold only keeps the intent from being picked up AGAIN. Nothing in this
//     file (or reachable from it) can touch git, a process, or a worker — the
//     leaf stays subprocess-free and internal-import-free, swept by the same
//     architest steer-overlay floor as the other overlay leaves.
//   - A pause with no release path is a LEAK that silently starves an intent
//     forever, so pause and resume ship as one row vocabulary in one ledger:
//     a leaf's hold state is the fold of its pause/resume rows, oldest first.
//   - The ledger is append-only and attributable: every row carries who and
//     when; rows are only ever appended, never rewritten — never rewrite a
//     peer's row. Loading the ledger is how paused time becomes VISIBLE
//     (a silently paused intent is indistinguishable from a finished one).

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// PauseSchema is the machine identifier for one pause-ledger row (both the
// pause and the resume action share it: they are one event vocabulary).
const PauseSchema = "fak.steerpr.pause.v1"

// The two actions a pause-ledger row can record. A leaf's hold state is the
// ledger-order fold of its rows: the latest action wins.
const (
	PauseActionPause  = "pause"
	PauseActionResume = "resume"
)

// Pause is one appended ledger row: an operator holding (or releasing) a
// unit's bound intent. A pause row binds the unit's bound issue ("#N") — that
// is the identity the dispatch hold lands on — while a resume row releases the
// leaf's hold and carries the issue only as context.
type Pause struct {
	Schema string `json:"schema"`
	Action string `json:"action"` // "pause" | "resume"
	Leaf   string `json:"leaf"`
	By     string `json:"by"`
	At     string `json:"at"`             // RFC3339 UTC: when the operator acted
	Note   string `json:"note,omitempty"` // optional: why the operator paused
	// Issue is the unit's bound issue ("#N"). REQUIRED on a pause row: the
	// dispatch loop holds work by issue number, so a pause that binds no issue
	// has nothing to hold and is refused rather than ledgered as a no-op.
	Issue string `json:"issue,omitempty"`
}

// NewPause builds a validated pause row. An unnamed unit, an unattributable
// pause (no by), or a missing/malformed bound issue is refused rather than
// defaulted: a pause is a specific person stopping the spend on a specific
// bound intent, and a row missing any leg holds nothing.
func NewPause(leaf, by, note, issue string, at time.Time) (Pause, error) {
	leaf = strings.TrimSpace(leaf)
	by = strings.TrimSpace(by)
	issue = strings.TrimSpace(issue)
	switch {
	case leaf == "":
		return Pause{}, fmt.Errorf("a pause must name the unit it holds")
	case by == "":
		return Pause{}, fmt.Errorf("a pause must be attributable: say who is pausing (--by, or set git config user.name)")
	case issueRefNumber(issue) == 0:
		return Pause{}, fmt.Errorf("a pause holds the unit's bound issue and %q is not a #N issue ref: with nothing bound there is nothing for the dispatch loop to hold", issue)
	}
	return Pause{
		Schema: PauseSchema,
		Action: PauseActionPause,
		Leaf:   leaf,
		By:     by,
		At:     at.UTC().Format(time.RFC3339),
		Note:   strings.TrimSpace(note),
		Issue:  issue,
	}, nil
}

// NewResume builds a validated resume row: the release half of the verb pair.
// The bound issue rides along as context when the pause it releases had one.
func NewResume(leaf, by, issue string, at time.Time) (Pause, error) {
	leaf = strings.TrimSpace(leaf)
	by = strings.TrimSpace(by)
	switch {
	case leaf == "":
		return Pause{}, fmt.Errorf("a resume must name the unit it releases")
	case by == "":
		return Pause{}, fmt.Errorf("a resume must be attributable: say who is releasing (--by, or set git config user.name)")
	}
	return Pause{
		Schema: PauseSchema,
		Action: PauseActionResume,
		Leaf:   leaf,
		By:     by,
		At:     at.UTC().Format(time.RFC3339),
		Issue:  strings.TrimSpace(issue),
	}, nil
}

// PauseLedgerPath is the overlay pause ledger's location under a repo root:
// gitignored runtime state beside the other .fak ledgers, one JSON row per
// line.
func PauseLedgerPath(root string) string {
	return filepath.Join(root, ".fak", "steer-pauses.jsonl")
}

// AppendPause appends one row to the ledger. Append-only by construction: the
// file is opened O_APPEND and rows are only ever added. An incomplete row is
// refused so every ledgered hold stays attributable and — for a pause —
// actually bound to an issue the dispatch loop can hold.
func AppendPause(path string, p Pause) error {
	switch {
	case strings.TrimSpace(p.Leaf) == "" || strings.TrimSpace(p.By) == "":
		return fmt.Errorf("refusing an incomplete pause-ledger row: it needs the leaf and who acted")
	case p.Action != PauseActionPause && p.Action != PauseActionResume:
		return fmt.Errorf("refusing a pause-ledger row with action %q: only %q and %q exist", p.Action, PauseActionPause, PauseActionResume)
	case p.Action == PauseActionPause && p.IssueNumber() == 0:
		return fmt.Errorf("refusing a pause row with no bound #N issue: the dispatch loop would have nothing to hold")
	}
	line, err := json.Marshal(p)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	_, werr := f.Write(append(line, '\n'))
	cerr := f.Close()
	if werr != nil {
		return werr
	}
	return cerr
}

// LoadPauses reads the ledger best-effort: a missing or unreadable file is an
// empty ledger, and a torn or foreign line is skipped rather than poisoning
// the rows around it. Failure never invents (or releases) a hold.
func LoadPauses(path string) []Pause {
	buf, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var out []Pause
	for _, line := range strings.Split(string(buf), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var p Pause
		if json.Unmarshal([]byte(line), &p) != nil {
			continue
		}
		if p.Schema != PauseSchema {
			continue
		}
		out = append(out, p)
	}
	return out
}

// ActivePauses folds the ledger, oldest first, into the leaves that are
// paused RIGHT NOW: a pause row sets the leaf's hold, a resume row clears it,
// and the returned map carries the pause row each held leaf is under (its By,
// At, Note, and bound Issue are what the dispatch hold and the render show).
// Rows that would not validate (a pause with no bound issue) hold nothing.
func ActivePauses(rows []Pause) map[string]Pause {
	active := map[string]Pause{}
	for _, p := range rows {
		leaf := strings.TrimSpace(p.Leaf)
		if leaf == "" {
			continue
		}
		switch p.Action {
		case PauseActionPause:
			if p.IssueNumber() != 0 {
				active[leaf] = p
			}
		case PauseActionResume:
			delete(active, leaf)
		}
	}
	return active
}

// PausedFor returns the active pause holding leaf, if any — the row `fak steer
// resume` releases and the render shows as "paused since".
func PausedFor(rows []Pause, leaf string) (Pause, bool) {
	p, ok := ActivePauses(rows)[strings.TrimSpace(leaf)]
	return p, ok
}

// IssueNumber is the bound issue's numeric id (the N of "#N"), or 0 when the
// row binds none — and 0 can never match a routed issue, so an unbound row can
// never hold anything.
func (p Pause) IssueNumber() int {
	return issueRefNumber(p.Issue)
}

// issueRefNumber parses a "#N" issue ref to N, or 0 for anything else.
func issueRefNumber(ref string) int {
	ref = strings.TrimSpace(ref)
	if !strings.HasPrefix(ref, "#") {
		return 0
	}
	n, err := strconv.Atoi(ref[1:])
	if err != nil || n <= 0 {
		return 0
	}
	return n
}
