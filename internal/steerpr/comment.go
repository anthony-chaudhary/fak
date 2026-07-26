package steerpr

// `fak steer comment` (#5029): the ANNOTATE rung of the steering ladder — the
// weakest, cheapest, most-used steering act.
//
// In a PR world the review comment costs nothing, blocks nothing, and lands
// where the work already lives. Continuous merge removed that place, so an
// operator's reasoning about a forming unit either evaporates or ends up in a
// Slack thread disconnected from the intent. `trajectory-control`'s ladder puts
// annotate one rung above observe and says the WEAKEST SUFFICIENT rung wins, so
// most steering should terminate here: building redirect (#5030) and pause
// (#5031) without this rung pushes operators up the ladder unnecessarily.
//
// Three rules keep the annotation honest:
//
//   - A comment is ANNOTATE-ONLY. Nothing in this file can move a Band, an ack,
//     a Verdict, or git: the leaf stays subprocess-free and internal-import-free
//     like its sibling rungs, and the record carries no field a fold could read
//     back as state. Its only outward act is a comment posted through the
//     trusted gh seam in the cmd/fak verb shell — GitHub moves, git does not.
//   - A comment anchors to what the operator actually READ: the unit's exact
//     member SHA set and its band at comment time ride on the record and on the
//     posted body, so the note can never drift onto commits the operator never
//     saw. A unit named without its SHA set is an unanchored opinion.
//   - A comment REQUIRES the unit's closure-grade binding (subject-bound
//     `Resolves` → "#N"). An unbound unit is refused rather than posted
//     somewhere plausible: a unit's Mentions are not a binding, and putting
//     operator reasoning on a merely-mentioned issue is worse than not posting.
//
// The ledger is append-only and attributable: every row carries who and when,
// and rows are only ever appended — never rewrite a peer's row. Loading the
// ledger is how the brief/loop can see that a unit received operator attention.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// CommentSchema is the machine identifier for one comment ledger row.
const CommentSchema = "fak.steerpr.comment.v1"

// Comment is one appended ledger row: an operator's note against a forming
// unit, anchored to the exact member SHA set and band the operator was reading
// and bound to the issue the unit itself declares. It is ADVISORY by
// construction — there is no field here that could touch a Verdict, a Band, an
// ack, or git.
type Comment struct {
	Schema string   `json:"schema"`
	Leaf   string   `json:"leaf"`
	By     string   `json:"by"`
	At     string   `json:"at"`   // RFC3339 UTC: when the operator annotated
	Note   string   `json:"note"` // REQUIRED: the operator's reasoning about this unit
	SHAs   []string `json:"shas"` // exact member set at comment time (sorted, deduped)
	Band   Band     `json:"band"` // band at comment time (anchor, never rewritten)
	// Issue is the unit's closure-grade bound issue ("#N"). REQUIRED: a comment
	// posts to the intent's own ticket or it does not post at all.
	Issue string `json:"issue"`
	// Posted is where the annotation landed (the comment URL gh reports),
	// recorded after the trusted gh seam posts it.
	Posted string `json:"posted,omitempty"`
}

// NewComment builds a validated ledger row. An unnamed unit, an unattributable
// comment (no by), an EMPTY NOTE, an empty member set, or a unit with no
// closure-grade "#N" binding is refused rather than defaulted: a comment is a
// specific person's reasoning about a specific read of a specific bound intent,
// and a row missing any leg annotates nothing.
func NewComment(leaf, by, note string, shas []string, band Band, issue string, at time.Time) (Comment, error) {
	leaf = strings.TrimSpace(leaf)
	by = strings.TrimSpace(by)
	note = strings.TrimSpace(note)
	issue = strings.TrimSpace(issue)
	set := normalizeSHASet(shas)
	switch {
	case leaf == "":
		return Comment{}, fmt.Errorf("a comment must name the unit it annotates")
	case by == "":
		return Comment{}, fmt.Errorf("a comment must be attributable: say who is annotating (--by, or set git config user.name)")
	case note == "":
		return Comment{}, fmt.Errorf("a comment must carry the operator's note (-m): an empty note annotates nothing")
	case len(set) == 0:
		return Comment{}, fmt.Errorf("a comment anchors to the unit's member SHA set, and an empty set anchors nothing")
	case issueRefNumber(issue) == 0:
		return Comment{}, fmt.Errorf("a comment posts to the unit's closure-grade bound issue and %q is not a #N issue ref: with nothing bound there is no honest place to post — a unit's mentions are not a binding", issue)
	}
	return Comment{
		Schema: CommentSchema,
		Leaf:   leaf,
		By:     by,
		At:     at.UTC().Format(time.RFC3339),
		Note:   note,
		SHAs:   set,
		Band:   band,
		Issue:  issue,
	}, nil
}

// CommentLedgerPath is the overlay comment ledger's location under a repo root:
// gitignored runtime state beside the other .fak ledgers, one JSON row per line.
func CommentLedgerPath(root string) string {
	return filepath.Join(root, ".fak", "steer-comments.jsonl")
}

// AppendComment appends one row to the ledger. Append-only by construction: the
// file is opened O_APPEND and rows are only ever added. An incomplete row is
// refused so every ledgered annotation stays attributable, anchored to a real
// SHA set, and bound to the issue it actually posted to.
func AppendComment(path string, c Comment) error {
	switch {
	case strings.TrimSpace(c.Leaf) == "" || strings.TrimSpace(c.By) == "" ||
		strings.TrimSpace(c.Note) == "" || len(normalizeSHASet(c.SHAs)) == 0:
		return fmt.Errorf("refusing an incomplete comment row: it needs the leaf, who annotated, the note, and the anchor SHA set")
	case c.IssueNumber() == 0:
		return fmt.Errorf("refusing a comment row with no bound #N issue: the annotation would have no honest place to live")
	}
	line, err := json.Marshal(c)
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

// LoadComments reads the ledger best-effort: a missing or unreadable file is an
// empty ledger, and a torn or foreign line is skipped rather than poisoning the
// rows around it. Failure never invents operator attention.
func LoadComments(path string) []Comment {
	buf, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var out []Comment
	for _, line := range strings.Split(string(buf), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var c Comment
		if json.Unmarshal([]byte(line), &c) != nil {
			continue
		}
		if c.Schema != CommentSchema {
			continue
		}
		out = append(out, c)
	}
	return out
}

// CommentsFor returns the ledgered comments against one unit, oldest first —
// the "has this unit received operator attention?" view the brief and the loop
// read: len(CommentsFor(...)) is how often an operator has annotated it.
func CommentsFor(rows []Comment, leaf string) []Comment {
	leaf = strings.TrimSpace(leaf)
	var out []Comment
	for _, c := range rows {
		if strings.TrimSpace(c.Leaf) == leaf {
			out = append(out, c)
		}
	}
	return out
}

// IssueNumber is the bound issue's numeric id (the N of "#N"), or 0 when the
// row binds none — and 0 can never name an issue, so an unbound row can never
// post anywhere.
func (c Comment) IssueNumber() int {
	return issueRefNumber(c.Issue)
}

// Body renders the annotation the cmd shell posts through the trusted gh seam:
// the operator's note plus the anchor — the exact member SHA set and band the
// operator was reading — so the note lands bound to what was actually read
// rather than to a unit NAME that may mean different commits tomorrow. Pure
// render: producing this text has no side effect.
func (c Comment) Body() string {
	var b strings.Builder
	fmt.Fprintf(&b, "## Operator comment (`fak steer comment`)\n\n%s\n\n", c.Note)
	b.WriteString("## Anchor — what was read when this note was written\n\n")
	fmt.Fprintf(&b, "- Unit: `%s`\n", c.Leaf)
	fmt.Fprintf(&b, "- Band at comment: %s\n", c.Band)
	b.WriteString("- Member SHA set at comment:\n")
	for _, sha := range normalizeSHASet(c.SHAs) {
		fmt.Fprintf(&b, "  - `%s`\n", sha)
	}
	b.WriteString("\nThis is an ANNOTATION: it changes no band, no ack, and nothing that landed.\n")
	b.WriteString("The commits above are witnessed, landed history and stay exactly as they are.\n")
	fmt.Fprintf(&b, "\n— commented by %s at %s\n", c.By, c.At)
	return b.String()
}
