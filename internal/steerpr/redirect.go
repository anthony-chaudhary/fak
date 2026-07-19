package steerpr

// `fak steer redirect` (#5030): re-aim an intent without touching the merge.
//
// When an operator reads a forming unit and concludes the fleet is building
// the wrong thing, the landed commits are NOT the problem — they are correct,
// witnessed, and merged. The thing that needs to change is the NEXT tick.
// A redirect records that steering act as a first-class, countable event.
// Three rules keep it honest:
//
//   - A redirect targets the intent's FUTURE, never its past. Nothing in this
//     file (or reachable from it) can revert, rewrite, force-push, or touch
//     the trunk: the leaf stays subprocess-free and internal-import-free, and
//     TestRedirectNeverReachesGitMutation makes that promise structural
//     rather than prose. The redirect's only outward act is a follow-up filed
//     through the trusted gh seam in the cmd/fak verb shell — GitHub moves,
//     git does not.
//   - A redirect anchors to what the operator actually read: the unit's exact
//     member SHA SET and its band at redirect time ride on the record and on
//     the filed follow-up, so the steer note can never drift onto commits the
//     operator never saw.
//   - The ledger is append-only and attributable: every row carries who and
//     when; rows are only ever appended, never rewritten — never rewrite a
//     peer's row. Loading the ledger is how the redirect becomes COUNTABLE
//     (input to the dogfood + effectiveness children of #5015).

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// RedirectSchema is the machine identifier for one redirect ledger row.
const RedirectSchema = "fak.steerpr.redirect.v1"

// Redirect is one appended ledger row: an operator's "aim the next tick over
// there" against a unit, anchored to the exact member SHA set and band the
// operator was reading. It is ADVISORY by construction — the record steers the
// intent's future and has no field that could touch a Verdict, a Band, or git.
type Redirect struct {
	Schema string   `json:"schema"`
	Leaf   string   `json:"leaf"`
	By     string   `json:"by"`
	At     string   `json:"at"`   // RFC3339 UTC: when the operator steered
	Note   string   `json:"note"` // REQUIRED: where the intent's next tick should aim
	SHAs   []string `json:"shas"` // exact member set at redirect time (sorted, deduped)
	Band   Band     `json:"band"` // band at redirect time (anchor, never rewritten)
	// Issue is the unit's closure-grade bound issue ("#N") when it has one —
	// the follow-up reopens/annotates it rather than filing fresh.
	Issue string `json:"issue,omitempty"`
	// FollowUp is where the filed follow-up landed (an issue ref or URL),
	// recorded after the trusted gh seam files it.
	FollowUp string `json:"follow_up,omitempty"`
}

// NewRedirect builds a validated ledger row. An unnamed unit, an
// unattributable redirect (no by), an EMPTY NOTE, or an empty member set is
// refused rather than defaulted: a redirect is a specific person re-aiming a
// specific intent somewhere specific, and a row missing any leg steers
// nothing.
func NewRedirect(leaf, by, note string, shas []string, band Band, issue string, at time.Time) (Redirect, error) {
	leaf = strings.TrimSpace(leaf)
	by = strings.TrimSpace(by)
	note = strings.TrimSpace(note)
	set := normalizeSHASet(shas)
	switch {
	case leaf == "":
		return Redirect{}, fmt.Errorf("a redirect must name the unit it re-aims")
	case by == "":
		return Redirect{}, fmt.Errorf("a redirect must be attributable: say who is steering (--by, or set git config user.name)")
	case note == "":
		return Redirect{}, fmt.Errorf("a redirect must say where the intent goes next (-m): an empty note steers nothing")
	case len(set) == 0:
		return Redirect{}, fmt.Errorf("a redirect anchors to the unit's member SHA set, and an empty set anchors nothing")
	}
	return Redirect{
		Schema: RedirectSchema,
		Leaf:   leaf,
		By:     by,
		At:     at.UTC().Format(time.RFC3339),
		Note:   note,
		SHAs:   set,
		Band:   band,
		Issue:  strings.TrimSpace(issue),
	}, nil
}

// RedirectLedgerPath is the overlay redirect ledger's location under a repo
// root: gitignored runtime state beside the other .fak ledgers, one JSON row
// per line.
func RedirectLedgerPath(root string) string {
	return filepath.Join(root, ".fak", "steer-redirects.jsonl")
}

// AppendRedirect appends one row to the ledger. Append-only by construction:
// the file is opened O_APPEND and rows are only ever added. An incomplete row
// is refused so every ledgered steering event stays attributable and anchored.
func AppendRedirect(path string, r Redirect) error {
	if strings.TrimSpace(r.Leaf) == "" || strings.TrimSpace(r.By) == "" ||
		strings.TrimSpace(r.Note) == "" || len(normalizeSHASet(r.SHAs)) == 0 {
		return fmt.Errorf("refusing an incomplete redirect row: it needs the leaf, who steered, the note, and the anchor SHA set")
	}
	line, err := json.Marshal(r)
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

// LoadRedirects reads the ledger best-effort: a missing or unreadable file is
// an empty ledger, and a torn or foreign line is skipped rather than poisoning
// the rows around it. Failure never invents a steering event.
func LoadRedirects(path string) []Redirect {
	buf, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var out []Redirect
	for _, line := range strings.Split(string(buf), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var r Redirect
		if json.Unmarshal([]byte(line), &r) != nil {
			continue
		}
		out = append(out, r)
	}
	return out
}

// RedirectsFor returns the ledgered redirects against one unit, oldest first —
// the countable steering-event view (#5030's "first-class, countable" leg):
// len(RedirectsFor(...)) is how often an operator has re-aimed this intent.
func RedirectsFor(rows []Redirect, leaf string) []Redirect {
	leaf = strings.TrimSpace(leaf)
	var out []Redirect
	for _, r := range rows {
		if strings.TrimSpace(r.Leaf) == leaf {
			out = append(out, r)
		}
	}
	return out
}

// FollowUpTitle renders the filed follow-up's title.
func (r Redirect) FollowUpTitle() string {
	return fmt.Sprintf("steer redirect: re-aim the %s intent's next tick", r.Leaf)
}

// FollowUpBody renders the follow-up the cmd shell files through the trusted
// gh seam: the operator's note plus the anchor — the exact member SHA set and
// band the operator was reading — so the steer lands bound to what was
// actually read. Pure render: producing this text has no side effect.
func (r Redirect) FollowUpBody() string {
	var b strings.Builder
	fmt.Fprintf(&b, "## Operator redirect (`fak steer redirect`)\n\n%s\n\n", r.Note)
	b.WriteString("## Anchor — what was read when this steer was recorded\n\n")
	fmt.Fprintf(&b, "- Unit: `%s`\n", r.Leaf)
	fmt.Fprintf(&b, "- Band at redirect: %s\n", r.Band)
	if r.Issue != "" {
		fmt.Fprintf(&b, "- Bound issue: %s\n", r.Issue)
	}
	b.WriteString("- Member SHA set at redirect:\n")
	for _, sha := range normalizeSHASet(r.SHAs) {
		fmt.Fprintf(&b, "  - `%s`\n", sha)
	}
	b.WriteString("\nThe commits above are witnessed, landed history and stay exactly as they are.\n")
	b.WriteString("A redirect re-aims the intent's NEXT tick — it never reverts, rewrites, or\ntouches the trunk.\n")
	fmt.Fprintf(&b, "\n— redirected by %s at %s\n", r.By, r.At)
	return b.String()
}
