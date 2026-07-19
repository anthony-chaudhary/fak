package steerpr

// `fak steer ack` (#5028): the operator's half of the RESIDUAL band.
//
// A RESIDUAL band says "an oracle could not confirm this claim, so operator
// attention buys something here". The ack is where that attention LANDS: a
// human records "I looked at this and satisfied myself". Three rules keep the
// record honest:
//
//   - An ack binds to the unit's exact member SHA SET at ack time, never to
//     the unit's name. A commit that joins (or leaves) the unit invalidates
//     the ack — the human never reviewed the new set — and the unit reads
//     RESIDUAL/unacked again. Otherwise an old look would silently bless code
//     that landed after the human stopped looking.
//   - An ack NEVER touches the machine band. Nothing in this file writes a
//     Verdict or a Band, and the fold has nowhere to take one (see the #5036
//     fences in antigaming_test.go and overlay_ack_nonforge_test.go). The
//     strongest render an ack can produce is "RESIDUAL (acked by X)" via
//     BandLabel — an additive suffix on the honest band, never CLEARED, and
//     it does not deflate the Residual count.
//   - The ledger is append-only and attributable: every row carries who and
//     when; rows are only ever appended, never rewritten — never rewrite a
//     peer's row.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// AckSchema is the machine identifier for one ack ledger row.
const AckSchema = "fak.steerpr.ack.v1"

// Ack is one appended ledger row: a human's "I looked" against a unit's exact
// member SHA set. It is a self-report — deliberately weaker than the machine's
// diff-witnessed bit — stored and rendered BESIDE the band, never in it.
type Ack struct {
	Schema string   `json:"schema"`
	Leaf   string   `json:"leaf"`
	By     string   `json:"by"`
	At     string   `json:"at"`             // RFC3339 UTC: when the human looked
	SHAs   []string `json:"shas"`           // exact member set at ack time (sorted, deduped)
	Note   string   `json:"note,omitempty"` // optional operator note
}

// NewAck builds a validated ledger row. An unattributable ack (no by), an
// unnamed unit, or an empty member set is refused rather than defaulted: an
// ack is a record of a specific person having reviewed a specific set of
// commits, and a row missing either leg is not evidence of anything.
func NewAck(leaf, by string, shas []string, note string, at time.Time) (Ack, error) {
	leaf = strings.TrimSpace(leaf)
	by = strings.TrimSpace(by)
	set := normalizeSHASet(shas)
	switch {
	case leaf == "":
		return Ack{}, fmt.Errorf("an ack must name the unit it lands on")
	case by == "":
		return Ack{}, fmt.Errorf("an ack must be attributable: say who looked (--by, or set git config user.name)")
	case len(set) == 0:
		return Ack{}, fmt.Errorf("an ack binds to the reviewed member SHA set, and an empty set reviewed nothing")
	}
	return Ack{
		Schema: AckSchema,
		Leaf:   leaf,
		By:     by,
		At:     at.UTC().Format(time.RFC3339),
		SHAs:   set,
		Note:   strings.TrimSpace(note),
	}, nil
}

// AckLedgerPath is the overlay ack ledger's location under a repo root:
// gitignored runtime state beside the other .fak ledgers, one JSON row per
// line.
func AckLedgerPath(root string) string {
	return filepath.Join(root, ".fak", "steer-acks.jsonl")
}

// AppendAck appends one row to the ledger. Append-only by construction: the
// file is opened O_APPEND and rows are only ever added. An incomplete row is
// refused so the ledger stays attributable end to end.
func AppendAck(path string, a Ack) error {
	if strings.TrimSpace(a.Leaf) == "" || strings.TrimSpace(a.By) == "" || len(normalizeSHASet(a.SHAs)) == 0 {
		return fmt.Errorf("refusing an incomplete ack row: it needs the leaf, who looked, and the member SHA set")
	}
	line, err := json.Marshal(a)
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

// LoadAcks reads the ledger best-effort: a missing or unreadable file is an
// empty ledger, and a torn or foreign line is skipped rather than poisoning
// the rows around it. Failure never invents an ack — a row that does not parse
// simply covers nothing.
func LoadAcks(path string) []Ack {
	buf, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var out []Ack
	for _, line := range strings.Split(string(buf), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var a Ack
		if json.Unmarshal([]byte(line), &a) != nil {
			continue
		}
		out = append(out, a)
	}
	return out
}

// Covers reports whether the ack's reviewed set is EXACTLY the given member
// set. Any drift — a member added, removed, or substituted — uncovers the ack:
// it was a review of a different SHA set, and letting it stand would bless
// code the human never saw.
func (a Ack) Covers(shas []string) bool {
	have := normalizeSHASet(a.SHAs)
	want := normalizeSHASet(shas)
	if len(want) == 0 || len(have) != len(want) {
		return false
	}
	for i := range have {
		if have[i] != want[i] {
			return false
		}
	}
	return true
}

// AckFor returns the LATEST ledger row that acks leaf's exact current member
// set. Later rows win because the ledger is append-only: a re-ack after the
// set changed and was re-reviewed is a newer fact about the same set.
func AckFor(acks []Ack, leaf string, shas []string) (Ack, bool) {
	leaf = strings.TrimSpace(leaf)
	var found Ack
	ok := false
	for _, a := range acks {
		if strings.TrimSpace(a.Leaf) == leaf && a.Covers(shas) {
			found, ok = a, true
		}
	}
	return found, ok
}

// UnitSHAs is the unit's current member SHA set — the thing an ack binds to.
func UnitSHAs(u Unit) []string {
	shas := make([]string, 0, len(u.Commits))
	for _, c := range u.Commits {
		shas = append(shas, c.SHA)
	}
	return normalizeSHASet(shas)
}

// BandLabel renders the operator-facing band with the acked state as a SUFFIX
// beside it: "RESIDUAL (acked by X)". The two facts stay side by side and
// unmerged — the Band half comes from the witness fold and only from it, so an
// acked residual can never read CLEARED, and this function returns prose that
// nothing feeds back into a fold.
func BandLabel(band Band, ack Ack, acked bool) string {
	if !acked {
		return string(band)
	}
	return fmt.Sprintf("%s (acked by %s)", band, ack.By)
}

// normalizeSHASet trims, drops empties, dedupes, and sorts, so set identity is
// order- and duplicate-insensitive.
func normalizeSHASet(shas []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(shas))
	for _, s := range shas {
		s = strings.TrimSpace(s)
		if s == "" || seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}
