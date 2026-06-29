package marketing

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// honesty.go — the CLAIMS.md cross-check. The witness rung (ship.go/claim.go) proves a
// commit LANDED; it does not prove the FEATURE that commit touches is actually shipped
// rather than a labeled stub. CLAIMS.md is the repo's honesty ledger: every capability line
// carries exactly one of [SHIPPED]/[SIMULATED]/[STUB]. A `feat(grammar): ...` commit can be
// a real, witnessed ship of PLUMBING toward a capability that CLAIMS.md still tags [STUB]
// (e.g. "the schema→token compiler is the named follow-on"). Marketing that capability as
// shipped — even hedged — would be an unshipped-feature boast on a public surface.
//
// So before a ship becomes a marketing claim it is cross-checked against the ledger: if a
// [STUB]/[SIMULATED] line names the ship's subsystem (its internal/<leaf>) or shares an
// issue ref (#N) with the ship's subject, the ship is EXCLUDED from the artifact (excluded,
// not softened — there is no "in progress" framing on a public channel in v1). Marketing
// only [SHIPPED]-witnessed leaves.
//
// The check is deliberately conservative: a false EXCLUDE (dropping a real ship because its
// leaf happens to appear in a stub line) costs one un-marketed feature; a false INCLUDE
// (marketing a stub) costs the honesty floor. We prefer the former.

// ledgerLineRE matches a CLAIMS.md capability line and captures its tag. Lines begin
// "- [TAG] ..." per the file's own lint contract (unit 96).
var ledgerLineRE = regexp.MustCompile(`^\s*-\s*\[(SHIPPED|SIMULATED|STUB)\]\s*(.*)$`)

// leafTokenRE pulls every internal/<leaf> or cmd/<leaf> subsystem token out of a ledger
// line — the join key against a ship's Leaf.
var leafTokenRE = regexp.MustCompile(`(?:internal|cmd)/([A-Za-z0-9][\w.\-]*)`)

// issueRefRE pulls every #N issue reference out of a line — the secondary join key.
var issueRefRE = regexp.MustCompile(`#(\d+)`)

// ClaimsLedger is the parsed CLAIMS.md: the set of subsystem leaves and issue refs that
// appear in any line NOT tagged [SHIPPED] (i.e. [SIMULATED] or [STUB]). A ship whose leaf
// or issue is in either set is unmarketable. Built once per Tick; pure after construction.
type ClaimsLedger struct {
	unshippedLeaves map[string]bool // lowercased <leaf> named by a [SIMULATED]/[STUB] line
	unshippedIssues map[int]bool    // #N named by a [SIMULATED]/[STUB] line
	loaded          bool            // false when CLAIMS.md was unreadable (then the gate is OPEN — see Marketable)
}

// LoadClaims reads CLAIMS.md at root and indexes the [SIMULATED]/[STUB] lines. An unreadable
// file yields a not-loaded ledger: Marketable then passes everything (the witness rung still
// holds; we don't fail the whole loop because one doc is missing). This mirrors the
// graceful-degrade posture readLaneTaxonomy takes on a missing dos.toml.
func LoadClaims(root string) ClaimsLedger {
	l := ClaimsLedger{unshippedLeaves: map[string]bool{}, unshippedIssues: map[int]bool{}}
	b, err := os.ReadFile(filepath.Join(root, "CLAIMS.md"))
	if err != nil {
		return l
	}
	l.loaded = true
	for _, raw := range strings.Split(string(b), "\n") {
		m := ledgerLineRE.FindStringSubmatch(raw)
		if m == nil {
			continue
		}
		tag, body := m[1], m[2]
		if tag == "SHIPPED" {
			continue // a [SHIPPED] line is the green light, not a block
		}
		for _, lm := range leafTokenRE.FindAllStringSubmatch(body, -1) {
			l.unshippedLeaves[strings.ToLower(lm[1])] = true
		}
		for _, im := range issueRefRE.FindAllStringSubmatch(body, -1) {
			l.unshippedIssues[atoiSafe(im[1])] = true
		}
	}
	return l
}

// Marketable reports whether a ship may be marketed as shipped, and a structured reason when
// not. The gate: a ship is UNmarketable if a [SIMULATED]/[STUB] CLAIMS.md line names its leaf
// or shares an issue ref with its subject. When the ledger did not load, the gate is open
// (every ship passes) — the witness rung is the floor, the ledger is the additional honesty
// rung on top of it.
func (l ClaimsLedger) Marketable(s Ship) (ok bool, reason string) {
	if !l.loaded {
		return true, ""
	}
	if s.Leaf != "" && l.unshippedLeaves[strings.ToLower(s.Leaf)] {
		return false, "leaf `" + s.Leaf + "` is named by a [SIMULATED]/[STUB] line in CLAIMS.md"
	}
	for _, n := range subjectIssueRefs(s.Subject) {
		if l.unshippedIssues[n] {
			return false, "issue #" + itoa(n) + " is named by a [SIMULATED]/[STUB] line in CLAIMS.md"
		}
	}
	return true, ""
}

// FilterMarketable splits ships into the marketable set and the excluded set (each excluded
// ship paired with its reason), so a caller can both build the honest artifact AND surface
// WHAT was withheld (a silent drop would read as "nothing to say" when the truth is "we held
// a stub back"). The marketable order is preserved.
func FilterMarketable(l ClaimsLedger, ships []Ship) (marketable []Ship, excluded []ExcludedShip) {
	for _, s := range ships {
		if ok, reason := l.Marketable(s); ok {
			marketable = append(marketable, s)
		} else {
			excluded = append(excluded, ExcludedShip{Ship: s, Reason: reason})
		}
	}
	return marketable, excluded
}

// ExcludedShip is a ship withheld from the artifact by the CLAIMS.md gate, with the reason —
// surfaced so the honesty hold is visible, never silent.
type ExcludedShip struct {
	Ship   Ship
	Reason string
}

// subjectIssueRefs returns the #N refs in a commit subject (the join key against the ledger's
// issue set).
func subjectIssueRefs(subject string) []int {
	var out []int
	for _, m := range issueRefRE.FindAllStringSubmatch(subject, -1) {
		out = append(out, atoiSafe(m[1]))
	}
	return out
}

// atoiSafe / itoa are tiny stdlib-only helpers (the package stays dependency-light); a
// non-numeric ref can't occur given the regex, so atoiSafe never needs to report an error.
func atoiSafe(s string) int {
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			return n
		}
		n = n*10 + int(c-'0')
	}
	return n
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
