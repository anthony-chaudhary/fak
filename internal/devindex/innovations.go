package devindex

// innovations.go folds docs/INNOVATIONS-INDEX.md's Status column into the leaf
// catalog and RECONCILES it against the CLAIMS.md rollup (C2 #1289). The two
// ledgers answer the same "is this shipped?" question from opposite ends - one
// hand-curated per-innovation, one lint-enforced per-claim - so where they
// disagree that disagreement is a FINDING an agent can act on, never a silent
// pick of whichever file happened to be read. It is a VIEW: it reads the bytes
// each authority already owns and never rewrites them, exactly like the
// CLAIMS.md join it reconciles against.

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// Innovation is one row of the INNOVATIONS-INDEX.md concept catalog: the
// innovation's name, the general concept it embodies, the lanes its Home
// package references bind to, and its raw Status cell. Home is resolved with the
// SAME LaneForPath the CLAIMS.md join uses, so both ledgers bind to one taxonomy;
// a Home token that names no declared lane is kept verbatim as an unbound label
// (HomeLanes carries only the bound lanes) rather than dropped or fatal.
type Innovation struct {
	Name      string   `json:"name"`
	Concept   string   `json:"concept,omitempty"`
	Home      []string `json:"home,omitempty"`       // raw Home tokens, backticks stripped
	HomeLanes []string `json:"home_lanes,omitempty"` // Home tokens that resolved to a declared lane
	Status    string   `json:"status"`
}

// ReconcileFinding is one disagreement between the INNOVATIONS-INDEX.md Status
// column and the CLAIMS.md maturity rollup for a single leaf. It is emitted, not
// resolved: the two sources are both authoritative-ish and a human (or the C6
// freshness gate, #1293) decides. Source names the ledger that made the claim,
// Detail the exact polarity, so the finding is self-contained.
type ReconcileFinding struct {
	Leaf   string `json:"leaf"`
	Source string `json:"source"`
	Detail string `json:"detail"`
}

// reconcileSourceInnovations is the Source label for findings where the
// INNOVATIONS-INDEX asserts a leaf is SHIPPED but CLAIMS.md binds no SHIPPED claim.
const reconcileSourceInnovations = "innovations-index"

// reconcileSourceClaims is the Source label for the converse: CLAIMS.md binds a
// SHIPPED claim to a leaf the INNOVATIONS-INDEX never marks as shipped.
const reconcileSourceClaims = "CLAIMS.md"

// innovationHeaderRE matches a Part 2 table header row. The header is repeated
// mid-file (once per subsection), so the parser re-arms on every occurrence - it
// cannot assume a single table. The four named columns are the contract; a table
// with different columns (Part 3's "General concept | Essence | Expressed |
// Lives in") is deliberately not this shape and is skipped.
var innovationHeaderRE = regexp.MustCompile(`^\|\s*Innovation\s*\|\s*Concept it embodies\s*\|\s*Home\s*\|\s*Status\s*\|\s*$`)

// mdTableSepRE matches a Markdown table separator row (`|---|---|---|`, with
// optional colons/whitespace) so the parser skips it between the header and data.
var mdTableSepRE = regexp.MustCompile(`^\|[\s:|-]+\|$`)

// backtickTokenRE extracts the backticked tokens of a table cell - the Home cell
// spells each package as “ `name` “, and (for the Part 3 rows) a doc path the
// same way. Non-backticked prose in the cell is ignored: only an explicitly named
// home counts.
var backtickTokenRE = regexp.MustCompile("`([^`]+)`")

// parseInnovations scans the INNOVATIONS-INDEX.md tables into c.Innovations. It
// tracks the header/separator/data row shape so a repeated header re-arms a new
// table rather than being read as data, and binds each row's Home tokens to lanes
// via LaneForPath. A row whose Home names no declared lane still parses (Home
// keeps the raw tokens; HomeLanes is empty) - the index is a hand-curated catalog
// and a novel package name must surface as an unbound label, never an error.
func (c *Catalog) parseInnovations(text string) {
	inTable := false
	pendingHeader := false // saw the header; waiting for the separator row
	for _, raw := range strings.Split(text, "\n") {
		line := strings.TrimSpace(raw)
		if !strings.HasPrefix(line, "|") {
			inTable, pendingHeader = false, false
			continue
		}
		if innovationHeaderRE.MatchString(line) {
			inTable, pendingHeader = true, true
			continue
		}
		if !inTable {
			continue
		}
		if pendingHeader {
			// The separator row immediately follows the header; anything else means
			// the "header" was actually prose and there is no table body.
			if mdTableSepRE.MatchString(line) {
				pendingHeader = false
			} else {
				inTable, pendingHeader = false, false
			}
			continue
		}
		cells := splitTableRow(line)
		if len(cells) != 4 {
			continue
		}
		home := backtickTokens(cells[2])
		row := Innovation{
			Name:    cells[0],
			Concept: cells[1],
			Home:    home,
			Status:  cells[3],
		}
		if row.Name == "" {
			continue
		}
		row.HomeLanes = c.lanesForHome(home)
		c.Innovations = append(c.Innovations, row)
	}
}

// splitTableRow splits a Markdown table data line into its trimmed cells. The
// outer pipes are dropped; an escaped `\|` is not used in this file, so a plain
// split on `|` is sufficient and cheaper than a full Markdown parser.
func splitTableRow(line string) []string {
	line = strings.TrimSpace(line)
	line = strings.TrimPrefix(line, "|")
	line = strings.TrimSuffix(line, "|")
	parts := strings.Split(line, "|")
	if len(parts) == 0 {
		return nil
	}
	out := make([]string, len(parts))
	for i, p := range parts {
		out[i] = strings.TrimSpace(p)
	}
	return out
}

// backtickTokens returns the backticked tokens of a cell in order, de-duplicated.
func backtickTokens(cell string) []string {
	var out []string
	seen := map[string]bool{}
	for _, m := range backtickTokenRE.FindAllStringSubmatch(cell, -1) {
		tok := strings.TrimSpace(m[1])
		if tok == "" || seen[tok] {
			continue
		}
		seen[tok] = true
		out = append(out, tok)
	}
	return out
}

// lanesForHome resolves each Home token to a lane, de-duplicated and sorted. A
// bare package name (`ctxmmu`) is tried as a lane key directly - the
// INNOVATIONS-INDEX Home column names packages, not paths - then through the
// standard path resolver (`internal/<name>`) for a path- or dir-shaped token.
// Tokens that resolve to nothing are omitted; they remain visible in Innovation.Home.
func (c *Catalog) lanesForHome(home []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, tok := range home {
		lane := c.laneForHomeToken(tok)
		if lane == "" || seen[lane] {
			continue
		}
		seen[lane] = true
		out = append(out, lane)
	}
	sort.Strings(out)
	return out
}

// laneForHomeToken resolves one Home token. A declared bare lane name wins (the
// Home column's common form); otherwise the token is treated as a repo-relative
// path and run through LaneForPath, which handles path-shaped references and the
// Part 3 doc-path entries.
func (c *Catalog) laneForHomeToken(tok string) string {
	t := strings.ToLower(strings.TrimSpace(tok))
	if t == "" {
		return ""
	}
	if c.declared[t] {
		return t
	}
	return c.LaneForPath(tok)
}

// shippedInInnovations reports whether the INNOVATIONS-INDEX Status cell is a
// PURE shipped claim: a bare `SHIPPED`, or a `SHIPPED (...)` whose qualifier does
// not itself carry an unresolved `[STUB]`/`[SIMULATED]` marker. `MIXED (...)` is
// never pure (it explicitly spans rungs), and a `SHIPPED` qualified by a `[STUB]`
// is not pure either - the qualifier is precisely the honesty caveat.
func shippedInInnovations(status string) bool {
	s := strings.ToUpper(strings.TrimSpace(status))
	if !strings.HasPrefix(s, "SHIPPED") {
		return false
	}
	rest := strings.TrimSpace(strings.TrimPrefix(s, "SHIPPED"))
	if !strings.HasPrefix(rest, "(") {
		return rest == ""
	}
	return !strings.Contains(rest, "[STUB]") && !strings.Contains(rest, "[SIMULATED]")
}

// Reconciliations computes, per declared leaf, the disagreements between the
// INNOVATIONS-INDEX.md Status column and the CLAIMS.md SHIPPED rollup. A leaf
// marked PURE SHIPPED in the index but carrying zero SHIPPED claims is one
// finding; the converse is another. `MIXED` rows never produce a finding by
// themselves. The result is sorted by leaf then source, so two runs over the same
// tree are byte-identical. Pure and stdlib-only: it reads only in-memory state.
func (c *Catalog) Reconciliations() []ReconcileFinding {
	if len(c.Leaves) == 0 {
		return nil
	}
	idxShipped := map[string]bool{}
	for _, in := range c.Innovations {
		if !shippedInInnovations(in.Status) {
			continue
		}
		for _, lane := range in.HomeLanes {
			idxShipped[lane] = true
		}
	}
	claimsShipped := map[string]bool{}
	for _, cl := range c.Claims {
		if cl.Tag != "SHIPPED" {
			continue
		}
		for _, lane := range cl.Lanes {
			claimsShipped[lane] = true
		}
	}

	var out []ReconcileFinding
	for _, l := range c.Leaves {
		idx, clm := idxShipped[l.Name], claimsShipped[l.Name]
		switch {
		case idx && !clm:
			out = append(out, ReconcileFinding{
				Leaf:   l.Name,
				Source: reconcileSourceInnovations,
				Detail: "INNOVATIONS-INDEX marks this leaf SHIPPED but CLAIMS.md binds no SHIPPED claim to it",
			})
		case clm && !idx:
			out = append(out, ReconcileFinding{
				Leaf:   l.Name,
				Source: reconcileSourceClaims,
				Detail: "CLAIMS.md binds a SHIPPED claim to this leaf but INNOVATIONS-INDEX never marks it SHIPPED",
			})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Leaf != out[j].Leaf {
			return out[i].Leaf < out[j].Leaf
		}
		return out[i].Source < out[j].Source
	})
	return out
}

// loadInnovations reads docs/INNOVATIONS-INDEX.md under root into the catalog. A
// missing or short file degrades to an empty catalog, never an error - the same
// discipline the CLAIMS.md ledger follows. The index is a hand-curated
// counterpart, and an agent with no CLI must still get the leaf/doc/claim views.
func (c *Catalog) loadInnovations() {
	b, err := os.ReadFile(filepath.Join(c.Root, "docs", "INNOVATIONS-INDEX.md"))
	if err != nil {
		return
	}
	c.parseInnovations(string(b))
}

// InnovationsForLeaf returns the innovation rows bound to the named
// (case-insensitive) leaf, in document order.
func (c *Catalog) InnovationsForLeaf(name string) []Innovation {
	n := strings.ToLower(strings.TrimSpace(name))
	var out []Innovation
	for _, in := range c.Innovations {
		for _, lane := range in.HomeLanes {
			if lane == n {
				out = append(out, in)
				break
			}
		}
	}
	return out
}
