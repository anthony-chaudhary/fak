package scorecard

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"
)

// Generated doc-folder file names.
const (
	ReadmeName = "README.md"
	IndexName  = "INDEX.md"
)

func frontMatter(title, desc string) []string {
	return []string{"---", fmt.Sprintf(`title: "%s"`, title), fmt.Sprintf(`description: "%s"`, desc), "---", ""}
}

// provenanceRow refuses an undated scorecard: an empty as_of or fak_version is
// treated as missing.
func provenanceRow(c *Corpus) (string, error) {
	asOf, version := "", ""
	if truthy(c.AsOf) {
		asOf = pyStrip(pyStr(c.AsOf))
	}
	if truthy(c.FakVersion) {
		version = pyStrip(pyStr(c.FakVersion))
	}
	var missing []string
	if asOf == "" {
		missing = append(missing, "as_of")
	}
	if version == "" {
		missing = append(missing, "fak_version")
	}
	if len(missing) > 0 {
		return "", fmt.Errorf("concept-disambiguation scorecard: refusing to render an undated scorecard — %s"+
			" is empty or absent in the data directory's _meta.json \"meta\" block. "+
			"Populate it the way the sibling scorecards do "+
			"(docs/industry-scorecard, docs/persona-fit-scorecard): an undated derived claim "+
			"is not a derived claim.", strings.Join(missing, ", "))
	}
	return fmt.Sprintf("| As of | %s (fak %s) |", asOf, version), nil
}

func stampComment(stamp string) string {
	return fmt.Sprintf("<!-- concept-disambiguation-scorecard: %s - process: tools/concept_disambiguation_scorecard.py - data: %s/ -->", stamp, DataDirRel)
}

// RenderDocIndex renders the generated README.md (without its trailing newline).
func (p *Payload) RenderDocIndex(stamp string) (string, error) {
	c, empty := p.view()
	cov := c.Coverage
	out := frontMatter(
		"fak concept-disambiguation scorecard - is every similar-sounding concept crystal-clear",
		"Inward naming scorecard: each confusable fak concept positioned on the grounded / defined / "+
			"disambiguated / anchored axes, with one clarity verdict per concept. Two driven numbers: "+
			"coverage (of the confusable concept space discovered in the tree) and disambiguation-debt.")
	out = append(out, "# Concept-disambiguation scorecard - crystal clarity across similar-sounding names", "")
	if stamp != "" {
		out = append(out, stampComment(stamp), "")
	}
	out = append(out, "The sibling scorecards grade fak's code, docs, and competitive standing. This one asks "+
		"the question that bites a reader as the system grows: **of the massive, growing set of "+
		"similar-sounding names (cache, vCache, KV cache, cachemeta, the provider prompt-cache), "+
		"is each distinct concept crystal-clear - one canonical name, a written definition, and an "+
		"explicit line drawn against the siblings it is confused with?** Every number below is "+
		"re-derived by `tools/concept_disambiguation_scorecard.py` and cross-checked against the "+
		"real tree (the grounding token must appear in the production corpus; the glossary anchor "+
		"must exist; a `distinct_from` reference must resolve). No verdict is hand-typed.", "")
	out = append(out, "> Regenerate: `fak concept generate` (the canonical writer preserves "+
		"classifications while refreshing every generated page).", "")
	out = append(out, "## Headline", "")
	out = append(out, "The driver is the UNBOUNDED disambiguation-debt (drive it to 0) plus the positive "+
		"counters that climb without a ceiling. The old bounded /100 score SATURATES near the "+
		"top - it hides the real, still-open work - so it is demoted to a labeled legacy line "+
		"below and is NOT the metric to optimize.", "")
	out = append(out, "| Metric (primary = unbounded driver) | Value |", "|---|---|")
	out = append(out, fmt.Sprintf("| **Disambiguation-debt (drive to 0)** | **%d** (clarity %d + coverage %d) |",
		c.DisambiguationDebt, c.ClarityDefects, c.CoverageDebt))
	out = append(out, fmt.Sprintf("| **Crystal-clear concepts (and climbing)** | **%d** crystal of %d positioned |",
		c.CrystalConcepts, c.Rows))
	out = append(out, fmt.Sprintf("| **Confusable tokens positioned (covered / discovered)** | **%d / %d** (%s%% of the discovered confusable space) |",
		cov.Covered, cov.Discovered, floatOr(empty, cov.CoveragePct, "0")))
	out = append(out, fmt.Sprintf("| **Undrawn twin-pairs (drive to 0)** | **%d** of %d confusable name-pairs |",
		c.Separation.Undrawn, c.Separation.ConfusablePairs))
	out = append(out, fmt.Sprintf("| **Ambiguous lookup names (drive to 0)** | **%d** of %d indexed names |",
		c.Index.AmbiguousKeys, c.Index.Keys))
	prov, err := provenanceRow(c)
	if err != nil {
		return "", err
	}
	out = append(out, prov)
	out = append(out, fmt.Sprintf("| Legacy bounded score (saturates; not the driver) | %s/100 (grade %s) |",
		floatOr(empty, c.Score, "0"), c.Grade), "")
	out = append(out, "> **Read this right.** The metric to optimize is the UNBOUNDED disambiguation-debt "+
		"(drive it toward 0) and the counters that climb without a ceiling (crystal concepts, "+
		"confusable tokens positioned). The bounded /100 score SATURATES - once the catalogued "+
		"namespace is clean it sits near 100 and can no longer tell you how much confusable "+
		"space is still un-disambiguated - so it is kept only as a labeled legacy line, not the "+
		"driver.", "")
	out = append(out, "## Standing at a glance", "", "```text", p.RenderChart(), "```", "")
	out = append(out, "## The clarity ladder", "", "| Verdict | Means |", "|---|---|",
		"| * crystal | defined + a line drawn against siblings + that line anchored in a doc that exists |",
		"| o defined | defined + a distinction line, but the line is not written in a discoverable doc |",
		"| ~ drifting | defined, but no line drawn against its siblings (you know what it is, not what it is NOT) |",
		"| = entangled | defined and draws SOME line, but not against the twin its own NAME is confusable with - per-row clean, pairwise still fog |",
		"| x colliding | shares a canonical name with another concept - a true ambiguity, fixable only by a rename |",
		"| . undocumented | the catalog gives no definition |", "")
	sp := c.Separation
	ix := c.Index
	out = append(out, "## Separation - is each concept disambiguated FROM THE OTHERS?", "")
	out = append(out, "Per-concept clarity is not the same question as pairwise separation. A concept "+
		"in a 250-member family satisfies `disambiguated` by naming **any one** sibling - "+
		"while the twin its name is genuinely mistakable for goes undrawn. So the "+
		"scorecard discovers, from the catalog's own names, the pairs a reader cannot "+
		"keep apart - **permuted** (`witnessPath` / `PathWitness`: the same words in a "+
		"different order) and **near** (`SessionRef` / `SessionRow`: a couple of edits "+
		"apart) - and asks whether the line is drawn between *those two specifically*. "+
		"Boundaries are directed, so for a confusable pair the line must be drawn from "+
		"**both** sides: `A.distinct_from = [B]` does nothing for a reader who arrived "+
		"at B. A concept that does not separate from its own twin is `entangled` - "+
		"clean per row, still fog pairwise.", "")
	out = append(out, "| Separation metric | Value |", "|---|---|")
	out = append(out, fmt.Sprintf("| Confusable name-pairs discovered | %d |", sp.ConfusablePairs))
	out = append(out, fmt.Sprintf("| **Separated from each other (drive to all)** | **%d / %d** (%d mutual, %d one-sided) |",
		sp.Separated, sp.ConfusablePairs, sp.Mutual, sp.OneSided))
	out = append(out, fmt.Sprintf("| **Undrawn twin-pairs (drive to 0)** | **%d** |", sp.Undrawn))
	out = append(out, fmt.Sprintf("| Entangled concepts (own twin undrawn) | %d |", sp.EntangledConcepts))
	out = append(out, fmt.Sprintf("| Boundaries drawn (mutual / total) | %d / %d |", sp.MutualBoundaries, sp.Boundaries))
	out = append(out, fmt.Sprintf("| Dangling `distinct_from` references (drive to 0) | %d |", sp.DanglingReferences), "")
	out = append(out, "## Indexing - can a reader who meets a NAME find the concept?", "")
	out = append(out, "The catalog is organised by concept; a reader arrives with a **spelling**. "+
		"[`INDEX.md`](INDEX.md) turns it around: every surface a name can arrive as - the "+
		"canonical, every alias, the raw grounding token - is a lookup key pointing at the "+
		"concept it denotes, carrying that concept's contrast set so *which one is this?* "+
		"and *how is it not the other one?* are answered together. A key that lands on two "+
		"concepts is an **index** defect the canonical names cannot show: both canonicals "+
		"may be unique while one row quietly claims the other's name as an alias.", "")
	out = append(out, "| Index metric | Value |", "|---|---|")
	out = append(out, fmt.Sprintf("| Lookup names indexed | %d over %d concepts |", ix.Keys, c.Rows))
	out = append(out, fmt.Sprintf("| Lookup names landing on several concepts | %d |", ix.AmbiguousKeys))
	out = append(out, fmt.Sprintf("| **Shared names whose concepts stay unseparated (drive to 0)** | **%d** |", ix.UnresolvedSharedNames))
	out = append(out, fmt.Sprintf("| Concepts carrying a contrast set | %d |", ix.ContrastConcepts), "")
	out = append(out, "## The concepts (best verdict first)", "")
	out = append(out, "| | Verdict | Kind | Family | Canonical - definition |", "|---|---|---|---|---|")
	for _, row := range sortedLeaderboard(c.Leaderboard) {
		out = append(out, fmt.Sprintf("| %s | %s | %s | %s | **%s** - %s |", markOf(row.Verdict), pyStr(row.Verdict),
			pyStr(row.Kind), pyStr(row.Family), pyStr(row.Canonical), pyStr(row.Definition)))
	}
	out = append(out, "")
	ru := c.Rollup
	if len(ru.Abstractions) > 0 {
		out = append(out, "## Concept roll-up (the namespace at its abstraction heads)", "")
		out = append(out, "Most readers cannot hold every concept at once. The optional `parent` forest lets "+
			"the catalog roll concepts up to the abstraction that HEADS them - and the roll-up is "+
			"**weakest-link**: an abstraction reads as crystal only when *every* concept beneath it "+
			"is. A single `defined` leaf keeps the whole head from rolling up to crystal, so the "+
			"collapsed view can never hide fog it contains. `!` marks a head whose declared verdict "+
			"reads clearer than its subtree supports.", "")
		out = append(out, "```text", p.RollupText(), "```", "")
		out = append(out, "| | Abstraction | Rolled | Head declares | Subtree | Debt | Weakest descendant |", "|---|---|---|---|---:|---:|---|")
		for _, a := range ru.Abstractions {
			if a.Depth != 0 {
				continue
			}
			flag := ""
			if a.Overclaim {
				flag = "!"
			}
			out = append(out, fmt.Sprintf("| %s%s | **%s** (`%s`) | %s | %s | %d | %d | %s = %s |", markOf(a.RolledVerdict), flag,
				pyStr(a.Canonical), a.ID, a.RolledVerdict, pyStr(a.DeclaredVerdict), a.SubtreeSize, a.SubtreeOwed,
				a.Weakest.ID, pyStr(a.Weakest.Verdict)))
		}
		out = append(out, "")
	}
	out = append(out, "## Per-KPI (disambiguation-debt = clarity of the rows that exist)", "")
	out = append(out, "| Group | KPI | Score | Debt | Detail |", "|---|---|---:|:--:|---|")
	for _, b := range c.Breakdown {
		out = append(out, fmt.Sprintf("| %s | `%s` | %d | %d | %s |", b.Group, b.Kpi, b.Score, b.Debt, b.Detail))
	}
	out = append(out, "")
	if pf := cov.PerFamily; len(pf) > 0 {
		out = append(out, "## Coverage by family (how much of each confusable space is positioned)", "")
		out = append(out, "| Family | Positioned | Discovered | Unpositioned |", "|---|---:|---:|---:|")
		for _, f := range sortedPerFamily(pf) {
			out = append(out, fmt.Sprintf("| %s | %d | %d | %d |", pyStr(f.Family), f.Covered, f.Discovered, f.Discovered-f.Covered))
		}
		out = append(out, "")
	}
	return strings.Join(out, "\n"), nil
}

func mdCell(v Value) string {
	s := ""
	if v != nil {
		s = pyStr(v)
	}
	return pyStrip(strings.ReplaceAll(strings.ReplaceAll(s, "|", `\|`), "\n", " "))
}

func indexBucket(key string) string {
	if key == "" {
		return "#"
	}
	r := []rune(key)[0]
	if unicode.IsLetter(r) {
		return strings.ToUpper(string(r))
	}
	return "0-9"
}

// RenderDocNameIndex renders the generated INDEX.md (without its trailing newline).
func (p *Payload) RenderDocNameIndex(stamp string) string {
	c, _ := p.view()
	ix := c.Index
	rows := p.indexRows()
	out := frontMatter(
		"fak concept name index - every spelling, the concept it denotes, and its contrast set",
		"A-Z lookup over every surface a fak concept name can arrive as (canonical, alias, "+
			"grounding token), pointing at the one concept it denotes plus the concepts it must "+
			"not be confused with. Generated - do not hand-edit.")
	out = append(out, "# fak concept name index - which concept is this name, and what is it NOT?", "")
	if stamp != "" {
		out = append(out, stampComment(stamp), "")
	}
	out = append(out, "The [scorecard](README.md) is organised by concept. A reader arrives with a "+
		"**spelling** - something they read in code, a metric name, a flag - and needs "+
		"the reverse map. This is it: every surface a name can arrive as (the canonical "+
		"name, every alias, the raw grounding token) resolves to the concept it denotes "+
		"and to that concept's **contrast set**, so *which one is this?* and *how is it "+
		"not the other one?* are answered in the same line.", "")
	out = append(out, "> Generated by `python tools/concept_disambiguation_scorecard.py "+
		"--markdown-dir docs/concept-disambiguation-scorecard`. Do not hand-edit: edit "+
		fmt.Sprintf("`%s/` and regenerate.", DataDirRel), "")
	out = append(out, fmt.Sprintf("**%d** lookup names over **%d** concepts; **%d** land on more than one concept (marked **!** - "+
		"tolerated only when the two concepts separate from each other in both "+
		"directions, so the index can honestly answer *both, and here is the difference*).", ix.Keys, c.Rows, ix.AmbiguousKeys), "")
	out = append(out, "The **not to be confused with** column is the union of the boundaries the concept "+
		"declares (`distinct_from`) and the twins the scorecard discovered by name - the "+
		"pairs whose spellings a reader cannot keep apart.", "")
	var order []string
	buckets := map[string][]IndexRow{}
	for _, e := range rows {
		b := indexBucket(e.Key)
		if _, ok := buckets[b]; !ok {
			order = append(order, b)
		}
		buckets[b] = append(buckets[b], e)
	}
	sort.Strings(order)
	for _, b := range order {
		out = append(out, "## "+b, "")
		out = append(out, "| | Name (and other spellings) | Concept | Family / kind | Not to be confused with |", "|---|---|---|---|---|")
		for _, e := range buckets[b] {
			flag := ""
			if e.Ambiguous {
				flag = "**!**"
			}
			spellings := mdCell(e.Spellings[0])
			if len(e.Spellings) > 1 {
				spellings += " <br><small>" + mdCell(strings.Join(e.Spellings[1:], ", ")) + "</small>"
			}
			for _, t := range e.TargetsFull {
				nc := "-"
				if len(t.NotToBeConfusedWith) > 0 {
					nc = mdCell(strings.Join(t.NotToBeConfusedWith, ", "))
				}
				out = append(out, fmt.Sprintf("| %s | `%s` | **%s** | %s / %s | %s |", flag, spellings, mdCell(t.Canonical),
					mdCell(t.Family), mdCell(t.Kind), nc))
			}
		}
		out = append(out, "")
	}
	return strings.Join(out, "\n")
}

// DocFolder renders README.md and INDEX.md as written (with trailing newline).
func (p *Payload) DocFolder(stamp string) (map[string][]byte, error) {
	readme, err := p.RenderDocIndex(stamp)
	if err != nil {
		return nil, err
	}
	return map[string][]byte{
		ReadmeName: []byte(readme + "\n"),
		IndexName:  []byte(p.RenderDocNameIndex(stamp) + "\n"),
	}, nil
}

// WriteDocFolder writes the generated doc folder into dir, creating it.
func (p *Payload) WriteDocFolder(dir, stamp string) error {
	files, err := p.DocFolder(stamp)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	for _, name := range []string{ReadmeName, IndexName} {
		if err := os.WriteFile(filepath.Join(dir, name), files[name], 0o644); err != nil {
			return err
		}
	}
	return nil
}
