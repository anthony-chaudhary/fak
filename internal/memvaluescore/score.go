// Package memvaluescore is the unbounded memory-value scorecard — the memory
// sibling of the cache-value P&L.
//
// The cache side of fak has durable value accounting: the savings ledger
// (docs/nightrun/cache-savings.jsonl), the two-track roll-up
// (docs/cache-value-rollup.md), real net rows per session. The memory side
// shipped its first loop-facing verb — `fak memory recall` (#2346 R1), which
// re-verifies every concrete artifact claim at page-in and WITHHOLDS a stale
// note with the failing claim named — but had zero value accounting: no ledger,
// no score, no trend. This card closes that gap by applying the established
// unbounded scoring triple (AGENTIC-DEV-ANTIPATTERNS-2026-07-02.md §4; concept
// note CONCEPT-MEMORY-VALUE-UNBOUNDED-SCORE-2026-07-03.md) to the memory store:
//
//   - memory_value_frontier (unbounded, higher = better): accumulated WITNESSED
//     value from the recall-events ledger, weighted on the {2,4,8} severity
//     scale — stale_withheld ×8 (a stale memory refused BEFORE injection: the
//     decision-corruption the raw-MEMORY.md alternative would have served as
//     fact — the net-true-value framing, measured against the real
//     alternative), lesson_distilled ×4 (reserved for the R3 rung),
//     fresh_rendered ×2 (a claim-verified orientation block delivered). It
//     moves ONLY on realized ledger events — never on store size
//     (unbounded-ephemera is not value) and never on capability presence. No
//     ledger ⇒ 0, reported as not-yet: the frontier fails LOW, never high.
//
//   - memory_rot_pressure (unbounded, lower = better): severity × live rot
//     instances over the store — stale artifact claims (sev 4: would corrupt a
//     decision if injected; the recall gate is what stands between them and a
//     turn) plus structural defects and unresolved wikilinks (sev 2 each). No
//     ceiling and no 0-100 clamp.
//
//   - memory_debt (floor 0, the ratchet axis, corpus key "memory_debt"): the
//     HARD, mechanically-mendable structural subset ONLY — dangling index rows,
//     orphan fact files, frontmatter-grammar violations. Two rot kinds are
//     DELIBERATELY excluded and stay soft pressure: stale claims (a peer's
//     ordinary commit can stale a note's SHA/path claim without anyone touching
//     memory — the same external-drift rule that keeps history-window counts
//     out of the antipatterns card's debt) and broken [[wikilinks]] (the store
//     grammar sanctions an unresolved link as a FORWARD REFERENCE — a memory
//     worth writing later — so its mend is a judgment call, not mechanical).
//
// Claim verification reuses the shipped recall grammar directly
// (recall.ExtractArtifactClaims + recall.DefaultArtifactVerifier — the SAME
// seam `fak memory recall` gates page-ins with), so this card and the recall
// verb can never disagree about what "stale" means. Deterministic + read-only:
// it reads the store, the ledger, and git — and edits nothing. The fold/grade
// machinery lives in pkg/scorecard; this package holds only the memory KPIs.
package memvaluescore

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/memoryread"
	"github.com/anthony-chaudhary/fak/internal/recall"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// Schema is the control-pane schema id.
const Schema = "fak-memory-value-scorecard/1"

// DebtKey is the headline ratchet integer the control pane folds.
const DebtKey = "memory_debt"

// LedgerSchema is the per-row schema of the recall-events ledger this card
// folds into the frontier. The append seam (in `fak memory recall`) is the
// parked next step; until it lands the ledger is absent and the frontier is 0.
const LedgerSchema = "fak-memory-value-ledger/1"

// DefaultStoreRel is the committed fleet-memory mirror, relative to the root.
const DefaultStoreRel = ".claude/memory"

// DefaultLedgerRel is the recall-events ledger path, relative to the root —
// the sibling of docs/nightrun/cache-savings.jsonl.
const DefaultLedgerRel = "docs/nightrun/memory-value.jsonl"

// Rot severities on the workspace's established {2,4,8} scale. Structural and
// wikilink rot weigh 2 (annoyance/tax); a stale claim weighs 4 (corrupts a
// decision if it reaches a turn uninjected — the recall gate is the mitigation).
const (
	SevStaleClaim  = 4
	SevStructural = 2
)

// FrontierUnits are the witnessed-event weights of the unbounded value
// frontier, same scale.
var FrontierUnits = map[string]int{
	"stale_withheld":   8, // a stale memory refused BEFORE injection
	"lesson_distilled": 4, // R3 rung: a witnessed-turn lesson admitted (0 today)
	"fresh_rendered":   2, // a claim-verified orientation block delivered
}

// RotItem is one rot finding over the store.
type RotItem struct {
	Kind   string `json:"kind"`
	File   string `json:"file"`
	Detail string `json:"detail"`
}

// StoreAudit is every rot finding over one store: HARD structural items (debt),
// SOFT rot (forward-reference wikilinks), stale claims (soft pressure), and the
// unverifiable-claim count (visibility only, never pressure).
type StoreAudit struct {
	Debt         []RotItem
	Soft         []RotItem
	Stale        []RotItem
	Unverifiable int
	Notes        int
	IndexRows    int
}

// LedgerFold is the recall-events ledger summed into frontier event counts.
// Unknown-schema / unparsable rows are counted in SkippedRows and REPORTED —
// never silently dropped.
type LedgerFold struct {
	Present     bool
	Rows        int
	SkippedRows int
	Events      map[string]int
}

var wikilinkRe = regexp.MustCompile(`\[\[([^\[\]|#]+?)(?:\|[^\[\]]*)?\]\]`)

// AuditStore runs every deterministic rot check over one markdown memory store
// (the memoryread grammar: MEMORY.md index + per-fact frontmatter files). A
// missing store is an empty corpus, never an error — the same contract as the
// memq notes backend. verifier nil selects recall.DefaultArtifactVerifier.
func AuditStore(ctx context.Context, store string, verifier recall.ArtifactVerifier) StoreAudit {
	if verifier == nil {
		verifier = recall.DefaultArtifactVerifier
	}
	var a StoreAudit

	indexBytes, err := os.ReadFile(filepath.Join(store, "MEMORY.md"))
	var indexed [][2]string
	if err == nil {
		indexed = memoryread.ParseIndex(string(indexBytes))
	}
	a.IndexRows = len(indexed)
	linked := map[string]bool{}
	for _, fact := range indexed {
		fname := fact[1]
		linked[fname] = true
		if _, statErr := os.Stat(filepath.Join(store, fname)); statErr != nil {
			a.Debt = append(a.Debt, RotItem{Kind: "dangling_index_row", File: "MEMORY.md",
				Detail: fmt.Sprintf("index links %q but no such fact file exists", fname)})
		}
	}

	entries, _ := os.ReadDir(store)
	var facts []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".md") && e.Name() != "MEMORY.md" {
			facts = append(facts, e.Name())
		}
	}
	sort.Strings(facts)
	a.Notes = len(facts)

	type parsed struct {
		file string
		body string
	}
	known := map[string]bool{}
	var bodies []parsed
	for _, fname := range facts {
		raw, readErr := os.ReadFile(filepath.Join(store, fname))
		if readErr != nil {
			continue
		}
		text := string(raw)
		stem := strings.TrimSuffix(fname, ".md")
		known[strings.ToLower(stem)] = true
		if name := frontmatterField(text, "name"); name != "" {
			known[strings.ToLower(name)] = true
		}
		bodies = append(bodies, parsed{file: fname, body: memoryread.StripFrontmatter(text)})

		if !linked[fname] {
			a.Debt = append(a.Debt, RotItem{Kind: "orphan_fact_file", File: fname,
				Detail: "fact file is not linked from the MEMORY.md index"})
		}
		a.Debt = append(a.Debt, frontmatterViolations(fname, stem, text)...)
	}

	for _, p := range bodies {
		for _, m := range wikilinkRe.FindAllStringSubmatch(p.body, -1) {
			target := strings.TrimSpace(m[1])
			if !known[strings.ToLower(target)] {
				a.Soft = append(a.Soft, RotItem{Kind: "broken_wikilink", File: p.file,
					Detail: fmt.Sprintf("[[%s]] resolves to no fact file yet (forward reference)", target)})
			}
		}
		claims := recall.ExtractArtifactClaims(p.body)
		if len(claims) == 0 {
			continue
		}
		for _, f := range verifier(ctx, claims) {
			switch f.Status {
			case recall.ArtifactStale:
				a.Stale = append(a.Stale, RotItem{Kind: string(f.Claim.Kind), File: p.file,
					Detail: fmt.Sprintf("%q: %s", f.Claim.Value, f.Detail)})
			case recall.ArtifactUnverifiable:
				a.Unverifiable++
			}
		}
	}

	sortRot(a.Debt)
	sortRot(a.Soft)
	sortRot(a.Stale)
	return a
}

func sortRot(items []RotItem) {
	sort.Slice(items, func(i, j int) bool {
		if items[i].Kind != items[j].Kind {
			return items[i].Kind < items[j].Kind
		}
		if items[i].File != items[j].File {
			return items[i].File < items[j].File
		}
		return items[i].Detail < items[j].Detail
	})
}

// frontmatterField reads one top-level frontmatter scalar (crude but exactly
// the grammar the store writes: `key: value` lines between --- fences).
func frontmatterField(text, key string) string {
	lines := strings.Split(text, "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) != "---" {
		return ""
	}
	for _, line := range lines[1:] {
		if strings.TrimSpace(line) == "---" {
			return ""
		}
		if strings.HasPrefix(line, " ") || strings.HasPrefix(line, "\t") {
			continue
		}
		k, v, ok := strings.Cut(line, ":")
		if ok && strings.TrimSpace(k) == key {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

// frontmatterViolations checks one fact file against the store grammar: a
// terminated frontmatter block carrying name (== the filename stem),
// description, and metadata.type.
func frontmatterViolations(fname, stem, text string) []RotItem {
	var out []RotItem
	lines := strings.Split(text, "\n")
	terminated := false
	if len(lines) > 0 && strings.TrimSpace(lines[0]) == "---" {
		for _, line := range lines[1:] {
			if strings.TrimSpace(line) == "---" {
				terminated = true
				break
			}
		}
	}
	if !terminated {
		return append(out, RotItem{Kind: "frontmatter_violation", File: fname,
			Detail: "missing or unterminated frontmatter block"})
	}
	name := frontmatterField(text, "name")
	if name == "" {
		out = append(out, RotItem{Kind: "frontmatter_violation", File: fname,
			Detail: "frontmatter is missing \"name\""})
	} else if name != stem {
		out = append(out, RotItem{Kind: "frontmatter_violation", File: fname,
			Detail: fmt.Sprintf("frontmatter name %q != filename stem %q", name, stem)})
	}
	if frontmatterField(text, "description") == "" {
		out = append(out, RotItem{Kind: "frontmatter_violation", File: fname,
			Detail: "frontmatter is missing \"description\""})
	}
	if metadataType(text) == "" {
		out = append(out, RotItem{Kind: "frontmatter_violation", File: fname,
			Detail: "frontmatter is missing \"metadata.type\""})
	}
	return out
}

// metadataType reads metadata:/type: from the frontmatter block.
func metadataType(text string) string {
	lines := strings.Split(text, "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) != "---" {
		return ""
	}
	inMetadata := false
	for _, line := range lines[1:] {
		if strings.TrimSpace(line) == "---" {
			return ""
		}
		indented := strings.HasPrefix(line, " ") || strings.HasPrefix(line, "\t")
		k, v, _ := strings.Cut(line, ":")
		k, v = strings.TrimSpace(k), strings.TrimSpace(v)
		if !indented {
			inMetadata = k == "metadata"
			continue
		}
		if inMetadata && k == "type" && v != "" {
			return v
		}
	}
	return ""
}

// FoldLedger sums the recall-events ledger into frontier event counts.
func FoldLedger(path string) LedgerFold {
	f := LedgerFold{Events: map[string]int{
		"fresh_rendered": 0, "stale_withheld": 0, "lesson_distilled": 0}}
	raw, err := os.ReadFile(path)
	if err != nil {
		return f
	}
	f.Present = true
	for _, line := range strings.Split(string(raw), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var row struct {
			Schema        string `json:"schema"`
			Fresh         int    `json:"fresh"`
			WithheldStale int    `json:"withheld_stale"`
			Lessons       int    `json:"lessons"`
		}
		if json.Unmarshal([]byte(line), &row) != nil || row.Schema != LedgerSchema {
			f.SkippedRows++
			continue
		}
		f.Rows++
		f.Events["fresh_rendered"] += max(0, row.Fresh)
		f.Events["stale_withheld"] += max(0, row.WithheldStale)
		f.Events["lesson_distilled"] += max(0, row.Lessons)
	}
	return f
}

// Frontier is the UNBOUNDED memory-value fold (higher = better; NOT a 0-100
// grade): Σ weight × witnessed event count over FrontierUnits, plus the
// per-term breakdown. Monotone in every term, no ceiling; a missing term
// counts as zero, so the frontier fails LOW, never high.
func Frontier(events map[string]int) (int, map[string]int) {
	byTerm := map[string]int{}
	total := 0
	for term, weight := range FrontierUnits {
		n := events[term]
		if n < 0 {
			n = 0
		}
		byTerm[term] = weight * n
		total += weight * n
	}
	return total, byTerm
}

// Pressure is the UNBOUNDED rot fold (lower = better): severity × live
// instances, per rot kind.
func Pressure(a StoreAudit) (int, map[string]int) {
	byTerm := map[string]int{}
	for _, it := range a.Debt {
		byTerm[it.Kind] += SevStructural
	}
	for _, it := range a.Soft {
		byTerm[it.Kind] += SevStructural
	}
	if n := len(a.Stale); n > 0 {
		byTerm["stale_claim"] = SevStaleClaim * n
	}
	total := 0
	for _, v := range byTerm {
		total += v
	}
	return total, byTerm
}

// Build audits the default store + ledger under root and folds the payload.
func Build(ctx context.Context, root string) scorecard.Payload {
	return BuildWith(ctx, filepath.Join(root, DefaultStoreRel),
		filepath.Join(root, DefaultLedgerRel), nil)
}

// BuildWith is Build with explicit store/ledger paths and an injectable
// verifier (nil = the real recall verifier) — the CLI and test seam.
func BuildWith(ctx context.Context, store, ledger string, verifier recall.ArtifactVerifier) scorecard.Payload {
	audit := AuditStore(ctx, store, verifier)
	fold := FoldLedger(ledger)
	frontier, frontierByTerm := Frontier(fold.Events)
	pressure, pressureByTerm := Pressure(audit)

	kpis := []scorecard.KPI{
		{
			Key: "index_file_bijection", Group: "store_integrity",
			Score:   binaryScore(countKinds(audit.Debt, "dangling_index_row", "orphan_fact_file") == 0),
			Detail:  "every MEMORY.md index row resolves to a fact file and every fact file is indexed",
			Defects: rotStrings(audit.Debt, "dangling_index_row", "orphan_fact_file"),
		},
		{
			Key: "frontmatter_grammar", Group: "store_integrity",
			Score:   binaryScore(countKinds(audit.Debt, "frontmatter_violation") == 0),
			Detail:  "every fact file carries a terminated name/description/metadata.type frontmatter, name == filename stem",
			Defects: rotStrings(audit.Debt, "frontmatter_violation"),
		},
		{
			Key: "claim_freshness", Group: "rot_watch",
			Score:  binaryScore(len(audit.Stale) == 0),
			Detail: "concrete artifact claims (the recall grammar: SHA/path/flag) still verify against the checkout — soft: external drift can stale a claim with no memory edit to mend",
			Soft:   rotStrings(audit.Stale),
		},
		{
			Key: "wikilink_resolution", Group: "rot_watch",
			Score:  binaryScore(len(audit.Soft) == 0),
			Detail: "[[wikilinks]] resolve to fact files — soft: an unresolved link is sanctioned forward-reference grammar",
			Soft:   rotStrings(audit.Soft),
		},
		{
			Key: "recall_value_witnessed", Group: "value",
			Score:  binaryScore(fold.Rows > 0),
			Detail: "the recall-events ledger has witnessed rows — not yet until the `fak memory recall` append seam lands (parked, cmd lane)",
			Soft:   ledgerSoft(fold),
		},
	}

	return scorecard.Fold(Schema, kpis, DebtKey, nil, scorecard.Messages{
		Finding:         "hard structural defects in the memory store — mend in-tree",
		FindingClean:    "store structurally sound; frontier and pressure are the trend numbers, not grades",
		NextAction:      "fix the defect list (index rows, orphans, frontmatter), then re-run",
		NextActionClean: "grow the frontier: land the recall-ledger append seam, then inject verified recall at loop-turn start (R2)",
		ExtraCorpus: map[string]any{
			"memory_value_frontier": frontier,
			"frontier_by_term":      frontierByTerm,
			"memory_rot_pressure":   pressure,
			"pressure_by_term":      pressureByTerm,
			"stale_claims":          len(audit.Stale),
			"unverifiable_claims":   audit.Unverifiable,
			"store_notes":           audit.Notes,
			"store_index_rows":      audit.IndexRows,
			"ledger_present":        fold.Present,
			"ledger_rows":           fold.Rows,
			"ledger_skipped_rows":   fold.SkippedRows,
		},
	})
}

func binaryScore(clean bool) float64 {
	if clean {
		return 100
	}
	return 0
}

func countKinds(items []RotItem, kinds ...string) int {
	n := 0
	for _, it := range items {
		for _, k := range kinds {
			if it.Kind == k {
				n++
			}
		}
	}
	return n
}

// rotStrings renders items (filtered to kinds, or all when none given) as the
// flat defect/soft strings the scorecard envelope carries.
func rotStrings(items []RotItem, kinds ...string) []string {
	var out []string
	for _, it := range items {
		if len(kinds) > 0 {
			match := false
			for _, k := range kinds {
				if it.Kind == k {
					match = true
					break
				}
			}
			if !match {
				continue
			}
		}
		out = append(out, fmt.Sprintf("%s %s: %s", it.Kind, it.File, it.Detail))
	}
	return out
}

func ledgerSoft(f LedgerFold) []string {
	if !f.Present {
		return []string{"ledger absent: no recall value witnessed yet (frontier 0 — fails low, never high)"}
	}
	if f.SkippedRows > 0 {
		return []string{fmt.Sprintf("%d row(s) skipped (unparsable or unknown schema) — reported, never silently dropped", f.SkippedRows)}
	}
	return nil
}
