package devcmd

// fak-dev index — the QUERYABLE SELF-INDEX (epic #1287 / C1 #1288). Instead of
// re-surveying dos.toml and INDEX.md every session, an agent ASKS:
//
//	fak-dev index <query>          the curated doc map (the default)
//	fak-dev index lane <path>...   which lane/leaf owns these paths (+ the commit stamp)
//	fak-dev index leaf [<query>]   the lane taxonomy, filtered by name/tree/description
//	fak-dev index claims <query>   the CLAIMS.md honesty ledger: shipped/simulated/stub
//	fak-dev index verbs [<query>]  the structured CLI-verb catalog (name/lane/synopsis)
//	fak-dev index work [<query>]   the selection surface: named issue views + the default's gh query
//	fak-dev index refs <pkg>.<Symbol>
//	                         direct + transitive dependents of a Go symbol before editing
//	fak-dev index ctxknobs     the manual-overlay counter: context flags/env/skills,
//	                         each operator-debug (fine) or user-required (defect) (#2199)
//	fak-dev index knobs        the knob census: every user-facing behavior knob,
//	                         each INTENT (promote) or HOUSEKEEPING (automate) (#2210)
//	fak-dev index generation [<query>]
//	                         the generation taxonomy: labels, milestones, evidence rules
//	fak-dev index freshness    the self-index drift report: undeclared leaves, dead doc
//	                         links, unknown verbs, and orphaned dated notes — is the
//	                         index still honest against the tree?
//	fak-dev index agents [<query>]
//	                         the sectioned AGENTS.md view (#3535): a compact resident
//	                         TOC by default, ranked sections for a query, --section <slug>
//	                         / --full for byte-exact bodies, --write-resident to (re)write
//	                         the resident TOC block into CLAUDE.md; --for <path> resolves
//	                         the trusted effective root-to-target hierarchy (#9391)
//
// It is a thin shell over internal/devindex, which reads the facts live from the
// files that already own them (dos.toml's [lanes.trees], the curated INDEX.md, the
// CLAIMS.md ledger), so the index is a VIEW, never a competing source of truth.
// --json on every subcommand makes the same answers machine- and MCP-consumable.

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

// laneAnswer is one path's resolved lane + the ship-stamp it implies (JSON shape).
type laneAnswer struct {
	Path  string `json:"path"`
	Lane  string `json:"lane"`
	Stamp string `json:"stamp,omitempty"`
}

func RunIndex(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		writeIndexUsage(stderr)
		return 2
	}
	sub := argv[0]
	restArgs := argv[1:]
	if !isIndexSubcommand(sub) {
		// Documentation lookup is the common orientation path. Keep every named
		// subcommand explicit, but let a free-text query use the curated doc map
		// without first requiring the caller to know the index taxonomy.
		sub = "docs"
		restArgs = argv
	}
	fs := flag.NewFlagSet("index "+sub, flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "index")
	root := fs.String("root", "", "repo root (default: search upward for dos.toml)")
	asJSON := fs.Bool("json", false, "emit the answer as JSON")
	limit := fs.Int("limit", 0, "cap the number of results (0 = all)")
	// `fak-dev index agents` selectors (unused by the other subverbs, like --limit is unused
	// by lane): a section slug, the whole-doc escape hatch, and the resident-TOC writer.
	agentsSection := fs.String("section", "", "fak-dev index agents: emit one AGENTS.md section by slug")
	agentsFull := fs.Bool("full", false, "fak-dev index agents: emit the whole AGENTS.md verbatim")
	agentsWriteResident := fs.Bool("write-resident", false, "fak-dev index agents: (re)write the resident TOC block into CLAUDE.md")
	agentsFor := fs.String("for", "", "fak-dev index agents: resolve effective instructions for a target path")
	var agentsFallbacks stringListFlag
	fs.Var(&agentsFallbacks, "fallback", "fak-dev index agents: lower-precedence instruction file name (repeatable)")
	agentsMaxBytes := fs.Int64("max-bytes", 0, "fak-dev index agents: shared effective-instruction byte budget")
	agentsTrust := fs.Bool("trust", false, "fak-dev index agents: mark caller-verified sources trusted")
	// `fak-dev index ownership` writer: without this shared-flag entry the
	// documented `fak-dev index ownership --write-manifest` regen command died
	// in Parse (flag provided but not defined) and the generated handoff
	// manifest could never be regenerated (#8924's own failure message names it).
	ownershipWriteManifest := fs.Bool("write-manifest", false, "fak-dev index ownership: (re)write the generated dev-handoff manifest")
	// Parse flags that may appear ANYWHERE around the positional query (the natural
	// `fak-dev index leaf cache --limit 6` order), not just before it. Go's flag package
	// stops at the first non-flag arg, so interleave Parse with positional collection.
	var args []string
	for rest := restArgs; ; {
		if err := fs.Parse(rest); err != nil {
			if errors.Is(err, flag.ErrHelp) {
				return 0
			}
			return 2
		}
		rest = fs.Args()
		if len(rest) == 0 {
			break
		}
		args = append(args, rest[0])
		rest = rest[1:]
	}

	rootDir := *root
	if rootDir == "" {
		rootDir = devindex.FindRoot(".")
	}
	cat, err := devindex.Load(rootDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev index: %v\n", err)
		return 1
	}

	switch sub {
	case "lane":
		return indexLane(stdout, stderr, cat, args, *asJSON)
	case "leaf", "leaves":
		return indexLeaf(stdout, stderr, cat, args, *asJSON, *limit)
	case "docs", "doc":
		return indexDocs(stdout, stderr, cat, args, *asJSON, *limit)
	case "benchmark":
		return indexDiscoveryBenchmark(stdout, stderr, cat, args, *asJSON)
	case "claims", "claim":
		return indexClaims(stdout, stderr, cat, args, *asJSON, *limit)
	case "verbs", "verb":
		return indexVerbs(stdout, stderr, cat, args, *asJSON, *limit)
	case "generation", "generations", "gen":
		return indexGeneration(stdout, stderr, cat, args, *asJSON, *limit)
	case "work", "views", "view":
		return indexWork(stdout, stderr, cat, args, *asJSON, *limit)
	case "refs", "ref":
		return indexRefs(stdout, stderr, rootDir, args, *asJSON, *limit)
	case "ctxplans", "ctxplan":
		if len(args) != 0 {
			fmt.Fprintln(stderr, "usage: fak-dev index ctxplans [--json] [--root PATH]")
			return 2
		}
		return indexCtxPlans(stdout, stderr, rootDir, *asJSON)
	case "ctxknobs", "ctxknob":
		return indexCtxKnobs(stdout, stderr, rootDir, *asJSON)
	case "knobs", "knob":
		return indexKnobs(stdout, stderr, rootDir, *asJSON)
	case "freshness", "fresh":
		return indexFreshness(stdout, stderr, cat, *asJSON, *limit)
	case "execaudit", "executables", "exec":
		return indexExecAudit(stdout, stderr, rootDir, *asJSON, *limit)
	case "agents", "agentsmd", "agent":
		return indexAgents(stdout, stderr, rootDir, args, *asJSON, *agentsSection, *agentsFull, *agentsWriteResident,
			*agentsFor, []string(agentsFallbacks), *agentsMaxBytes, *agentsTrust)

	case "ownership":
		return indexOwnership(stdout, stderr, rootDir, args, *asJSON, *ownershipWriteManifest)
	case "policy", "enforce":
		return runIndexPolicy(stdout, stderr, rootDir, args, *asJSON)
	case "graph":
		if len(args) != 0 {
			fmt.Fprintln(stderr, "usage: fak-dev index graph [--json] [--root PATH]")
			return 2
		}
		return indexGraphMain(stdout, stderr, rootDir, boolArg(*asJSON))
	default:
		fmt.Fprintf(stderr, "fak-dev index: unknown subcommand %q\n", sub)
		writeIndexUsage(stderr)
		return 2
	}
}

type stringListFlag []string

func (v *stringListFlag) String() string { return strings.Join(*v, ",") }
func (v *stringListFlag) Set(value string) error {
	*v = append(*v, value)
	return nil
}

func isIndexSubcommand(s string) bool {
	switch s {
	case "lane", "leaf", "leaves", "docs", "doc", "claims", "claim", "verbs", "verb",
		"generation", "generations", "gen", "work", "views", "view", "refs", "ref",
		"ctxplans", "ctxplan", "ctxknobs", "ctxknob", "knobs", "knob", "freshness", "fresh",
		"execaudit", "executables", "exec", "agents", "agentsmd", "agent", "ownership",
		"policy", "enforce", "graph", "benchmark":
		return true
	default:
		return false
	}
}

func indexGeneration(stdout, stderr io.Writer, cat *devindex.Catalog, args []string, asJSON bool, limit int) int {
	hits := capGenerations(cat.SearchGenerations(joinArgs(args)), limit)
	return indexRenderHits(stdout, stderr, hits, asJSON, "fak-dev index generation", "no matching generation",
		func(tw *tabwriter.Writer, g devindex.Generation) {
			fmt.Fprintf(tw, "%s\t%s\t%s\t%s\tpromote: %s\tdemote: %s\n",
				g.Stream, g.Label, g.Milestone, truncRunes(g.Meaning, 88),
				truncRunes(g.PromotionEvidence, 80), truncRunes(g.DemotionEvidence, 80))
		})
}

func indexLane(stdout, stderr io.Writer, cat *devindex.Catalog, paths []string, asJSON bool) int {
	if len(paths) == 0 {
		fmt.Fprintln(stderr, "fak-dev index lane: needs at least one path")
		return 2
	}
	answers := make([]laneAnswer, 0, len(paths))
	for _, p := range paths {
		lane := cat.LaneForPath(p)
		answers = append(answers, laneAnswer{Path: p, Lane: lane, Stamp: cat.SuggestStamp(p)})
	}
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, answers, "fak-dev index lane")
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	for _, a := range answers {
		lane := a.Lane
		if lane == "" {
			lane = "(no lane — root file?)"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\n", a.Path, lane, a.Stamp)
	}
	return flushTab(tw, stderr, "fak-dev index lane")
}

// indexRenderHits is the shared post-search rendering for the index subcommands (leaf,
// claims, docs): emit JSON when asked, the "no matching ..." line when the result set is
// empty, else a tab-aligned table whose rows renderRow writes. cmdLabel labels the
// JSON-encode/flush errors; emptyMsg is the no-results line.
func indexRenderHits[T any](stdout, stderr io.Writer, hits []T, asJSON bool, cmdLabel, emptyMsg string, renderRow func(tw *tabwriter.Writer, row T)) int {
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, hits, cmdLabel)
	}
	if len(hits) == 0 {
		fmt.Fprintln(stdout, emptyMsg)
		return 0
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	for _, row := range hits {
		renderRow(tw, row)
	}
	return flushTab(tw, stderr, cmdLabel)
}

func indexLeaf(stdout, stderr io.Writer, cat *devindex.Catalog, args []string, asJSON bool, limit int) int {
	hits := capLeaves(cat.SearchLeaves(joinArgs(args)), limit)
	return indexRenderHits(stdout, stderr, hits, asJSON, "fak-dev index leaf", "no matching leaf",
		func(tw *tabwriter.Writer, l devindex.Leaf) {
			mark := "ok"
			if !l.Exists {
				mark = "MISSING"
			}
			desc := l.Desc
			if desc == "" {
				desc = l.Tree
			}
			fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", l.Name, mark, statusBadge(l.Status), desc)
		})
}

// statusBadge renders a leaf's CLAIMS.md maturity rollup as a compact, model- and
// human-readable cell — "[4 shipped · 1 stub]" — or "" when the ledger names no
// capability for the leaf (so the column stays empty rather than noisy).
func statusBadge(s devindex.Status) string {
	if s.Total() == 0 {
		return ""
	}
	var parts []string
	if s.Shipped > 0 {
		parts = append(parts, fmt.Sprintf("%d shipped", s.Shipped))
	}
	if s.Simulated > 0 {
		parts = append(parts, fmt.Sprintf("%d sim", s.Simulated))
	}
	if s.Stub > 0 {
		parts = append(parts, fmt.Sprintf("%d stub", s.Stub))
	}
	return "[" + strings.Join(parts, " · ") + "]"
}

// indexNeedsQuery is the shared empty-query guard for the subcommands whose search
// query is mandatory (claims, docs — unlike leaf/verbs, where empty lists all): with
// no positional query it writes the subcommand's usage line to stderr and reports
// true, and the caller exits 2 (usage error).
func indexNeedsQuery(stderr io.Writer, args []string, usage string) bool {
	if len(args) > 0 {
		return false
	}
	fmt.Fprintln(stderr, usage)
	return true
}

func indexClaims(stdout, stderr io.Writer, cat *devindex.Catalog, args []string, asJSON bool, limit int) int {
	if indexNeedsQuery(stderr, args, "fak-dev index claims: needs a search query (a lane, a token, or a capability)") {
		return 2
	}
	hits := capClaims(cat.SearchClaims(joinArgs(args)), limit)
	return indexRenderHits(stdout, stderr, hits, asJSON, "fak-dev index claims", "no matching claim",
		func(tw *tabwriter.Writer, cl devindex.Claim) {
			lanes := strings.Join(cl.Lanes, ",")
			if lanes == "" {
				lanes = "-"
			}
			fmt.Fprintf(tw, "%s\t%s\t%s\n", cl.Tag, lanes, truncRunes(cl.Text, 96))
		})
}

// indexVerbs answers `fak-dev index verbs [<query>]` from the structured C3 verb manifest
// (#1290) — the parseable replacement for grepping usage.go's freeform prose. An empty
// query lists the whole catalog (the SearchVerbs convention), matching `fak-dev index leaf`.
func indexVerbs(stdout, stderr io.Writer, cat *devindex.Catalog, args []string, asJSON bool, limit int) int {
	hits := capVerbs(cat.SearchVerbs(joinArgs(args)), limit)
	return indexRenderHits(stdout, stderr, hits, asJSON, "fak-dev index verbs", "no matching verb",
		func(tw *tabwriter.Writer, v devindex.Verb) {
			lane := v.Lane
			if lane == "" {
				lane = "-"
			}
			tier := string(v.Tier)
			if tier == "" {
				tier = "-" // a curated entry whose verb is not (yet) dispatched
			}
			fmt.Fprintf(tw, "fak %s\t%s\t%s\t%s\n", v.Name, tier, lane, v.Synopsis)
		})
}

func indexDocs(stdout, stderr io.Writer, cat *devindex.Catalog, args []string, asJSON bool, limit int) int {
	if indexNeedsQuery(stderr, args, "fak-dev index docs: needs a search query") {
		return 2
	}
	hits := capDocs(cat.SearchDocs(joinArgs(args)), limit)
	return indexRenderHits(stdout, stderr, hits, asJSON, "fak-dev index docs", "no matching doc",
		func(tw *tabwriter.Writer, d devindex.Doc) {
			fmt.Fprintf(tw, "%s\t%s\n", d.Path, d.Title)
		})
}

func indexDiscoveryBenchmark(stdout, stderr io.Writer, cat *devindex.Catalog, args []string, asJSON bool) int {
	if len(args) != 0 {
		fmt.Fprintln(stderr, "fak-dev index benchmark: accepts no query")
		return 2
	}
	report := cat.RunDiscoveryBenchmark(devindex.DefaultDiscoveryQuestions())
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak-dev index benchmark")
	}
	fmt.Fprintf(stdout, "discovery benchmark: %s\n", report.Summary())
	for _, c := range report.Cases {
		verdict := "MISS"
		if c.Success {
			verdict = fmt.Sprintf("HIT@%d", c.Rank)
		}
		fmt.Fprintf(stdout, "%s\t%s\t%s\t%s\n", verdict, c.Category, c.ID, c.Query)
	}
	return 0
}

// indexWork answers `fak-dev index work [<query>]` from .github/issue-views.json — the
// curated DEFAULT selection surface ("what should I work on"), the API-readable
// mirror of the GitHub saved views. With no query it leads with the default view's
// ready-to-run `gh issue list --search` command, then lists every named view; a query
// filters the views (by slug/title/note). --json round-trips the full surface (every
// view's gh query) for tooling/MCP. This is the work-view #1291 / WORK-MAP's "no
// single live view of in-flight work" drift, as a one-call query.
func indexWork(stdout, stderr io.Writer, cat *devindex.Catalog, args []string, asJSON bool, limit int) int {
	views, err := cat.IssueViews()
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev index work: %v\n", err)
		return 1
	}
	if asJSON {
		if len(args) == 0 {
			return encodeJSONOrFail(stdout, stderr, views, "fak-dev index work")
		}
		return encodeJSONOrFail(stdout, stderr, capViews(views.SearchViews(joinArgs(args)), limit), "fak-dev index work")
	}
	hits := capViews(views.SearchViews(joinArgs(args)), limit)
	if len(hits) == 0 {
		fmt.Fprintln(stdout, "no matching view")
		return 0
	}
	// With no query, lead with the default view's ready-to-run command — the literal
	// "what should I work on" answer an agent can paste.
	if len(args) == 0 {
		if def, ok := views.DefaultView(); ok {
			fmt.Fprintf(stdout, "default: %s — %s\n", def.Slug, def.Title)
			fmt.Fprintf(stdout, "  gh issue list --search %q --limit %d\n\n", def.Query, views.PageLimit())
		}
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	for _, iv := range hits {
		slug := iv.Slug
		if iv.Slug == views.Default {
			slug = iv.Slug + " *" // mark the default selection surface
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\n", slug, iv.Title, truncRunes(iv.Note, 80))
	}
	return flushTab(tw, stderr, "fak-dev index work")
}

// indexFreshness answers `fak-dev index freshness` from internal/devindex.CheckFreshness:
// every way the self-index disagrees with its live sources — an undeclared leaf, a dead
// INDEX.md doc link, a main.go verb missing from the catalog, or a dated docs/notes/ note
// INDEX.md never lists. It is a READ-ONLY query (always exit 0), not a gate: it surfaces
// the drift an agent should fix, in one place, for humans (a kind/subject/reason table)
// and LLMs (--json / MCP). A clean tree prints a single reassuring line. The build-redding
// gate over the same findings lives in *_test.go, out of this shell's lane.
func indexFreshness(stdout, stderr io.Writer, cat *devindex.Catalog, asJSON bool, limit int) int {
	drift := cat.CheckFreshness()
	if limit > 0 && len(drift) > limit {
		drift = drift[:limit]
	}
	return indexRenderHits(stdout, stderr, drift, asJSON, "fak-dev index freshness",
		"index fresh: no drift — the catalog agrees with the tree",
		func(tw *tabwriter.Writer, d devindex.Drift) {
			fmt.Fprintf(tw, "%s\t%s\t%s\n", d.Kind, d.Subject, d.Reason)
		})
}

func writeIndexUsage(w io.Writer) {
	fmt.Fprint(w, `usage:
  fak-dev index <query>           search the curated doc map (default; same as index docs)
  fak-dev index lane <path>...    which lane/leaf owns each path, + the (fak <leaf>) commit stamp
  fak-dev index leaf [<query>]    the lane taxonomy (+ shipped/sim/stub rollup), filtered by name/tree/desc
  fak-dev index claims <query>    the CLAIMS.md honesty ledger, ranked by relevance (shipped/simulated/stub)
  fak-dev index verbs [<query>]   the structured CLI-verb catalog (name/owning-lane/synopsis)
  fak-dev index generation [<q>]  generation labels, milestones, issue-body signals, and evidence rules
  fak-dev index work [<query>]    the selection surface ("what should I work on"): named issue views + the default's gh query
  fak-dev index refs <pkg>.<Sym>  direct + transitive dependents of a Go symbol before editing
  fak-dev index ctxknobs          the manual-overlay counter: context flags/env/skills classified operator-debug vs user-required (#2199)
  fak-dev index knobs             the knob census: every user-facing behavior knob classified INTENT (promote) vs HOUSEKEEPING (automate) (#2210)
  fak-dev index freshness         the self-index drift report: undeclared leaves, dead doc links, unknown verbs, orphaned dated notes
  fak-dev index execaudit         executable packages that build but have no adjacent test or no invocation edge outside themselves (#5648)
  fak-dev index agents [<query>]  sectioned AGENTS.md view; --for PATH resolves the effective hierarchy
  fak-dev index graph             HEAD-only Markdown reachability census under named resolver rules
  flags: --json  --limit N  --root DIR  |  agents: --section <slug>  --full  --write-resident  --for PATH  --fallback NAME  --max-bytes N  --trust
`)
}

func joinArgs(args []string) string {
	out := ""
	for i, a := range args {
		if i > 0 {
			out += " "
		}
		out += a
	}
	return out
}

func capLeaves(ls []devindex.Leaf, limit int) []devindex.Leaf {
	if limit > 0 && len(ls) > limit {
		return ls[:limit]
	}
	return ls
}

func capDocs(ds []devindex.Doc, limit int) []devindex.Doc {
	if limit > 0 && len(ds) > limit {
		return ds[:limit]
	}
	return ds
}

func capClaims(cs []devindex.Claim, limit int) []devindex.Claim {
	if limit > 0 && len(cs) > limit {
		return cs[:limit]
	}
	return cs
}

func capVerbs(vs []devindex.Verb, limit int) []devindex.Verb {
	if limit > 0 && len(vs) > limit {
		return vs[:limit]
	}
	return vs
}

func capGenerations(gs []devindex.Generation, limit int) []devindex.Generation {
	if limit > 0 && len(gs) > limit {
		return gs[:limit]
	}
	return gs
}

func capViews(vs []devindex.IssueView, limit int) []devindex.IssueView {
	if limit > 0 && len(vs) > limit {
		return vs[:limit]
	}
	return vs
}

// truncRunes shortens s to at most n runes (UTF-8-safe — a claim line carries em
// dashes and middots that a byte slice would split), appending an ellipsis when it
// cuts. The package-wide truncate (benchmarks.go) is byte-based and ASCII-only.
func truncRunes(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n-1]) + "…"
}

func flushTab(tw *tabwriter.Writer, stderr io.Writer, label string) int {
	if err := tw.Flush(); err != nil {
		fmt.Fprintf(stderr, "%s: %v\n", label, err)
		return 1
	}
	return 0
}

func indexOwnership(stdout, stderr io.Writer, root string, args []string, inheritedJSON bool, writeManifestFlag bool) int {
	asJSON := inheritedJSON
	writeManifest := writeManifestFlag
	for _, arg := range args {
		switch arg {
		case "--json":
			asJSON = true
		case "--write-manifest":
			writeManifest = true
		default:
			fmt.Fprintf(stderr, "fak-dev index ownership: unexpected argument %q\n", arg)
			return 2
		}
	}
	if writeManifest {
		if asJSON {
			fmt.Fprintln(stderr, "fak-dev index ownership: --write-manifest and --json are mutually exclusive")
			return 2
		}
		if err := devindex.WriteDevHandoffManifest(root); err != nil {
			fmt.Fprintf(stderr, "fak-dev index ownership: write manifest: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, devindex.DevHandoffManifestPath)
		return 0
	}
	// One implementation only (#6022). This dispatch used to build its own report
	// with pattern "./...", which counts the WHOLE MODULE's dependency closure
	// instead of what the runtime binary links — so packages=/internal= reported
	// numbers no runtime artifact has. RunOwnership is the migrated body and it
	// asks for "./cmd/fak"; parse the flags here and hand off, so the pattern
	// cannot drift apart in two places again.
	return RunOwnership(stdout, stderr, root, asJSON)
}
