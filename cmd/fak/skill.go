package main

// cmdSkill is the operator + RSI surface for the queried skill loader (epic
// #1103, child C7 / issue #1110). It fronts the capindex keystone — the
// protocol-blind Capability index already shipped by C1–C6 — with three verbs:
//
//	fak skill query <intent> [--budget N] [--mcp] [--json]
//	    run the in-kernel query: rank the at-rest cards by intent, fault in the
//	    top winners up to the budget, and print the working set + the paged cost.
//	fak skill residency [--mcp] [--journal PATH] [--json]
//	    show the resident capability cards (states + digests), the version
//	    page-table pins, and the loader-journal reconciliation (the C6 read).
//	fak skill swap <name> <version> [--from VER] [--json]
//	    hot-swap a skill's active version (C4), printing the pre-flip blast
//	    radius, and persist the page-table flip so residency reads it back.
//
// The verbs work against .claude/skills/ (the skill Resolver) and, with --mcp,
// fold in the MCP-tool Resolver — the catalog is protocol-blind by construction
// (C1/C5), so one query ranks skills and MCP tools together.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/capindex"
	"github.com/anthony-chaudhary/fak/internal/capindexgw"
	"github.com/anthony-chaudhary/fak/internal/ctxresidency"
	"github.com/anthony-chaudhary/fak/internal/skillenv"
	"github.com/anthony-chaudhary/fak/internal/toolcatalog"
)

func cmdSkill(args []string) {
	if len(args) == 0 {
		skillUsage(os.Stderr)
		os.Exit(2)
	}
	var code int
	switch args[0] {
	case "query":
		code = runSkillQuery(os.Stdout, os.Stderr, args[1:])
	case "residency":
		code = runSkillResidency(os.Stdout, os.Stderr, args[1:])
	case "footprint":
		code = runSkillFootprint(os.Stdout, os.Stderr, args[1:])
	case "value":
		code = runSkillValue(os.Stdout, os.Stderr, args[1:])
	case "swap":
		code = runSkillSwap(os.Stdout, os.Stderr, args[1:])
	case "compile":
		code = runSkillCompile(os.Stdout, os.Stderr, args[1:])
	case "-h", "--help", "help":
		skillUsage(os.Stderr)
		return
	default:
		fmt.Fprintf(os.Stderr, "fak skill: unknown subcommand %q\n", args[0])
		skillUsage(os.Stderr)
		os.Exit(2)
	}
	if code != 0 {
		os.Exit(code)
	}
}

func skillUsage(w io.Writer) {
	fmt.Fprint(w, `usage: fak skill <subcommand>

The queried skill loader — 0 cost for ∞ skills, paged on demand (epic #1103, C7).

  fak skill query <intent> [--budget N] [--mcp] [--json]
      rank the at-rest capability cards by intent, fault in the top N winners
      (the working set), and print the paged cost in bytes. --mcp folds in the
      MCP-tool resolver so one query ranks skills and MCP tools together.

  fak skill residency [--mcp] [--journal PATH] [--json]
      show the resident capability cards (name, version, digest, state), the
      version page-table pins, and (with --journal) the loader-journal
      reconciliation against the kernel counters — the C6 audit read.

  fak skill swap <name> <version> [--from VER] [--json]
      hot-swap a skill's active version and print the pre-flip blast radius.
      --from guards the remap (refuses if the pinned version differs); the flip
      is persisted to .claude/skill-page-table.json so residency reads it back.

  fak skill compile <SKILL.md> [--source NAME] [--expose NAME ...] [--dialect NAME] [--json]
      compile only an explicit versioned fak-program block; registration and
      model exposure are separate, and --expose selects the request-visible tool.

  fak skill footprint [--top N] [--mcp] [--json]
      the userland resident-floor scorecard: per-skill resident description
      bytes + at-rest card bytes, the floor total (the /context Skills slice),
      and the top-N heaviest. Deterministic and offline (epic #3229 / #3234).

  fak skill value [--ledger PATH] [--basis PATH] [--json]
      the per-skill outcome-VALUE ledger (not a usage count): measured pass /
      cost / latency lift of sessions that LOADED a skill vs matched same-class
      sessions that did not. Flags net-negative skills for auto-revert and any
      active skill promoted with no valuation basis (#2796 mirror; issue #2873).

Without a live kernel the faulted/evicted residency states and the eviction
blast radius are zero (a one-shot CLI holds no resident pages); the verbs still
exercise the real query / page-table / journal primitives over .claude/skills.
`)
}

// buildSkillCatalog wires the skill Resolver over .claude/skills (and, with
// includeMCP, the MCP-tool Resolver), then Syncs the hash-diff index. It is the
// one constructor every skill verb shares.
func buildSkillCatalog(root string, includeMCP bool) *capindex.Catalog {
	cat := capindex.NewCatalog()
	cat.AddResolver(capindex.CapKindSkill, capindex.NewSkillResolver(skillDir(root)))
	if includeMCP {
		// NewMCPResolver(nil) is fine: Index()/Fault() read the package-level
		// gateway tool-descriptor registry, not a live server connection.
		cat.AddResolver(capindex.CapKindMCPTool, capindexgw.NewMCPResolver(nil))
	}
	cat.Sync()
	return cat
}

// skillDir returns the .claude/skills directory under root.
func skillDir(root string) string { return filepath.Join(root, ".claude", "skills") }

// partitionArgs splits an argv into Go flag tokens and positional words, so a
// free-form positional (the query intent, or the swap name+version) can sit
// before flags: `fak skill query <intent> --budget N`. valueFlags names the
// flags that take a value (everything else is treated as boolean). A `--` ends
// flag scanning. This is the only way to keep Go's stdlib flag (which otherwise
// stops at the first positional) while honoring the documented verb syntax.
func partitionArgs(argv []string, valueFlags map[string]bool) (flagArgs, rest []string) {
	flagArgs = []string{}
	rest = []string{}
	i := 0
	for i < len(argv) {
		a := argv[i]
		if a == "--" {
			rest = append(rest, argv[i+1:]...)
			break
		}
		if len(a) > 2 && a[:2] == "--" {
			name := a[2:]
			if eq := strings.IndexByte(name, '='); eq >= 0 {
				flagArgs = append(flagArgs, a)
				i++
				continue
			}
			flagArgs = append(flagArgs, a)
			if valueFlags[name] {
				if i+1 < len(argv) {
					flagArgs = append(flagArgs, argv[i+1])
					i += 2
					continue
				}
			}
			i++
			continue
		}
		rest = append(rest, a)
		i++
	}
	return flagArgs, rest
}

// --- query -----------------------------------------------------------------

func runSkillQuery(out, errw io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak skill query", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	budget := fs.Int("budget", 1, "max winners to fault in (working-set size)")
	includeMCP := fs.Bool("mcp", false, "fold in the MCP-tool resolver")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	flagArgs, rest := partitionArgs(argv, map[string]bool{"budget": true})
	if err := fs.Parse(flagArgs); err != nil {
		fmt.Fprintln(errw, err)
		return 2
	}
	intent := strings.TrimSpace(strings.Join(rest, " "))
	if intent == "" {
		fmt.Fprintln(errw, "fak skill query: an intent string is required")
		skillUsage(errw)
		return 2
	}

	root := repoRoot()
	cat := buildSkillCatalog(root, *includeMCP)
	ranked := cat.RankCards(intent)

	limit := *budget
	if limit < 0 {
		limit = 0
	}

	type winner struct {
		Ref       capindex.CapRef `json:"ref"`
		Digest    string          `json:"digest"`
		BodyBytes int             `json:"body_bytes"`
	}
	var winners []winner
	totalBytes := 0
	for i := range ranked {
		if len(winners) >= limit {
			break
		}
		c, err := cat.Lookup(ranked[i].Ref)
		if err != nil {
			continue
		}
		body := c.Materialize()
		winners = append(winners, winner{Ref: ranked[i].Ref, Digest: ranked[i].Digest, BodyBytes: len(body)})
		totalBytes += len(body)
	}

	if *asJSON {
		_ = writeIndentedJSONNoEscape(out, map[string]any{
			"schema":      "fak-skill-query/1",
			"intent":      intent,
			"include_mcp": *includeMCP,
			"budget":      limit,
			"working_set": cardSummaries(ranked),
			"winners":     winners,
			"cost_bytes":  totalBytes,
		})
		return 0
	}

	fmt.Fprintf(out, "skill query: intent=%q budget=%d mcp=%v\n", intent, limit, *includeMCP)
	if len(ranked) == 0 {
		fmt.Fprintln(out, "  working set: (empty — no card scored above zero)")
		fmt.Fprintln(out, "  cost: 0 bytes faulted")
		return 0
	}
	fmt.Fprintln(out, "  working set (ranked, cards only — no body paged):")
	for i, c := range ranked {
		marker := "  "
		if i < limit {
			marker = "->"
		}
		fmt.Fprintf(out, "  %s #%d %s %s  trigger=%q\n", marker, i+1, c.Ref.Kind, refLabel(c.Ref), skillTruncate(c.Trigger, 70))
	}
	fmt.Fprintf(out, "  faulted %d winner(s): %d bytes paged\n", len(winners), totalBytes)
	return 0
}

// --- residency -------------------------------------------------------------

func runSkillResidency(out, errw io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak skill residency", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	includeMCP := fs.Bool("mcp", false, "fold in the MCP-tool resolver")
	journalPath := fs.String("journal", "", "loader audit journal to reconcile against kernel counters")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if err := fs.Parse(argv); err != nil {
		fmt.Fprintln(errw, err)
		return 2
	}

	root := repoRoot()
	cards := catalogCards(root, *includeMCP)
	pins := loadSkillPageTable(root)
	recon := reconcileLoaderJournal(*journalPath)

	atRestBytes := 0
	for _, c := range cards {
		atRestBytes += len(c.CardBytes)
	}

	if *asJSON {
		_ = writeIndentedJSONNoEscape(out, map[string]any{
			"schema":        "fak-skill-residency/1",
			"include_mcp":   *includeMCP,
			"cards":         cardSummaries(cards),
			"at_rest_bytes": atRestBytes,
			"page_table":    pins,
			"journal":       recon,
		})
		return 0
	}

	fmt.Fprintf(out, "skill residency: %d card(s) at rest (%d bytes); mcp=%v\n", len(cards), atRestBytes, *includeMCP)
	for _, c := range cards {
		fmt.Fprintf(out, "  %-10s %-28s digest=%s state=at-rest trigger=%q\n",
			c.Ref.Kind, refLabel(c.Ref), shortDigest(c.Digest), skillTruncate(c.Trigger, 50))
	}
	if len(pins) > 0 {
		fmt.Fprintln(out, "  page-table pins:")
		for _, name := range sortedStringKeys(pins) {
			fmt.Fprintf(out, "    %s -> %s\n", name, pins[name])
		}
	} else {
		fmt.Fprintln(out, "  page-table pins: (none — no swap has flipped an entry)")
	}
	if *journalPath != "" {
		fmt.Fprintf(out, "  journal %s: faults=%d evictions=%d version_binds=%d reconciled=%v\n",
			*journalPath, recon.Faults, recon.Evictions, recon.VersionBinds, recon.Reconciled)
	}
	return 0
}

// --- swap ------------------------------------------------------------------

func runSkillSwap(out, errw io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak skill swap", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	fromVersion := fs.String("from", "", "expected current version (guards the remap; empty = unguarded pin)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	flagArgs, rest := partitionArgs(argv, map[string]bool{"from": true})
	if err := fs.Parse(flagArgs); err != nil {
		fmt.Fprintln(errw, err)
		return 2
	}
	args := rest
	if len(args) < 2 {
		fmt.Fprintln(errw, "fak skill swap: requires <name> <version>")
		skillUsage(errw)
		return 2
	}
	name, toVersion := args[0], args[1]

	root := repoRoot()
	pins := loadSkillPageTable(root)

	// Resolve the current version: a pin wins, else the skill's declared version.
	cur := pins[name]
	if cur == "" {
		if v, found := resolveSkillVersion(root, name); found {
			cur = v
		}
	}
	if *fromVersion != "" && cur != *fromVersion {
		fmt.Fprintf(errw, "fak skill swap: swap refused: skill %s is at %s, not %s\n", name, firstString(cur, "<unknown>"), *fromVersion)
		return 1
	}

	// Exercise the real C4 primitive. With no live kernel the blast radius is
	// zero (skillenv.Table returns BlastRadius{} when mmu/kvctx are nil); we
	// surface that honestly rather than inventing a cost.
	table := skillenv.New(nil, nil, nil)
	var blast ctxresidency.BlastRadius
	var err error
	if *fromVersion != "" {
		// The from-guard asserts the skill is currently at --from; the table's
		// own guard only knows its in-memory pins (empty here), so the CLI
		// checks the resolved current version (pin, else declared frontmatter).
		if cur != *fromVersion {
			fmt.Fprintf(errw, "fak skill swap: refused: %s is at %q, not %q\n", name, cur, *fromVersion)
			return 1
		}
		if _, blast, err = table.Swap(name, *fromVersion, toVersion); err != nil {
			fmt.Fprintf(errw, "fak skill swap: %v\n", err)
			return 1
		}
	} else {
		if _, blast, err = table.Pin(name, toVersion); err != nil {
			fmt.Fprintf(errw, "fak skill swap: %v\n", err)
			return 1
		}
	}
	emitSwap(out, *asJSON, name, cur, toVersion, blast)

	// Persist the flip so residency reads it back.
	pins[name] = toVersion
	if err := saveSkillPageTable(root, pins); err != nil {
		fmt.Fprintf(errw, "fak skill swap: persist page-table: %v\n", err)
		return 1
	}
	return 0
}

func emitSwap(out io.Writer, asJSON bool, name, from, to string, blast ctxresidency.BlastRadius) {
	if asJSON {
		_ = writeIndentedJSONNoEscape(out, map[string]any{
			"schema":       "fak-skill-swap/1",
			"name":         name,
			"from_version": from,
			"to_version":   to,
			"blast_radius": map[string]any{
				"tokens":            blast.Tokens,
				"dependent_entries": blast.DependentEntries,
			},
			"live_kernel": false,
		})
		return
	}
	fmt.Fprintf(out, "skill swap: %s  %q -> %q\n", name, from, to)
	fmt.Fprintf(out, "  pre-flip blast radius: tokens=%d dependent_entries=%d (zero without a live kernel)\n",
		blast.Tokens, blast.DependentEntries)
}

// --- shared helpers --------------------------------------------------------

// catalogCards returns the at-rest cards the catalog holds, via the resolver
// Index() calls Sync folded in. (capindex.Index has no public enumerator, so we
// re-list from the resolvers — the same cards Sync registered.)
func catalogCards(root string, includeMCP bool) []capindex.CapCard {
	cards := capindex.NewSkillResolver(skillDir(root)).Index()
	if includeMCP {
		cards = append(cards, capindexgw.NewMCPResolver(nil).Index()...)
	}
	sort.Slice(cards, func(i, j int) bool {
		if cards[i].Ref.Kind != cards[j].Ref.Kind {
			return cards[i].Ref.Kind < cards[j].Ref.Kind
		}
		return cards[i].Ref.Name < cards[j].Ref.Name
	})
	return cards
}

// resolveSkillVersion reads a skill's declared frontmatter version, if any.
func resolveSkillVersion(root, name string) (string, bool) {
	for _, c := range capindex.NewSkillResolver(skillDir(root)).Index() {
		if c.Ref.Name == name {
			return c.Ref.Version, c.Ref.Version != ""
		}
	}
	return "", false
}

func cardSummaries(cards []capindex.CapCard) []map[string]any {
	out := make([]map[string]any, 0, len(cards))
	for _, c := range cards {
		out = append(out, map[string]any{
			"kind":    string(c.Ref.Kind),
			"name":    c.Ref.Name,
			"version": c.Ref.Version,
			"digest":  c.Digest,
			"trigger": c.Trigger,
			"tags":    c.Tags,
		})
	}
	return out
}

// reconcileLoaderJournal folds the durable audit journal (the C6 trust floor)
// against zero kernel counters. Without a live kernel the authoritative counts
// are zero, so a non-empty journal surfaces a discrepancy; an absent journal
// reconciles vacuously true.
func reconcileLoaderJournal(path string) ctxresidency.LoaderSnapshot {
	if path == "" {
		return ctxresidency.LoaderSnapshot{Reconciled: true}
	}
	snap, err := ctxresidency.LoaderJournal(path, 0, 0, 0)
	if err != nil {
		return ctxresidency.LoaderSnapshot{Reconciled: false}
	}
	return snap
}

// --- page-table persistence ------------------------------------------------

// skillPageTablePath is the persisted version page-table: skill name -> active
// version. It is the on-disk form of skillenv.Table.List(), so a CLI swap
// survives across invocations and residency reads it back.
func skillPageTablePath(root string) string {
	return filepath.Join(root, ".claude", "skill-page-table.json")
}

func loadSkillPageTable(root string) map[string]string {
	b, err := os.ReadFile(skillPageTablePath(root))
	if err != nil {
		return map[string]string{}
	}
	var m map[string]string
	if err := json.Unmarshal(b, &m); err != nil {
		return map[string]string{}
	}
	if m == nil {
		m = map[string]string{}
	}
	return m
}

func saveSkillPageTable(root string, m map[string]string) error {
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(skillPageTablePath(root), b, 0o644)
}

// --- small format helpers --------------------------------------------------

func refLabel(r capindex.CapRef) string {
	if r.Version == "" {
		return r.Name
	}
	return r.Name + "@" + r.Version
}

func shortDigest(d string) string {
	if len(d) > 16 {
		return d[:16]
	}
	return d
}

func skillTruncate(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}

func sortedStringKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func runSkillCompile(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("skill compile", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var source, dialect string
	var expose skillStringListFlag
	var asJSON bool
	fs.StringVar(&source, "source", "", "stable source identity (defaults to path)")
	fs.StringVar(&dialect, "dialect", "openai", "model/harness tool-name dialect")
	fs.Var(&expose, "expose", "canonical tool name to expose (repeatable)")
	fs.BoolVar(&asJSON, "json", false, "emit machine-readable JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "fak skill compile: pass exactly one SKILL.md path")
		return 2
	}
	path := fs.Arg(0)
	if source == "" {
		source = filepath.ToSlash(filepath.Clean(path))
	}
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak skill compile: %v\n", err)
		return 1
	}
	reg, err := toolcatalog.CompileSkill(data, source)
	if err != nil {
		fmt.Fprintf(stderr, "fak skill compile: %v\n", err)
		return 1
	}
	view, err := toolcatalog.Expose([]toolcatalog.Registration{reg}, expose, dialect)
	if err != nil {
		fmt.Fprintf(stderr, "fak skill compile: %v\n", err)
		return 1
	}
	result := struct {
		Registration toolcatalog.Registration `json:"registration"`
		ModelView    toolcatalog.Snapshot     `json:"model_view"`
	}{reg, view}
	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(result); err != nil {
			fmt.Fprintf(stderr, "fak skill compile: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "%s %s\n", reg.Program.Name, reg.Digest)
	fmt.Fprintf(stdout, "registered: executable via %q\n", reg.Program.Executor.Argv)
	if len(view.Tools) == 0 {
		fmt.Fprintln(stdout, "model-visible: no (registration does not imply exposure)")
	} else {
		fmt.Fprintf(stdout, "model-visible: yes as %s (%s)\n", view.Tools[0].Name, view.Digest)
	}
	return 0
}

type skillStringListFlag []string

func (f *skillStringListFlag) String() string { return strings.Join(*f, ",") }
func (f *skillStringListFlag) Set(value string) error {
	*f = append(*f, value)
	return nil
}
