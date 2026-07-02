// help.go — the `fak help` verb and the compact usage overview.
//
// Taste contract (the reason this file exists): `fak --help`, bare `fak`, and a
// mistyped verb used to dump the full ~650-line usage wall. The wall's depth is
// real documentation, but it belongs one level down. The shape now:
//
//	fak / fak --help / fak help   -> the compact curated overview (below)
//	fak help <verb>               -> that verb's synopsis + its usage-wall block
//	fak help --all                -> one line per verb, the whole catalog
//	fak help --full               -> the original full wall, unchanged
//	fak <typo>                    -> three lines: unknown verb, did-you-mean, next step
//
// The overview is hand-curated text (membership is taste, not derivation) but it
// cannot drift: help_test.go asserts every overview verb is live in the dispatch
// switch and that the overview stays compact. Per-verb depth is carved from the
// usage.go wall constants at runtime, so there is exactly one authored copy of
// each verb's documentation. The devindex catalog (when the repo is readable)
// supplies the --all index and upgrades did-you-mean; outside a repo help still
// works from the compiled-in wall alone.
package main

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/devindex"
)

// overviewEntry is one curated line of the compact overview. The blurb is
// hand-written for scanability at a glance — shorter than the catalog synopsis,
// same voice. help_test.go gates the width so the overview stays columnar.
type overviewEntry struct {
	name  string
	blurb string
}

// overviewGroups is the curated compact overview: the verbs an operator or an
// agent reaches for in a normal working session, grouped by what they are doing.
// Membership is a taste call, deliberately small — everything else is one
// `fak help --all` away. help_test.go asserts every name here is a live verb.
var overviewGroups = []struct {
	title   string
	entries []overviewEntry
}{
	{"the work loop", []overviewEntry{
		{"orient", "conventions for the paths you are about to touch: lane, tier, tests, stamp"},
		{"whats-changed", "what peers changed under your target paths since the session base"},
		{"affected", "run only the go tests your working-tree change can affect"},
		{"test", "host-aware test runner: fast | full | race | <pkg>"},
		{"done", "pre-claim self-check: paths clean + tests + claims-lint + witness"},
		{"commit", "safe shared-trunk commit by explicit pathspec (never git add -A)"},
		{"sweep", "drive a dirty multi-session tree toward zero, one lane at a time"},
		{"sync", "safe fast-forward sync for a dirty shared worktree; never pull/stash/reset"},
		{"merge", "predict a shared-trunk merge before starting it"},
	}},
	{"guard + serve (the front doors)", []overviewEntry{
		{"guard", "wrap an agent harness: adjudicate every tool call in-process"},
		{"serve", "the OpenAI-compatible gateway in front of a local or remote model"},
		{"run", "run an agent turn (or a recorded trace) through the kernel"},
		{"preflight", "adjudicate one tool call against a policy, no model in the loop"},
		{"policy", "dump / check the deployable capability floor"},
	}},
	{"observe + operate", []overviewEntry{
		{"ps", "live served-session process table ('fak top' = --watch)"},
		{"session", "read, cancel, or steer a served session in flight"},
		{"resume", "what happens to the prompt cache on resume, and what to do"},
		{"doctor", "operator diagnostic: answer-shape witness + kernel admit verdict"},
		{"scorecard", "the control pane: every metric's debt, grade, and trend"},
		{"recover", "map a refusal reason token to concrete recovery commands"},
	}},
	{"self-index", []overviewEntry{
		{"index", "queryable self-index: lane / leaf / docs / claims / verbs"},
		{"feature", "the unified self-feature catalog"},
		{"version", "print the fak version"},
	}},
}

// usageCompact prints the curated overview — what `fak`, `fak -h`, and `fak help`
// show. Kept deliberately far under one screen; the gate test holds the line.
func usageCompact(w io.Writer) {
	fmt.Fprintf(w, "fak - the Fused Agent Kernel (v%s)\n", appversion.Current())
	for _, g := range overviewGroups {
		fmt.Fprintf(w, "\n%s:\n", g.title)
		for _, e := range g.entries {
			fmt.Fprintf(w, "  %-14s %s\n", e.name, e.blurb)
		}
	}
	fmt.Fprintln(w)
	if cat := helpCatalog(); cat != nil {
		fmt.Fprintf(w, "%d verbs in this build. ", len(cat.Verbs()))
	}
	fmt.Fprintln(w, "'fak help --all' lists every verb;")
	fmt.Fprintln(w, "'fak help <verb>' explains one in depth; 'fak <verb> -h' lists its flags.")
}

// cmdHelp implements `fak help [verb | --all | --full]` (also reached via
// `fak -h` / `fak --help`). Requested help goes to stdout; only the error path
// (an unknown verb argument) writes stderr and exits 2.
func cmdHelp(args []string) {
	if len(args) == 0 {
		usageCompact(os.Stdout)
		return
	}
	switch args[0] {
	case "--all", "-a", "all":
		usageAllVerbs(os.Stdout)
	case "--full", "full":
		usageWall(os.Stdout)
	default:
		if printVerbHelp(os.Stdout, args[0]) {
			return
		}
		fmt.Fprintf(os.Stderr, "fak help: no verb %q\n", args[0])
		if s := suggestVerb(args[0]); s != "" {
			fmt.Fprintf(os.Stderr, "  did you mean 'fak help %s'?\n", s)
		}
		fmt.Fprintln(os.Stderr, "  'fak help --all' lists every verb.")
		os.Exit(2)
	}
}

// usageAllVerbs prints the one-line-per-verb index of the whole catalog: the
// devindex live view (dispatch-derived coverage, curated synopses) when the repo
// is readable, else the compiled-in wall.
func usageAllVerbs(w io.Writer) {
	cat := helpCatalog()
	if cat == nil {
		usageWall(w)
		return
	}
	verbs := cat.Verbs()
	fmt.Fprintf(w, "fak - the Fused Agent Kernel (v%s) — %d verbs\n\n", appversion.Current(), len(verbs))
	for _, v := range verbs {
		name := v.Name
		if len(v.Aliases) > 0 {
			name += " (" + strings.Join(v.Aliases, ", ") + ")"
		}
		fmt.Fprintf(w, "  %-34s %s\n", name, v.Synopsis)
	}
	fmt.Fprintln(w, "\n'fak help <verb>' explains one in depth; 'fak <verb> -h' lists its flags.")
}

// printVerbHelp prints one verb's deep help: the catalog synopsis line (when
// available) over the verb's block(s) carved from the usage wall. Reports false
// when neither source knows the verb.
func printVerbHelp(w io.Writer, tok string) bool {
	tok = strings.ToLower(strings.TrimSpace(tok))
	spellings := []string{tok}
	var header string
	cat := helpCatalog()
	if cat == nil {
		// VerbByName reads only the curated manifest, so an unloaded catalog
		// still answers — help works outside a repo.
		cat = &devindex.Catalog{}
	}
	if v, ok := cat.VerbByName(tok); ok {
		spellings = v.Spellings()
		header = fmt.Sprintf("fak %s — %s", v.Name, v.Synopsis)
		if len(v.Aliases) > 0 {
			header += "\naliases: " + strings.Join(v.Aliases, ", ")
		}
		if v.Doc != "" {
			header += "\nsee also: " + v.Doc
		}
	}
	sections := verbWallSections(spellings)
	if header == "" && len(sections) == 0 {
		return false
	}
	if header != "" {
		fmt.Fprintln(w, header)
	}
	for _, s := range sections {
		fmt.Fprintln(w)
		fmt.Fprint(w, s)
	}
	if len(sections) == 0 {
		fmt.Fprintln(w)
	}
	fmt.Fprintf(w, "\nflags: fak %s -h\n", tok)
	return true
}

// verbWallSections carves the usage wall (the three usage.go constants) into the
// block(s) documenting the given verb spellings: each run of lines from a
// `  fak <verb> ...` synopsis header through its indented continuation/paragraph
// lines. A verb documented in more than one wall section yields multiple blocks.
func verbWallSections(spellings []string) []string {
	want := map[string]bool{}
	for _, s := range spellings {
		want[strings.ToLower(s)] = true
	}
	var sections []string
	var cur []string
	inSection := false
	flush := func() {
		if len(cur) > 0 {
			sections = append(sections, strings.Join(cur, "\n")+"\n")
			cur = nil
		}
		inSection = false
	}
	for _, line := range strings.Split(usageWallText(), "\n") {
		if tok, ok := wallHeaderVerb(line); ok {
			if want[tok] {
				if !inSection {
					inSection = true
				}
				cur = append(cur, line)
			} else {
				flush()
			}
			continue
		}
		if inSection {
			// Continuation lines (synopsis wraps and the paragraph) are deeply
			// indented; a blank or unindented line ends the block.
			if strings.HasPrefix(line, "    ") {
				cur = append(cur, line)
			} else {
				flush()
			}
		}
	}
	flush()
	return sections
}

// wallHeaderVerb reports whether the wall line is a verb synopsis header
// (`  fak <verb> ...`) and returns the lowercased verb token if so.
func wallHeaderVerb(line string) (string, bool) {
	if !strings.HasPrefix(line, "  fak ") {
		return "", false
	}
	rest := strings.TrimSpace(line[len("  fak "):])
	if rest == "" {
		return "", false
	}
	tok := strings.Fields(rest)[0]
	return strings.ToLower(tok), true
}

// suggestVerb proposes the closest known verb for a mistyped token: first by
// edit distance over every catalog spelling plus every wall header token, then
// (in-repo) by the catalog's lexical verb search, so `fak docs` can still point
// at `fak index` even with no near-miss spelling. Empty when nothing is close.
func suggestVerb(tok string) string {
	tok = strings.ToLower(strings.TrimSpace(tok))
	if tok == "" {
		return ""
	}
	best, bestDist := "", 3
	consider := func(name string) {
		if name == "" {
			return
		}
		if d := levenshtein(tok, strings.ToLower(name)); d < bestDist || (d == bestDist && best != "" && name < best) {
			best, bestDist = name, d
		}
	}
	cat := helpCatalog()
	if cat != nil {
		for _, v := range cat.Verbs() {
			for _, sp := range v.Spellings() {
				consider(sp)
			}
		}
	}
	for _, line := range strings.Split(usageWallText(), "\n") {
		if t, ok := wallHeaderVerb(line); ok {
			consider(t)
		}
	}
	// Short tokens need a tight radius or the suggestion is noise.
	maxDist := 2
	if len(tok) <= 3 {
		maxDist = 1
	}
	if best != "" && bestDist <= maxDist {
		return best
	}
	if cat != nil {
		if hits := cat.SearchVerbs(tok); len(hits) > 0 {
			return hits[0].Name
		}
	}
	return ""
}

// helpCatalog loads the devindex catalog when the repo is readable (dos.toml
// found from cwd), else nil — help then runs from the compiled-in wall alone.
func helpCatalog() *devindex.Catalog {
	root := devindex.FindRoot(".")
	cat, err := devindex.Load(root)
	if err != nil {
		return nil
	}
	return cat
}
