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
// each verb's documentation. Runtime help has no repository-development catalog
// dependency; the separate fak-dev artifact owns that inventory.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/appversion"
)

// overviewEntry is one curated line of the compact overview. The blurb is
// hand-written for scanability at a glance — shorter than the catalog synopsis,
// same voice. help_test.go gates the width so the overview stays columnar.
type overviewEntry struct {
	name  string
	blurb string
}

// overviewGroups is the compact runtime-product overview, grouped by what an
// operator or adopter is doing. Repository-development inventory and search are
// exposed by the separately built fak-dev artifact.
var overviewGroups = []struct {
	title   string
	entries []overviewEntry
}{
	{"spend fewer tokens + turns", []overviewEntry{
		{"capabilities", "query token, turn, cache, routing, and session-control outcomes"},
		{"ablate", "same-trace cache ablation: attribute savings instead of guessing"},
		{"resume", "price full replay vs cut/reset when resuming a long context"},
		{"session", "budget turns/tokens/context; steer or stop without another prompt turn"},
		{"info", "live reused-token, effective-cost, and total-savings overlay"},
	}},
	{"manage + serve", []overviewEntry{
		{"agent", "the offline proof: run one managed-agent task end to end ('fak agent --offline')"},
		{"manage", "wrap an agent harness: manage every tool call in-process ('fak m'; legacy: guard)"},
		{"serve", "the OpenAI-compatible gateway in front of a local or remote model"},
		{"run", "run an agent turn (or a recorded trace / 'fak replay') through the kernel"},
		{"codex", "launch OpenAI Codex routed through the kernel"},
	}},
	{"supporting capability floor", []overviewEntry{
		{"preflight", "adjudicate one tool call against a policy, no model in the loop"},
		{"policy", "dump / check the deployable capability floor"},
		{"attest", "compliance attestation: prove the policy floor from preflight"},
		{"audit", "verify / export a guard decision journal's hash chain"},
		{"egress", "prove the network-egress floor (cloud-metadata / SSRF class)"},
	}},
	{"observe + operate", []overviewEntry{
		{"ps", "live served-session process table ('fak top' = --watch)"},
		{"signal", "job control for a running session: pause / resume / stop / steer"},
		{"doctor", "operator diagnostic: answer-shape witness + kernel admit verdict"},
		{"recover", "map a refusal reason token to concrete recovery commands"},
	}},
	{"models + housekeeping", []overviewEntry{
		{"model", "resolve / cache an hf:// model ('fak pull' / 'fak ls' aliases)"},
		{"self-update", "converge a built-from-source fak binary on origin/main"},
		{"version", "print the fak version"},
		{"help", "this overview; 'help <verb>' for depth, 'help --all' for the catalog"},
	}},
}

// frontdoorCompanions maps a frontdoor companion spelling to the concept it folds
// into for the compact overview: `fak replay` is `fak run --trace`, `fak top` is
// `fak ps --watch`, `fak pull`/`fak ls` are `fak model` sub-spellings. Each is a
// frontdoor-tier verb in its own right (C1), but listing it separately would just
// reproduce its primary; instead the primary's blurb names it. The set-equality
// gate treats a companion as covered when its primary is in the overview AND the
// companion's spelling appears in that primary's blurb — folding never hides a verb.
var frontdoorCompanions = map[string]string{
	"guard":  "manage",
	"m":      "manage",
	"replay": "run",
	"top":    "ps",
	"pull":   "model",
	"ls":     "model",
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
	fmt.Fprintln(w, "repository-development tooling is in the separate 'fak-dev' executable.")
	fmt.Fprintln(w, "'fak help --all' lists runtime usage;")
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

// usageAllVerbs prints runtime fak's complete usage wall. The development command
// catalog belongs to fak-dev and is intentionally absent from this binary.
func usageAllVerbs(w io.Writer) {
	usageWall(w)
}

// printVerbHelp prints one verb's deep help: the catalog synopsis line (when
// available) over the verb's block(s) carved from the usage wall. Reports false
// when neither source knows the verb.
func printVerbHelp(w io.Writer, tok string) bool {
	tok = strings.ToLower(strings.TrimSpace(tok))
	if !verbDeepHelpBody(w, tok) {
		return false
	}
	flagTok := tok
	if tok == "m" || tok == "guard" {
		flagTok = "manage"
	}
	fmt.Fprintf(w, "\nflags: fak %s -h\n", flagTok)
	return true
}

// verbDeepHelpBody writes the verb's deep help body (catalog synopsis/aliases/doc
// line over the wall-carved section(s)) WITHOUT the trailing "flags: fak X -h"
// hint — the shared core behind both `fak help <verb>` (printVerbHelp, which adds
// the hint) and verbFlagUsage (which follows the body with the real flag dump
// instead of a hint pointing at itself). Reports false when neither the catalog
// nor the wall knows the verb.
func verbDeepHelpBody(w io.Writer, tok string) bool {
	tok = strings.ToLower(strings.TrimSpace(tok))
	if tok == "capabilities" {
		writeCapabilitiesUsage(w)
		return true
	}
	if tok == "manage" || tok == "m" || tok == "guard" {
		// Lead every spelling with the canonical manage block, then retain the
		// complete historical guard wall so no invocation or flag becomes
		// undiscoverable during the compatibility sunset.
		manageSections := verbWallSections([]string{"manage"})
		for _, section := range manageSections {
			fmt.Fprint(w, section)
		}
		if tok == "guard" {
			fmt.Fprintln(w, "DEPRECATED: fak guard is the legacy compatibility spelling; migrate to fak manage (or fak m).")
			fmt.Fprintln(w, "Sunset notice: guard remains fully compatible during migration; no removal date is set.")
			fmt.Fprintln(w)
		}
		legacySections := verbWallSections([]string{"guard"})
		for _, section := range legacySections {
			fmt.Fprint(w, section)
		}
		return len(manageSections)+len(legacySections) > 0
	}
	sections := verbWallSections([]string{tok})
	if len(sections) == 0 {
		return false
	}
	for _, section := range sections {
		fmt.Fprint(w, section)
	}
	return true
}

// verbFlagUsage sets fs.Usage so `fak <verb> --help` / `-h` prints the verb's deep
// help (the catalog synopsis + the wall-carved block) above the flag defaults,
// instead of the bare `flag` package dump ("Usage of <verb>: ..."). tok is the
// catalog verb name to look up — for a sub-command FlagSet (e.g. "session ls")
// pass the parent verb ("session") so the lookup hits the verb's own wall entry.
// Call this right after flag.NewFlagSet, before Parse.
func verbFlagUsage(fs *flag.FlagSet, tok string) {
	fs.Usage = func() {
		w := fs.Output()
		if !verbDeepHelpBody(w, tok) {
			fmt.Fprintf(w, "Usage of %s:\n", fs.Name())
			fs.PrintDefaults()
			return
		}
		fmt.Fprintln(w)
		fs.PrintDefaults()
	}
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
		matchName := strings.ToLower(name)
		preferred := name
		if matchName == "guard" || matchName == "m" {
			preferred = "manage"
		}
		if d := levenshtein(tok, matchName); d < bestDist || (d == bestDist && best != "" && preferred < best) {
			best, bestDist = preferred, d
		}
	}
	for _, group := range overviewGroups {
		for _, entry := range group.entries {
			consider(entry.name)
		}
	}
	for _, line := range strings.Split(usageWallText(), "\n") {
		if name, ok := wallHeaderVerb(line); ok {
			consider(name)
		}
	}
	maxDist := 2
	if len(tok) <= 3 {
		maxDist = 1
	}
	if best != "" && bestDist <= maxDist {
		return best
	}
	return ""
}

// suggestVerbSpelling is the TIER-AWARE did-you-mean for a mistyped token at the
// TOP level (`fak <typo>`): it resolves the closest verb (suggestVerb) and returns
// the CANONICAL spelling to suggest — `dev <verb>` for a dev-tier verb, the bare
// `<verb>` for frontdoor. Empty when nothing is close. Keeping the tier fold in
// ONE place is what makes the C5 enforcement flip a one-line change: today
// `fak swep` already answers "did you mean 'fak dev sweep'?" (help stays ungated,
// so `fak help <typo>` keeps suggesting the bare `fak help <verb>` spelling).
func suggestVerbSpelling(tok string) string {
	return suggestVerb(tok)
}
