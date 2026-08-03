package main

// dev.go — `fak dev`, C2 of epic #2228 (#2231): the namespace verb of the DEV
// tier. Most of fak's ~270 verbs are internal dev/fleet tooling, not product;
// `fak dev <verb>` is their canonical spelling, and this file is the whole
// namespace:
//
//	fak dev                      -> the one-line-per-verb dev-tier listing
//	fak dev -h | --help          -> same listing
//	fak dev <devverb> [args...]  -> dispatch the verb EXACTLY as its bare spelling
//	fak dev <frontdoorverb>      -> refuse: a frontdoor verb has exactly one spelling
//	fak dev <unknown>            -> did-you-mean, tier-aware
//
// Dispatch parity is structural, not re-implemented: main() calls resolveDevVerb
// BEFORE the dispatch switch and, on a dev-tier hit, rewrites os.Args to the
// underlying verb so the very same case arm runs in the same process — no
// re-exec, no second parser, nothing to drift. The bare spelling keeps working
// unchanged until the C5 evidence-gated flip; the usage journal records this
// path as the composite verb ("dev commit" vs "commit"), which is the
// bare-vs-gated adoption evidence that flip reads.
//
// The tier answer comes compiled-in from internal/devindex (TierOf), so the
// namespace works outside a repo; only the listing's synopses degrade to the
// curated manifest when the live catalog is unreadable.

import (
	"fmt"
	"io"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

// resolveDevVerb folds one `fak dev ...` argv into a decision. exit == -1 means
// DISPATCH: the caller rewrites os.Args to (verb, rest) and falls through to the
// dispatch switch. Any other exit means the invocation is complete (listing
// printed, or a refusal written to stderr) and the caller exits with that code.
// Pure in/out over the writers so the contract is unit-testable without a
// process spawn.
func resolveDevVerb(argv []string, stdout, stderr io.Writer) (verb string, rest []string, exit int) {
	if len(argv) == 0 {
		printDevListing(stdout)
		return "", nil, 0
	}
	tok := strings.ToLower(strings.TrimSpace(argv[0]))
	if strings.HasPrefix(tok, "-") {
		if tok == "-h" || tok == "--help" {
			printDevListing(stdout)
			return "", nil, 0
		}
		fmt.Fprintf(stderr, "fak dev: unknown flag %q — 'fak dev' (or --help) lists the dev tier\n", argv[0])
		return "", nil, 2
	}
	if isDevOnlyVerb(tok) {
		return tok, argv[1:], -1
	}
	if tier, ok := devindex.TierOf(tok); ok {
		switch tier {
		case devindex.TierDev:
			return tok, argv[1:], -1
		case devindex.TierFrontdoor:
			// One spelling per frontdoor verb, or the frontdoor/dev ambiguity is
			// reproduced one level down.
			fmt.Fprintf(stderr, "fak dev: %q is a frontdoor verb — it has exactly one spelling: 'fak %s'\n", tok, tok)
			return "", nil, 2
		}
		// TierHidden falls through to the unknown path: an internal re-exec seam
		// is not advertised, under this namespace or anywhere else.
	}
	fmt.Fprintf(stderr, "fak dev: unknown verb %q\n", argv[0])
	if s := suggestVerb(tok); s != "" && !strings.EqualFold(s, tok) {
		switch t, _ := devindex.TierOf(s); t {
		case devindex.TierDev:
			fmt.Fprintf(stderr, "  did you mean 'fak dev %s'?\n", s)
		case devindex.TierFrontdoor:
			fmt.Fprintf(stderr, "  did you mean 'fak %s'?\n", s)
		}
	}
	fmt.Fprintln(stderr, "  'fak dev' lists the dev tier; 'fak help --all' lists every verb.")
	return "", nil, 2
}

func dispatchDevOnlyVerb(verb string, argv []string) bool {
	switch verb {
	case "gh-spam-comments":
		cmdGHSpamComments(argv)
		return true
	case "workspace":
		cmdDevWorkspace(argv)
		return true
	default:
		return false
	}
}

func isDevOnlyVerb(verb string) bool {
	for _, v := range devOnlyVerbs() {
		if v.Name == verb {
			return true
		}
	}
	return false
}

// printDevListing renders the dev tier, one line per verb with its catalog
// synopsis — the `fak help --all` treatment scoped to this namespace. Deliberately
// NOT curated-compact: the dev tier is the long tail, and its listing is the
// catalog pane, while the compact `fak help` stays the frontdoor's (C3).
func printDevListing(w io.Writer) {
	verbs := devTierVerbs()
	fmt.Fprintf(w, "fak dev — internal dev/fleet tooling, %d verbs. usage: fak dev <verb> [args...]\n", len(verbs))
	fmt.Fprintln(w, "(runs the verb exactly as its bare spelling; the product front door is 'fak help')")
	// The adopter line (#5465): someone who installed fak to front a model reaches this
	// listing and reasonably asks what these are and whether they are safe to ship. Answer
	// it here rather than making them find INSTALL.md. Column 0 on purpose — dev_test.go's
	// set check parses only the indented verb rows.
	fmt.Fprintln(w, "these develop the fak repository itself and are inert for an adopter: a verb runs")
	fmt.Fprintln(w, "only when you type it. INSTALL.md 'What's in the binary' has the full answer.")
	fmt.Fprintln(w)
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	for _, v := range verbs {
		fmt.Fprintf(tw, "  %s\t%s\n", v.Name, v.Synopsis)
	}
	tw.Flush()
	fmt.Fprintln(w, "\n'fak help <verb>' explains one in depth; 'fak dev <verb> -h' lists its flags.")
}

// devTierVerbs returns the dev-tier slice of the verb catalog, in the catalog's
// name order. In-repo the set derives from the live dispatch switch; outside a
// repo it degrades to the curated manifest — same tiers either way (compiled-in).
func devTierVerbs() []devindex.Verb {
	cat := helpCatalog()
	if cat == nil {
		cat = &devindex.Catalog{}
	}
	var out []devindex.Verb
	for _, v := range cat.Verbs() {
		if v.Tier == devindex.TierDev {
			out = append(out, v)
		}
	}
	out = append(out, devOnlyVerbs()...)
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func devOnlyVerbs() []devindex.Verb {
	return []devindex.Verb{
		{
			Name:     "gh-spam-comments",
			Synopsis: "scan GitHub issue/PR comments for untrusted spam across reusable abuse families (release archives, fake patch/fix lures) and optionally hide it as SPAM",
			Lane:     "cmd",
			Tier:     devindex.TierDev,
		},
		{
			Name:     "workspace",
			Synopsis: "map the local agentic-dev workspace spine and read the guard floor's live decision stream (allowed/blocked/quarantined/witnessed) — the #3426 preview",
			Lane:     "cmd",
			Tier:     devindex.TierDev,
		},
	}
}
