package devcmd

// fak wiki — the fak-native, witness-verified repo wiki (epic #4277, mined from
// deepwiki-open @ 16f35a0 — inspire, clean-room). Two deterministic subverbs, the
// two things every LLM wiki generator in the field structurally cannot do:
//
//	fak wiki structure        emit the section→page tree projected from the
//	                          self-index (leaves/lanes/docs) — NOT LLM-inferred.
//	                          --json prints the stable wiki.json. (L1 #4278)
//	fak wiki verify <page.md> resolve every `Sources:[path:line]` code citation in
//	                          the page against the working tree; exit nonzero on any
//	                          dangling cite (missing file / out-of-range line). A CI
//	                          gate the whole pipeline runs before publish. (L3 #4280)
//
// Both are pure views over internal/wiki + internal/devindex: no LLM, no clock, no
// network. `structure` seeds the tree from ground truth; `verify` witnesses cites.

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/devindex"
	"github.com/anthony-chaudhary/fak/internal/wiki"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func RunWiki(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		writeWikiUsage(stderr)
		return 2
	}
	sub := argv[0]
	switch sub {
	case "structure", "struct", "tree":
		return wikiStructure(stdout, stderr, argv[1:])
	case "verify", "check":
		return wikiVerify(stdout, stderr, argv[1:])
	case "fresh", "freshness":
		return wikiFresh(stdout, stderr, argv[1:])
	case "score":
		return wikiScore(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		writeWikiUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak wiki: unknown subcommand %q\n", sub)
		writeWikiUsage(stderr)
		return 2
	}
}

func writeWikiUsage(w io.Writer) {
	fmt.Fprint(w, `fak wiki — witness-verified repo wiki

usage:
  fak wiki structure [--root DIR] [--json]   section→page tree from the self-index
  fak wiki verify <page.md> [--root DIR] [--json]
                                             resolve Sources:[path:line] cites vs the tree
  fak wiki fresh <page.md> [--root DIR] [--json]
                                             flag the page stale if any cited file moved since its generated_at_sha
  fak wiki score [--pages DIR] [--root DIR] [--check] [--json]
                                             citation-resolve + leaf-coverage + freshness score (a CI gate)

`)
}

// wikiStructure prints the deterministic section→page tree. Text by default (a
// navigable outline), --json for the stable wiki.json payload.
func wikiStructure(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wiki structure", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wiki")
	root := fs.String("root", "", "repo root (default: search upward for dos.toml)")
	asJSON := fs.Bool("json", false, "emit the stable wiki.json")
	if err := fs.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	rootDir := *root
	if rootDir == "" {
		rootDir = devindex.FindRoot(".")
	}
	cat, err := devindex.Load(rootDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak wiki: %v\n", err)
		return 1
	}
	tree := wiki.Structure(cat)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, tree, "fak wiki")
	}
	return renderWikiTree(stdout, tree)
}

func renderWikiTree(stdout io.Writer, tree wiki.Tree) int {
	fmt.Fprintf(stdout, "wiki: %s (%d pages)\n", tree.Repo, tree.PageCount())
	for _, s := range tree.Sections {
		fmt.Fprintf(stdout, "\n%s (%d)\n", s.Title, len(s.Pages))
		for _, p := range s.Pages {
			files := ""
			if len(p.RelevantFiles) > 0 {
				files = "  ← " + joinCap(p.RelevantFiles, 3)
			}
			fmt.Fprintf(stdout, "  - %s%s\n", p.Title, files)
		}
	}
	return 0
}

// wikiVerify resolves the page's code citations against the tree. It exits 0 when
// every cite resolves (or the page cites nothing), 1 on any dangler — the gate.
func wikiVerify(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wiki verify", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wiki")
	root := fs.String("root", "", "repo root the cites resolve against (default: search upward for dos.toml)")
	asJSON := fs.Bool("json", false, "emit the danglers as JSON")
	if err := fs.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if fs.NArg() == 0 {
		fmt.Fprintln(stderr, "fak wiki verify: needs a <page.md>")
		return 2
	}
	page := fs.Arg(0)
	md, err := os.ReadFile(page)
	if err != nil {
		fmt.Fprintf(stderr, "fak wiki verify: %v\n", err)
		return 1
	}
	rootDir := *root
	if rootDir == "" {
		rootDir = devindex.FindRoot(".")
	}
	dangs := wiki.VerifyCitations(rootDir, md)
	if *asJSON {
		code := encodeJSONOrFail(stdout, stderr, dangs, "fak wiki verify")
		if code == 0 && len(dangs) > 0 {
			return 1
		}
		return code
	}
	if len(dangs) == 0 {
		fmt.Fprintf(stdout, "ok: %s — every code citation resolves against %s\n", page, rootDir)
		return 0
	}
	tw := tabwriter.NewWriter(stdout, 0, 2, 2, ' ', 0)
	fmt.Fprintf(tw, "DANGLING\tLINE\tREASON\tCITE\n")
	for _, d := range dangs {
		detail := string(d.Reason)
		if d.Reason == wiki.ReasonLineOutOfRange {
			detail = fmt.Sprintf("%s (file has %d lines)", d.Reason, d.Lines)
		}
		fmt.Fprintf(tw, "%s\tL%d\t%s\t%s\n", d.Path, d.Line, detail, d.Raw)
	}
	tw.Flush()
	fmt.Fprintf(stderr, "fak wiki verify: %d dangling citation(s) in %s\n", len(dangs), page)
	return 1
}

// wikiFresh witnesses one page's freshness (L4 #4281): it parses the page's
// generated_at_sha + cited_files frontmatter, asks git which files changed since
// that SHA, and flags the page stale if any cited file is in that set. Exit 0 fresh,
// 1 stale (the gate), 2 usage.
func wikiFresh(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wiki fresh", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wiki")
	root := fs.String("root", "", "repo root the SHA/cited files resolve against (default: search upward for dos.toml)")
	asJSON := fs.Bool("json", false, "emit the freshness verdict as JSON")
	if err := fs.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if fs.NArg() == 0 {
		fmt.Fprintln(stderr, "fak wiki fresh: needs a <page.md>")
		return 2
	}
	page := fs.Arg(0)
	md, err := os.ReadFile(page)
	if err != nil {
		fmt.Fprintf(stderr, "fak wiki fresh: %v\n", err)
		return 1
	}
	rootDir := *root
	if rootDir == "" {
		rootDir = devindex.FindRoot(".")
	}
	meta := wiki.ParseFrontmatter(md)

	// `git diff --name-only <sha>` compares the pinned build commit to the working
	// tree, so it catches BOTH committed and uncommitted drift of the cited code
	// since the page was generated — the freshness question exactly.
	var changed []string
	if strings.TrimSpace(meta.GeneratedAtSHA) != "" {
		out, gerr := gitOut(rootDir, "diff", "--name-only", meta.GeneratedAtSHA)
		if gerr != nil {
			fmt.Fprintf(stderr, "fak wiki fresh: %v\n", gerr)
			return 1
		}
		changed = nonEmptyLines(out)
	}
	sp, stale := wiki.DriftStaleWikiPage(page, meta, changed)

	if *asJSON {
		verdict := struct {
			Page   string          `json:"page"`
			Stale  bool            `json:"stale"`
			Detail *wiki.StalePage `json:"detail,omitempty"`
		}{Page: page, Stale: stale}
		if stale {
			verdict.Detail = &sp
		}
		code := encodeJSONOrFail(stdout, stderr, verdict, "fak wiki fresh")
		if code == 0 && stale {
			return 1
		}
		return code
	}

	if !stale {
		fmt.Fprintf(stdout, "fresh: %s — pinned at %s, no cited file has moved\n", page, short(meta.GeneratedAtSHA))
		return 0
	}
	switch sp.Reason {
	case wiki.ReasonNoSHA:
		fmt.Fprintf(stdout, "stale: %s — no generated_at_sha; a generated page must pin its build commit\n", page)
	case wiki.ReasonCitedCodeMoved:
		fmt.Fprintf(stdout, "stale: %s — %d cited file(s) moved since %s:\n", page, len(sp.Touched), short(sp.SHA))
		for _, f := range sp.Touched {
			fmt.Fprintf(stdout, "  - %s\n", f)
		}
	}
	return 1
}

// wikiScore folds the generated pages under --pages into the wiki quality report
// (L7 #4284): citation-resolve rate (L3), leaf coverage (L1), freshness (L4). With
// --check it exits nonzero when the score falls below the floors — the CI gate that
// keeps the wiki honest as the tree moves.
func wikiScore(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wiki score", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wiki")
	root := fs.String("root", "", "repo root the cites resolve against (default: search upward for dos.toml)")
	pagesDir := fs.String("pages", "docs/wiki", "directory of generated wiki pages (*.md)")
	check := fs.Bool("check", false, "exit nonzero when the score is below the floors")
	minResolve := fs.Float64("min-resolve", 1.0, "minimum citation-resolve rate for --check")
	minCoverage := fs.Float64("min-coverage", 0.0, "minimum leaf-coverage rate for --check")
	asJSON := fs.Bool("json", false, "emit the score as JSON")
	if err := fs.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	rootDir := *root
	if rootDir == "" {
		rootDir = devindex.FindRoot(".")
	}
	cat, err := devindex.Load(rootDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak wiki: %v\n", err)
		return 1
	}
	pages, absDir, err := loadWikiPages(rootDir, *pagesDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak wiki score: %v\n", err)
		return 1
	}
	score := wiki.ComputeScore(rootDir, cat, pages)
	pass := score.Passes(*minResolve, *minCoverage)

	if *asJSON {
		code := encodeJSONOrFail(stdout, stderr, score, "fak wiki score")
		if code == 0 && *check && !pass {
			return 1
		}
		return code
	}

	renderScore(stdout, score, absDir)
	if *check && !pass {
		fmt.Fprintf(stderr, "fak wiki score: below floor (resolve %.0f%% < %.0f%% or coverage %.0f%% < %.0f%%)\n",
			100*score.CitationResolveRate, 100**minResolve, 100*score.LeafCoverage, 100**minCoverage)
		return 1
	}
	return 0
}

func renderScore(stdout io.Writer, s wiki.Score, dir string) {
	if s.Pages == 0 {
		fmt.Fprintf(stdout, "wiki: %s — no generated pages under %s (nothing scored)\n", s.Repo, dir)
	} else {
		fmt.Fprintf(stdout, "wiki: %s — %d pages under %s\n", s.Repo, s.Pages, dir)
	}
	tw := tabwriter.NewWriter(stdout, 0, 2, 2, ' ', 0)
	fmt.Fprintf(tw, "METRIC\tVALUE\tDETAIL\n")
	fmt.Fprintf(tw, "citation-resolve\t%.0f%%\t%d/%d cites resolve\n", 100*s.CitationResolveRate, s.CitationsResolved, s.Citations)
	fmt.Fprintf(tw, "leaf-coverage\t%.0f%%\t%d/%d leaves have a page\n", 100*s.LeafCoverage, s.LeavesCovered, s.Leaves)
	fmt.Fprintf(tw, "freshness\t%.0f%%\t%d/%d pages pin a generated_at_sha\n", 100*s.FreshRate, s.FreshPages, s.Pages)
	tw.Flush()
	if len(s.Danglers) > 0 {
		fmt.Fprintf(stdout, "\n%d dangling citation(s) — run `fak wiki verify <page>` to locate\n", len(s.Danglers))
	}
}

// loadWikiPages walks pagesRel under root for *.md files and reads each into a
// PageInput whose RelID is the slash path under the pages dir with ".md" stripped
// (matching the Structure page ID scheme). A missing pages dir is not an error — it
// returns zero pages, the honest "nothing generated yet" state. It returns the
// absolute pages dir for the human-facing render.
func loadWikiPages(root, pagesRel string) ([]wiki.PageInput, string, error) {
	absDir := pagesRel
	if !filepath.IsAbs(absDir) {
		absDir = filepath.Join(root, filepath.FromSlash(pagesRel))
	}
	info, err := os.Stat(absDir)
	if err != nil || !info.IsDir() {
		return nil, absDir, nil // no wiki dir yet: 0 pages, not a failure
	}
	var pages []wiki.PageInput
	walkErr := filepath.WalkDir(absDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.EqualFold(filepath.Ext(path), ".md") {
			return nil
		}
		rel, rerr := filepath.Rel(absDir, path)
		if rerr != nil {
			return rerr
		}
		body, rerr := os.ReadFile(path)
		if rerr != nil {
			return rerr
		}
		id := strings.TrimSuffix(filepath.ToSlash(rel), ".md")
		pages = append(pages, wiki.PageInput{RelID: id, Markdown: body})
		return nil
	})
	if walkErr != nil {
		return nil, absDir, walkErr
	}
	return pages, absDir, nil
}

// joinCap joins up to n entries with ", ", appending a "+k more" tail when the
// slice is longer — keeps the text tree from spilling a huge glob list per page.
func joinCap(xs []string, n int) string {
	if len(xs) <= n {
		return joinComma(xs)
	}
	return joinComma(xs[:n]) + fmt.Sprintf(", +%d more", len(xs)-n)
}

func joinComma(xs []string) string {
	out := ""
	for i, x := range xs {
		if i > 0 {
			out += ", "
		}
		out += x
	}
	return out
}

func verbFlagUsage(fs *flag.FlagSet, _ string) {
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "Usage of %s:\n", fs.Name())
		fs.PrintDefaults()
	}
}

func encodeJSONOrFail(stdout, stderr io.Writer, v any, label string) int {
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		fmt.Fprintf(stderr, "%s: encode json: %v\n", label, err)
		return 1
	}
	return 0
}

func gitOut(root string, args ...string) (string, error) {
	cmd := exec.Command("git", append([]string{"-C", root}, args...)...)
	windowgate.ConfigureBackgroundCommand(cmd)
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = io.Discard
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("git %s: %w", strings.Join(args, " "), err)
	}
	return out.String(), nil
}

func nonEmptyLines(s string) []string {
	var rows []string
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimRight(line, "\r")
		if strings.TrimSpace(line) != "" {
			rows = append(rows, line)
		}
	}
	return rows
}

func short(sha string) string {
	if len(sha) > 12 {
		return sha[:12]
	}
	return sha
}
