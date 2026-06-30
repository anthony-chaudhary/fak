package main

import (
	"fmt"
	"io"
	"os"
	"path"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/sotamatrix"
)

// `fak sota`  -  the agent-facing prior-art lookup an agent runs BEFORE writing a
// kernel. This repo has a documented failure mode: an agent reaches for "implement
// the Mac Q6_K fused MLP from scratch" or "hand-roll the amd64 kquant SIMD"
// without first checking that llama.cpp's GGML kernels, Marlin / CUTLASS /
// FlashInfer, or a named paper already solved the same contraction. This verb
// reads internal/sotamatrix (the in-binary source of truth) and answers, for one
// operation OR one kernel file path:
//
//	fak sota [list]            the table of every operation: slug, title, route, SOTA
//	fak sota <slug>            the full reference for one op (SOTA stack, link, route, oracle, papers)
//	fak sota <file>            the op(s) whose FileGlobs match a kernel path
//
// So `fak sota awq-int4-gemm` or `fak sota internal/model/awq.go` prints
// "Marlin / AutoAWQ, route=borrow, oracle=cpuref+HF, read https://github.com/IST-DASLab/marlin"
// before the author hand-rolls what is known art.
func cmdSota(argv []string) { os.Exit(runSota(os.Stdout, os.Stderr, argv)) }

func runSota(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		return sotaList(stdout, stderr, nil)
	}
	switch argv[0] {
	case "list", "ls":
		return sotaList(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		sotaUsage(stdout)
		return 0
	default:
		return sotaLookup(stdout, stderr, argv[0])
	}
}

func sotaUsage(w io.Writer) {
	fmt.Fprint(w, `fak sota  -  the prior-art lookup: read the SOTA reference BEFORE writing a kernel

usage:
  fak sota [list] [--json]   every operation: slug, title, route, SOTA stack
  fak sota <slug>            the full reference for one op (slug from the list)
  fak sota <file>            the op(s) whose kernel files match a path

Run this before hand-rolling a contraction  -  almost no op should be written
from scratch. Each row's route says how to relate to the reference:
  borrow        study the reference kernel and adapt its technique (after a witness exists)
  bind          bind to the production library/format rather than re-implement it
  stay-minimal  fak's value is the bit-exact contract, not beating it at throughput
`)
}

// sotaList prints every operation as a table (or JSON). The table is the
// discovery surface: which op, how fak relates to the SOTA, and what the SOTA is.
func sotaList(stdout, stderr io.Writer, argv []string) int {
	asJSON := false
	for _, a := range argv {
		switch a {
		case "--json", "-json":
			asJSON = true
		default:
			fmt.Fprintf(stderr, "fak sota list: unknown flag %q\n", a)
			return 2
		}
	}

	ops := sotamatrix.Operations()
	if asJSON {
		// NoEscape: the matrix carries UTF-8 (Δ, ≥) that should survive verbatim.
		_ = writeIndentedJSONNoEscape(stdout, ops)
		return 0
	}

	fmt.Fprintf(stdout, "%d kernel operations  -  read the reference before writing one from scratch.\n", len(ops))
	fmt.Fprintf(stdout, "Tip: `fak sota <slug>` for one op's full reference; `fak sota <file>` to look up by kernel path.\n\n")

	tw := tabwriter.NewWriter(stdout, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "SLUG\tTITLE\tROUTE\tSOTA")
	for _, o := range ops {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", o.Slug, o.Title, o.Route, truncate(o.SOTA, 48))
	}
	_ = tw.Flush()
	return 0
}

// sotaLookup resolves an argument to one or more operations. It first tries an
// exact slug, then falls back to matching the argument as a kernel FILE PATH
// against every row's FileGlobs. Nothing matching is exit 1 with the slug list.
func sotaLookup(stdout, stderr io.Writer, arg string) int {
	if op, ok := sotamatrix.BySlug(arg); ok {
		sotaDetail(stdout, op)
		return 0
	}

	matches := sotaMatchPath(arg)
	if len(matches) == 0 {
		fmt.Fprintf(stderr, "fak sota: %q is not a known op slug and matches no kernel file.\n\n", arg)
		fmt.Fprintln(stderr, "Available operations:")
		for _, o := range sotamatrix.Operations() {
			fmt.Fprintf(stderr, "  %s\n", o.Slug)
		}
		fmt.Fprintln(stderr, "\nRun `fak sota list` for the table, or `fak sota <slug>` for one op.")
		return 1
	}

	for i, op := range matches {
		if i > 0 {
			fmt.Fprintln(stdout)
		}
		sotaDetail(stdout, op)
	}
	return 0
}

// sotaMatchPath returns the operations whose FileGlobs match a repo-relative file
// path. Backslashes are normalized to forward slashes first (the matrix globs are
// forward-slash, agents on Windows type backslashes). A row matches if the path
// matches any glob via path.Match, OR if the path is under a glob's non-wildcard
// prefix directory (so internal/metalgemm/sub/foo.metal still finds the
// internal/metalgemm/* row, which a single-segment path.Match wildcard misses).
func sotaMatchPath(arg string) []sotamatrix.Op {
	p := strings.ReplaceAll(arg, "\\", "/")
	var out []sotamatrix.Op
	for _, op := range sotamatrix.Operations() {
		if sotaGlobsMatch(op.FileGlobs, p) {
			out = append(out, op)
		}
	}
	return out
}

func sotaGlobsMatch(globs []string, p string) bool {
	for _, g := range globs {
		if matchKernelGlob(p, g) {
			return true
		}
	}
	return false
}

// matchKernelGlob reports whether path p matches a sotamatrix FileGlob, with the
// SAME precise semantics as the PRIOR_ART gate's matcher (internal/hooks): only a
// glob whose LAST segment is a bare directory wildcard ("internal/metalgemm/*")
// is treated as a prefix match (so a deeper path under that dir still matches);
// every other glob — including a filename pattern like "internal/model/awq*.go" —
// is matched with path.Match exactly. This is the bug-fix for the earlier
// prefix-dir rule that rooted "awq*.go" at "internal/model/" and so matched every
// file in internal/model/ against the AWQ row.
func matchKernelGlob(p, glob string) bool {
	p = strings.ReplaceAll(p, "\\", "/")
	glob = strings.ReplaceAll(glob, "\\", "/")
	// A literal "dir/*" glob: accept any path under that prefix directory (the bare
	// trailing "*" is a directory wildcard, not a filename pattern).
	if strings.HasSuffix(glob, "/*") {
		return strings.HasPrefix(p, strings.TrimSuffix(glob, "*")) // prefix keeps trailing "/"
	}
	ok, err := path.Match(glob, p)
	return err == nil && ok
}

// sotaDetail prints the full reference for one operation  -  what an author reads
// before writing the kernel: where fak does it, the SOTA stack to study, the link
// to actually open, the route + its gloss, the verification oracle, and papers.
func sotaDetail(w io.Writer, op sotamatrix.Op) {
	fmt.Fprintf(w, "%s  [%s]\n", op.Title, op.Slug)
	fmt.Fprintf(w, "  fak path:    %s\n", op.FakPath)
	fmt.Fprintf(w, "  SOTA stack:  %s\n", op.SOTA)
	fmt.Fprintf(w, "  read:        %s\n", op.PrimaryLink)
	fmt.Fprintf(w, "  route:       %s  -  %s\n", op.Route, op.Note)
	fmt.Fprintf(w, "  oracle:      %s\n", op.Oracle)
	if len(op.Papers) > 0 {
		fmt.Fprintln(w, "  papers:")
		for _, pp := range op.Papers {
			fmt.Fprintf(w, "    - %s\n", pp)
		}
	}
}
