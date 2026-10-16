// Command negframescan is a throwaway harness (NOT shipped) that exercises internal/negframe
// against real repo prose while cmd/fak is wedged by unrelated in-progress breakage elsewhere in
// the tree. It prints the mechanical reframes and judgement nudges for the given paths (or the
// default steer-prose corpus). Delete after use.
package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/negframe"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

func run(root string, paths []string, stdout io.Writer) int {
	if len(paths) == 0 {
		paths = negframe.ResolveTargets(root)
		fmt.Fprintf(stdout, "# default corpus: %d file(s)\n", len(paths))
	}
	mech, judge := 0, 0
	for _, rel := range paths {
		text := scorecard.SafeRead(filepath.Join(root, filepath.FromSlash(rel)))
		d := negframe.ScoreDoc(rel, text)
		if d.Negatives() == 0 {
			continue
		}
		fmt.Fprintf(stdout, "\n== %s  (mechanical=%d judgement=%d, %d prose lines) ==\n", rel, d.Mechanical, d.Judgement, d.Sentences)
		for _, f := range d.Findings {
			if f.Mechanical() {
				mech++
				fmt.Fprintf(stdout, "  L%-4d [%s] MECH\n     - %s\n     + %s\n", f.Line, f.Category, f.Text, f.Suggest)
			} else {
				judge++
				fmt.Fprintf(stdout, "  L%-4d [%s] soft: %s\n     %s\n", f.Line, f.Category, f.Hint, f.Text)
			}
		}
	}
	fmt.Fprintf(stdout, "\n# TOTAL mechanical=%d judgement=%d across %d path(s)\n", mech, judge, len(paths))
	return 0
}

func main() {
	root, _ := os.Getwd()
	os.Exit(run(root, os.Args[1:], os.Stdout))
}
