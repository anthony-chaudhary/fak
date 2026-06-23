package main

// fak codelint — the kernel's LANGUAGE-SERVER-PACK linter for CODE. It is to
// agent-written code what `fak lint` is to the kernel's own tool registry: a
// definition-/write-time check run at the kernel boundary instead of trusting the
// model's output. Point it at files or directories; each file is routed to the pack
// that owns its extension (Go/JSON in-process, Python/CUDA via their toolchains,
// degrading to no-opinion where a checker is absent). Exit 0 = clean, 1 = a hard
// (parse/compile) error finding, 2 = usage error — so it composes as a pipeline or
// CI gate. The same Lint the SWE-bench fleet runs on every agent file write.

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/codelint"
)

func cmdCodelint(argv []string) {
	os.Exit(runCodelint(os.Stdout, os.Stderr, argv))
}

// runCodelint is the testable core: it returns the process exit code instead of
// calling os.Exit, and takes its streams explicitly.
func runCodelint(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("codelint", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the findings as a JSON array")
	list := fs.Bool("list", false, "list the languages this build can lint, then exit")
	errorsOnly := fs.Bool("errors-only", false, "report only hard (parse/compile) errors, suppressing warnings")
	if err := fs.Parse(argv); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}

	reg := codelint.DefaultRegistry()
	if *list {
		fmt.Fprintf(stdout, "codelint can lint: %s\n", strings.Join(reg.Langs(), ", "))
		return 0
	}
	if fs.NArg() == 0 {
		fmt.Fprintln(stderr, "fak codelint: give one or more files or directories to lint (or --list)")
		return 2
	}

	files, err := expandCodeTargets(fs.Args())
	if err != nil {
		fmt.Fprintf(stderr, "fak codelint: %v\n", err)
		return 2
	}

	findings := []codelint.Finding{} // non-nil so --json renders [] (not null) on a clean run
	for _, f := range files {
		fs, err := reg.LintFile(context.Background(), f)
		if err != nil {
			fmt.Fprintf(stderr, "fak codelint: %s: %v\n", f, err)
			continue
		}
		for _, x := range fs {
			if *errorsOnly && x.Severity != codelint.Error {
				continue
			}
			findings = append(findings, x)
		}
	}

	if *asJSON {
		b, _ := json.MarshalIndent(findings, "", "  ")
		fmt.Fprintln(stdout, string(b))
	} else {
		writeCodelintHuman(stdout, len(files), findings)
	}
	if codelint.HasError(findings) {
		return 1
	}
	return 0
}

// expandCodeTargets resolves each arg to a list of files: a file is kept as-is; a
// directory is walked for files a pack owns (skipping VCS/vendor/hidden trees so a
// `fak codelint .` over a repo stays bounded and fast).
func expandCodeTargets(args []string) ([]string, error) {
	reg := codelint.DefaultRegistry()
	var out []string
	for _, a := range args {
		info, err := os.Stat(a)
		if err != nil {
			return nil, err
		}
		if !info.IsDir() {
			out = append(out, a)
			continue
		}
		err = filepath.WalkDir(a, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				if skipDirName(d.Name()) && p != a {
					return filepath.SkipDir
				}
				return nil
			}
			if _, ok := reg.PackFor(p); ok {
				out = append(out, p)
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func skipDirName(name string) bool {
	switch name {
	case ".git", "vendor", "node_modules", ".cache", "testdata":
		return true
	}
	return strings.HasPrefix(name, ".") && name != "."
}

func writeCodelintHuman(w io.Writer, nFiles int, findings []codelint.Finding) {
	if len(findings) == 0 {
		fmt.Fprintf(w, "codelint: %d file(s) checked, no findings\n", nFiles)
		return
	}
	errs := 0
	for _, f := range findings {
		if f.Severity == codelint.Error {
			errs++
		}
	}
	fmt.Fprintf(w, "codelint: %d file(s) checked, %d finding(s) (%d error, %d warning)\n",
		nFiles, len(findings), errs, len(findings)-errs)
	fmt.Fprintln(w, codelint.Summary(findings))
}
