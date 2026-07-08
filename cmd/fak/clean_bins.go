package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

// cmdCleanBins — `fak clean-bins`: the safe, idempotent, witnessed prune of the stray
// go-build binaries that `go build ./cmd/<name>` drops at the module root. Each such
// binary is a regenerable artifact (`.gitignore` matches `*.exe`, `*.exe~`, and the
// anchored bare-binary names like `/batchbench`) — they accumulate across bench/demo
// builds (hundreds of MB) and never get swept, so this verb is the loop that sweeps them.
//
// It is the build-artifact twin of `fak git-maint`: a housekeeping verb a dev-loop or
// CI runs to keep the tree from bloating. It NEVER deletes anything git does not ignore
// (the gitignore gate is the safety net — a tracked file is skipped even if its name
// looks like a build target), and it protects the live `fak` binary at the root unless
// --all is passed. Default is APPLY (the point of the verb is to clean); --dry-run lists
// what WOULD be removed and mutates nothing.
//
//	fak clean-bins            remove stray root-level build binaries (keep the live fak)
//	fak clean-bins --dry-run  list what would be removed; delete nothing
//	fak clean-bins --all      also remove the live fak binary at the root
//	fak clean-bins --json     machine-readable witness (schema fak-clean-bins/1)
func cmdCleanBins(argv []string) {
	fs := flag.NewFlagSet("clean-bins", flag.ExitOnError)
	verbFlagUsage(fs, "clean-bins")
	dryRun := fs.Bool("dry-run", false, "list what would be removed and mutate nothing")
	all := fs.Bool("all", false, "also remove the live fak binary at the module root")
	asJSON := fs.Bool("json", false, "emit a machine-readable result (schema fak-clean-bins/1)")
	root := fs.String("root", "", "module root to sweep (default: discover the git repo root from cwd)")
	_ = fs.Parse(argv)

	repoRoot := strings.TrimSpace(*root)
	if repoRoot == "" {
		repoRoot = discoverRepoRoot()
	}
	if repoRoot == "" {
		fmt.Fprintln(os.Stderr, "clean-bins: could not resolve a git repo root (pass --root)")
		os.Exit(2)
	}

	res := runCleanBins(cleanBinsOptions{
		Root:        repoRoot,
		Apply:       !*dryRun,
		IncludeLive: *all,
		CmdDirs:     cmdDirSet(repoRoot),
		IsIgnored:   func(name string) (bool, error) { return gitPathIgnored(repoRoot, name) },
	})

	if *asJSON {
		if err := renderCleanBinsJSON(os.Stdout, res); err != nil {
			fmt.Fprintf(os.Stderr, "clean-bins: encode json: %v\n", err)
			os.Exit(1)
		}
	} else {
		renderCleanBinsText(os.Stdout, res)
	}

	// Exit nonzero only on a GENUINE removal error (an I/O fault) — a locked / in-use
	// artifact is reported as a skip, not an error, so a housekeeping loop that runs
	// alongside live fak processes stays green.
	if len(res.Errors) > 0 {
		os.Exit(1)
	}
}

// cleanBinsOptions is the pure-data input to the testable core. IsIgnored is the
// gitignore gate (the production wiring shells to `git check-ignore`; tests inject a
// map lookup) — a candidate is only ever removed when it returns true, so the core can
// never touch a tracked file.
type cleanBinsOptions struct {
	Root        string
	Apply       bool                            // false = dry-run (report only)
	IncludeLive bool                            // also remove the live fak binary
	CmdDirs     map[string]bool                 // set of cmd/<name> dirs, for bare-name (Unix) artifacts
	IsIgnored   func(name string) (bool, error) // gitignore gate; required
	Remove      func(path string) error         // removal hook (nil -> os.Remove); injected in tests
}

type removedBin struct {
	Name  string `json:"name"`
	Bytes int64  `json:"bytes"`
}

// cleanBinsResult is the witness: what was removed (or would be, in dry-run), what was
// protected, and any per-file errors. Removed/Skipped are sorted for a stable report.
type cleanBinsResult struct {
	Root       string       `json:"root"`
	Apply      bool         `json:"apply"`
	Removed    []removedBin `json:"removed"`
	Skipped    []string     `json:"skipped,omitempty"`
	Errors     []string     `json:"errors,omitempty"`
	TotalBytes int64        `json:"total_bytes"`
}

// runCleanBins is the testable core. It enumerates the module root (non-recursive),
// selects the regular files that are go-build artifacts AND git-ignored, protects the
// live fak binary unless IncludeLive, and (when Apply) removes them — tallying bytes
// freed. It is idempotent: a second run over a swept tree finds nothing.
func runCleanBins(opts cleanBinsOptions) cleanBinsResult {
	res := cleanBinsResult{Root: opts.Root, Apply: opts.Apply}
	live := liveBinaryName()
	remove := opts.Remove
	if remove == nil {
		remove = os.Remove
	}

	entries, err := os.ReadDir(opts.Root)
	if err != nil {
		res.Errors = append(res.Errors, fmt.Sprintf("read dir %s: %v", opts.Root, err))
		return res
	}

	for _, e := range entries {
		if !e.Type().IsRegular() {
			continue
		}
		name := e.Name()
		if !isBuildArtifact(name, opts.CmdDirs) {
			continue
		}
		if name == live && !opts.IncludeLive {
			res.Skipped = append(res.Skipped, name+" (live binary; pass --all to remove)")
			continue
		}
		ignored, err := opts.IsIgnored(name)
		if err != nil {
			res.Errors = append(res.Errors, fmt.Sprintf("%s: check-ignore: %v", name, err))
			continue
		}
		if !ignored {
			// Tracked (or un-ignored) file that merely looks like a build target — never touch it.
			res.Skipped = append(res.Skipped, name+" (not git-ignored; refusing to delete)")
			continue
		}

		var size int64
		if info, err := e.Info(); err == nil {
			size = info.Size()
		}
		if opts.Apply {
			if err := remove(filepath.Join(opts.Root, name)); err != nil {
				// A locked / in-use artifact (a memory-mapped .exe on Windows — a running
				// process's own image, e.g. the .exe~ a self-rotate leaves behind) is not a
				// failure: it is regenerable and a later run reclaims it once the holder
				// exits. Report it as a skip so a housekeeping loop stays green; only a
				// genuine I/O error is an incident worth a nonzero exit.
				if os.IsPermission(err) {
					res.Skipped = append(res.Skipped, fmt.Sprintf("%s (in use / locked; will retry next run)", name))
				} else {
					res.Errors = append(res.Errors, fmt.Sprintf("%s: remove: %v", name, err))
				}
				continue
			}
		}
		res.Removed = append(res.Removed, removedBin{Name: name, Bytes: size})
		res.TotalBytes += size
	}

	sort.Slice(res.Removed, func(i, j int) bool { return res.Removed[i].Name < res.Removed[j].Name })
	sort.Strings(res.Skipped)
	return res
}

// isBuildArtifact reports whether a root-level file name is a go-build artifact:
//   - any *.exe or *.exe~ (Windows build output / the ~ scratch left when a mapped binary
//     can't be atomically replaced), or
//   - a bare (extension-less) name that matches a cmd/<name> directory — the Unix binary
//     `go build ./cmd/<name>` drops at the root.
//
// The gitignore gate in runCleanBins is the actual safety net; this is the intent filter.
func isBuildArtifact(name string, cmdDirs map[string]bool) bool {
	switch {
	case strings.HasSuffix(name, ".exe~"):
		return true
	case strings.HasSuffix(name, ".exe"):
		return true
	case filepath.Ext(name) == "":
		return cmdDirs[name]
	default:
		return false
	}
}

// liveBinaryName is the product binary this verb protects by default: `fak.exe` on
// Windows, `fak` elsewhere.
func liveBinaryName() string {
	if runtime.GOOS == "windows" {
		return "fak.exe"
	}
	return "fak"
}

// cmdDirSet reads cmd/ and returns the set of subdirectory names, so a bare root-level
// binary can be recognized as the output of `go build ./cmd/<name>`.
func cmdDirSet(root string) map[string]bool {
	set := map[string]bool{}
	entries, err := os.ReadDir(filepath.Join(root, "cmd"))
	if err != nil {
		return set
	}
	for _, e := range entries {
		if e.IsDir() {
			set[e.Name()] = true
		}
	}
	return set
}

// gitPathIgnored reports whether root/name is ignored by git. `git check-ignore -q`
// exits 0 when the path is ignored, 1 when it is not, and >1 on a real error.
func gitPathIgnored(root, name string) (bool, error) {
	_, code, err := gitRunner(context.Background(), root, "check-ignore", "-q", "--", name)
	if err != nil {
		return false, err
	}
	switch code {
	case 0:
		return true, nil
	case 1:
		return false, nil
	default:
		return false, fmt.Errorf("git check-ignore exited %d", code)
	}
}

func renderCleanBinsJSON(w io.Writer, res cleanBinsResult) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(struct {
		Schema string `json:"schema"`
		cleanBinsResult
	}{Schema: "fak-clean-bins/1", cleanBinsResult: res})
}

func renderCleanBinsText(w io.Writer, res cleanBinsResult) {
	mode := "apply"
	if !res.Apply {
		mode = "dry-run"
	}
	verb := "removed"
	if !res.Apply {
		verb = "would remove"
	}
	fmt.Fprintf(w, "clean-bins (%s) — %s\n", mode, res.Root)

	if len(res.Removed) == 0 {
		fmt.Fprintln(w, "no stray build binaries found — tree is clean")
	} else {
		for _, r := range res.Removed {
			fmt.Fprintf(w, "  - %s: %s (%s)\n", verb, r.Name, humanBytes(r.Bytes))
		}
	}
	for _, s := range res.Skipped {
		fmt.Fprintf(w, "  · skipped %s\n", s)
	}
	for _, e := range res.Errors {
		fmt.Fprintf(w, "  ! error %s\n", e)
	}
	fmt.Fprintf(w, "%s %d binary(ies), %s freed\n", verb, len(res.Removed), humanBytes(res.TotalBytes))
}
