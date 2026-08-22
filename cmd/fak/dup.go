package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/clonescan"
	"github.com/anthony-chaudhary/fak/internal/tokencache"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// cmdDup is `fak dup` — the AUTHORING-TIME dedup query. Where the code-slop
// scorecard grades the whole tree a cycle after code lands (as debt), `fak dup
// query` inverts the same normalized-token clone engine into a forward question:
// given a candidate Go block, which tracked sites already hold a token-similar
// block? Run it BEFORE writing a new helper, so a clone is prevented instead of
// counted later. See docs/notes/DEDUP-EARLIER-AND-MORE-OFTEN-2026-07-03.md.
//
//	query --file F [--json] [--k N]   — sites in the tracked tree similar to F's blocks
//	query --stdin [--json] [--k N]    — same, reading the candidate block from stdin
//	cache-maintain [--json]            — converge the shared token-cache retention bounds
func cmdDup(args []string) {
	if len(args) == 0 {
		dupUsage()
		os.Exit(2)
	}
	switch args[0] {
	case "query":
		cmdDupQuery(args[1:])
	case "guard":
		cmdDupGuard(args[1:])
	case "cache-maintain":
		os.Exit(runDupCacheMaintain(os.Stdout, os.Stderr, args[1:]))
	case "-h", "--help", "help":
		dupUsage()
	default:
		fmt.Fprintf(os.Stderr, "fak dup: unknown subcommand %q\n", args[0])
		dupUsage()
		os.Exit(2)
	}
}

func dupUsage() {
	fmt.Fprintln(os.Stderr, "usage: fak dup query --file <candidate.go> [--k 5] [--json]   (tracked sites similar to the candidate)")
	fmt.Fprintln(os.Stderr, "       fak dup query --stdin [--k 5] [--json]                  (read the candidate block from stdin)")
	fmt.Fprintln(os.Stderr, "       fak dup guard [--staged | --range A..B] [--gate N] [--json] (warn if added Go blocks clone a tracked site)")
	fmt.Fprintln(os.Stderr, "       fak dup cache-maintain [--repo DIR] [--max-bytes N] [--max-entries N] [--temp-grace DURATION] [--json]")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Ask \"does a block like this already exist?\" BEFORE writing it. Same normalized")
	fmt.Fprintln(os.Stderr, "token-window clone definition as the code-slop scorecard, run as a forward query.")
	fmt.Fprintln(os.Stderr, "guard is advisory unless --gate N is set; then it exits 1 when warned files exceed N.")
}

func runDupCacheMaintain(stdout, stderr io.Writer, args []string) int {
	defaults := tokencache.MaintenanceDefaults()
	fs := flag.NewFlagSet("dup cache-maintain", flag.ContinueOnError)
	fs.SetOutput(stderr)
	repo := fs.String("repo", ".", "repository whose git-common-dir owns the shared token cache")
	maxBytes := fs.Int64("max-bytes", defaults.MaxBytes, "immutable JSON byte ceiling")
	maxEntries := fs.Int("max-entries", defaults.MaxEntries, "immutable JSON entry-count ceiling")
	tempGrace := fs.Duration("temp-grace", defaults.TempGrace, "minimum age before an atomic-write temp is stale")
	asJSON := fs.Bool("json", false, "emit the maintenance receipt as JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || *maxBytes <= 0 || *maxEntries <= 0 || *tempGrace <= 0 {
		fmt.Fprintln(stderr, "fak dup cache-maintain: ceilings and temp grace must be positive")
		return 2
	}
	receipt := tokencache.Maintain(*repo, tokencache.MaintenanceOptions{
		MaxBytes:   *maxBytes,
		MaxEntries: *maxEntries,
		TempGrace:  *tempGrace,
	})
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(receipt); err != nil {
			fmt.Fprintf(stderr, "fak dup cache-maintain: encode receipt: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, "token cache maintenance")
		fmt.Fprintf(stdout, "before: %d bytes / %d entries\n", receipt.BeforeBytes, receipt.BeforeEntries)
		fmt.Fprintf(stdout, "after: %d bytes / %d entries\n", receipt.AfterBytes, receipt.AfterEntries)
		fmt.Fprintf(stdout, "removed: %d entries / %d bytes\n", receipt.RemovedEntries, receipt.RemovedBytes)
		fmt.Fprintf(stdout, "stale temps: %d removed / %d remain / %d bytes removed\n", receipt.StaleTempsRemoved, receipt.StaleTempsAfter, receipt.StaleTempBytesRemoved)
		fmt.Fprintf(stdout, "skipped locked files: %d\n", receipt.SkippedLockedFiles)
		fmt.Fprintf(stdout, "complete: %t\n", receipt.Complete)
		fmt.Fprintf(stdout, "verdict: %s\n", receipt.Verdict)
	}
	switch receipt.Verdict {
	case tokencache.VerdictError, tokencache.VerdictUnsafePath, tokencache.VerdictUnavailable:
		if receipt.Detail != "" {
			fmt.Fprintf(stderr, "fak dup cache-maintain: %s\n", receipt.Detail)
		}
		return 1
	default:
		return 0
	}
}

// cmdDupQuery answers the query against the git-tracked .go tree.
func cmdDupQuery(args []string) {
	fs := flag.NewFlagSet("dup query", flag.ExitOnError)
	file := fs.String("file", "", "candidate Go file to check against the tracked tree")
	stdin := fs.Bool("stdin", false, "read the candidate block from stdin instead of --file")
	k := fs.Int("k", 5, "how many matching sites to return (0 = all)")
	asJSON := fs.Bool("json", false, "emit matches as JSON")
	_ = fs.Parse(args)

	if (*file == "") == (!*stdin) {
		fmt.Fprintln(os.Stderr, "fak dup query: pass exactly one of --file F or --stdin")
		os.Exit(2)
	}

	var candidate string
	var selfPath string
	if *stdin {
		b, err := io.ReadAll(os.Stdin)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak dup query: read stdin: %v\n", err)
			os.Exit(1)
		}
		candidate = string(b)
	} else {
		b, err := os.ReadFile(*file)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak dup query: %v\n", err)
			os.Exit(1)
		}
		candidate = string(b)
		// If the candidate file is itself tracked, exclude its own path so it is
		// not reported as a duplicate of itself.
		selfPath = trackedRelPath(*file)
	}

	tree, err := trackedGoTree()
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak dup query: %v\n", err)
		os.Exit(1)
	}

	matches := clonescan.Query(candidate, tree, selfPath, *k)
	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(matches)
		return
	}
	if len(matches) == 0 {
		fmt.Println("no token-similar block found in the tracked tree — clear to write")
		return
	}
	fmt.Printf("%d tracked site(s) already hold a token-similar block (most overlap first):\n", len(matches))
	for _, m := range matches {
		fmt.Printf("  %-3d windows  %s:%d-%d\n", m.Windows, m.File, m.StartLine, m.EndLine)
	}
	fmt.Println("\nreview these before adding the block — a shared helper may already exist.")
}

// cmdDupGuard is the DURABLE, more-often half of the dedup query: instead of an
// author remembering to run `dup query`, the guard runs it automatically over the
// ADDED Go lines of a diff (staged, or an explicit range) and warns when a new
// block token-clones an existing tracked site. It is strictly ADVISORY — it prints
// warnings and ALWAYS exits 0, so a false positive on an idiom can never wedge the
// shared trunk. Wire it as an opt-in pre-commit hook; the trunk guard blocks, this
// one only nudges.
func cmdDupGuard(args []string) {
	fs := flag.NewFlagSet("dup guard", flag.ExitOnError)
	staged := fs.Bool("staged", false, "check the staged (git diff --cached) added Go lines")
	rng := fs.String("range", "", "check the added Go lines of a commit range, e.g. origin/main..HEAD")
	asJSON := fs.Bool("json", false, "emit warnings as JSON")
	gate := fs.Int("gate", -1, "exit 1 when warned file count exceeds N; omitted keeps advisory exit 0")
	_ = fs.Parse(args)

	if *gate < -1 {
		fmt.Fprintln(os.Stderr, "fak dup guard: --gate must be >= 0")
		os.Exit(2)
	}
	if (*staged) == (*rng != "") {
		fmt.Fprintln(os.Stderr, "fak dup guard: pass exactly one of --staged or --range A..B")
		os.Exit(2)
	}

	added, err := addedGoByFile(*staged, *rng)
	if err != nil {
		// A guard must never fail the caller; report and exit 0.
		fmt.Fprintf(os.Stderr, "fak dup guard: %v (advisory — skipping)\n", err)
		return
	}

	tree, err := trackedGoTree()
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak dup guard: %v (advisory — skipping)\n", err)
		return
	}

	// Tokenize the whole tracked tree ONCE, then query each added file against the
	// prebuilt index. Previously each added file re-tokenized all ~5.7k tracked
	// files, so the scan cost grew with (added files × tree size); building the
	// index once makes it (tree size + added files × cheap intersection).
	//
	// A persisted, fleet-shared WindowCache (#4330) memoizes each tracked file's
	// tokenization under its content hash, so the ~5.7k-file re-lex this verb pays on
	// every run collapses to a directory of lookups for the files that did not change.
	// A nil cache (FAK_TOKEN_CACHE=off, or common dir unresolvable) degrades to the
	// exact uncached path.
	index := clonescan.BuildTreeIndex(tree, tokencache.Open("."))

	type warning struct {
		AddedIn string            `json:"added_in"`
		Matches []clonescan.Match `json:"matches"`
	}
	addedFiles := make([]string, 0, len(added))
	for rel := range added {
		addedFiles = append(addedFiles, rel)
	}
	sort.Strings(addedFiles)
	var warnings []warning
	for _, rel := range addedFiles {
		// Exclude the file itself: a block appearing in both the added lines and the
		// committed file is the same code, not a duplicate.
		matches := index.Query(clonescan.CandidateKeys(added[rel]), rel, 5)
		if len(matches) > 0 {
			warnings = append(warnings, warning{AddedIn: rel, Matches: matches})
		}
	}

	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(warnings)
		if dupGuardGateExitCode(len(warnings), *gate) != 0 {
			os.Exit(1)
		}
		return
	}
	if len(warnings) == 0 {
		fmt.Println("dup guard: no added Go block clones an existing tracked site — clear.")
		return
	}
	fmt.Printf("dup guard (advisory): %d added file(s) hold a block that clones a tracked site:\n", len(warnings))
	for _, w := range warnings {
		fmt.Printf("  %s adds a block already at:\n", w.AddedIn)
		for _, m := range w.Matches {
			fmt.Printf("      %-3d windows  %s:%d-%d\n", m.Windows, m.File, m.StartLine, m.EndLine)
		}
	}
	fmt.Println("\na shared helper may already exist — this is advisory, the commit is not blocked.")
	if *gate >= 0 {
		fmt.Printf("dup guard gate: %d warned file(s), threshold %d.\n", len(warnings), *gate)
	}
	if dupGuardGateExitCode(len(warnings), *gate) != 0 {
		os.Exit(1)
	}
}

func dupGuardGateExitCode(warnedFiles, gate int) int {
	if gate >= 0 && warnedFiles > gate {
		return 1
	}
	return 0
}

// addedGoByFile returns, per changed .go file, the concatenated ADDED source lines
// of the diff (staged or a range). Only added lines (diff '+') are kept, so a small
// edit to an existing file yields only its new code as the candidate.
func addedGoByFile(staged bool, rng string) (map[string]string, error) {
	var cmd *exec.Cmd
	if staged {
		cmd = exec.Command("git", "diff", "--cached", "--unified=0", "--", "*.go")
	} else {
		cmd = exec.Command("git", "diff", "--unified=0", rng, "--", "*.go")
	}
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git diff: %w", err)
	}
	return parseAddedGo(string(out)), nil
}

// parseAddedGo extracts, per file, the concatenated added ('+') source lines from
// unified `git diff` text. Split out from the git invocation so the diff-parsing is
// pure and testable. File headers ('+++ b/PATH') switch the current file; '+++'
// itself is never treated as an added line.
func parseAddedGo(diff string) map[string]string {
	added := make(map[string]string)
	var cur string
	var sb strings.Builder
	flush := func() {
		if cur != "" && sb.Len() > 0 {
			added[cur] += sb.String()
		}
		sb.Reset()
	}
	for _, line := range strings.Split(diff, "\n") {
		if strings.HasPrefix(line, "+++ b/") {
			flush()
			cur = filepath.ToSlash(strings.TrimPrefix(line, "+++ b/"))
			continue
		}
		if strings.HasPrefix(line, "+") && !strings.HasPrefix(line, "+++") {
			sb.WriteString(strings.TrimPrefix(line, "+"))
			sb.WriteByte('\n')
		}
	}
	flush()
	return added
}

// trackedGoTree returns the git-tracked *.go files as rel-path -> source text.
func trackedGoTree() (map[string]string, error) {
	cmd := exec.Command("git", "ls-files", "*.go")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git ls-files: %w", err)
	}
	tree := make(map[string]string)
	for _, rel := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		rel = strings.TrimSpace(rel)
		if rel == "" {
			continue
		}
		b, err := os.ReadFile(rel)
		if err != nil {
			continue // a tracked-but-deleted file; skip
		}
		tree[filepath.ToSlash(rel)] = string(b)
	}
	return tree, nil
}

// trackedRelPath returns the slash-form repo-relative path of a file if it is
// inside the repo, else "" (an untracked candidate has no self to exclude).
func trackedRelPath(path string) string {
	abs, err := filepath.Abs(path)
	if err != nil {
		return ""
	}
	cmd := exec.Command("git", "rev-parse", "--show-toplevel")
	windowgate.ConfigureBackgroundCommand(cmd)
	root, err := cmd.Output()
	if err != nil {
		return ""
	}
	rel, err := filepath.Rel(strings.TrimSpace(string(root)), abs)
	if err != nil || strings.HasPrefix(rel, "..") {
		return ""
	}
	return filepath.ToSlash(rel)
}
