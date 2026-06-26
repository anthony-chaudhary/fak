package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

// commitFn is the seam the CLI shim calls; it defaults to the real safecommit.Commit and
// is overridden in tests so runCommit is exercised without a real git or repo.
var commitFn = safecommit.Commit

func cmdCommit(argv []string) { os.Exit(runCommit(os.Stdout, os.Stderr, argv)) }

// pathList is a repeatable --path flag (the loopKVList shape): each --path appends one
// repo-relative pathspec.
type pathList []string

func (p *pathList) String() string { return strings.Join(*p, ",") }
func (p *pathList) Set(v string) error {
	v = strings.TrimSpace(v)
	if v == "" {
		return fmt.Errorf("empty --path")
	}
	*p = append(*p, v)
	return nil
}

// runCommit is the `fak commit` shim: it assembles a safecommit.Options from flags
// (message from -m / -F / stdin; paths from repeated --path AND/OR positionals after --),
// runs the safe-commit algorithm, and reports the structured Result. Exit codes mirror the
// loop verb's discipline: 0 success; 2 usage error; 3 a PRE-commit refusal (blocked, safe
// to retry/replan); 1 a POST-attempt failure (the commit ran but its result is bad — halt)
// or an infrastructure error.
func runCommit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("commit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var paths pathList
	fs.Var(&paths, "path", "a repo-relative path to commit (repeatable); paths may also be given after --")
	msg := fs.String("m", "", "commit message (mutually exclusive with -F)")
	msgFile := fs.String("F", "", "read the commit message from this file ('-' = stdin)")
	dir := fs.String("dir", "", "repo directory (default: discover from cwd)")
	trunk := fs.String("trunk", "", "expected trunk branch (default: main)")
	push := fs.Bool("push", false, "push after a VERIFIED commit (plain push, never --force)")
	noSignoff := fs.Bool("no-signoff", false, "do not add the DCO sign-off (-s is the default)")
	asJSON := fs.Bool("json", false, "emit the commit result as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	*dir = pathutil.ExpandTilde(*dir)
	paths = append(paths, fs.Args()...)
	if len(paths) == 0 {
		fmt.Fprintln(stderr, "fak commit: at least one --path (or a path after --) is required")
		return 2
	}

	message, code := assembleMessage(stdin(), *msg, *msgFile, stderr)
	if code != 0 {
		return code
	}

	res, err := commitFn(context.Background(), safecommit.Options{
		Dir:     *dir,
		Paths:   paths,
		Message: message,
		Trunk:   *trunk,
		SignOff: !*noSignoff,
		Push:    *push,
	})
	if err != nil {
		// Infrastructure failure (git not executable, lock unopenable): not a refusal.
		fmt.Fprintf(stderr, "fak commit: %v\n", err)
		return 1
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if encErr := enc.Encode(res); encErr != nil {
			fmt.Fprintf(stderr, "fak commit: %v\n", encErr)
			return 1
		}
	} else {
		renderCommitResult(stdout, res)
	}
	return commitExitCode(res)
}

// stdin is overridable in tests; defaults to os.Stdin.
var stdin = func() io.Reader { return os.Stdin }

// assembleMessage resolves the commit message from exactly one source: -m, -F <file>
// (or -F - for stdin). Returns (message, 0) on success or ("", exitCode) on a usage error.
func assembleMessage(in io.Reader, m, file string, stderr io.Writer) (string, int) {
	m = strings.TrimSpace(m)
	file = strings.TrimSpace(file)
	switch {
	case m != "" && file != "":
		fmt.Fprintln(stderr, "fak commit: -m and -F are mutually exclusive")
		return "", 2
	case m != "":
		return m, 0
	case file == "-":
		b, err := io.ReadAll(in)
		if err != nil {
			fmt.Fprintf(stderr, "fak commit: read message from stdin: %v\n", err)
			return "", 2
		}
		return string(b), 0
	case file != "":
		b, err := os.ReadFile(file)
		if err != nil {
			fmt.Fprintf(stderr, "fak commit: read message file: %v\n", err)
			return "", 2
		}
		return string(b), 0
	default:
		fmt.Fprintln(stderr, "fak commit: a message is required (-m STR, or -F FILE/-)")
		return "", 2
	}
}

// commitExitCode maps a Result to the process exit code. PRE-commit refusals are exit 3
// ("blocked — retry or replan"); a commit that ran but produced a bad result (race, push
// rejection, hook refusal) is exit 1 ("ran, result is bad — halt").
func commitExitCode(res safecommit.Result) int {
	switch res.Reason {
	case "":
		return 0
	case safecommit.ReasonNoPath, safecommit.ReasonEmptyMessage:
		return 2
	case safecommit.ReasonNotARepo, safecommit.ReasonOffTrunk,
		safecommit.ReasonMergeInProgress, safecommit.ReasonNothingStaged,
		safecommit.ReasonLockBusy, safecommit.ReasonWindowFull:
		return 3
	default: // PATHSPEC_RACE, HOOK_REFUSED, PUSH_REJECTED
		return 1
	}
}

func renderCommitResult(stdout io.Writer, res safecommit.Result) {
	if res.Reason == "" {
		fmt.Fprintf(stdout, "committed %s (%d path(s))%s\n", short(res.SHA), len(res.Paths), pushedSuffix(res))
		return
	}
	fmt.Fprintf(stdout, "%s", res.Reason)
	if res.Detail != "" {
		fmt.Fprintf(stdout, ": %s", res.Detail)
	}
	fmt.Fprintln(stdout)
	if len(res.RacedExtra) > 0 {
		fmt.Fprintf(stdout, "  raced extra paths: %s\n", strings.Join(res.RacedExtra, ", "))
		if res.SHA != "" {
			fmt.Fprintf(stdout, "  commit %s left intact for review (was %s)\n", short(res.SHA), short(res.HeadBefore))
		}
	}
}

func pushedSuffix(res safecommit.Result) string {
	if res.Pushed {
		return " and pushed"
	}
	return ""
}

func short(sha string) string {
	if len(sha) > 12 {
		return sha[:12]
	}
	return sha
}
