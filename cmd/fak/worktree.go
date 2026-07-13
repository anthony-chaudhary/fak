package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
	"github.com/anthony-chaudhary/fak/internal/worktreewitness"
)

// cmdWorktree — `fak worktree <sub>`: the guarded, on-trunk-safe worktree verbs.
// Today it hosts one sub, `witness`, with room to grow (a future `worktree doctor`
// could front tools/worktree_doctor.py the same way).
func cmdWorktree(argv []string) {
	if len(argv) == 0 {
		worktreeUsage()
		os.Exit(2)
	}
	switch argv[0] {
	case "witness":
		cmdWorktreeWitness(argv[1:])
	case "-h", "--help", "help":
		worktreeUsage()
	default:
		fmt.Fprintf(os.Stderr, "fak worktree: unknown subcommand %q\n", argv[0])
		worktreeUsage()
		os.Exit(2)
	}
}

func worktreeUsage() {
	fmt.Fprintln(os.Stderr, strings.TrimSpace(`
fak worktree <subcommand>

  witness -- <cmd...>   Run <cmd> in a transient detached worktree pinned at
                        origin/main, so the verdict reflects the TRUNK TIP and not
                        this checkout's dirty tree. Prints GREEN/RED + the witnessed
                        SHA, then self-reaps (archiving if the run dirtied the tree).
`))
}

// cmdWorktreeWitness is the peer-dirty-proof green check. The shared working tree
// is almost always dirty with some peer's uncommitted WIP here, so a whole-package
// `go test`/`go build` can red on a tree the trunk tip compiles clean — a FALSE
// red. This runs the check where a peer's edits cannot reach it: a fresh detached
// worktree at origin/main. It is the safe-worktree envelope made executable
// (transient + detached-origin/main + observe-only + self-reaping) — see AGENTS.md.
//
// Usage: fak worktree witness [flags] -- <cmd> [args...]
//
//	fak worktree witness -- go test ./cmd/fak
//	fak worktree witness --json -- go build ./...
func cmdWorktreeWitness(argv []string) {
	fs := flag.NewFlagSet("worktree witness", flag.ExitOnError)
	asJSON := fs.Bool("json", false, "emit a machine-readable witness result")
	ref := fs.String("ref", worktreewitness.DefaultRef, "ref to pin and detach at")
	fetch := fs.Bool("fetch", false, "git fetch the ref before resolving it (freshest trunk tip)")
	root := fs.String("root", "", "repo root (default: discover from cwd)")
	archiveDir := fs.String("archive-dir", "", "save a dirtied worktree's diff+untracked here before reap ('' disables archiving)")
	fs.Parse(argv)

	// The witnessed command is everything after `--`; flag.Parse stops at it and
	// leaves it in Args(). An empty command is a usage error, not a witness.
	cmd := fs.Args()
	if len(cmd) == 0 {
		fmt.Fprintln(os.Stderr, "fak worktree witness: no command to witness (usage: fak worktree witness [flags] -- <cmd> [args...])")
		os.Exit(2)
	}

	repoRoot := strings.TrimSpace(*root)
	if repoRoot == "" {
		repoRoot = discoverRepoRoot()
	}
	if repoRoot == "" {
		fmt.Fprintln(os.Stderr, "fak worktree witness: could not resolve a git repo root (pass --root)")
		os.Exit(2)
	}

	res, err := worktreewitness.Run(worktreewitness.Config{
		Repo:       repoRoot,
		Ref:        *ref,
		Fetch:      *fetch,
		Command:    cmd,
		ArchiveDir: strings.TrimSpace(*archiveDir),
	}, gitWitnessRunner, cmdWitnessRunner)
	if err != nil {
		// A harness failure (rev-parse/worktree-add/command-not-started) — distinct
		// from a red verdict. Surface it and exit 2 (harness), not 1 (red).
		if *asJSON {
			_ = json.NewEncoder(os.Stdout).Encode(witnessJSON(res, err))
		} else {
			fmt.Fprintf(os.Stderr, "fak worktree witness: %v\n", err)
		}
		os.Exit(2)
	}

	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(witnessJSON(res, nil))
	} else {
		renderWitnessText(res)
	}

	// The verdict IS the exit code: GREEN -> 0, RED -> 1, so the witness composes
	// in a shell (`fak worktree witness -- go test ./... && fak sync push`).
	if !res.Green {
		os.Exit(1)
	}
}

// witnessResultJSON is the machine-readable envelope. It embeds the core Result and
// adds a harness-error field so a JSON consumer can tell a red verdict (green:false,
// error:"") from a harness failure (error non-empty).
type witnessResultJSON struct {
	Schema                 string `json:"schema"`
	Error                  string `json:"error,omitempty"`
	worktreewitness.Result        // green, exit_code, ref, sha, short_sha, command, ...
}

func witnessJSON(res worktreewitness.Result, err error) witnessResultJSON {
	out := witnessResultJSON{Schema: "fak-worktree-witness/1", Result: res}
	if err != nil {
		out.Error = err.Error()
	}
	return out
}

func renderWitnessText(res worktreewitness.Result) {
	verdict := "RED"
	if res.Green {
		verdict = "GREEN"
	}
	fmt.Printf("%s  %s @ %s (exit %d)\n", verdict, res.Command, res.ShortSHA, res.ExitCode)
	if res.Archived != "" {
		fmt.Printf("  archived dirtied worktree -> %s\n", res.Archived)
	}
	if res.ReapNote != "" {
		fmt.Printf("  reap: %s\n", res.ReapNote)
	}
	// On a red verdict, echo the tail of the output so the caller sees WHY without
	// re-running — the whole point is not having to reproduce it on a clean tree.
	if !res.Green && strings.TrimSpace(res.Output) != "" {
		fmt.Fprintln(os.Stderr, "--- witnessed output (tail) ---")
		fmt.Fprintln(os.Stderr, tailLines(res.Output, 40))
	}
}

// tailLines returns the last n lines of s, for a compact red diagnostic.
func tailLines(s string, n int) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	if len(lines) <= n {
		return strings.Join(lines, "\n")
	}
	return "…\n" + strings.Join(lines[len(lines)-n:], "\n")
}

// gitWitnessRunner drives git for the witness. It matches worktreewitness.Runner:
// combined output + exit code, error only when git could not be STARTED (a non-zero
// git exit is reported via code, never err — the witness core decides what that
// means). GIT_OPTIONAL_LOCKS=0 keeps the burst-time reads off the shared index lock
// a peer may hold, same as gitRunner.
func gitWitnessRunner(dir, name string, args ...string) (string, int, error) {
	cmd := exec.Command(name, args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Env = append(os.Environ(), "GIT_OPTIONAL_LOCKS=0")
	if dir != "" {
		cmd.Dir = dir
	}
	out, err := cmd.CombinedOutput()
	if err == nil {
		return string(out), 0, nil
	}
	if ee, ok := err.(*exec.ExitError); ok {
		return string(out), ee.ExitCode(), nil
	}
	return string(out), -1, err
}

// cmdWitnessRunner runs the witnessed command in the worktree module dir. Same
// contract: a non-zero exit is the SIGNAL (code), not an error; err is set only when
// the command could not be started (binary missing), which the core turns into a
// harness failure rather than a clean red.
func cmdWitnessRunner(dir, name string, args ...string) (string, int, error) {
	cmd := exec.Command(name, args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	if dir != "" {
		cmd.Dir = dir
	}
	out, err := cmd.CombinedOutput()
	if err == nil {
		return string(out), 0, nil
	}
	if ee, ok := err.(*exec.ExitError); ok {
		return string(out), ee.ExitCode(), nil
	}
	return string(out), -1, err
}
