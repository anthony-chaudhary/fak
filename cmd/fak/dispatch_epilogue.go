package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func dispatchEpilogueUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak dispatch epilogue <subcommand> [flags]")
	fmt.Fprintln(w, "Subcommands:")
	fmt.Fprintln(w, "  submit   submit an epilogue to the landing queue")
	fmt.Fprintln(w, "  list     list submitted epilogues")
	fmt.Fprintln(w, "  drain    drain pending epilogues sequentially into commits")
	fmt.Fprintln(w, "  status   show summary status of epilogues")
}

func runDispatchEpilogue(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		dispatchEpilogueUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "submit":
		return runDispatchEpilogueSubmit(stdout, stderr, argv[1:])
	case "list":
		return runDispatchEpilogueList(stdout, stderr, argv[1:])
	case "drain":
		return runDispatchEpilogueDrain(stdout, stderr, argv[1:])
	case "status":
		return runDispatchEpilogueStatus(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		dispatchEpilogueUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak dispatch epilogue: unknown subcommand %q (want submit, list, drain, status)\n", argv[0])
		dispatchEpilogueUsage(stderr)
		return 2
	}
}

func runDispatchEpilogueSubmit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch epilogue submit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	issue := fs.Int("issue", 0, "GitHub issue number")
	lane := fs.String("lane", "", "lane or worker domain")
	var paths pathList
	fs.Var(&paths, "path", "repo-relative path (repeatable); paths may also follow --")
	var msg messageList
	fs.Var(&msg, "m", "commit message paragraph (repeatable)")
	msgFile := fs.String("F", "", "read commit message from file ('-' = stdin)")
	runsDir := fs.String("runs-dir", "", "dispatch runs directory (default: .dispatch-runs)")
	baseSHA := fs.String("base", "", "base commit SHA")
	worktreeDir := fs.String("worktree-dir", "", "worker worktree directory")
	patchFile := fs.String("patch", "", "patch file containing diff ('-' = stdin)")
	asJSON := fs.Bool("json", false, "output JSON record")

	if !parseFlags(fs, argv) {
		return 2
	}
	paths = append(paths, fs.Args()...)

	message := strings.TrimSpace(msg.Joined())
	if *msgFile != "" {
		if *msgFile == "-" {
			b, err := io.ReadAll(os.Stdin)
			if err != nil {
				fmt.Fprintf(stderr, "fak dispatch epilogue submit: read -F stdin: %v\n", err)
				return 2
			}
			message = strings.TrimSpace(string(b))
		} else {
			b, err := os.ReadFile(*msgFile)
			if err != nil {
				fmt.Fprintf(stderr, "fak dispatch epilogue submit: read -F file: %v\n", err)
				return 2
			}
			message = strings.TrimSpace(string(b))
		}
	}
	if message == "" && *issue > 0 {
		message = fmt.Sprintf("resolve(#%d): update paths (fak %s)", *issue, firstNonEmpty(*lane, "lane"))
	}

	var patchContent string
	if *patchFile != "" {
		if *patchFile == "-" {
			b, err := io.ReadAll(os.Stdin)
			if err != nil {
				fmt.Fprintf(stderr, "fak dispatch epilogue submit: read patch stdin: %v\n", err)
				return 2
			}
			patchContent = string(b)
		} else {
			b, err := os.ReadFile(*patchFile)
			if err != nil {
				fmt.Fprintf(stderr, "fak dispatch epilogue submit: read patch file: %v\n", err)
				return 2
			}
			patchContent = string(b)
		}
	} else if len(paths) > 0 {
		dir := *worktreeDir
		if dir == "" {
			dir = "."
		}
		cmdArgs := append([]string{"diff", "HEAD", "--"}, paths...)
		cmd := exec.Command("git", cmdArgs...)
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if err == nil && len(out) > 0 {
			patchContent = string(out)
		} else {
			cmdCached := exec.Command("git", append([]string{"diff", "--cached", "--"}, paths...)...)
			cmdCached.Dir = dir
			cachedOut, _ := cmdCached.CombinedOutput()
			if len(cachedOut) > 0 {
				patchContent = string(cachedOut)
			}
		}
	}

	actualRunsDir := *runsDir
	if actualRunsDir == "" {
		actualRunsDir = filepath.Join(".", dispatchtick.RunsDirName)
	}

	rec, err := dispatchtick.SubmitEpilogue(actualRunsDir, dispatchtick.EpilogueRecord{
		Issue:       *issue,
		Lane:        *lane,
		WorkerPID:   os.Getpid(),
		BaseSHA:     *baseSHA,
		Paths:       []string(paths),
		Message:     message,
		WorktreeDir: *worktreeDir,
		Patch:       patchContent,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch epilogue submit: %v\n", err)
		return 1
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rec); err != nil {
			fmt.Fprintf(stderr, "fak dispatch epilogue submit: %v\n", err)
			return 1
		}
		return 0
	}

	fmt.Fprintf(stdout, "epilogue %s submitted for issue #%d (%d path(s))\n", rec.ID, rec.Issue, len(rec.Paths))
	return 0
}

func runDispatchEpilogueList(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch epilogue list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	runsDir := fs.String("runs-dir", "", "dispatch runs directory (default: .dispatch-runs)")
	status := fs.String("status", "", "filter by status (pending, landing, landed, conflict, failed)")
	asJSON := fs.Bool("json", false, "output JSON array")

	if !parseFlags(fs, argv) {
		return 2
	}

	actualRunsDir := *runsDir
	if actualRunsDir == "" {
		actualRunsDir = filepath.Join(".", dispatchtick.RunsDirName)
	}

	recs, err := dispatchtick.ListEpilogues(actualRunsDir, dispatchtick.EpilogueStatus(*status))
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch epilogue list: %v\n", err)
		return 1
	}

	if *asJSON {
		if recs == nil {
			recs = []dispatchtick.EpilogueRecord{}
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(recs); err != nil {
			fmt.Fprintf(stderr, "fak dispatch epilogue list: %v\n", err)
			return 1
		}
		return 0
	}

	if len(recs) == 0 {
		fmt.Fprintln(stdout, "no epilogues found")
		return 0
	}

	w := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "ID\tSTATUS\tISSUE\tLANE\tPATHS\tSUBMITTED\tLANDED_SHA")
	for _, r := range recs {
		sha := r.LandedSHA
		if len(sha) > 10 {
			sha = sha[:10]
		}
		fmt.Fprintf(w, "%s\t%s\t#%d\t%s\t%d\t%s\t%s\n",
			r.ID, r.Status, r.Issue, r.Lane, len(r.Paths),
			r.SubmittedAt.Format("15:04:05"), sha)
	}
	_ = w.Flush()
	return 0
}

func runDispatchEpilogueDrain(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch epilogue drain", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "repo workspace root (default: current directory)")
	runsDir := fs.String("runs-dir", "", "dispatch runs directory (default: <workspace>/.dispatch-runs)")
	push := fs.Bool("push", false, "push landed commits through git push")
	limit := fs.Int("limit", 0, "maximum epilogues to drain in this run (0 = all)")
	dryRun := fs.Bool("dry-run", false, "simulate drain without applying or committing")
	asJSON := fs.Bool("json", false, "output result as JSON")

	if !parseFlags(fs, argv) {
		return 2
	}

	root := *workspace
	if root == "" {
		root = resolveRoot(".")
	}
	actualRunsDir := *runsDir
	if actualRunsDir == "" {
		actualRunsDir = filepath.Join(root, dispatchtick.RunsDirName)
	}

	res, err := dispatchtick.DrainEpilogues(root, actualRunsDir, dispatchtick.EpilogueDrainOptions{
		Limit:  *limit,
		DryRun: *dryRun,
		Push:   *push,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch epilogue drain: %v\n", err)
		return 1
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(res); err != nil {
			fmt.Fprintf(stderr, "fak dispatch epilogue drain: %v\n", err)
			return 1
		}
		return 0
	}

	fmt.Fprintf(stdout, "drained %d epilogue(s): %d landed, %d conflicted, %d failed (total %d pending)\n",
		res.Landed+res.Conflicted+res.Failed, res.Landed, res.Conflicted, res.Failed, res.Total)
	return 0
}

func runDispatchEpilogueStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch epilogue status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	runsDir := fs.String("runs-dir", "", "dispatch runs directory (default: .dispatch-runs)")
	asJSON := fs.Bool("json", false, "output JSON status summary")

	if !parseFlags(fs, argv) {
		return 2
	}

	actualRunsDir := *runsDir
	if actualRunsDir == "" {
		actualRunsDir = filepath.Join(".", dispatchtick.RunsDirName)
	}

	recs, err := dispatchtick.ListEpilogues(actualRunsDir, "")
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch epilogue status: %v\n", err)
		return 1
	}

	counts := map[string]int{
		"pending":  0,
		"landing":  0,
		"landed":   0,
		"conflict": 0,
		"failed":   0,
		"total":    len(recs),
	}
	for _, r := range recs {
		counts[string(r.Status)]++
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(counts); err != nil {
			fmt.Fprintf(stderr, "fak dispatch epilogue status: %v\n", err)
			return 1
		}
		return 0
	}

	fmt.Fprintf(stdout, "dispatch epilogues status (%d total):\n", counts["total"])
	fmt.Fprintf(stdout, "  pending:   %d\n", counts["pending"])
	fmt.Fprintf(stdout, "  landing:   %d\n", counts["landing"])
	fmt.Fprintf(stdout, "  landed:    %d\n", counts["landed"])
	fmt.Fprintf(stdout, "  conflict:  %d\n", counts["conflict"])
	fmt.Fprintf(stdout, "  failed:    %d\n", counts["failed"])
	return 0
}
