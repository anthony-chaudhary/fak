package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/branchrole"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

const recoverSchema = "fak.recover.v1"

type recoveryStep struct {
	Argv    []string `json:"argv"`
	Summary string   `json:"summary"`
	Safe    bool     `json:"safe"`
}

type recoveryPlan struct {
	Reason     string         `json:"reason"`
	Summary    string         `json:"summary"`
	Executable bool           `json:"executable"`
	Steps      []recoveryStep `json:"steps"`
	Notes      []string       `json:"notes,omitempty"`
}

type recoveryResult struct {
	Schema  string          `json:"schema"`
	Reason  string          `json:"reason"`
	Mode    string          `json:"mode"`
	Plan    recoveryPlan    `json:"plan"`
	Results []stepRunResult `json:"results,omitempty"`
}

type stepRunResult struct {
	Argv     []string `json:"argv"`
	ExitCode int      `json:"exit_code"`
}

var recoverRunStep = runRecoverStep

func cmdRecover(argv []string) {
	os.Exit(runRecover(os.Stdout, os.Stderr, argv))
}

func runRecover(stdout, stderr io.Writer, argv []string) int {
	reasonArg := ""
	if len(argv) > 0 && !strings.HasPrefix(argv[0], "-") {
		reasonArg = argv[0]
		argv = argv[1:]
	}
	fs := flag.NewFlagSet("recover", flag.ContinueOnError)
	fs.SetOutput(stderr)
	execute := fs.Bool("execute", false, "run the safe recovery commands (default: dry-run)")
	dryRun := fs.Bool("dry-run", false, "print the recovery commands without running them (default)")
	asJSON := fs.Bool("json", false, "emit JSON")
	list := fs.Bool("list", false, "list known recovery reasons")
	dir := fs.String("dir", ".", "repo directory")
	trunk := fs.String("trunk", "", "configured trunk/development branch override")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	*dir = pathutil.ExpandTilde(*dir)
	if *execute && *dryRun {
		fmt.Fprintln(stderr, "fak recover: choose either --execute or --dry-run, not both")
		return 2
	}
	if strings.TrimSpace(*trunk) == "" {
		roles, err := branchrole.Load(*dir)
		if err == nil {
			*trunk = roles.DevelopmentBranch
		}
	}
	if strings.TrimSpace(*trunk) == "" {
		*trunk = "main"
	}
	plans := recoveryPlans(*trunk)
	if *list {
		return renderRecoverList(stdout, plans, *asJSON)
	}
	if reasonArg == "" {
		if fs.NArg() == 1 {
			reasonArg = fs.Arg(0)
		} else {
			fmt.Fprintln(stderr, "usage: fak recover <REASON> [--dry-run|--execute] [--json]")
			return 2
		}
	} else if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak recover <REASON> [--dry-run|--execute] [--json]")
		return 2
	}
	token := normalizeRecoveryReason(reasonArg)
	plan, ok := plans[token]
	if !ok {
		fmt.Fprintf(stderr, "fak recover: unknown recovery reason %q (run `fak recover --list`)\n", reasonArg)
		return 2
	}
	mode := "dry-run"
	if *execute {
		mode = "execute"
	}
	result := recoveryResult{Schema: recoverSchema, Reason: token, Mode: mode, Plan: plan}
	if !*execute {
		if *asJSON {
			return encodeJSONOrFail(stdout, stderr, result, "fak recover")
		}
		renderRecoveryPlan(stdout, plan, false)
		return 0
	}
	if !plan.Executable {
		if *asJSON {
			_ = encodeJSONOrFail(stdout, stderr, result, "fak recover")
		}
		fmt.Fprintf(stderr, "fak recover: %s has no safe executable recovery; use the dry-run notes\n", token)
		return 3
	}
	for _, step := range plan.Steps {
		if !step.Safe {
			continue
		}
		fmt.Fprintf(stdout, "+ %s\n", shellJoin(step.Argv))
		code := recoverRunStep(*dir, step.Argv, stdout, stderr)
		result.Results = append(result.Results, stepRunResult{Argv: step.Argv, ExitCode: code})
		if code != 0 {
			if *asJSON {
				_ = encodeJSONOrFail(stdout, stderr, result, "fak recover")
			}
			return code
		}
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, result, "fak recover")
	}
	for _, note := range plan.Notes {
		fmt.Fprintf(stdout, "note: %s\n", note)
	}
	return 0
}

func recoveryPlans(trunk string) map[string]recoveryPlan {
	originTrunk := "origin/" + trunk
	return map[string]recoveryPlan{
		"OFF_TRUNK": {
			Reason:     "OFF_TRUNK",
			Summary:    "reconcile the configured trunk in place; do not open a branch or worktree",
			Executable: true,
			Steps: []recoveryStep{
				{Argv: []string{"git", "fetch", "origin", trunk}, Summary: "refresh the configured trunk ref", Safe: true},
				{Argv: []string{"git", "merge", "--no-edit", originTrunk}, Summary: "merge the trunk tip into this checkout in place", Safe: true},
			},
			Notes: []string{"resolve conflicts in place if merge stops; never force-push"},
		},
		"STALE_BASE_DELETION": {
			Reason:     "STALE_BASE_DELETION",
			Summary:    "refresh and merge the trunk so path-scoped commit sees peer-added blocks",
			Executable: true,
			Steps: []recoveryStep{
				{Argv: []string{"git", "fetch", "origin", trunk}, Summary: "refresh the configured trunk ref", Safe: true},
				{Argv: []string{"git", "merge", "--no-edit", originTrunk}, Summary: "merge the trunk tip before retrying the path commit", Safe: true},
			},
			Notes: []string{"retry the original path-scoped commit after the merge is clean"},
		},
		"MERGE_IN_PROGRESS": {
			Reason:     "MERGE_IN_PROGRESS",
			Summary:    "drop your staged paths and wait unless this is your merge to finish",
			Executable: true,
			Steps: []recoveryStep{
				{Argv: []string{"git", "restore", "--staged"}, Summary: "unstage your pending pathspec so the peer merge can finish", Safe: true},
			},
			Notes: []string{"if MERGE_HEAD is yours, finish it promptly; if it is a peer's, wait for MERGE_HEAD to clear"},
		},
		"STALE_RECALL": {
			Reason:     "STALE_RECALL",
			Summary:    "refresh the source witness and discard the stale recalled digest",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"dos", "status"}, Summary: "refresh live DOS status from the source witness"},
				{Argv: []string{"dos", "commit-audit", "HEAD"}, Summary: "refresh git ancestry evidence for the current tip"},
			},
			Notes: []string{"replace recalled memory with the fresh witness before retrying"},
		},
		"COLLISION_RISK": {
			Reason:     "COLLISION_RISK",
			Summary:    "wait for the live lease or choose a disjoint lane/region",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"dos", "top"}, Summary: "inspect live leases and workers"},
				{Argv: []string{"dos", "arbitrate"}, Summary: "retry arbitration with a disjoint region"},
			},
			Notes: []string{"do not bypass the lease; repartition or wait"},
		},
		"OUT_OF_TREE_WRITE": {
			Reason:     "OUT_OF_TREE_WRITE",
			Summary:    "rerun inside the workspace or use an explicit temp directory",
			Executable: false,
			Notes:      []string{"rewrite the command target so it stays under the repo root; never use ../sibling repos for scratch"},
		},
		"PUBLIC_LEAK": {
			Reason:     "PUBLIC_LEAK",
			Summary:    "redact the staged leak needle before committing",
			Executable: false,
			Notes:      []string{"remove or redact the secret/private needle; use the one-shot override only for intentional adversarial fixtures"},
		},
		"FILE_ADMISSION": {
			Reason:     "FILE_ADMISSION",
			Summary:    "remove private-only, loose operational, generated, or oversized artifacts from the staged set",
			Executable: false,
			Notes:      []string{"move private-only material to fak-private or mark one-off ops notes operator-private"},
		},
	}
}

func normalizeRecoveryReason(s string) string {
	s = strings.TrimSpace(s)
	s = strings.ReplaceAll(s, "-", "_")
	s = strings.ReplaceAll(s, " ", "_")
	return strings.ToUpper(s)
}

func renderRecoverList(w io.Writer, plans map[string]recoveryPlan, asJSON bool) int {
	keys := make([]string, 0, len(plans))
	for k := range plans {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	if asJSON {
		out := make([]recoveryPlan, 0, len(keys))
		for _, k := range keys {
			out = append(out, plans[k])
		}
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		if err := enc.Encode(out); err != nil {
			return 1
		}
		return 0
	}
	for _, k := range keys {
		p := plans[k]
		exec := "manual"
		if p.Executable {
			exec = "executable"
		}
		fmt.Fprintf(w, "%-24s %-10s %s\n", p.Reason, exec, p.Summary)
	}
	return 0
}

func renderRecoveryPlan(w io.Writer, plan recoveryPlan, execute bool) {
	mode := "dry-run"
	if execute {
		mode = "execute"
	}
	fmt.Fprintf(w, "recover %s (%s)\n", plan.Reason, mode)
	fmt.Fprintf(w, "reason: %s\n", plan.Summary)
	if len(plan.Steps) > 0 {
		fmt.Fprintln(w, "commands:")
		for _, step := range plan.Steps {
			fmt.Fprintf(w, "  %s\n", shellJoin(step.Argv))
			if step.Summary != "" {
				fmt.Fprintf(w, "    # %s\n", step.Summary)
			}
		}
	}
	for _, note := range plan.Notes {
		fmt.Fprintf(w, "note: %s\n", note)
	}
}

func runRecoverStep(dir string, argv []string, stdout, stderr io.Writer) int {
	if len(argv) == 0 {
		return 0
	}
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Dir = dir
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode()
		}
		fmt.Fprintf(stderr, "fak recover: %v\n", err)
		return 1
	}
	return 0
}

func shellJoin(argv []string) string {
	parts := make([]string, len(argv))
	for i, arg := range argv {
		parts[i] = shellQuote(arg)
	}
	return strings.Join(parts, " ")
}

func shellQuote(arg string) string {
	if arg == "" || strings.ContainsAny(arg, " \t\n\"'") {
		return strconv.Quote(arg)
	}
	return arg
}
