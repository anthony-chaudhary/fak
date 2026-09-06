package main

import (
	"encoding/json"
	"errors"
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

const (
	recoverSchema      = "fak.recover.v1"
	recoverExitRefusal = 3
)

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
	verbFlagUsage(fs, "recover")
	execute := fs.Bool("execute", false, "run the safe recovery commands (default: dry-run)")
	dryRun := fs.Bool("dry-run", false, "print the recovery commands without running them (default)")
	asJSON := fs.Bool("json", false, "emit JSON")
	list := fs.Bool("list", false, "list known recovery reasons")
	var set repeatedStringFlag
	fs.Var(&set, "set", "bind a catalog placeholder: NAME=VALUE substitutes <NAME> in the plan's commands and notes (repeatable). Config-class bails print the recovery pre-bound.")
	dir := fs.String("dir", ".", "repo directory")
	trunk := fs.String("trunk", "", "configured trunk/development branch override")
	if !parseFlags(fs, argv) {
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
	bindings, err := parseRecoveryBindings(set.Values())
	if err != nil {
		fmt.Fprintf(stderr, "fak recover: %v\n", err)
		return 2
	}
	plan = bindRecoveryPlan(plan, bindings)
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
		return recoverExitRefusal
	}
	// A placeholder is legible in a dry run and unrunnable in an execute: shelling
	// out a literal <path> passes an argument the operator never chose. Refuse and
	// name what to bind, which is the next step rather than a dead end.
	if missing := unboundRecoveryPlaceholders(plan); len(missing) > 0 {
		fmt.Fprintf(stderr, "fak recover: %s needs %s bound before --execute can run its commands\n", token, strings.Join(missing, ", "))
		for _, name := range missing {
			fmt.Fprintf(stderr, "  next: fak recover %s --execute --set %s=<value>\n", token, name)
		}
		return 2
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

// recoveryPlans is the whole closed recovery vocabulary: the TREE class below
// (guard/DOS refusals over a working tree) merged with the CONFIG class from
// recover_config.go (the bails whose cause is a flag, an env var, or a file).
// The two are kept in separate files because they read differently — a tree
// recovery runs complete commands, a config recovery carries placeholders the
// bail site binds — but they share one namespace so `fak recover <TOKEN>`
// resolves any reason fak emits, whichever half produced it.
// emittedRecoveryReasons is the refusal vocabulary recover must cover. Adding a
// refusal token without an actionable recovery must fail CI, not a live agent turn.
var emittedRecoveryReasons = []string{
	"BEHIND",
	"BEHIND_FASTFORWARDABLE",
	"BUDGET_RECEIPT_INCOMPLETE",
	"BUILD_CHECK_TIMEOUT",
	"COMMITTED_RED",
	"CONCEPT_ADMISSION",
	"CONCEPT_FRESHNESS",
	"DEFAULT_DENY",
	"DIRTY_WRITE_OVERLAP",
	"DISAMBIGUATION_TIMEOUT",
	"DIVERGED_DISJOINT",
	"DIVERGED_OVERLAP",
	"INTEGRITY_REFUTED",
	"ISSUE_NOT_DISPATCH_LEAF",
	"ISSUE_UNROUTED",
	"LEASE_HELD",
	"LEASE_OWNER_UNAVAILABLE",
	"LOCK_BUSY",
	"MALFORMED",
	"MERGE_ACTIVE_PEER_OWNED",
	"MISROUTE",
	"OVERSIZE",
	"PARENT_TOKEN_BUDGET_EXCEEDED",
	"PARENT_WALL_DEADLINE_EXCEEDED",
	"PATHSPEC_RACE",
	"PII_EXFIL",
	"PII_REDACTED",
	"POLICY_BLOCK",
	"PROMPT_INJECTION",
	"QUEUED_AWAITING_QUIESCENCE",
	"RATE_LIMITED",
	"REQUIRE_WITNESS",
	"RESULT_SECRET_DISCOVERED",
	"SCOPE_CROSSING",
	"SECRET_EXFIL",
	"SECRET_REDACTED",
	"SELF_MODIFY",
	"SHELL_DIALECT",
	"SYSTEM_COMMIT_HEADROOM",
	"TAINT_EGRESS",
	"TARGET_MOVED",
	"TRUST_VIOLATION",
	"UNKNOWN_TOOL",
	"UNWITNESSED",
}

func recoveryPlans(trunk string) map[string]recoveryPlan {
	plans := treeRecoveryPlans(trunk)
	for token, plan := range configRecoveryPlans() {
		plans[token] = plan
	}
	root := guardFindReasonRoot()
	if root != "" {
		for token, doc := range guardReadReasonDocs(root) {
			if _, exists := plans[token]; !exists {
				notes := []string{}
				if strings.TrimSpace(doc.Fix) != "" {
					notes = append(notes, strings.TrimSpace(doc.Fix))
				}
				plans[token] = recoveryPlan{
					Reason:     token,
					Summary:    strings.TrimSpace(doc.Summary),
					Executable: false,
					Notes:      notes,
				}
			}
		}
	}
	return plans
}

func treeRecoveryPlans(trunk string) map[string]recoveryPlan {
	originTrunk := "origin/" + trunk
	return map[string]recoveryPlan{
		"BEHIND": {
			Reason:     "BEHIND",
			Summary:    "legacy sync divergence token; superseded by closed typed reasons",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"fak", "sync", "check", "--fetch", "--remote", "origin", "--branch", trunk}, Summary: "classify divergence into typed reasons (such as BEHIND_FASTFORWARDABLE)"},
			},
			Notes: []string{
				"BEHIND is a compatibility transition alias; use typed reasons like BEHIND_FASTFORWARDABLE or DIVERGED_DISJOINT",
				"run `fak sync check --fetch` to determine whether changes are fast-forwardable or diverged",
			},
		},
		"BEHIND_FASTFORWARDABLE": {
			Reason:     "BEHIND_FASTFORWARDABLE",
			Summary:    "the local branch is behind origin and can be cleanly fast-forwarded",
			Executable: true,
			Steps: []recoveryStep{
				{Argv: []string{"fak", "sync", "apply", "--fetch", "--remote", "origin", "--branch", trunk}, Summary: "fast-forward the local branch to the remote tracking ref", Safe: true},
			},
			Notes: []string{
				"fast-forward only runs when the working tree write set is clean",
				"retry push after the fast-forward is applied",
			},
		},
		"TARGET_MOVED": {
			Reason:     "TARGET_MOVED",
			Summary:    "the remote tracking ref moved concurrently during pre-push evaluation",
			Executable: true,
			Steps: []recoveryStep{
				{Argv: []string{"fak", "sync", "check", "--fetch", "--remote", "origin", "--branch", trunk}, Summary: "refresh and verify alignment against the moved target ref", Safe: true},
			},
			Notes: []string{
				"re-verify divergence status and retry the push once the target settles",
			},
		},
		"DIVERGED_DISJOINT": {
			Reason:     "DIVERGED_DISJOINT",
			Summary:    "the local branch and remote have diverged with disjoint changed file sets",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"fak", "sync", "check", "--fetch", "--remote", "origin", "--branch", trunk}, Summary: "preview the disjoint integration changes in dry-run mode"},
			},
			Notes: []string{
				"changes touch non-overlapping files; integrate in place with merge or rebase",
				"do not force-push; verify tests pass after integration",
			},
		},
		"DIVERGED_OVERLAP": {
			Reason:     "DIVERGED_OVERLAP",
			Summary:    "the local branch and remote have diverged with overlapping modified paths",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"git", "diff", originTrunk + "...HEAD"}, Summary: "inspect overlapping differences between local branch and remote"},
			},
			Notes: []string{
				"resolve content conflicts in place before committing and re-pushing",
				"never force-push, reset, or discard peer work",
			},
		},
		"MERGE_ACTIVE_PEER_OWNED": {
			Reason:     "MERGE_ACTIVE_PEER_OWNED",
			Summary:    "a peer-owned merge is currently in progress on the shared working tree",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"git", "status"}, Summary: "inspect in-progress merge state"},
			},
			Notes: []string{
				"unstage your pending paths with `git restore --staged` and wait for the peer merge to complete",
				"if MERGE_HEAD belongs to you, finish it; otherwise do not abort or commit a peer's merge",
			},
		},
		"DIRTY_WRITE_OVERLAP": {
			Reason:     "DIRTY_WRITE_OVERLAP",
			Summary:    "uncommitted local changes overlap with incoming remote changes",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"git", "status"}, Summary: "inspect dirty working tree changes"},
			},
			Notes: []string{
				"checkpoint finished local changes with `fak wip checkpoint` before syncing",
				"ensure working tree paths are clean or disjoint from remote changes",
			},
		},
		"QUEUED_AWAITING_QUIESCENCE": {
			Reason:     "QUEUED_AWAITING_QUIESCENCE",
			Summary:    "high concurrent trunk activity; operations queued awaiting repository quiescence",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"fak", "sync", "drain", "--remote", "origin", "--branch", trunk}, Summary: "drain the queue and flush changes when trunk activity quiesces"},
			},
			Notes: []string{
				"wait for concurrent workers to settle before retrying sync or push",
				"the drain command safely batches pending changes once a quiet window opens",
			},
		},
		"LEASE_OWNER_UNAVAILABLE": {
			Reason:     "LEASE_OWNER_UNAVAILABLE",
			Summary:    "the writer or lane lease owner is unresponsive or unreachable",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"dos", "top"}, Summary: "inspect active leases and worker status"},
			},
			Notes: []string{
				"verify whether the lease holder is still active or has terminated",
				"reclaim the lease only after confirming the holder is stale; do not bypass lease safety",
			},
		},
		"REQUIRE_WITNESS": {Reason: "REQUIRE_WITNESS", Summary: "the claimed effect has no independently inspectable witness", Notes: []string{"capture the failure-class witness (test, render, live read-back, or dos verify) and retry with that artifact; do not self-certify the effect"}},
		"SYSTEM_COMMIT_HEADROOM": {Reason: "SYSTEM_COMMIT_HEADROOM", Summary: "the host's operating-system commit reserve is at or below the managed-worker safety floor", Notes: []string{
			"let an in-flight managed worker finish, then rerun dispatch preflight so the host is measured again",
			"if the pressure persists, move the next worker to another sanctioned fleet node or add host commit capacity through the operator's normal OS change process",
			"do not lower FAK_SYSTEM_COMMIT_HEADROOM_MB, terminate unrelated processes, or add launch retries to route around the refusal",
		}},
		"PATHSPEC_RACE":     {Reason: "PATHSPEC_RACE", Summary: "a peer changed the index while the path-scoped commit was being sealed", Steps: []recoveryStep{{Argv: []string{"git", "show", "--stat", "--oneline", "HEAD"}, Summary: "inspect the intact commit and verify which paths landed"}}, Notes: []string{"do not amend or force-push; if an extra peer path landed, report the intact commit and let its owner reconcile it"}, Executable: true},
		"BUILD_CHECK_TIMEOUT": {
			Reason:     "BUILD_CHECK_TIMEOUT",
			Summary:    "prospective commit validation exceeded its observation deadline before completing",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"fak", "commit", "status"}, Summary: "inspect live committer ownership and active child validation process state before retrying"},
				{Argv: []string{"fak", "validate", "--mine", "<path>"}, Summary: "diagnose phase progress with bounded standalone validation; standalone success does not bypass or substitute for the commit gate"},
			},
			Notes: []string{
				"observation timeout vs terminal receipt: a timeout indicates the validation check deadline expired while in-flight, distinct from a terminal receipt where build/vet/test definitively failed red; the tree was never proven green",
				"inspect live owner and phase before retry: check whether the child validation process is still active or completed before re-running commit, avoiding duplicate competing validation runs",
				"permit a declared finite validation budget: if the validation workload requires more time, retry with an explicit bounded validation timeout budget once configured rather than unbounded attempts",
				"fail-closed: retain the refusal if prospective validation does not complete within the declared budget; never fail open or assume unverified code is safe",
				"do not bypass the gate: standalone validation output (`fak validate`) diagnoses phase health but is not automatically consumed or authorized by `fak commit`; the prospective committed tree must be verified by the commit gate",
			},
		},
		"COMMITTED_RED":     {Reason: "COMMITTED_RED", Summary: "the committed tip fails its isolated build or formatting gate", Steps: []recoveryStep{{Argv: []string{"fak-dev", "ci-preflight"}, Summary: "reproduce the committed-tip failure outside the peer-dirty tree"}}, Notes: []string{"fix the committed failure before attempting another commit or push"}, Executable: true},
		"LOCK_BUSY":         {Reason: "LOCK_BUSY", Summary: "another committer owns the serialized commit lock", Steps: []recoveryStep{{Argv: []string{"fak", "commit", "--reclaim-stale-commit-lock"}, Summary: "probe only the commit lock and reclaim only when its recorded owner is proven stale"}}, Notes: []string{"the actuator is a dry-run unless you explicitly add --apply; if the owner is live, wait and retry; never delete the lock by hand"}, Executable: true},
		"CONCEPT_ADMISSION": {Reason: "CONCEPT_ADMISSION", Summary: "a new concept-family identifier is absent from the staged concept corpus", Notes: []string{"add or reuse the concept-corpus row named by the refusal, then stage that evidence in the same commit as the identifier"}},
		"CONCEPT_FRESHNESS": {Reason: "CONCEPT_FRESHNESS", Summary: "the staged concept corpus is stale relative to the staged source tree", Steps: []recoveryStep{{Argv: []string{"fak", "concept", "generate", "--staged"}, Summary: "regenerate against the exact staged tree"}}, Notes: []string{"stage the generated corpus paths in the same commit; the gate intentionally evaluates the staged tree"}, Executable: true},
		"DISAMBIGUATION_TIMEOUT": {
			Reason:     "DISAMBIGUATION_TIMEOUT",
			Summary:    "the exact whole-tree disambiguation oracle exceeded its bounded pre-CAS deadline",
			Executable: false,
			Steps: []recoveryStep{{
				Argv:    []string{"fak", "worktree", "worker", "land", "<same-args>", "--disambiguation-timeout-ms", "<milliseconds>"},
				Summary: "rerun the same managed land once with an explicit bounded deadline",
			}},
			Notes: []string{
				"<milliseconds> must be an integer in the inclusive range 1..900000; the worker-land command passes it through the oracle's existing timeout resolver",
				"omitting the flag preserves the 120000 ms default",
				"the override changes only the one shared deadline around the same three witnesses; it does not skip, weaken, or retry any witness",
				"the timeout refusal happens before trunk CAS, so the managed worker and trunk/index remain unchanged",
			},
		},
		"BUDGET_RECEIPT_INCOMPLETE": {
			Reason:     "BUDGET_RECEIPT_INCOMPLETE",
			Summary:    "one or more child workers did not return provider-authoritative token usage",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"fak", "ultracode", "status"}, Summary: "inspect the ultracode run to verify child worker completion and usage authority"},
			},
			Notes: []string{
				"ensure all launched child workers finish cleanly and report provider-authoritative token usage before admitting the aggregate envelope receipt",
				"re-run with completed child receipts or inspect failed child workers",
			},
		},
		"PARENT_TOKEN_BUDGET_EXCEEDED": {
			Reason:     "PARENT_TOKEN_BUDGET_EXCEEDED",
			Summary:    "aggregate token consumption across child workers exceeded the parent budget envelope",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"fak", "ultracode", "status"}, Summary: "inspect child token allocation and consumption breakdown"},
			},
			Notes: []string{
				"increase the parent token budget envelope, reduce the child worker count, or constrain individual child context/prompt sizes",
				"do not tamper with child usage records to bypass the parent token envelope",
			},
		},
		"PARENT_WALL_DEADLINE_EXCEEDED": {
			Reason:     "PARENT_WALL_DEADLINE_EXCEEDED",
			Summary:    "ultracode run elapsed wall-clock time exceeded the parent deadline envelope",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"fak", "ultracode", "status"}, Summary: "inspect child worker durations and execution bottlenecks"},
			},
			Notes: []string{
				"extend the parent wall budget duration or optimize/parallelize slow child execution steps",
				"verify worker responsiveness and network latency across compute nodes",
			},
		},
		"ISSUE_NOT_DISPATCH_LEAF": {Reason: "ISSUE_NOT_DISPATCH_LEAF", Summary: "the selected issue is a parent, research item, or other non-leaf unit", Notes: []string{"decompose it into independently shippable child issues with done conditions, then dispatch one child leaf"}},
		"ISSUE_UNROUTED":          {Reason: "ISSUE_UNROUTED", Summary: "the issue lacks enough lane, path, or scope evidence for safe dispatch", Notes: []string{"add an explicit owned path/lane and bounded done condition to the issue, then rerun issue contract/dispatch planning"}},
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
		"STALE_UNTRACKED": {
			Reason:     "STALE_UNTRACKED",
			Summary:    "a requested path is untracked here but already on the trunk with different content",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"git", "fetch", "origin", trunk}, Summary: "refresh the trunk ref before comparing", Safe: true},
				{Argv: []string{"git", "show", originTrunk + ":<path>"}, Summary: "read the trunk copy content-to-content; git diff shows an untracked path as wholly deleted, so its line counts are the trunk file's own"},
			},
			Notes: []string{
				"merging while the path is untracked can stop on an overwrite refusal: move your copy aside, merge, then re-apply only the parts that are genuinely yours",
				"if the trunk copy is the one to keep, discard the local copy rather than committing it",
				"to supersede the trunk copy deliberately, having read it, re-run the commit with FAK_STALE_BASE_GUARD=warn",
			},
		},
		"FRESH_DELETION": {
			Reason:     "FRESH_DELETION",
			Summary:    "a staged commit deletes a recently-added path without naming it in the message",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"git", "restore", "--staged", "<path>"}, Summary: "unstage the suspicious deletion while preserving the working tree", Safe: true},
				{Argv: []string{"git", "restore", "--", "<path>"}, Summary: "restore the path if the deletion was collateral", Safe: true},
			},
			Notes: []string{"if the deletion is intentional, include the deleted path in the commit message or override once with FLEET_ALLOW_FRESH_DELETE=1"},
		},
		"MESSAGE_RACE": {
			Reason:     "MESSAGE_RACE",
			Summary:    "a commit landed with a different subject/body than the requested message",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"git", "show", "--stat", "HEAD"}, Summary: "inspect the unpushed mis-bound commit and keep it intact for review"},
				{Argv: []string{"git", "log", "-1", "--format=%B"}, Summary: "read the landed message before deciding the remediation path"},
			},
			Notes: []string{"do not push the mis-bound commit as verified; avoid raw git commit on the shared tree and use fak commit by explicit path"},
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
				{Argv: []string{"dos", "status", "<run-id>"}, Summary: "refresh live DOS status from the source witness"},
				{Argv: []string{"dos", "commit-audit", "HEAD"}, Summary: "refresh git ancestry evidence for the current tip"},
			},
			Notes: []string{"replace recalled memory with the fresh witness before retrying"},
		},
		"COLLISION_RISK": {
			Reason:     "COLLISION_RISK",
			Summary:    "checkpoint finished work, or wait for the live lease, or choose a disjoint lane/region",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"fak", "wip", "checkpoint"}, Summary: "park an already-finished delta durably so the lane can be released"},
				{Argv: []string{"dos", "top"}, Summary: "inspect live leases and workers"},
				{Argv: []string{"dos", "arbitrate"}, Summary: "retry arbitration with a disjoint region"},
			},
			Notes: []string{
				"do not bypass the lease; repartition, park, or wait",
				"the other two routes assume the work is not written yet: waiting leaves a finished delta dirty in a shared checkout where a peer's broad `git add` sweeps it (see `fak wip sweep-guard`), and an already-written change cannot be re-aimed at a disjoint lane",
				"if the change is already written and green, checkpoint it first, then land it with `fak wip land` once the lane frees",
			},
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
		"POLICY_BLOCK": {
			Reason:     "POLICY_BLOCK",
			Summary:    "an explicit policy rule or dangerous gotcha denied the tool call",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"fak", "preflight"}, Summary: "inspect recent capability floor denials and matching rules", Safe: true},
			},
			Notes: []string{
				"the command was blocked by a dangerous gotchas pattern, an explicit deny rule, or an arg predicate",
				"for autonomous agents: practice scoped fail-to-abstain — pivot to permitted tools, isolate safe non-gated subtasks (tests, diagnostics), commit partial verified deliverables, and record a structured ABSTAIN verdict with exact boundary specifications for operator escalation",
				"(Operator only — do not attempt from autonomous agent): if the operation is benign, run with --posture default_open or configure an operator overlay with `fak guard allow`",
			},
		},
		"TRUST_VIOLATION": {
			Reason:     "TRUST_VIOLATION",
			Summary:    "legacy umbrella refusal for capability, taint, context-MMU quarantine, or witness contradiction",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"fak", "preflight"}, Summary: "inspect the recent trust violation details and quarantine witness", Safe: true},
			},
			Notes: []string{
				"this legacy token covers four physical sub-cases: TAINT_EGRESS, SCOPE_CROSSING, PROMPT_INJECTION, and INTEGRITY_REFUTED",
				"for TAINT_EGRESS: sanitize data before dispatch or use internal IPC (e.g. send_input, a2achan) or supply override_reason",
				"for SCOPE_CROSSING: confine payloads to declared scope or residency boundaries",
				"for PROMPT_INJECTION: page out quarantined context or read sanitized stubs via context-MMU",
				"for INTEGRITY_REFUTED: ground claims in verified git/filesystem state rather than self-report",
			},
		},
		"DEFAULT_DENY": {
			Reason:     "DEFAULT_DENY",
			Summary:    "the proposed tool is not permitted by any loaded capability floor rule",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"fak", "preflight"}, Summary: "inspect recent capability floor denials", Safe: true},
			},
			Notes: []string{
				"the tool call failed closed under strict posture because it was not in the policy allowlist",
				"for autonomous agents: decompose the task into S0/S1 leaves, pivot to sanctioned tools, land verified partial progress, and emit a structured ABSTAIN record describing the required unpermitted tool",
				"(Operator only — do not attempt from autonomous agent): run with `--posture default_open` for permissive execution or configure an operator overlay via `fak guard allow`",
			},
		},
		"SELF_MODIFY": {
			Reason:     "SELF_MODIFY",
			Summary:    "the call attempted to modify guarded kernel code, ABI definitions, or workspace configuration",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"fak", "preflight"}, Summary: "inspect the blocked self-modification path and policy rule", Safe: true},
			},
			Notes: []string{
				"the target path reaches guarded files (internal/adjudicator, internal/kernel, internal/abi, dos.toml, or self_modify_globs)",
				"for autonomous agents: scope abstention strictly to the guarded boundary — land safe non-gated changes (tests, documentation, caller leaves), record a structured ABSTAIN verdict for the locked kernel/ABI path, and escalate to an operator",
				"target an unguarded path, or run in a detached worker worktree via `fak worktree worker prepare`",
				"(Operator only — do not attempt from autonomous agent): adjust policy rules via an explicit operator overlay",
			},
		},
		"LEASE_HELD": {
			Reason:     "LEASE_HELD",
			Summary:    "the requested file tree or lane is locked by another in-flight lease",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"dos", "top"}, Summary: "inspect active lane leases and holders", Safe: true},
			},
			Notes: []string{
				"wait for the conflicting lease to clear or pick a disjoint lane with `dos arbitrate`",
				"checkpoint finished progress with `fak wip checkpoint` while waiting",
			},
		},
		"MALFORMED": {
			Reason:     "MALFORMED",
			Summary:    "the tool call arguments or payload failed schema or grammar validation",
			Executable: false,
			Notes: []string{
				"inspect the tool schema and ensure all required arguments and types conform to expectations",
				"check argument JSON formatting and arity before re-invoking",
			},
		},
		"MISROUTE": {
			Reason:     "MISROUTE",
			Summary:    "the call used an inappropriate tool or argument shape for the intended action",
			Executable: false,
			Notes: []string{
				"consult tool documentation and choose the specialized tool designed for this action",
				"re-route file operations to specialized file tools rather than generic shell scripts",
			},
		},
		"RATE_LIMITED": {
			Reason:     "RATE_LIMITED",
			Summary:    "tool or provider invocation rate exceeded the configured throughput limit",
			Executable: false,
			Notes: []string{
				"back off and wait before retrying the operation",
				"adjust rate limit thresholds or concurrency limits if authorized",
			},
		},
		"SECRET_EXFIL": {
			Reason:     "SECRET_EXFIL",
			Summary:    "tool arguments or payload matched a secret or credential pattern destined for egress",
			Executable: false,
			Notes: []string{
				"redact API keys, tokens, or credentials from command arguments before transmission",
				"store secrets in environment variables or private keyrings rather than inline arguments",
			},
		},
		"UNWITNESSED": {
			Reason:     "UNWITNESSED",
			Summary:    "a required verification witness or proof artifact is missing",
			Executable: false,
			Notes: []string{
				"provide a corroborating test run, captured render, or commit witness before claiming completion",
				"run the required verification command and capture its exit receipt",
			},
		},
		"OVERSIZE": {
			Reason:     "OVERSIZE",
			Summary:    "the payload exceeded the context admission budget or token limit",
			Executable: false,
			Notes: []string{
				"reduce the request payload size or compact preceding session turns",
				"target smaller file chunks or specific slices rather than whole directory dumps",
			},
		},
		"UNKNOWN_TOOL": {
			Reason:     "UNKNOWN_TOOL",
			Summary:    "the proposed tool name is not registered in the tool catalog",
			Executable: false,
			Notes: []string{
				"verify the tool name against available tools registered in the current session",
				"check for typos or missing namespace prefixes in the tool name",
			},
		},
		"RESULT_SECRET_DISCOVERED": {
			Reason:     "RESULT_SECRET_DISCOVERED",
			Summary:    "a tool result contained sensitive credentials caught during discovery inspection",
			Executable: false,
			Notes: []string{
				"mask or scrub sensitive credentials before propagating tool results into LLM context",
				"configure redaction rules in secretgate policy to automatically mask matched patterns",
			},
		},
		"SECRET_REDACTED": {
			Reason:     "SECRET_REDACTED",
			Summary:    "a credential span in a tool result was masked in place by policy",
			Executable: false,
			Notes: []string{
				"the credential was safely redacted; non-sensitive output remains available in context",
				"proceed with workflow using the masked output, or supply explicit authorization if unredacted secret is required",
			},
		},
		"SHELL_DIALECT": {
			Reason:     "SHELL_DIALECT",
			Summary:    "a command was executed under an incompatible shell interpreter dialect",
			Executable: false,
			Notes: []string{
				"re-route PowerShell commands and cmdlets to the PowerShell execution tool",
				"use standard POSIX commands when targeting the Bash execution tool",
			},
		},
		"PII_REDACTED": {
			Reason:     "PII_REDACTED",
			Summary:    "personally identifiable information (PII) was masked in place by policy",
			Executable: false,
			Notes: []string{
				"the PII was redacted according to privacy policy while preserving surrounding text",
				"proceed using sanitized data; do not attempt to reconstruct masked identity fields",
			},
		},
		"PII_EXFIL": {
			Reason:     "PII_EXFIL",
			Summary:    "a tool call attempted to egress unredacted personally identifiable information (PII)",
			Executable: false,
			Notes: []string{
				"sanitize or mask PII fields before sending data to external destinations or sinks",
				"verify privacy compliance rules governing sensitive customer or user data",
			},
		},
		"TAINT_EGRESS": {
			Reason:     "TAINT_EGRESS",
			Summary:    "a tool call attempted external network egress or sensitive sink invocation carrying tainted data",
			Executable: false,
			Notes: []string{
				"sanitize or untaint data flow before dispatching to external network endpoints",
				"use internal IPC channels (e.g. send_input, a2a_send) for worker delegation within the workspace",
			},
		},
		"SCOPE_CROSSING": {
			Reason:     "SCOPE_CROSSING",
			Summary:    "a payload or message was routed outside its permitted isolation or residency boundary",
			Executable: false,
			Notes: []string{
				"confine payload routing to its declared scope boundary (agent, fleet, or tenant)",
				"ensure tenant-scoped data remains within local or designated compute engines",
			},
		},
		"PROMPT_INJECTION": {
			Reason:     "PROMPT_INJECTION",
			Summary:    "untrusted input or tool result matched prompt injection patterns",
			Executable: false,
			Notes: []string{
				"inspect untrusted content and strip jailbreak or instruction override markers",
				"quarantine the payload and use structured data extraction rather than raw instruction ingestion",
			},
		},
		"INTEGRITY_REFUTED": {
			Reason:     "INTEGRITY_REFUTED",
			Summary:    "an external witness or verification check refuted the claimed state",
			Executable: false,
			Notes: []string{
				"verify claims against ground-truth git history, test runs, or external reporters",
				"never fabricate evidence or self-report completion without verifiable artifacts",
			},
		},
	}
}

func normalizeRecoveryReason(s string) string {
	s = strings.TrimSpace(s)
	s = strings.ReplaceAll(s, "-", "_")
	s = strings.ReplaceAll(s, " ", "_")
	s = strings.ToUpper(s)
	if s == "BEHIND_FAST_FORWARDABLE" {
		return "BEHIND_FASTFORWARDABLE"
	}
	return s
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
	// Width the reason column to the widest token actually present rather than a
	// fixed 24: the config class added UNAUTHENTICATED_OFF_HOST_BIND at 29, whose
	// overflow shunted that row's mode and summary out of alignment with every
	// other row — in the one listing an operator reads to find their token.
	width := 0
	for _, k := range keys {
		if n := len(plans[k].Reason); n > width {
			width = n
		}
	}
	for _, k := range keys {
		p := plans[k]
		exec := "manual"
		if p.Executable {
			exec = "executable"
		}
		fmt.Fprintf(w, "%-*s %-10s %s\n", width, p.Reason, exec, p.Summary)
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
	cmd := exec.Command(recoveryArgv0(argv[0]), argv[1:]...)
	cmd.Dir = dir
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode()
		}
		if errors.Is(err, exec.ErrNotFound) {
			// A recovery that cannot find its own tool is the dead end this whole
			// path exists to remove, so it gets the same treatment as any other
			// config bail rather than a bare exec error.
			writeConfigBail(stderr, configBail{
				Verb:    "fak recover",
				Summary: fmt.Sprintf("the recovery step %q is not on PATH, so it could not be run", argv[0]),
				Knobs:   []bailKnob{bailEnv("PATH", "does not contain "+argv[0]).want("a directory holding " + argv[0])},
				Check:   "re-run this recovery with --dry-run to print the commands and run them yourself",
			})
			return 1
		}
		fmt.Fprintf(stderr, "fak recover: %v\n", err)
		return 1
	}
	return 0
}

// recoveryArgv0 resolves a step's command. A plan that says `fak` means THIS
// fak: the operator is already running it, so making the recovery depend on a
// second copy being installed on PATH is a failure mode invented by the
// recovery itself. Every other command (git, and anything a future plan names)
// resolves through PATH as written.
//
// os.Executable can fail on an exotic platform; the bare name is then still
// correct, just PATH-dependent again.
func recoveryArgv0(name string) string {
	if name != "fak" {
		return name
	}
	self, err := os.Executable()
	if err != nil || strings.TrimSpace(self) == "" {
		return name
	}
	return self
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
