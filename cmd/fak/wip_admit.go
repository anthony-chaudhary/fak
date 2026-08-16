package main

// wip_admit.go — the command shell for the START-OF-TASK admission (#3879 family).
//
// WHY THIS FILE EXISTS. Every other WIP surface in this repo is RETROSPECTIVE: `wip
// attribute` grades hunks that exist, `wip sweep-guard` warns about staging hunks that
// exist, `wip blocked` / `wip reconcile` / `tree-doctor` recover hunks that exist, and
// the flowmetrics local_wip KPI scores a tree that has already accumulated. The pure
// prospective fold — wipattr.AdmitStart — shipped in cf2d5d9558 with NO caller, so the
// one moment the work still costs nothing (before the first edit) stayed unguarded.
// This file is that caller, and nothing more: it gathers the git facts and hands them
// to the fold.
//
// IT GATHERS THE SAME INPUTS AS SWEEP-GUARD ON PURPOSE. wipSweepGuard folds
// (attributions, live sessions); this folds (attributions, live sessions, untracked).
// Sharing the gatherers is what makes the prospective and retrospective gates unable to
// disagree about who owns what — a path that sweep-guard would call a HAZARD is the same
// path admit HOLDs on, because both read one attribution set.
//
// THE UNTRACKED SET IS THE THIRD INPUT BECAUSE ATTRIBUTION CANNOT SEE IT. A checkpoint
// captures the TRACKED delta, so an untracked file is invisible to Attribute and reads
// as "clean" — the exact silence that let cmd/fak/serve_ggufplan.go sit unowned on this
// shared checkout for a day breaking `go build ./cmd/...` for every peer. Feeding
// wipUntrackedPaths in is what turns that silence into PATH_UNTRACKED_WIP.
//
// EXIT CODES follow sweep-guard: 0 ADMIT, 3 HOLD, 1 runtime error, 2 usage error. A
// HOLD is a distinct code (not 1) so a wrapper can branch on "the gate refused" without
// having to tell it apart from "the gate broke".

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/wipattr"
)

// wipAdmitHoldExit is the exit code for a HOLD verdict, matching sweep-guard's hazard
// code so one wrapper can treat "this gate refused" uniformly across both.
const wipAdmitHoldExit = 3

// runWipAdmit answers whether a task may BEGIN on the declared paths. Read-only: it
// inspects git and decides, it never writes a ref, stages, or edits the tree.
func runWipAdmit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wip admit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wip")
	repo := fs.String("C", "", "run in this git repo (default: cwd)")
	session := fs.String("session", "", "the self session id (default: $CLAUDE_CODE_SESSION_ID, else $FAK_SESSION_ID)")
	var intends pathList
	// No backticks in this usage string: flag.UnquoteUsage reads the first backticked
	// span as the flag's VALUE NAME.
	fs.Var(&intends, "path", "a repo-relative path this task INTENDS to touch (repeatable); undeclared means UNCHECKED, not clear")
	strict := fs.Bool("strict", false, "promote every soft finding (undeclared intent, unlanded self-WIP) to a hard HOLD")
	ceiling := fs.Int("ceiling", 0, "override how many unlanded paths this session may already hold (default: 3)")
	asJSON := fs.Bool("json", false, "emit the admission report as JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	self := strings.TrimSpace(*session)
	if self == "" {
		self = firstNonEmpty(os.Getenv("CLAUDE_CODE_SESSION_ID"), os.Getenv("FAK_SESSION_ID"))
	}
	if self == "" {
		fmt.Fprintln(stderr, "fak wip admit: --session is required when no session environment is set")
		return 2
	}

	rep, err := wipAdmit(context.Background(), *repo, self, intends, *strict, *ceiling)
	if err != nil {
		fmt.Fprintf(stderr, "fak wip admit: %v\n", err)
		return 1
	}
	if *asJSON {
		if code := encodeJSONOrFail(stdout, stderr, rep, "fak wip admit"); code != 0 {
			return code
		}
		if rep.Verdict == wipattr.AdmitHold {
			return wipAdmitHoldExit
		}
		return 0
	}
	return wipAdmitRender(stdout, stderr, rep)
}

// wipAdmit gathers the three git facts the fold needs and applies it. The gatherers are
// the SAME ones sweep-guard and wip owner already use, so no new view of the tree is
// invented here.
func wipAdmit(ctx context.Context, repo, self string, intends []string, strict bool, ceiling int) (wipattr.AdmitReport, error) {
	attrs, err := wipBuildAttributions(ctx, repo)
	if err != nil {
		return wipattr.AdmitReport{}, err
	}
	live, err := wipLiveSessions(ctx, repo)
	if err != nil {
		return wipattr.AdmitReport{}, err
	}
	untracked, err := wipUntrackedPaths(ctx, repo)
	if err != nil {
		return wipattr.AdmitReport{}, err
	}
	return wipattr.AdmitStart(wipattr.AdmitInput{
		Self:           self,
		Intends:        intends,
		Attrs:          attrs,
		Untracked:      untracked,
		Live:           live,
		SelfWIPCeiling: ceiling,
		Strict:         strict,
	}), nil
}

// wipAdmitRender writes the human form. An ADMIT goes to stdout (it is the answer); a
// HOLD goes to stderr (it is a refusal a caller may be piping past), and every finding
// prints its reason token so a reader can route the remedy without parsing prose.
func wipAdmitRender(stdout, stderr io.Writer, rep wipattr.AdmitReport) int {
	if rep.Verdict == wipattr.AdmitOK && len(rep.Findings) == 0 {
		fmt.Fprintf(stdout, "admit: ADMIT — %s clear to start (self holds %d/%d unlanded path(s))\n",
			wipAdmitIntentLabel(rep.Intends), rep.SelfDirty, rep.Ceiling)
		return 0
	}
	w := stdout
	if rep.Verdict == wipattr.AdmitHold {
		w = stderr
	}
	fmt.Fprintf(w, "admit: %s — %s (self holds %d/%d unlanded path(s))\n",
		rep.Verdict, wipAdmitIntentLabel(rep.Intends), rep.SelfDirty, rep.Ceiling)
	for _, f := range rep.Findings {
		kind := "warn"
		if f.Hard {
			kind = "HOLD"
		}
		where := f.Path
		if where == "" {
			where = "(session)"
		}
		fmt.Fprintf(w, "  %s\t%s\t%s\n\t%s\n", kind, where, f.Reason, f.Detail)
	}
	if rep.Next != "" {
		fmt.Fprintf(w, "next: %s\n", rep.Next)
	}
	if rep.Verdict == wipattr.AdmitHold {
		return wipAdmitHoldExit
	}
	return 0
}

// wipAdmitIntentLabel renders the declared set for the summary line, naming the
// unchecked case explicitly rather than printing an empty list.
func wipAdmitIntentLabel(intends []string) string {
	if len(intends) == 0 {
		return "no path declared"
	}
	return strings.Join(intends, ", ")
}
