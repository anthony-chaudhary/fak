package main

// fak complain — the agent's APPEAL channel against the kernel. When a governed
// agent judges a `fak guard` decision wrong (a false positive, an over-broad gate,
// a confusing or slow refusal), this files a structured, deduplicating GitHub issue
// in the agent's own words AND attaches the WITNESSED verdict pulled from the guard
// decision journal — because a self-report is not a witness, but the hash-chained
// journal row is. Repeat appeals about the same class fold onto ONE escalating issue
// (an occurrence count) so a recurring false positive reads as the stronger signal
// it is, instead of a pile of duplicates.
//
// It is the SUBJECTIVE complement to the objective guard RSI loop
// (`fak guard-verdict-rsi` / internal/guardroute): a false-positive DENY is
// byte-identical to a correct one in the journal, so only the agent that made the
// call knows it was legitimate. The decision layer lives in internal/guardcomplaint;
// this is the thin CLI shim over it, reusing the gh issue-create/update plumbing in
// internal/dogfoodissues.
//
// Safe by default: a dry-run that prints the planned appeal and never touches the
// network; --live is the explicit opt-in that fetches existing issues and shells out
// to `gh`. The friction taxonomy and routing this verb belongs to are documented in
// docs/notes/CONCEPT-AGENT-FRICTION-COMPLAINT-CHANNELS-2026-06-29.md.
//
//	fak complain --summary "floor blocked a legit docs/notes commit" \
//	    --reason FILE_ADMISSION --tool Bash --from-journal \
//	    --rationale "the path is a curated note, not operator-private telemetry; the marker heuristic misfired" --live

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
	"github.com/anthony-chaudhary/fak/internal/guardcomplaint"
)

func cmdComplain(argv []string) { os.Exit(runComplain(os.Stdout, os.Stderr, argv)) }

// runComplain is the testable core: it returns the process exit code instead of
// calling os.Exit, and takes its streams explicitly. Exit codes: 0 ok, 1 a live gh
// create/edit failed, 2 usage/parse/IO error.
func runComplain(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("complain", flag.ContinueOnError)
	fs.SetOutput(stderr)
	kind := fs.String("kind", guardcomplaint.DefaultKind, "complaint class: false-positive | over-broad | latency | confusing | other")
	reason := fs.String("reason", "", "the guard reason token being appealed (e.g. FILE_ADMISSION, OUT_OF_TREE_WRITE)")
	tool := fs.String("tool", "", "the refused tool (e.g. Bash, Write)")
	summary := fs.String("summary", "", "one-line headline of the complaint (required; drives the dedup key)")
	rationale := fs.String("rationale", "", "why the agent judges the guard wrong — and the recovery it wanted")
	fromJournal := fs.Bool("from-journal", false, "attach the latest matching DENY/QUARANTINE verdict from the guard decision journal as the witness (filtered by --reason/--tool)")
	journal := fs.String("journal", "", "explicit guard-audit journal path to pull the witness from (default: discover under --workspace and the user config dir)")
	workspace := fs.String("workspace", ".", "workspace root for journal discovery when --journal is not given")
	repo := fs.String("repo", "", "owner/repo for gh; default is the current repo")
	limit := fs.Int("limit", 300, "existing issue scan limit for live/fetch modes")
	existingJSON := fs.String("existing-json", "", "fixture list of existing gh issues for dry-run tests")
	fetchExisting := fs.Bool("fetch-existing", false, "dry-run but query gh to classify create vs update")
	live := fs.Bool("live", false, "create/update the GitHub issue with gh")
	asJSON := fs.Bool("json", false, "emit machine-readable plan/result")
	var labels stringList
	fs.Var(&labels, "label", "label to add to a newly-created complaint; repeatable (default: "+guardcomplaint.Label+")")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	if strings.TrimSpace(*summary) == "" {
		fmt.Fprintln(stderr, "fak complain: --summary is required (the one-line headline that identifies the complaint)")
		return 2
	}
	normKind, err := guardcomplaint.NormalizeKind(*kind)
	if err != nil {
		fmt.Fprintf(stderr, "fak complain: %v\n", err)
		return 2
	}

	c := guardcomplaint.Complaint{
		Kind:      normKind,
		Reason:    strings.TrimSpace(*reason),
		Tool:      strings.TrimSpace(*tool),
		Summary:   strings.TrimSpace(*summary),
		Rationale: strings.TrimSpace(*rationale),
	}

	// Pull the witnessed verdict from the journal unless told not to. An absent
	// witness is disclosed honestly in the body (never fabricated), so we only note
	// the miss on stderr and proceed — the appeal still files on the agent's rationale.
	if *fromJournal {
		paths := guardcomplaint.DiscoverJournals(*workspace, *journal)
		if ev := guardcomplaint.LatestDenial(paths, c.Reason, c.Tool); ev != nil {
			c.Evidence = ev
		} else {
			fmt.Fprintln(stderr, "fak complain: --from-journal found no matching DENY/QUARANTINE row; filing on the rationale alone (the body discloses the missing witness)")
		}
	}

	var existing []dogfoodissues.Issue
	switch {
	case *existingJSON != "":
		b, readErr := os.ReadFile(*existingJSON)
		if readErr != nil {
			fmt.Fprintf(stderr, "fak complain: %v\n", readErr)
			return 2
		}
		if err := json.Unmarshal(b, &existing); err != nil {
			fmt.Fprintf(stderr, "fak complain: --existing-json must contain a JSON list: %v\n", err)
			return 2
		}
	case *live || *fetchExisting:
		existing, err = guardcomplaint.FetchExisting(*repo, *limit)
		if err != nil {
			fmt.Fprintf(stderr, "fak complain: %v\n", err)
			return 2
		}
	}

	row := guardcomplaint.BuildPlan(c, existing)
	mode := "dry-run"
	if *live {
		mode = "live"
	}
	result := guardcomplaint.Result{
		Schema:  guardcomplaint.Schema,
		Mode:    mode,
		Planned: []guardcomplaint.PlanRow{row},
		Synced:  []dogfoodissues.SyncRow{},
	}
	if *live {
		useLabels := []string(labels)
		if len(useLabels) == 0 {
			useLabels = []string{guardcomplaint.Label}
		}
		result.Synced = []dogfoodissues.SyncRow{guardcomplaint.Sync(row, *repo, useLabels, nil)}
	}

	if *asJSON {
		if err := writeIndentedJSON(stdout, result); err != nil {
			fmt.Fprintf(stderr, "fak complain: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, guardcomplaint.Render(result))
	}

	if *live {
		for _, s := range result.Synced {
			if !s.OK {
				return 1
			}
		}
	}
	return 0
}
