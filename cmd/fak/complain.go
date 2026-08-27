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
//	    --journal-seq 42 \
//	    --rationale "the path is a curated note, not operator-private telemetry; the marker heuristic misfired" --live

import (
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
	domain := fs.String("domain", guardcomplaint.DefaultDomain, "complaint domain: guard (capability-floor appeal) | workflow (non-guard dev friction: shared-tree-clobber, tool-timeout, lane-collision)")
	kind := fs.String("kind", "", "complaint class within the domain (empty = the domain default). guard: false-positive|over-broad|latency|confusing|other. workflow: shared-tree-clobber|tool-timeout|lane-collision|other")
	reason := fs.String("reason", "", "the guard reason token being appealed (e.g. FILE_ADMISSION, OUT_OF_TREE_WRITE); in the workflow domain, free context for the friction")
	tool := fs.String("tool", "", "the refused tool (e.g. Bash, Write)")
	summary := fs.String("summary", "", "one-line headline of the complaint (required; drives the dedup key)")
	rationale := fs.String("rationale", "", "why the agent judges the guard wrong — and the recovery it wanted")
	fromJournal := fs.Bool("from-journal", false, "attach one matching DENY/QUARANTINE verdict from the guard decision journal as the witness; ambiguous reason/tool matches attach nothing until disambiguated")
	journal := fs.String("journal", "", "explicit guard-audit journal path to pull the witness from (default: discover under --workspace and the user config dir)")
	journalSeq := fs.Uint64("journal-seq", 0, "select the exact journal row sequence (requires --from-journal)")
	traceID := fs.String("trace-id", "", "select the exact denial trace id (requires --from-journal)")
	argsDigest := fs.String("args-digest", "", "select the exact denial args digest (requires --from-journal)")
	workspace := fs.String("workspace", ".", "workspace root for journal discovery when --journal is not given")
	repo := fs.String("repo", "", "owner/repo for gh; default is the current repo")
	limit := fs.Int("limit", 300, "existing issue scan limit for live/fetch modes")
	existingJSON := fs.String("existing-json", "", "fixture list of existing gh issues for dry-run tests")
	fetchExisting := fs.Bool("fetch-existing", false, "dry-run but query gh to classify create vs update")
	live := fs.Bool("live", false, "create/update the GitHub issue with gh (or set FAK_COMPLAIN_LIVE=1 to auto-file every complaint fleet-wide)")
	asJSON := fs.Bool("json", false, "emit machine-readable plan/result")
	var labels stringList
	fs.Var(&labels, "label", "label to add to a newly-created complaint; repeatable (default: "+guardcomplaint.Label+")")
	if !parseFlags(fs, argv) {
		return 2
	}

	if strings.TrimSpace(*summary) == "" {
		fmt.Fprintln(stderr, "fak complain: --summary is required (the one-line headline that identifies the complaint)")
		return 2
	}
	if !*fromJournal && (*journalSeq != 0 || strings.TrimSpace(*traceID) != "" || strings.TrimSpace(*argsDigest) != "") {
		fmt.Fprintln(stderr, "fak complain: --journal-seq, --trace-id, and --args-digest require --from-journal")
		return 2
	}
	normDomain, err := guardcomplaint.NormalizeDomain(*domain)
	if err != nil {
		fmt.Fprintf(stderr, "fak complain: %v\n", err)
		return 2
	}
	normKind, err := guardcomplaint.NormalizeKindFor(normDomain, *kind)
	if err != nil {
		fmt.Fprintf(stderr, "fak complain: %v\n", err)
		return 2
	}

	// Resolve whether this complaint actually files a gh ticket. --live is the
	// explicit per-call opt-in; FAK_COMPLAIN_LIVE=1 is the fleet-wide default that
	// makes the appeal channel file automatically. Without one of them a complaint
	// is a dry-run that records NOTHING durable — the "worked around it and
	// journaled it" failure the friction-complaint doctrine exists to kill.
	liveMode := complainLiveMode(*live, os.Getenv)

	c := guardcomplaint.Complaint{
		Domain:    normDomain,
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
		selection := guardcomplaint.SelectDenial(paths, guardcomplaint.DenialSelector{
			Reason:     c.Reason,
			Tool:       c.Tool,
			Seq:        *journalSeq,
			TraceID:    strings.TrimSpace(*traceID),
			ArgsDigest: strings.TrimSpace(*argsDigest),
		})
		switch {
		case selection.Evidence != nil:
			c.Evidence = selection.Evidence
		case selection.Ambiguous:
			fmt.Fprintf(stderr, "fak complain: --from-journal found %d matching DENY/QUARANTINE rows; refusing to attach an ambiguous witness. Pass --journal-seq, --trace-id, or --args-digest to select the refused call being appealed\n", selection.Matches)
		default:
			fmt.Fprintln(stderr, "fak complain: --from-journal found no matching DENY/QUARANTINE row; filing on the rationale alone (the body discloses the missing witness)")
		}
	}

	var existing []dogfoodissues.Issue
	switch {
	case *existingJSON != "":
		if err := readJSONFileInto(*existingJSON, &existing); err != nil {
			fmt.Fprintf(stderr, "fak complain: --existing-json must contain a JSON list: %v\n", err)
			return 2
		}
	case liveMode || *fetchExisting:
		existing, err = guardcomplaint.FetchExisting(*repo, *limit)
		if err != nil {
			fmt.Fprintf(stderr, "fak complain: %v\n", err)
			return 2
		}
	}

	row := guardcomplaint.BuildPlan(c, existing)
	mode := "dry-run"
	if liveMode {
		mode = "live"
	}
	result := guardcomplaint.Result{
		Schema:  guardcomplaint.Schema,
		Mode:    mode,
		Planned: []guardcomplaint.PlanRow{row},
		Synced:  []dogfoodissues.SyncRow{},
	}
	if liveMode {
		useLabels := []string(labels)
		if len(useLabels) == 0 {
			useLabels = []string{guardcomplaint.LabelFor(normDomain)}
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

	if liveMode {
		for _, s := range result.Synced {
			if !s.OK {
				return 1
			}
		}
	} else {
		// A dry-run files nothing. Disclose that loudly so the complaint is not
		// mistaken for a filed ticket, and name BOTH ways to actually file.
		fmt.Fprintln(stderr, "fak complain: dry-run — NO gh ticket was filed. Add --live to file now, or set FAK_COMPLAIN_LIVE=1 to auto-file every complaint.")
	}
	return 0
}

// complainLiveMode resolves whether a complaint should actually file a gh ticket.
// The explicit --live flag is the per-call opt-in; FAK_COMPLAIN_LIVE (1/true/yes)
// is the fleet-wide default that makes the appeal channel file GitHub tickets
// automatically, so a complaint is a tracked, deduped record rather than a
// dry-run that leaves nothing behind. getenv is injected so the resolution is
// unit-testable without touching the process environment or the network.
func complainLiveMode(flagLive bool, getenv func(string) string) bool {
	if flagLive {
		return true
	}
	v := strings.TrimSpace(getenv("FAK_COMPLAIN_LIVE"))
	return v == "1" || strings.EqualFold(v, "true") || strings.EqualFold(v, "yes")
}
