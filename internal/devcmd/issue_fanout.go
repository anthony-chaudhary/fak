package devcmd

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuefanout"
)

// runIssueFanout expands one shipped working spine into its contract-ready
// follow-on backlog (QA / dogfood / productization / observability /
// integration / docs / release) — the "3..50+ follow-ons at creation time"
// default. By default it only plans; --live files the planned candidates as
// GitHub issues via gh (milestone + labels at creation), skipping any
// candidate whose fanout-<leaf>-<slug> marker key already appears in an
// existing issue body — so a rerun files zero and spams nothing (#2531).
// Offline alternatives: file by hand with gh, or wave-plan via
// `fak-dev issue cohort --from-plan`.
func runIssueFanout(stdout, stderr io.Writer, argv []string) int {
	return runIssueFanoutWith(stdout, stderr, argv, nil)
}

// runIssueFanoutWith is runIssueFanout with an injectable gh runner so tests
// exercise the --live path without a real gh.
func runIssueFanoutWith(stdout, stderr io.Writer, argv []string, gh issueCreateRunner) int {
	fs := flag.NewFlagSet("issue fanout", flag.ContinueOnError)
	fs.SetOutput(stderr)
	title := fs.String("title", "", "human name of the shipped spine")
	leaf := fs.String("leaf", "", "owning leaf/lane (stamps keys, lane, default paths)")
	spine := fs.String("spine", "", "spine witness: commit SHA, demo command, or doc path")
	parent := fs.String("parent", "", "epic/issue ref the fan-out hangs off (default: --spine)")
	parentIssue := fs.Int("parent-issue", 0, "parent issue number for project-work denominator binding")
	parentBaseline := fs.Float64("parent-baseline-points", 0, "declared parent production-scope baseline points")
	completionStandard := fs.String("completion-standard", "production", "generated child maturity (default production)")
	targetEnvelope := fs.String("target-envelope", "", "production target operating envelope")
	witnessedEnvelope := fs.String("witnessed-envelope", "", "currently witnessed operating envelope")
	paths := fs.String("paths", "", "comma-separated file trees (default internal/<leaf>/)")
	areas := fs.String("areas", "", "comma-separated area filter ("+strings.Join(issuefanout.AreaNames(), ",")+")")
	maxN := fs.Int("max", 0, "cap candidates (0 = full taxonomy; floor "+fmt.Sprint(issuefanout.MinFanout)+")")
	asJSON := fs.Bool("json", false, "emit the machine-readable fan-out plan (feed to fak-dev issue cohort --from-plan)")
	adoption := fs.Bool("adoption", false, "measure the default instead of planning: report which --leaves cleared the fan-out floor vs gaps (exit 1 on any gap)")
	leaves := fs.String("leaves", "", "with --adoption: comma-separated shipped leaves to audit")
	markers := fs.String("markers", "", "with --adoption: comma-separated filed fan-out marker keys (fanout-<leaf>-<slug>)")
	coverage := fs.Bool("coverage", false, "score the defaults on the REAL repo: gather witnesses from git + gh and report spine coverage and fan-out coverage with per-leaf evidence (exit 1 if either rate is short)")
	since := fs.String("since", "90 days ago", "with --coverage: the window of 'recently shipped' (any git --since selector)")
	scanCap := fs.Int("scan-cap", issuefanout.DefaultCoverageScanCap, "with --coverage: tracker export size for the marker scan (fan-out markers live in OLDER issues, so this is much larger than --dedupe-cap; a scan that hits the cap reports NOT PROVEN)")
	live := fs.Bool("live", false, "file the planned candidates as GitHub issues via gh, after marker-key dedupe (default: plan only)")
	repo := fs.String("repo", "", "with --live: owner/repo for gh (default: current repo)")
	dedupeCap := fs.Int("dedupe-cap", issuefanout.DefaultDedupeCap, "with --live: bounded existing-issue scan for marker-key dedupe")
	existingJSON := fs.String("existing-json", "", "with --live: fixture of existing issues (JSON [{number,body}]) instead of querying gh")
	if !parseFlags(fs, argv) {
		return 2
	}

	if *coverage {
		if fs.NArg() != 0 {
			fmt.Fprintln(stderr, "fak-dev issue fanout --coverage: takes no positional args (witnesses are gathered from git and gh)")
			return 2
		}
		if *adoption {
			fmt.Fprintln(stderr, "fak-dev issue fanout: --coverage and --adoption are alternative meters — --coverage gathers its own witnesses, --adoption takes them via --leaves/--markers")
			return 2
		}
		return emitFanoutCoverage(stdout, stderr, *since, *repo, *scanCap, *asJSON, fanoutCoverageDeps{gh: gh})
	}

	if *adoption {
		if fs.NArg() != 0 {
			fmt.Fprintln(stderr, "fak-dev issue fanout --adoption: takes no positional args (pass --leaves and --markers)")
			return 2
		}
		rep := issuefanout.Adoption(issueFanoutSplit(*leaves), issueFanoutSplit(*markers))
		if *asJSON {
			if err := writeIndentedJSON(stdout, rep); err != nil {
				fmt.Fprintf(stderr, "fak-dev issue fanout: encode json: %v\n", err)
				return 1
			}
		} else {
			fmt.Fprint(stdout, issuefanout.RenderAdoption(rep))
		}
		if !rep.OK {
			return 1 // a shipped leaf is a gap — the honesty meter fails the gate
		}
		return 0
	}

	if fs.NArg() != 0 || *title == "" || *leaf == "" || *spine == "" {
		fmt.Fprintln(stderr, "fak-dev issue fanout: --title, --leaf and --spine are required (the spine witness comes first; no spine yet means the spine itself is the issue to file)")
		return 2
	}

	plan, err := issuefanout.Build(issuefanout.Input{
		Title:              *title,
		Leaf:               *leaf,
		SpineRef:           *spine,
		ParentRef:          *parent,
		Paths:              issueFanoutSplit(*paths),
		Areas:              issueFanoutSplit(*areas),
		Max:                *maxN,
		ParentIssue:        *parentIssue,
		ParentBaseline:     *parentBaseline,
		CompletionStandard: *completionStandard,
		TargetEnvelope:     *targetEnvelope,
		WitnessedEnvelope:  *witnessedEnvelope,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue fanout: %v\n", err)
		return 2
	}

	if *live {
		if *dedupeCap <= 0 {
			fmt.Fprintln(stderr, "fak-dev issue fanout: --live needs a positive --dedupe-cap — the bounded marker-key scan is the anti-spam contract")
			return 2
		}
		var existing []issuefanout.Issue
		if *existingJSON != "" {
			b, err := os.ReadFile(*existingJSON)
			if err != nil {
				fmt.Fprintf(stderr, "fak-dev issue fanout: %v\n", err)
				return 2
			}
			if err := json.Unmarshal(b, &existing); err != nil {
				fmt.Fprintf(stderr, "fak-dev issue fanout: --existing-json must contain a JSON list of {number,body}: %v\n", err)
				return 2
			}
		} else {
			existing, err = fetchFanoutExisting(*repo, *dedupeCap, gh)
			if err != nil {
				fmt.Fprintf(stderr, "fak-dev issue fanout: %v\n", err)
				return 2
			}
		}
		run := gh
		if run == nil {
			run = runTaskHandoffGH
		}
		res, err := issuefanout.FileLive(plan, existing, issuefanout.LiveOptions{
			Repo:      *repo,
			DedupeCap: *dedupeCap,
			Runner:    issuefanout.Runner(run),
		})
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue fanout: %v\n", err)
			return 2
		}
		if *asJSON {
			if err := writeIndentedJSON(stdout, res); err != nil {
				fmt.Fprintf(stderr, "fak-dev issue fanout: encode json: %v\n", err)
				return 1
			}
		} else {
			fmt.Fprint(stdout, issuefanout.RenderLive(res))
		}
		if res.Failed > 0 {
			return 1
		}
		return 0
	}

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, plan, "fak-dev issue fanout")
	}
	fmt.Fprint(stdout, issuefanout.Render(plan))
	return 0
}

// fetchFanoutExisting runs the bounded dedupe scan (`gh issue list --state all`)
// and decodes the {number,body} rows the marker-key scan reads.
func fetchFanoutExisting(repo string, dedupeCap int, gh issueCreateRunner) ([]issuefanout.Issue, error) {
	run := gh
	if run == nil {
		run = runTaskHandoffGH
	}
	args := issuefanout.ListExistingArgs(repo, dedupeCap)
	stdout, stderr, ok := run(args)
	if !ok {
		return nil, fmt.Errorf("gh %s: %s", strings.Join(args, " "), strings.TrimSpace(stderr))
	}
	if strings.TrimSpace(stdout) == "" {
		return nil, nil
	}
	var issues []issuefanout.Issue
	if err := json.Unmarshal([]byte(stdout), &issues); err != nil {
		return nil, fmt.Errorf("parse gh issue list JSON: %w", err)
	}
	return issues, nil
}

func issueFanoutSplit(csv string) []string {
	var out []string
	for _, part := range strings.Split(csv, ",") {
		if part = strings.TrimSpace(part); part != "" {
			out = append(out, part)
		}
	}
	return out
}
