package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/stalework"
)

var staleWorkIssueRunner = func(out, errw io.Writer, args []string) int {
	return runDevHandoff(strings.NewReader(""), out, errw, args)
}

var staleWorkDispatchRunner = func(out, errw io.Writer, args []string) int {
	return runDispatch(out, errw, args)
}

func runStaleWorkLoop(args []string, out, errw io.Writer) int {
	fs := flag.NewFlagSet("stale-work loop", flag.ContinueOnError)
	fs.SetOutput(errw)
	root := fs.String("root", ".", "repository root (used when --packet is omitted)")
	packetPath := fs.String("packet", "", "#6613 packet JSON (default: scan the repository)")
	issuesPath := fs.String("issues", "", "open plus recently-closed issue snapshot JSON")
	statePath := fs.String("state", "", "prior adjudication state JSON")
	witnessPath := fs.String("witnesses", "", "independent git/issue/test witness JSON")
	stateOut := fs.String("state-out", "", "write next adjudication state (explicit persistence)")
	maxWave := fs.Int("max-wave", 0, "maximum collision-free workers per wave (0 = natural width)")
	liveIssues := fs.Bool("live-issues", false, "create missing dedicated issues (explicit GitHub mutation)")
	liveLaunch := fs.Bool("live-launch", false, "launch contract-valid issue workers (explicit process mutation)")
	selfcheck := fs.Bool("selfcheck", false, "run the deterministic orchestration spine")
	_ = fs.Bool("json", true, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(errw, "fak stale-work loop: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *selfcheck {
		return staleWorkLoopSelfcheck(out, errw)
	}
	if *liveIssues && strings.TrimSpace(*issuesPath) == "" {
		fmt.Fprintln(errw, "fak stale-work loop: --live-issues requires --issues with the dedupe read-back")
		return 2
	}

	packet, err := loadStaleWorkPacket(*packetPath, *root)
	if err != nil {
		fmt.Fprintf(errw, "fak stale-work loop: packet: %v\n", err)
		return 1
	}
	var issues []stalework.IssueSnapshot
	if err := readOptionalJSON(*issuesPath, &issues); err != nil {
		fmt.Fprintf(errw, "fak stale-work loop: issues: %v\n", err)
		return 1
	}
	var state stalework.LoopState
	if err := readOptionalJSON(*statePath, &state); err != nil {
		fmt.Fprintf(errw, "fak stale-work loop: state: %v\n", err)
		return 1
	}
	var witnesses []stalework.WitnessRecord
	if err := readOptionalJSON(*witnessPath, &witnesses); err != nil {
		fmt.Fprintf(errw, "fak stale-work loop: witnesses: %v\n", err)
		return 1
	}

	opt := stalework.LoopOptions{
		Issues: issues, State: state, Witnesses: witnesses, MaxWave: *maxWave,
		LiveIssueCreate: *liveIssues, LiveLaunch: *liveLaunch,
	}
	plan := stalework.BuildLoop(packet, opt)
	if *liveIssues {
		for i := range plan.Units {
			unit := &plan.Units[i]
			if unit.Issue.Action != "create" {
				continue
			}
			number, url, code := executeStaleWorkIssue(unit.Issue.Command, errw)
			if code != 0 {
				return code
			}
			issues = append(issues, stalework.IssueSnapshot{
				Number: number, Title: unit.Issue.Title, Body: unit.Issue.Body,
				State: "OPEN", URL: url,
			})
		}
		opt.Issues = issues
		opt.LiveIssueCreate = false
		plan = stalework.BuildLoop(packet, opt)
		plan.Mode = "live"
	}
	if *liveLaunch {
		for _, wave := range plan.Waves {
			for i := range plan.Units {
				unit := &plan.Units[i]
				if unit.Dispatch.Status != stalework.DispatchReady || unit.Dispatch.Wave != wave.Index {
					continue
				}
				var stdout, stderr bytes.Buffer
				command := unit.Dispatch.Command
				if len(command) < 3 || command[0] != "fak" || command[1] != "dispatch" {
					fmt.Fprintf(errw, "fak stale-work loop: invalid dispatch command for %s\n", unit.DedupeKey)
					return 1
				}
				code := staleWorkDispatchRunner(&stdout, &stderr, command[2:])
				unit.Dispatch.ExitCode = code
				unit.Dispatch.Launched = code == 0
				if code != 0 {
					fmt.Fprintf(errw, "fak stale-work loop: launch %s: %s\n", unit.DedupeKey, strings.TrimSpace(stderr.String()))
					return code
				}
				plan.Counts.Launches++
			}
		}
		plan.Mode = "live"
	}
	if strings.TrimSpace(*stateOut) != "" {
		if err := writeLoopState(*stateOut, plan.NextState); err != nil {
			fmt.Fprintf(errw, "fak stale-work loop: state-out: %v\n", err)
			return 1
		}
	}
	enc := json.NewEncoder(out)
	enc.SetIndent("", "  ")
	if err := enc.Encode(plan); err != nil {
		return 1
	}
	return 0
}

func loadStaleWorkPacket(path, root string) (stalework.Packet, error) {
	if strings.TrimSpace(path) != "" {
		var packet stalework.Packet
		if err := readOptionalJSON(path, &packet); err != nil {
			return packet, err
		}
		return packet, nil
	}
	return stalework.Scan(context.Background(), stalework.Options{Root: root, Limit: 10})
}

func readOptionalJSON(path string, dst any) error {
	if strings.TrimSpace(path) == "" {
		return nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(raw, dst)
}

func executeStaleWorkIssue(command []string, errw io.Writer) (int, string, int) {
	if len(command) < 3 || command[0] != "fak" || command[1] != "issue" || command[2] != "create" {
		fmt.Fprintln(errw, "fak stale-work loop: invalid issue-create command")
		return 0, "", 1
	}
	var stdout, stderr bytes.Buffer
	code := staleWorkIssueRunner(&stdout, &stderr, command[1:])
	if code != 0 {
		fmt.Fprintf(errw, "fak stale-work loop: create issue: %s\n", strings.TrimSpace(stderr.String()))
		return 0, "", code
	}
	var result struct {
		OK  bool   `json:"ok"`
		URL string `json:"url"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil || !result.OK {
		fmt.Fprintf(errw, "fak stale-work loop: create issue returned invalid JSON: %s\n", strings.TrimSpace(stdout.String()))
		return 0, "", 1
	}
	number, err := issueNumberFromURL(result.URL)
	if err != nil {
		fmt.Fprintf(errw, "fak stale-work loop: create issue URL: %v\n", err)
		return 0, "", 1
	}
	return number, result.URL, 0
}

func issueNumberFromURL(url string) (int, error) {
	part := strings.TrimRight(strings.TrimSpace(url), "/")
	if i := strings.LastIndexByte(part, '/'); i >= 0 {
		part = part[i+1:]
	}
	number, err := strconv.Atoi(part)
	if err != nil || number <= 0 {
		return 0, fmt.Errorf("cannot parse issue number from %q", url)
	}
	return number, nil
}

func writeLoopState(path string, state stalework.LoopState) error {
	raw, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return os.WriteFile(path, append(raw, '\n'), 0o644)
}

func staleWorkLoopSelfcheck(out, errw io.Writer) int {
	c := stalework.Candidate{
		Path: "docs/operator.md", Batch: "docs/operator.md", Score: 50, Status: "candidate",
		Components:         []stalework.Component{{Name: "dependency_drift", Points: 50, Provenance: "git", Evidence: "one dependency commit"}},
		LastSemanticCommit: "old", ExcerptSHA256: "excerpt",
		DedupeKey:  "stale-work:docs/operator.md",
		VerifyWith: "fak stale-work --path docs/operator.md --json",
	}
	plan := stalework.BuildLoop(stalework.Packet{Head: "head", Candidates: []stalework.Candidate{c}}, stalework.LoopOptions{})
	if len(plan.Units) != 1 || !plan.Units[0].Issue.Review.OK ||
		plan.Units[0].Dispatch.Reason != stalework.ReasonIssueRequired || plan.Counts.Launches != 0 {
		fmt.Fprintf(errw, "stale-work loop selfcheck failed: %+v\n", plan.Counts)
		return 1
	}
	fmt.Fprintln(out, "PASS contract-valid dedicated issue planned; dispatch refused before issue; zero launches")
	return 0
}
