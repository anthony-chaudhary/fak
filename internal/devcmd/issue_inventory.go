package devcmd

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"time"

	"github.com/anthony-chaudhary/fak/internal/issuededup"
	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

var issueInventoryNow = time.Now

type issueInventoryInput struct {
	Schema       string                         `json:"schema,omitempty"`
	GeneratedAt  string                         `json:"generated_at,omitempty"`
	Observations []issuepolicy.ErrorObservation `json:"observations"`
}

func RunIssueInventory(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("issue inventory", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fromIssues := fs.String("from-issues", "", "read a cached gh issue array from PATH or '-'")
	fromObservations := fs.String("from-observations", "", "read trusted error observations from PATH")
	asJSON := fs.Bool("json", false, "emit fak.issue-error-inventory/1 JSON")
	requireActionable := fs.Int("require-actionable", 0, "gate dispatch for one issue number")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 || *fromIssues == "" || *fromObservations == "" {
		fmt.Fprintln(stderr, "fak-dev issue inventory: --from-issues and --from-observations are required")
		return 2
	}
	issueRaw, err := readIssueDedupInput(*fromIssues)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue inventory: %v\n", err)
		return 2
	}
	issues, err := issuededup.ParseBacklog(issueRaw)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue inventory: %v\n", err)
		return 2
	}
	obsRaw, err := os.ReadFile(*fromObservations)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue inventory: %v\n", err)
		return 2
	}
	input, err := decodeIssueInventoryInput(obsRaw)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue inventory: %v\n", err)
		return 2
	}
	if err := validateInventoryIssueSet(issues, input.Observations); err != nil {
		fmt.Fprintf(stderr, "fak-dev issue inventory: %v\n", err)
		return 2
	}
	generatedAt := issueInventoryNow().UTC()
	if input.GeneratedAt != "" {
		generatedAt, err = time.Parse(time.RFC3339, input.GeneratedAt)
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue inventory: invalid generated_at: %v\n", err)
			return 2
		}
	}
	digest, err := issueInventoryDigest(issues, input.Observations)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue inventory: %v\n", err)
		return 1
	}
	rep, err := issuepolicy.BuildErrorInventory(issuepolicy.ErrorInventoryInput{GeneratedAt: generatedAt, SnapshotDigest: digest, Observations: input.Observations})
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue inventory: %v\n", err)
		return 2
	}
	if *asJSON {
		if code := encodeJSONOrFail(stdout, stderr, rep, "fak-dev issue inventory"); code != 0 {
			return code
		}
	} else {
		renderIssueInventory(stdout, rep)
	}
	if *requireActionable != 0 {
		for _, row := range rep.Issues {
			if row.Issue == *requireActionable {
				return issuepolicy.ActionabilityExit(row.Disposition)
			}
		}
		fmt.Fprintf(stderr, "fak-dev issue inventory: issue %d is not in the inventory\n", *requireActionable)
		return 2
	}
	return 0
}

func decodeIssueInventoryInput(b []byte) (issueInventoryInput, error) {
	var in issueInventoryInput
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&in); err != nil {
		return in, fmt.Errorf("parse observations: %w", err)
	}
	if in.Schema != "" && in.Schema != issuepolicy.ErrorInventorySchema {
		return in, fmt.Errorf("unsupported observations schema %q", in.Schema)
	}
	if len(in.Observations) == 0 {
		return in, fmt.Errorf("observations are required")
	}
	return in, nil
}

func validateInventoryIssueSet(issues []issuededup.BacklogIssue, obs []issuepolicy.ErrorObservation) error {
	known := make(map[int]bool, len(issues))
	for _, issue := range issues {
		known[issue.Number] = true
	}
	for _, row := range obs {
		if !known[row.Issue] {
			return fmt.Errorf("observation issue %d is absent from --from-issues", row.Issue)
		}
	}
	return nil
}

func issueInventoryDigest(issues []issuededup.BacklogIssue, obs []issuepolicy.ErrorObservation) (string, error) {
	sort.Slice(issues, func(i, j int) bool { return issues[i].Number < issues[j].Number })
	sort.Slice(obs, func(i, j int) bool { return obs[i].Issue < obs[j].Issue })
	b, err := json.Marshal(struct {
		Issues       []issuededup.BacklogIssue      `json:"issues"`
		Observations []issuepolicy.ErrorObservation `json:"observations"`
	}{issues, obs})
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

func renderIssueInventory(w io.Writer, rep issuepolicy.ErrorInventory) {
	fmt.Fprintf(w, "issue error inventory %s\n", rep.SnapshotDigest)
	for _, row := range rep.Issues {
		fmt.Fprintf(w, "#%s %-20s canonical=#%s %s\n", strconv.Itoa(row.Issue), row.Disposition, strconv.Itoa(row.CanonicalIssue), row.ReasonCodes[0])
	}
}
