package main

// unwitnessed_claim.go is the thin I/O shell for `fak dispatch
// unwitnessed-claim`: it reads one issue's state + comments via `gh issue
// view`, folds them through internal/unwitnessedclaim's pure Evaluate, and
// either prints the would-post comment (default, dry-run) or posts it via
// `gh issue comment` when --live is passed. It never closes the issue --
// only ancestry does that (#1816).

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os/exec"

	"github.com/anthony-chaudhary/fak/internal/unwitnessedclaim"
)

func runDispatchUnwitnessedClaim(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch unwitnessed-claim", flag.ContinueOnError)
	fs.SetOutput(stderr)
	issue := fs.Int("issue", 0, "the GitHub issue number to check (required)")
	live := fs.Bool("live", false, "actually post the comment via `gh issue comment` (default: dry-run, print only)")
	asJSON := fs.Bool("json", false, "emit the raw Report JSON instead of the human summary")
	if err := fs.Parse(argv); err != nil {
		return 2 // flag already printed the error
	}
	if *issue <= 0 {
		fmt.Fprintln(stderr, "fak dispatch unwitnessed-claim: --issue N is required")
		return 2
	}

	raw, err := ghIssueViewJSON(*issue)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch unwitnessed-claim: %v\n", err)
		return 1
	}
	in, err := parseGhIssueUnwitnessedInput(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch unwitnessed-claim: %v\n", err)
		return 1
	}
	rep := unwitnessedclaim.Evaluate(in)

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rep, "fak dispatch unwitnessed-claim")
	}
	if !rep.Flagged {
		fmt.Fprintf(stdout, "fak dispatch unwitnessed-claim: issue #%d -- no unwitnessed completion claim\n", *issue)
		return 0
	}
	fmt.Fprintf(stdout, "fak dispatch unwitnessed-claim: issue #%d -- unwitnessed claim by @%s\n\n%s\n", *issue, rep.ClaimAuthor, rep.CommentBody)
	if !*live {
		fmt.Fprintln(stdout, "\n(dry-run -- pass --live to actually post this comment)")
		return 0
	}
	if err := ghPostIssueComment(*issue, rep.CommentBody); err != nil {
		fmt.Fprintf(stderr, "fak dispatch unwitnessed-claim: post comment: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, "\nposted.")
	return 0
}

// ghIssueCommentJSON / ghIssueViewJSONShape mirror the subset of `gh issue
// view --json number,state,comments` this shell needs.
type ghIssueCommentJSON struct {
	Author struct {
		Login string `json:"login"`
	} `json:"author"`
	Body string `json:"body"`
}

type ghIssueViewJSONShape struct {
	Number   int                  `json:"number"`
	State    string               `json:"state"`
	Comments []ghIssueCommentJSON `json:"comments"`
}

// parseGhIssueUnwitnessedInput is the pure half of the shell: turning the
// `gh issue view` JSON into unwitnessedclaim.Input is deterministic and
// needs no process I/O, so it is tested directly against canned JSON rather
// than only through a live `gh` call.
func parseGhIssueUnwitnessedInput(raw []byte) (unwitnessedclaim.Input, error) {
	var v ghIssueViewJSONShape
	if err := json.Unmarshal(raw, &v); err != nil {
		return unwitnessedclaim.Input{}, fmt.Errorf("parse gh issue view json: %w", err)
	}
	in := unwitnessedclaim.Input{IssueNumber: v.Number, Open: v.State == "OPEN"}
	for _, c := range v.Comments {
		in.Comments = append(in.Comments, unwitnessedclaim.Comment{Author: c.Author.Login, Body: c.Body})
	}
	return in, nil
}

func ghIssueViewJSON(issue int) ([]byte, error) {
	cmd := exec.Command("gh", "issue", "view", fmt.Sprint(issue), "--json", "number,state,comments")
	configureDispatchHelperCommand(cmd)
	var out, errBuf bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errBuf
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("gh issue view %d: %w: %s", issue, err, errBuf.String())
	}
	return out.Bytes(), nil
}

func ghPostIssueComment(issue int, body string) error {
	cmd := exec.Command("gh", "issue", "comment", fmt.Sprint(issue), "--body", body)
	configureDispatchHelperCommand(cmd)
	var errBuf bytes.Buffer
	cmd.Stderr = &errBuf
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("%w: %s", err, errBuf.String())
	}
	return nil
}
