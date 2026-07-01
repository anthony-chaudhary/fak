package main

// dispatch_commit_links.go is the thin I/O shell for `fak dispatch
// commit-links`: it shells out to `git log` over a revision range, parses
// each commit into internal/commitissuelink's pure Commit fact, and folds
// them into a Report. All classification logic lives in
// internal/commitissuelink; this file only does the disk/process I/O and
// rendering (#1812).

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/commitissuelink"
)

// commitLinkFieldSep / commitLinkRecordSep are the same %x1f/%x1e idiom
// internal/marketing/ship.go uses to split `git log` output without
// colliding with commit-message content.
const (
	commitLinkFieldSep  = "\x1f"
	commitLinkRecordSep = "\x1e"
)

func runDispatchCommitLinks(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch commit-links", flag.ContinueOnError)
	fs.SetOutput(stderr)
	rng := fs.String("range", "HEAD~50..HEAD", "git revision range to scan (git rev-list syntax)")
	witnessJSON := fs.String("witness-json", "", "read commit-linked issue witness facts and bucket unresolved close-gate failures")
	asJSON := fs.Bool("json", false, "emit the raw Report JSON instead of the human table")
	if err := fs.Parse(argv); err != nil {
		return 2 // flag already printed the error
	}

	if strings.TrimSpace(*witnessJSON) != "" {
		rows, err := readCommitLinkedIssueWitnesses(*witnessJSON)
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch commit-links: %v\n", err)
			return 1
		}
		rep := commitissuelink.FoldUnresolvedCommitLinkedIssues(rows)
		if *asJSON {
			return encodeJSONOrFail(stdout, stderr, rep, "fak dispatch commit-links")
		}
		renderUnresolvedCommitLinks(stdout, rep)
		return 0
	}

	raw, err := gitLogForCommitLinks(*rng)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch commit-links: %v\n", err)
		return 1
	}
	commits := parseCommitLinkLog(raw)
	rep := commitissuelink.Fold(commits)

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rep, "fak dispatch commit-links")
	}
	renderCommitLinks(stdout, rep)
	return 0
}

// gitLogForCommitLinks reads SHA + full raw commit message for every
// non-merge commit in rng, in the %x1f/%x1e delimited form parseCommitLinkLog
// expects.
func gitLogForCommitLinks(rng string) (string, error) {
	cmd := exec.Command("git", "log", "--no-merges", "--format=%H"+commitLinkFieldSep+"%B"+commitLinkRecordSep, rng)
	configureDispatchHelperCommand(cmd)
	var out, errBuf bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errBuf
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("git log %s: %w: %s", rng, err, strings.TrimSpace(errBuf.String()))
	}
	return out.String(), nil
}

func readCommitLinkedIssueWitnesses(path string) ([]commitissuelink.CommitLinkedIssue, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read witness json %s: %w", path, err)
	}
	var arr []commitissuelink.CommitLinkedIssue
	if err := json.Unmarshal(raw, &arr); err == nil {
		return arr, nil
	}
	var obj struct {
		Issues []commitissuelink.CommitLinkedIssue `json:"issues"`
	}
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, fmt.Errorf("parse witness json %s: %w", path, err)
	}
	return obj.Issues, nil
}

// parseCommitLinkLog is the pure half of the shell: splitting the delimited
// `git log` text into commitissuelink.Commit facts is deterministic and
// needs no process I/O, so it is tested directly against canned strings
// rather than only through a live git repo.
func parseCommitLinkLog(raw string) []commitissuelink.Commit {
	var commits []commitissuelink.Commit
	for _, rec := range strings.Split(raw, commitLinkRecordSep) {
		rec = strings.Trim(rec, "\n")
		if rec == "" {
			continue
		}
		parts := strings.SplitN(rec, commitLinkFieldSep, 2)
		if len(parts) != 2 {
			continue
		}
		sha := parts[0]
		msg := strings.TrimPrefix(parts[1], "\n")
		subject := msg
		body := ""
		if i := strings.IndexByte(msg, '\n'); i >= 0 {
			subject = msg[:i]
			body = strings.TrimLeft(msg[i+1:], "\n")
		}
		commits = append(commits, commitissuelink.Commit{SHA: sha, Subject: subject, Body: body})
	}
	return commits
}

func renderCommitLinks(w io.Writer, rep commitissuelink.Report) {
	fmt.Fprintf(w, "fak dispatch commit-links: scanned %d commit(s)\n", rep.Scanned)
	if len(rep.Findings) == 0 {
		fmt.Fprintln(w, "  no missing subject-line issue links")
		return
	}
	for _, f := range rep.Findings {
		sha := f.SHA
		if len(sha) > 12 {
			sha = sha[:12]
		}
		if f.GuessedIssue != "" {
			fmt.Fprintf(w, "  %s  %s  (likely #%s, from a body trailer not the subject)\n", sha, f.Subject, f.GuessedIssue)
		} else {
			fmt.Fprintf(w, "  %s  %s  (no #N anywhere -- unlinked)\n", sha, f.Subject)
		}
	}
}

func renderUnresolvedCommitLinks(w io.Writer, rep commitissuelink.UnresolvedReport) {
	fmt.Fprintf(w, "fak dispatch commit-links witness-failures: scanned %d issue(s)\n", rep.Scanned)
	if len(rep.Findings) == 0 {
		fmt.Fprintln(w, "  no unresolved commit-linked issues")
		return
	}
	for _, f := range rep.Findings {
		fmt.Fprintf(w, "  #%d  %s  %s  %s\n", f.Number, f.SHA, f.Reason, f.Detail)
	}
}
