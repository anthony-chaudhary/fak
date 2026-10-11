package safesync

// duplicatefix.go is the pre-push DUPLICATE-FIX advisory (2026-10-10 ctxmmu incident). Two
// sessions repaired the same red package within the hour by renaming the same source file to
// two DIFFERENT names; the second learned only after the fact, and merging both would have
// redefined the moved symbols twice and reddened the trunk again. Before landing, compare what
// the local commits do to each source file since the merge-base against what the upstream
// commits since that merge-base did: the same source renamed or deleted on both sides, with a
// different outcome, names the peer commit to converge on instead of landing a second fix.
//
// Advisory only — it never refuses a push. It reads the upstream tracking ref as fetched; a
// stale ref simply under-reports.

import (
	"context"
	"fmt"
	"io"
	"sort"
	"strings"
)

// DuplicateFix is one source file that the local commits and an upstream peer commit both
// renamed or deleted, to different results.
type DuplicateFix struct {
	Source       string `json:"source"`
	LocalStatus  string `json:"local_status"`           // "R" or "D"
	LocalTarget  string `json:"local_target,omitempty"` // rename destination; empty for a delete
	PeerSHA      string `json:"peer_sha"`
	PeerSubject  string `json:"peer_subject,omitempty"`
	PeerStatus   string `json:"peer_status"`
	PeerTarget   string `json:"peer_target,omitempty"`
	MergeBaseSHA string `json:"merge_base"`
}

// PeerCommit is one upstream commit (since a base) touching a set of paths.
type PeerCommit struct {
	SHA     string `json:"sha"`
	Subject string `json:"subject"`
}

type nameChange struct{ status, target string }

// DuplicateFixAdvisories compares HEAD with upstream (e.g. "origin/main") since their
// merge-base. nil run uses RealRunner. A missing upstream ref or no merge-base yields no
// advisories and no error: there is nothing to compare against.
func DuplicateFixAdvisories(ctx context.Context, run Runner, repo, upstream string) ([]DuplicateFix, error) {
	if run == nil {
		run = RealRunner
	}
	mbRes := run(ctx, repo, "merge-base", "HEAD", upstream)
	if mbRes.Err != nil {
		return nil, mbRes.Err
	}
	mb := strings.TrimSpace(string(mbRes.Stdout))
	if mbRes.Code != 0 || mb == "" {
		return nil, nil
	}
	diff := run(ctx, repo, "-c", "core.quotePath=false", "diff", "--name-status", "-M", mb, "HEAD")
	if diff.Err != nil || diff.Code != 0 {
		return nil, gitFailure("diff", diff)
	}
	local := map[string]nameChange{}
	for _, line := range strings.Split(string(diff.Stdout), "\n") {
		if src, ch, ok := parseNameStatus(line); ok {
			local[src] = ch
		}
	}
	if len(local) == 0 {
		return nil, nil
	}
	logRes := run(ctx, repo, "-c", "core.quotePath=false", "log", "-M", "--name-status",
		"--format=%x1e%H%x1f%s", mb+".."+upstream)
	if logRes.Err != nil || logRes.Code != 0 {
		return nil, gitFailure("log", logRes)
	}
	var out []DuplicateFix
	for _, rec := range strings.Split(string(logRes.Stdout), "\x1e") {
		lines := strings.Split(rec, "\n")
		sha, subject, ok := strings.Cut(lines[0], "\x1f")
		if !ok {
			continue
		}
		for _, line := range lines[1:] {
			src, peer, ok := parseNameStatus(line)
			if !ok {
				continue
			}
			mine, both := local[src]
			if !both || mine == peer {
				continue // untouched locally, or the identical change (merges cleanly)
			}
			out = append(out, DuplicateFix{
				Source: src, LocalStatus: mine.status, LocalTarget: mine.target,
				PeerSHA: strings.TrimSpace(sha), PeerSubject: strings.TrimSpace(subject),
				PeerStatus: peer.status, PeerTarget: peer.target, MergeBaseSHA: mb,
			})
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Source < out[j].Source })
	return out, nil
}

// PeerCommitsTouching lists upstream commits after base that touch any of paths (files or
// directories), newest first. A red-trunk repair checks this before starting: a commit here
// may already be the fix.
func PeerCommitsTouching(ctx context.Context, run Runner, repo, base, upstream string, paths []string) ([]PeerCommit, error) {
	if run == nil {
		run = RealRunner
	}
	if len(paths) == 0 || strings.TrimSpace(base) == "" {
		return nil, nil
	}
	args := append([]string{"log", "--format=%H%x1f%s", base + ".." + upstream, "--"}, paths...)
	res := run(ctx, repo, args...)
	if res.Err != nil || res.Code != 0 {
		return nil, gitFailure("log", res)
	}
	var out []PeerCommit
	for _, line := range strings.Split(strings.TrimSpace(string(res.Stdout)), "\n") {
		if sha, subject, ok := strings.Cut(line, "\x1f"); ok {
			out = append(out, PeerCommit{SHA: strings.TrimSpace(sha), Subject: strings.TrimSpace(subject)})
		}
	}
	return out, nil
}

// RenderDuplicateFixes writes the advisory lines for fixes (nothing when empty).
func RenderDuplicateFixes(w io.Writer, upstream string, fixes []DuplicateFix) {
	if len(fixes) == 0 {
		return
	}
	fmt.Fprintf(w, "fak sync: DUPLICATE_FIX advisory: %s already changed %d source file(s) this push also renames/deletes differently:\n", upstream, len(fixes))
	for _, f := range fixes {
		fmt.Fprintf(w, "  %s: local %s, peer %s %s (%s)\n", f.Source, describeChange(f.LocalStatus, f.LocalTarget),
			shortSHA(f.PeerSHA), describeChange(f.PeerStatus, f.PeerTarget), f.PeerSubject)
	}
	fmt.Fprintln(w, "  both landing can define the same symbols twice; converge on the peer's names (or drop this fix) before pushing.")
}

func describeChange(status, target string) string {
	if status == "D" {
		return "deletes it"
	}
	return "renames it -> " + target
}

func shortSHA(sha string) string {
	if len(sha) > 12 {
		return sha[:12]
	}
	return sha
}

// parseNameStatus reads one --name-status line, keeping only renames and deletes keyed by
// their source path.
func parseNameStatus(line string) (string, nameChange, bool) {
	fields := strings.Split(strings.TrimRight(line, "\r"), "\t")
	if len(fields) < 2 || fields[0] == "" {
		return "", nameChange{}, false
	}
	switch fields[0][0] {
	case 'R':
		if len(fields) < 3 {
			return "", nameChange{}, false
		}
		return fields[1], nameChange{status: "R", target: fields[2]}, true
	case 'D':
		return fields[1], nameChange{status: "D"}, true
	}
	return "", nameChange{}, false
}

func gitFailure(verb string, res RunResult) error {
	if res.Err != nil {
		return res.Err
	}
	return fmt.Errorf("git %s exited %d: %s", verb, res.Code, strings.TrimSpace(string(res.Stderr)))
}
