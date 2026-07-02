package hooks

import (
	"context"
	"fmt"
	"path"
	"strconv"
	"strings"
)

const (
	freshDeletionGateName = "FRESH_DELETION"
	freshDeletionWindow   = 50
)

// FreshDeletionFindings is the commit-msg backstop for stale-snapshot commits:
// a staged deletion of a path added in recent history is suspicious unless the
// proposed message names that path. It runs at commit-msg time because that is
// the first hook point that can see BOTH the staged deletion set and the
// proposed subject/body.
func FreshDeletionFindings(root, msg string) ([]Finding, error) {
	return freshDeletionFindingsWith(context.Background(), realRunner, root, msg, freshDeletionWindow)
}

func freshDeletionFindingsWith(ctx context.Context, run Runner, root, msg string, window int) ([]Finding, error) {
	if strings.TrimSpace(root) == "" {
		return nil, ErrCouldNotRun
	}
	out, code, err := run(ctx, root, "diff", "--cached", "--name-only", "--diff-filter=D")
	if err != nil || code != 0 {
		return nil, ErrCouldNotRun
	}
	var findings []Finding
	for _, p := range splitNonEmptyLines(out) {
		addSHA, ok := recentlyAddedPath(ctx, run, root, p, window)
		if !ok || messageMentionsDeletedPath(msg, p) {
			continue
		}
		short := addSHA
		if len(short) > 12 {
			short = short[:12]
		}
		findings = append(findings, Finding{
			Gate: freshDeletionGateName,
			File: p,
			Detail: fmt.Sprintf(
				"staged deletion removes a path added within the last %d commits (added at %s), but the commit message does not mention it; name %q in the message or restore the path before committing",
				window, short, p,
			),
		})
	}
	return findings, nil
}

func recentlyAddedPath(ctx context.Context, run Runner, root, p string, window int) (string, bool) {
	addOut, code, err := run(ctx, root, "log", "--diff-filter=A", "--format=%H", "--max-count=1", "HEAD", "--", p)
	if err != nil || code != 0 {
		return "", false
	}
	addSHA := strings.TrimSpace(addOut)
	if addSHA == "" {
		return "", false
	}
	countOut, code, err := run(ctx, root, "rev-list", "--count", addSHA+"..HEAD")
	if err != nil || code != 0 {
		return "", false
	}
	n, err := strconv.Atoi(strings.TrimSpace(countOut))
	if err != nil || n > window {
		return "", false
	}
	return addSHA, true
}

func messageMentionsDeletedPath(msg, p string) bool {
	text := normalizePathMention(msg)
	base := path.Base(p)
	stem := strings.TrimSuffix(base, path.Ext(base))
	candidates := []string{
		normalizePathMention(p),
		normalizePathMention(base),
		normalizePathMention(stem),
		normalizePathMention(stripDatedSuffix(stem)),
	}
	for _, c := range candidates {
		if c != "" && strings.Contains(text, c) {
			return true
		}
	}
	return false
}

func stripDatedSuffix(stem string) string {
	if len(stem) < len("x-2006-01-02") {
		return stem
	}
	i := len(stem) - len("-2006-01-02")
	if stem[i] != '-' {
		return stem
	}
	date := stem[i+1:]
	if len(date) == 10 && date[4] == '-' && date[7] == '-' &&
		allDigits(date[:4]) && allDigits(date[5:7]) && allDigits(date[8:]) {
		return stem[:i]
	}
	return stem
}

func allDigits(s string) bool {
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return s != ""
}

func normalizePathMention(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	repl := strings.NewReplacer("\\", "/", "_", " ", "-", " ")
	s = repl.Replace(s)
	return strings.Join(strings.Fields(s), " ")
}

func splitNonEmptyLines(s string) []string {
	var out []string
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			out = append(out, line)
		}
	}
	return out
}
