package main

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

func renderReleaseShip(stdout, stderr io.Writer, result releaseShipResult) {
	if result.OK {
		fmt.Fprintf(stdout, "release-ship: OK")
		if result.Tag != "" {
			fmt.Fprintf(stdout, " %s", result.Tag)
		}
		fmt.Fprintln(stdout)
	} else {
		fmt.Fprintf(stderr, "release-ship: REFUSED")
		if result.Tag != "" {
			fmt.Fprintf(stderr, " %s", result.Tag)
		}
		fmt.Fprintln(stderr)
	}
	if result.Worktree != "" {
		fmt.Fprintf(stdout, "  worktree: %s\n", result.Worktree)
	}
	if result.CommitSHA != "" {
		fmt.Fprintf(stdout, "  commit: %s\n", result.CommitSHA)
	}
	if result.SourceSHA != "" {
		status := ""
		if result.SourceCI != nil {
			status = stringFromAny(result.SourceCI["status"])
		}
		if status != "" {
			fmt.Fprintf(stdout, "  source: %s %s (ci=%s)\n", result.SourceBranch, result.SourceSHA, status)
		} else {
			fmt.Fprintf(stdout, "  source: %s %s\n", result.SourceBranch, result.SourceSHA)
		}
	}
	if result.TargetSHA != "" {
		status := ""
		if result.TargetAncestry != nil {
			status = stringFromAny(result.TargetAncestry["status"])
		}
		if status != "" {
			fmt.Fprintf(stdout, "  target: %s %s (ancestry=%s)\n", result.TargetBranch, result.TargetSHA, status)
		} else {
			fmt.Fprintf(stdout, "  target: %s %s\n", result.TargetBranch, result.TargetSHA)
		}
	}
	if result.RemoteBranch != nil {
		lease := ""
		if result.RemoteBranchPush != nil {
			if expected := stringFromAny(result.RemoteBranchPush["lease_expected_sha"]); expected != "" {
				lease = " (force-with-lease: " + expected + ")"
			}
		}
		fmt.Fprintf(stdout, "  pushed: %s/%s %s%s\n", result.RemoteBranch["remote"], result.RemoteBranch["trunk"], result.RemoteBranch["sha"], lease)
	}
	if n := len(result.PushRetries); n > 0 {
		fmt.Fprintf(stdout, "  push retries: %d (re-cut onto advanced trunk)\n", n)
	}
	if result.PromotionBranchPush != nil {
		branch := stringFromAny(result.PromotionBranchPush["branch"])
		sha := stringFromAny(result.PromotionBranchPush["sha"])
		if branch != "" && sha != "" {
			lease := ""
			if used, _ := result.PromotionBranchPush["used_force_lease"].(bool); used {
				if absent, _ := result.PromotionBranchPush["lease_expected_absent"].(bool); absent {
					lease = " (force-with-lease: absent)"
				} else if expected := stringFromAny(result.PromotionBranchPush["lease_expected_sha"]); expected != "" {
					lease = " (force-with-lease: " + expected + ")"
				} else {
					lease = " (force-with-lease)"
				}
			}
			fmt.Fprintf(stdout, "  promotion branch: %s %s%s\n", branch, sha, lease)
		}
	}
	if result.Publish != nil {
		if gh, ok := result.Publish["github_release"].(map[string]any); ok {
			if url := stringFromAny(gh["url"]); url != "" {
				fmt.Fprintf(stdout, "  github: %s\n", url)
			} else if status := stringFromAny(gh["status"]); status != "" {
				fmt.Fprintf(stdout, "  github: %s\n", status)
			}
		}
	}
	if result.PullRequest != nil {
		if url := stringFromAny(result.PullRequest["url"]); url != "" {
			fmt.Fprintf(stdout, "  pr: %s\n", url)
		} else if title := stringFromAny(result.PullRequest["title"]); title != "" {
			fmt.Fprintf(stdout, "  pr (preview): %s -> %s %q\n", stringFromAny(result.PullRequest["head"]), stringFromAny(result.PullRequest["base"]), title)
		}
	}
	for _, warning := range result.Warnings {
		fmt.Fprintf(stderr, "  warning: %s\n", warning)
	}
	for _, err := range result.Errors {
		fmt.Fprintf(stderr, "  ERROR: %s\n", err)
	}
}

func stringFromAny(value any) string {
	if s, ok := value.(string); ok {
		return s
	}
	return ""
}

func sameSHA(a, b string) bool {
	a = strings.TrimSpace(strings.ToLower(a))
	b = strings.TrimSpace(strings.ToLower(b))
	return a != "" && b != "" && (a == b || strings.HasPrefix(a, b) || strings.HasPrefix(b, a))
}

func jsonTail(value map[string]any) string {
	raw, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprint(value)
	}
	return tail(string(raw))
}

func tail(s string) string {
	s = strings.TrimSpace(s)
	if len(s) <= 500 {
		return s
	}
	return s[len(s)-500:]
}
