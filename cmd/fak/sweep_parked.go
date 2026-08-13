package main

import (
	"fmt"
	"io"
	"sort"
	"strings"
)

type sweepParkedSummary struct {
	Count       int               `json:"count"`
	Stashes     []sweepParkedItem `json:"stashes,omitempty"`
	Refs        []sweepParkedItem `json:"unmerged_refs,omitempty"`
	Worktrees   []sweepParkedItem `json:"other_worktrees,omitempty"`
	Diagnostics []string          `json:"diagnostics,omitempty"`
}
type sweepParkedItem struct {
	Kind    string `json:"kind"`
	Name    string `json:"name"`
	Summary string `json:"summary,omitempty"`
}

func collectSweepParked(root string) sweepParkedSummary {
	var out sweepParkedSummary
	if raw, err := gitOutput(root, "stash", "list", "--format=%gd%x09%s"); err != nil {
		out.Diagnostics = append(out.Diagnostics, "stash inventory: "+err.Error())
	} else {
		for _, line := range nonemptyLines(raw) {
			name, summary, _ := strings.Cut(line, "\t")
			out.Stashes = append(out.Stashes, sweepParkedItem{"stash", name, summary})
		}
	}
	if raw, err := gitOutput(root, "for-each-ref", "--format=%(refname)%09%(objectname:short)%09%(subject)", "refs/heads", "refs/remotes"); err != nil {
		out.Diagnostics = append(out.Diagnostics, "ref inventory: "+err.Error())
	} else {
		for _, line := range nonemptyLines(raw) {
			parts := strings.SplitN(line, "\t", 3)
			if len(parts) < 2 || parts[0] == "refs/heads/main" || strings.HasSuffix(parts[0], "/HEAD") || strings.HasSuffix(parts[0], "/main") {
				continue
			}
			if _, err := gitOutput(root, "merge-base", "--is-ancestor", parts[0], "main"); err == nil {
				continue
			}
			summary := parts[1]
			if len(parts) == 3 && parts[2] != "" {
				summary += " " + parts[2]
			}
			out.Refs = append(out.Refs, sweepParkedItem{"ref", parts[0], summary})
		}
	}
	if raw, err := gitOutput(root, "worktree", "list", "--porcelain"); err != nil {
		out.Diagnostics = append(out.Diagnostics, "worktree inventory: "+err.Error())
	} else {
		blocks := strings.Split(strings.TrimSpace(raw), "\n\n")
		for i, block := range blocks {
			if i == 0 || strings.TrimSpace(block) == "" {
				continue
			}
			var path, head string
			for _, line := range strings.Split(block, "\n") {
				if strings.HasPrefix(line, "worktree ") {
					path = strings.TrimPrefix(line, "worktree ")
				}
				if strings.HasPrefix(line, "HEAD ") {
					head = strings.TrimPrefix(line, "HEAD ")
				}
			}
			out.Worktrees = append(out.Worktrees, sweepParkedItem{"worktree", path, head})
		}
	}
	sort.Slice(out.Refs, func(i, j int) bool { return out.Refs[i].Name < out.Refs[j].Name })
	sort.Slice(out.Worktrees, func(i, j int) bool { return out.Worktrees[i].Name < out.Worktrees[j].Name })
	out.Count = len(out.Stashes) + len(out.Refs) + len(out.Worktrees)
	return out
}
func nonemptyLines(raw string) []string {
	var lines []string
	for _, line := range strings.Split(strings.ReplaceAll(raw, "\r\n", "\n"), "\n") {
		if line = strings.TrimSpace(line); line != "" {
			lines = append(lines, line)
		}
	}
	return lines
}
func writeSweepParkedText(b io.Writer, parked sweepParkedSummary) {
	fmt.Fprintf(b, "\nPARKED / HIDDEN WORK  %d item(s) outside the main working-tree diff\n", parked.Count)
	for _, group := range [][]sweepParkedItem{parked.Stashes, parked.Refs, parked.Worktrees} {
		for _, item := range group {
			fmt.Fprintf(b, "  %-8s %-28s %s\n", item.Kind, item.Name, item.Summary)
		}
	}
	for _, d := range parked.Diagnostics {
		fmt.Fprintf(b, "  warning  %s\n", d)
	}
	if parked.Count == 0 {
		io.WriteString(b, "  none detected\n")
	} else {
		io.WriteString(b, "  inspect before dropping: git stash show -p <stash> / git show <ref> / fak worktree worker list\n")
	}
}
