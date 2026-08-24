package main

import (
	"context"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/wiprecon"
)

type sweepParkedSummary struct {
	Count       int               `json:"count"`
	Stashes     []sweepParkedItem `json:"stashes,omitempty"`
	Refs        []sweepParkedItem `json:"unmerged_refs,omitempty"`
	Checkpoints []sweepParkedItem `json:"wip_checkpoints,omitempty"`
	Worktrees   []sweepParkedItem `json:"other_worktrees,omitempty"`
	Diagnostics []string          `json:"diagnostics,omitempty"`
}
type sweepParkedItem struct {
	Kind            string   `json:"kind"`
	Name            string   `json:"name"`
	Summary         string   `json:"summary,omitempty"`
	Paths           int      `json:"paths,omitempty"`
	Action          string   `json:"action,omitempty"`
	CheckpointClass string   `json:"checkpoint_class,omitempty"`
	Replication     string   `json:"replication,omitempty"`
	NextCommand     string   `json:"next_command,omitempty"`
	ReviewCommands  []string `json:"review_commands,omitempty"`
}

func collectSweepParked(root string) sweepParkedSummary {
	var out sweepParkedSummary
	if raw, ok := sweepParkedGitOutput(root, &out, "stash inventory", "stash", "list", "--format=%gd%x09%s"); ok {
		for _, line := range nonemptyLines(raw) {
			name, summary, _ := strings.Cut(line, "\t")
			paths := 0
			if changed, countErr := gitOutput(root, "diff", "--name-only", name+"^1", name); countErr == nil {
				paths = len(nonemptyLines(changed))
			}
			out.Stashes = append(out.Stashes, sweepParkedItem{Kind: "stash", Name: name, Summary: summary, Paths: paths})
		}
	}
	if raw, ok := sweepParkedGitOutput(root, &out, "ref inventory", "for-each-ref", "--format=%(refname)%09%(objectname:short)%09%(subject)", "refs/heads", "refs/remotes"); ok {
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
			out.Refs = append(out.Refs, sweepParkedItem{Kind: "ref", Name: parts[0], Summary: summary})
		}
	}
	checkpointDecisions := map[string]wiprecon.Decision{}
	if reconciled, err := wipReconcile(context.Background(), root); err != nil {
		out.Diagnostics = append(out.Diagnostics, "WIP checkpoint classification: "+err.Error())
	} else {
		for _, decision := range reconciled.Decisions {
			checkpointDecisions[decision.Session] = decision
		}
	}
	if raw, ok := sweepParkedGitOutput(root, &out, "WIP checkpoint inventory", "for-each-ref", "--format=%(refname)%09%(objectname:short)%09%(subject)", "refs/fak/wip"); ok {
		for _, line := range nonemptyLines(raw) {
			parts := strings.SplitN(line, "\t", 3)
			if len(parts) < 2 {
				continue
			}
			summary := parts[1]
			if len(parts) == 3 && parts[2] != "" {
				summary += " " + parts[2]
			}
			item := sweepParkedItem{Kind: "checkpoint", Name: parts[0], Summary: summary}
			if decision, ok := checkpointDecisions[strings.TrimPrefix(parts[0], "refs/fak/wip/")]; ok {
				item.Action = string(decision.Action)
				item.CheckpointClass = decision.CheckpointClass
				item.Replication = decision.Replication
				item.NextCommand = decision.NextCommand
				item.ReviewCommands = append([]string(nil), decision.ReviewCommands...)
			}
			out.Checkpoints = append(out.Checkpoints, item)
		}
	}
	if raw, ok := sweepParkedGitOutput(root, &out, "worktree inventory", "worktree", "list", "--porcelain"); ok {
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
			out.Worktrees = append(out.Worktrees, sweepParkedItem{Kind: "worktree", Name: path, Summary: head})
		}
	}
	sort.Slice(out.Refs, func(i, j int) bool { return out.Refs[i].Name < out.Refs[j].Name })
	sort.Slice(out.Worktrees, func(i, j int) bool { return out.Worktrees[i].Name < out.Worktrees[j].Name })
	out.Count = len(out.Stashes) + len(out.Refs) + len(out.Checkpoints) + len(out.Worktrees)
	return out
}

func sweepParkedGitOutput(root string, out *sweepParkedSummary, diagnostic string, args ...string) (string, bool) {
	raw, err := gitOutput(root, args...)
	if err != nil {
		out.Diagnostics = append(out.Diagnostics, diagnostic+": "+err.Error())
		return "", false
	}
	return raw, true
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
	for _, group := range [][]sweepParkedItem{parked.Stashes, parked.Refs, parked.Checkpoints, parked.Worktrees} {
		for _, item := range group {
			detail := item.Summary
			if item.Paths > 0 {
				detail = fmt.Sprintf("%d changed path(s); %s", item.Paths, detail)
			}
			fmt.Fprintf(b, "  %-8s %-28s %s\n", item.Kind, item.Name, detail)
			if item.Kind == "checkpoint" {
				fmt.Fprintf(b, "           lifecycle=%s class=%s replication=%s\n", firstNonEmpty(item.Action, "unknown"), firstNonEmpty(item.CheckpointClass, "unknown"), firstNonEmpty(item.Replication, "unknown"))
				if item.NextCommand != "" {
					fmt.Fprintf(b, "           next: %s\n", item.NextCommand)
				}
				for _, command := range item.ReviewCommands {
					fmt.Fprintf(b, "           review: %s\n", command)
				}
			}
		}
	}
	for _, d := range parked.Diagnostics {
		fmt.Fprintf(b, "  warning  %s\n", d)
	}
	if parked.Count == 0 {
		io.WriteString(b, "  none detected\n")
	} else {
		io.WriteString(b, "  inspect before dropping: git stash show -p <stash> / git show <ref> / fak wip reconcile / fak wip lifecycle list / fak worktree worker list\n")
	}
}
