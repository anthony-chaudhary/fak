package main

import (
	"fmt"
	"os"
	"strings"
)

const dispatchRepoPulseEnv = "FAK_DISPATCH_REPO_PULSE"

// dispatchRepoPulseOrientation is the default-on startup orientation seam for
// detached coding agents. It fails open: launch correctness never depends on a
// git read, while a successful pulse saves the child from repeating three raw
// parent-visible tool turns.
func dispatchRepoPulseOrientation(root string) (string, repoPulseReport, error) {
	if v, ok := os.LookupEnv(dispatchRepoPulseEnv); ok {
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "0", "false", "off", "no":
			return "", repoPulseReport{}, nil
		}
	}
	r, err := runRepoPulse(root)
	if err != nil {
		return "", repoPulseReport{}, err
	}
	if r.Verdict != "PASS" {
		return "", r, fmt.Errorf("repo pulse verdict %s", r.Verdict)
	}
	return fmt.Sprintf("%s\nreceipt: inline_tokens=%d folded_tokens=%d saved_tokens=%d parent_tool_turns=%d->%d tool_turns_skipped=%d journal_rows=%d", r.Collapsed, r.InlineTokens, r.FoldedTokens, r.SavedTokens, r.ParentToolTurnsInline, r.ParentToolTurnsCollapsed, r.ToolTurnsSkipped, r.JournalRows), r, nil
}
