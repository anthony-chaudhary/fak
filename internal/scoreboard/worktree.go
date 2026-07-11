package scoreboard

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

type WorktreeStatus struct {
	IsolatedWorktrees int `json:"isolated_worktrees"`
	PoisonIncidents   int `json:"poison_incidents"`
}

// FoldWorktreeStatus reads auditor-authored git porcelain plus the wave poison
// ledger; no worker self-report enters either count.
func FoldWorktreeStatus(gitPorcelain, poisonLedger string) WorktreeStatus {
	out := WorktreeStatus{}
	scan := bufio.NewScanner(strings.NewReader(gitPorcelain))
	for scan.Scan() {
		line := strings.TrimSpace(scan.Text())
		if strings.HasPrefix(line, "worktree ") {
			path := strings.TrimSpace(strings.TrimPrefix(line, "worktree "))
			base := path
			if i := strings.LastIndexAny(base, "/\\"); i >= 0 {
				base = base[i+1:]
			}
			if strings.HasPrefix(base, "fak-worker-wt-") {
				out.IsolatedWorktrees++
			}
		}
	}
	if b, err := os.ReadFile(poisonLedger); err == nil {
		for _, line := range strings.Split(string(b), "\n") {
			if strings.Contains(line, `"tree_poisoned"`) || strings.Contains(line, `"verdict":"TREE_POISONED"`) {
				out.PoisonIncidents++
			}
		}
	}
	return out
}
func (s WorktreeStatus) Line() string {
	return strconv.Itoa(s.IsolatedWorktrees) + " isolated worktrees, " + strconv.Itoa(s.PoisonIncidents) + " poison incidents"
}
func WithWorktreeStatus(u Update, s WorktreeStatus) Update {
	u.Lines = append(u.Lines, s.Line())
	return u
}
