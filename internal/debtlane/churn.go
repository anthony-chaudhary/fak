package debtlane

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"math"
	"os/exec"
	"path"
	"strings"
	"time"
)

// DefaultChurnDays is the git-history window the CLI uses to weight wave
// priority by recent change activity.
const DefaultChurnDays = 30

// maxChurnFilesPerCommit drops bulk commits (vendoring, mass renames, and the
// boundary commit of a shallow clone, which lists the whole tree) so they do
// not flatten the churn signal across every lane.
const maxChurnFilesPerCommit = 200

// maxChurnFactor caps how far recent churn can lift a lane's wave priority.
const maxChurnFactor = 3.0

// gitChurnLog is the git log hook; tests replace it to stay hermetic.
var gitChurnLog = func(root string, days int) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", "-C", root, "log",
		fmt.Sprintf("--since=%d.days", days), "--no-merges", "--name-only", "--format=@@%H")
	return cmd.Output()
}

// ParseChurnLog splits `git log --name-only --format=@@%H` output into the
// file lists of each commit, dropping bulk commits above the per-commit cap.
func ParseChurnLog(out []byte) [][]string {
	var commits [][]string
	var cur []string
	inCommit := false
	flush := func() {
		if inCommit && len(cur) > 0 && len(cur) <= maxChurnFilesPerCommit {
			commits = append(commits, cur)
		}
		cur = nil
	}
	sc := bufio.NewScanner(bytes.NewReader(out))
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "@@") {
			flush()
			inCommit = true
			continue
		}
		cur = append(cur, line)
	}
	flush()
	return commits
}

// AttachChurn sets RecentCommits on each lane to the number of commits that
// touched a file whose deepest owning unit of work is that lane's. Lanes that
// share one unit of work are each credited.
func AttachChurn(lanes []DebtLane, commits [][]string) {
	byUnit := make(map[string][]int)
	for i := range lanes {
		u := normalizeUnit(lanes[i].UnitOfWork)
		if u == "" || u == "." {
			continue
		}
		byUnit[u] = append(byUnit[u], i)
	}
	counts := make([]int, len(lanes))
	for _, files := range commits {
		touched := make(map[int]bool)
		for _, f := range files {
			for dir := path.Dir(normalizeUnit(f)); dir != "." && dir != "/" && dir != ""; dir = path.Dir(dir) {
				if idx, ok := byUnit[dir]; ok {
					for _, i := range idx {
						touched[i] = true
					}
					break
				}
			}
		}
		for i := range touched {
			counts[i]++
		}
	}
	for i := range lanes {
		lanes[i].RecentCommits = counts[i]
	}
}

// ChurnFactor maps a lane's recent commit count to a wave-priority multiplier:
// 1.0 with no churn, rising logarithmically and capped at maxChurnFactor.
// Debt in code that is actively changing is paid on every change, so it is
// retired first among lanes of comparable principal.
func ChurnFactor(commits int) float64 {
	if commits <= 0 {
		return 1.0
	}
	f := 1.0 + 0.5*math.Log2(1+float64(commits))
	if f > maxChurnFactor {
		f = maxChurnFactor
	}
	return f
}

// PriorityScore is the wave-planning priority of a lane: total debt weighted
// by recent churn. With no churn data it equals TotalDebt.
func PriorityScore(l DebtLane) float64 {
	return l.TotalDebt * ChurnFactor(l.RecentCommits)
}

func normalizeUnit(p string) string {
	p = strings.TrimSpace(strings.ReplaceAll(p, "\\", "/"))
	p = strings.TrimSuffix(p, "/**")
	p = strings.TrimSuffix(p, "/")
	return path.Clean(p)
}

// attachRepoChurn loads churn for each repository root and attaches it to the
// lanes whose Repo maps to that root. Git failures leave churn at zero.
func attachRepoChurn(lanes []DebtLane, rootFor func(repo string) string, days int) {
	if days <= 0 || len(lanes) == 0 {
		return
	}
	groups := make(map[string][]int)
	for i := range lanes {
		root := rootFor(lanes[i].Repo)
		if root != "" {
			groups[root] = append(groups[root], i)
		}
	}
	for root, idx := range groups {
		out, err := gitChurnLog(root, days)
		if err != nil {
			continue
		}
		commits := ParseChurnLog(out)
		sub := make([]DebtLane, len(idx))
		for k, i := range idx {
			sub[k] = lanes[i]
		}
		AttachChurn(sub, commits)
		for k, i := range idx {
			lanes[i].RecentCommits = sub[k].RecentCommits
		}
	}
}
