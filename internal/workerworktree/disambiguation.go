package workerworktree

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/conceptcatalog"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// DisambiguationWitness is one independently materialized view used by land.
type DisambiguationWitness struct {
	Tree           string             `json:"tree"`
	Fresh          bool               `json:"fresh"`
	SemanticValid  bool               `json:"semantic_valid"`
	CriticalClean  bool               `json:"critical_clean"`
	Coverage       float64            `json:"coverage"`
	CoverageDebt   int                `json:"coverage_debt"`
	FamilyCoverage map[string]float64 `json:"family_coverage,omitempty"`
	Detail         string             `json:"detail,omitempty"`
}
type DisambiguationWitnesses struct {
	Before    DisambiguationWitness `json:"before"`
	Worktree  DisambiguationWitness `json:"worktree"`
	PostApply DisambiguationWitness `json:"post_apply"`
}

func disambiguationRelevant(paths []string) bool {
	for _, p := range paths {
		p = filepath.ToSlash(p)
		if conceptcatalog.RelevantPath(p) {
			return true
		}
		if strings.HasPrefix(p, "internal/") || strings.HasPrefix(p, "cmd/") || strings.HasPrefix(p, "docs/") || strings.HasPrefix(p, "tools/") {
			return strings.HasSuffix(p, ".go") || strings.HasSuffix(p, ".md") || strings.HasSuffix(p, ".json")
		}

	}
	return false
}

var readDisambiguation = readDisambiguationWitness

func verifyAppliedDisambiguation(root, wtPath, treeSHA string) (*DisambiguationWitnesses, bool) {
	before := readDisambiguation(root, "HEAD")
	worktree := readDisambiguation(wtPath, "HEAD")
	post := readDisambiguation(root, treeSHA)
	all := &DisambiguationWitnesses{Before: before, Worktree: worktree, PostApply: post}
	ok := post.Fresh && post.SemanticValid && post.CriticalClean && post.Coverage+0.0001 >= before.Coverage && post.CoverageDebt <= before.CoverageDebt && !coverageFamilyRegressed(before.FamilyCoverage, post.FamilyCoverage)
	if !ok && post.Detail == "" {
		all.PostApply.Detail = fmt.Sprintf("coverage %.2f -> %.2f; coverage debt %d -> %d", before.Coverage, post.Coverage, before.CoverageDebt, post.CoverageDebt)
	}
	return all, ok
}

func coverageFamilyRegressed(before, after map[string]float64) bool {
	for family, b := range before {
		if a, ok := after[family]; ok && a+0.0001 < b {
			return true
		}
	}
	return false
}

func readDisambiguationWitness(repo, tree string) DisambiguationWitness {
	w := DisambiguationWitness{Tree: tree}
	tmp, err := os.MkdirTemp("", "fak-disambiguation-tree-*")
	if err != nil {
		w.Detail = err.Error()
		return w
	}
	defer os.RemoveAll(tmp)
	cmd := exec.Command("git", "archive", "--format=tar", tree)
	cmd.Dir = repo
	windowgate.ConfigureBackgroundCommand(cmd)
	archive, err := cmd.Output()
	if err != nil {
		w.Detail = err.Error()
		return w
	}
	tarPath := filepath.Join(tmp, "tree.tar")
	if err = os.WriteFile(tarPath, archive, 0600); err != nil {
		w.Detail = err.Error()
		return w
	}
	out := filepath.Join(tmp, "tree")
	_ = os.MkdirAll(out, 0755)
	tar := exec.Command("tar", "-xf", tarPath, "-C", out)
	windowgate.ConfigureBackgroundCommand(tar)
	if b, err := tar.CombinedOutput(); err != nil {
		w.Detail = fmt.Sprintf("extract candidate tree: %v: %s", err, b)
		return w
	}
	inv, err := conceptcatalog.CheckInvariant(out)
	if err != nil {
		w.Detail = err.Error()
		return w
	}
	w.Fresh = inv.Freshness.Fresh
	w.SemanticValid = inv.SemanticValid
	w.CriticalClean = inv.CriticalClean
	w.Coverage = inv.Coverage
	w.CoverageDebt = inv.CoverageDebt
	w.FamilyCoverage = inv.FamilyCoverage
	w.Detail = inv.Detail
	return w
}

func (w *DisambiguationWitnesses) compactDetail() string { b, _ := json.Marshal(w); return string(b) }
