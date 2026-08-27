package workerworktree

import (
	"archive/tar"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
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
	ClarityDebt    int                `json:"clarity_debt"`
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
	// Freshness is a NON-REGRESSION check, mirroring the coverage terms beside it: a land must
	// never turn a fresh HEAD stale, but it is not required to repair a peer's pre-existing
	// staleness. Requiring absolute post.Fresh over-refused every isolated land whenever an
	// unrelated doc-regen left the tree stale at HEAD (before.Fresh already false), even for a
	// diff that regressed nothing. See #5359.
	freshNonRegress := post.Fresh || !before.Fresh
	// Clarity is also a non-regression gate. A clean HEAD must remain clean, while a HEAD
	// with pre-existing clarity debt may land an unrelated change provided the candidate
	// does not increase that debt. Semantic validity remains an absolute requirement.
	clarityNonRegress := post.CriticalClean || (!before.CriticalClean && post.ClarityDebt <= before.ClarityDebt)
	ok := freshNonRegress && post.SemanticValid && clarityNonRegress && post.Coverage+0.0001 >= before.Coverage && post.CoverageDebt <= before.CoverageDebt && !coverageFamilyRegressed(before.FamilyCoverage, post.FamilyCoverage)
	if !ok && post.Detail == "" {
		all.PostApply.Detail = fmt.Sprintf("clarity debt %d -> %d; coverage %.2f -> %.2f; coverage debt %d -> %d", before.ClarityDebt, post.ClarityDebt, before.Coverage, post.Coverage, before.CoverageDebt, post.CoverageDebt)
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
	// Extract the archive in-process rather than shelling out to `tar`: GNU tar
	// reads a Windows drive path (C:\...) as a remote host:path and fails with
	// "Cannot connect to C:", which reddened every disambiguation witness on
	// Windows. This mirrors the in-process extraction in conceptcatalog.CheckGitTree.
	out := filepath.Join(tmp, "tree")
	if err = extractTar(archive, out); err != nil {
		w.Detail = fmt.Sprintf("extract candidate tree: %v", err)
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
	w.ClarityDebt = inv.ClarityDebt
	w.Coverage = inv.Coverage
	w.CoverageDebt = inv.CoverageDebt
	w.FamilyCoverage = inv.FamilyCoverage
	w.Detail = inv.Detail
	return w
}

// extractTar unpacks a tar archive (as produced by `git archive --format=tar`)
// into dst using the stdlib reader, so extraction is free of the external `tar`
// binary and its platform quirks (notably GNU tar treating a Windows C:\ path as
// a remote host). Path traversal outside dst is rejected.
func extractTar(archive []byte, dst string) error {
	if err := os.MkdirAll(dst, 0755); err != nil {
		return err
	}
	tr := tar.NewReader(bytes.NewReader(archive))
	for {
		h, e := tr.Next()
		if errors.Is(e, io.EOF) {
			break
		}
		if e != nil {
			return e
		}
		clean := filepath.Clean(filepath.FromSlash(h.Name))
		if filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
			return fmt.Errorf("unsafe archive path %q", h.Name)
		}
		target := filepath.Join(dst, clean)
		if h.FileInfo().IsDir() {
			if e := os.MkdirAll(target, 0755); e != nil {
				return e
			}
			continue
		}
		if h.Typeflag != tar.TypeReg && h.Typeflag != tar.TypeRegA {
			continue
		}
		if e := os.MkdirAll(filepath.Dir(target), 0755); e != nil {
			return e
		}
		f, e := os.OpenFile(target, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, h.FileInfo().Mode())
		if e != nil {
			return e
		}
		_, copyErr := io.Copy(f, tr)
		closeErr := f.Close()
		if copyErr != nil {
			return copyErr
		}
		if closeErr != nil {
			return closeErr
		}
	}
	return nil
}

func (w *DisambiguationWitnesses) compactDetail() string { b, _ := json.Marshal(w); return string(b) }
