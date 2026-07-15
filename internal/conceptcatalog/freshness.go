package conceptcatalog

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
	"runtime"
	"sort"
	"strings"
)

const (
	GeneratedReadme   = "docs/concept-disambiguation-scorecard/README.md"
	RegenerateCommand = "python tools/concept_disambiguation_scorecard.py --markdown-dir docs/concept-disambiguation-scorecard"
)

// FreshnessResult describes deterministic generated-artifact freshness.
type FreshnessResult struct {
	Fresh      bool     `json:"fresh"`
	StalePaths []string `json:"stale_paths,omitempty"`
	Regenerate string   `json:"regenerate"`
}
type InvariantResult struct {
	Freshness      FreshnessResult    `json:"freshness"`
	SemanticValid  bool               `json:"semantic_valid"`
	CriticalClean  bool               `json:"critical_clean"`
	ClarityDebt    int                `json:"clarity_debt"`
	Coverage       float64            `json:"coverage"`
	CoverageDebt   int                `json:"coverage_debt"`
	FamilyCoverage map[string]float64 `json:"family_coverage,omitempty"`
	Detail         string             `json:"detail,omitempty"`
}

// RelevantPath reports whether a change can affect the disambiguation snapshot.
func RelevantPath(path string) bool {
	p := filepath.ToSlash(path)
	return p == "tools/concept_disambiguation_scorecard.py" ||
		p == "docs/fak/concept-glossary.md" ||
		p == GeneratedReadme ||
		strings.HasPrefix(p, "tools/concept_disambiguation_scorecard.data/")
}

// CheckFresh regenerates in scratch space and compares every tracked generated artifact.
func CheckFresh(root string) (FreshnessResult, error) {
	out, err := os.MkdirTemp("", "fak-concept-fresh-*")
	if err != nil {
		return FreshnessResult{}, err
	}
	defer os.RemoveAll(out)
	generated := filepath.Join(out, "generated")
	if err := generate(root, generated); err != nil {
		return FreshnessResult{}, err
	}
	result := FreshnessResult{Fresh: true, Regenerate: RegenerateCommand}
	expected, err := os.ReadFile(filepath.Join(generated, "README.md"))
	if err != nil {
		return result, fmt.Errorf("read generated README: %w", err)
	}
	actual, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(GeneratedReadme)))
	if err != nil || !bytes.Equal(actual, expected) {
		result.Fresh = false
		result.StalePaths = append(result.StalePaths, GeneratedReadme)
	}
	sort.Strings(result.StalePaths)
	return result, nil
}

// CheckGitTree checks a committed or staged git tree, immune to peer working-tree files.
// An empty treeish means the current index (git write-tree).
func CheckGitTree(root, treeish string) (FreshnessResult, error) {
	if treeish == "" {
		b, err := git(root, "write-tree")
		if err != nil {
			return FreshnessResult{}, err
		}
		treeish = strings.TrimSpace(string(b))
	}
	tmp, err := os.MkdirTemp("", "fak-concept-tree-*")
	if err != nil {
		return FreshnessResult{}, err
	}
	defer os.RemoveAll(tmp)
	cmd := exec.Command("git", "archive", "--format=tar", treeish)
	cmd.Dir = root
	raw, err := cmd.Output()
	if err != nil {
		return FreshnessResult{}, fmt.Errorf("git archive %s: %w", treeish, err)
	}
	tr := tar.NewReader(bytes.NewReader(raw))
	for {
		h, e := tr.Next()
		if errors.Is(e, io.EOF) {
			break
		}
		if e != nil {
			return FreshnessResult{}, e
		}
		clean := filepath.Clean(filepath.FromSlash(h.Name))
		if filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
			return FreshnessResult{}, fmt.Errorf("unsafe archive path %q", h.Name)
		}
		dst := filepath.Join(tmp, clean)
		if h.FileInfo().IsDir() {
			if e := os.MkdirAll(dst, 0755); e != nil {
				return FreshnessResult{}, e
			}
			continue
		}
		if h.Typeflag != tar.TypeReg && h.Typeflag != tar.TypeRegA {
			continue
		}
		if e := os.MkdirAll(filepath.Dir(dst), 0755); e != nil {
			return FreshnessResult{}, e
		}
		f, e := os.OpenFile(dst, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, h.FileInfo().Mode())
		if e != nil {
			return FreshnessResult{}, e
		}
		_, copyErr := io.Copy(f, tr)
		closeErr := f.Close()
		if copyErr != nil {
			return FreshnessResult{}, copyErr
		}
		if closeErr != nil {
			return FreshnessResult{}, closeErr
		}
	}
	return CheckFresh(tmp)
}

func generate(root, out string) error {
	script := filepath.Join(root, "tools", "concept_disambiguation_scorecard.py")
	python := "python3"
	if runtime.GOOS == "windows" {
		python = "python"
	}
	cmd := exec.Command(python, script, "--workspace", root, "--markdown-dir", out)
	cmd.Dir = root
	var stderr bytes.Buffer
	cmd.Stdout = io.Discard
	cmd.Stderr = &stderr
	err := cmd.Run()
	// The scorecard intentionally exits 1 when its current verdict is ACTION;
	// generation itself succeeded if the artifact exists.
	if err != nil {
		if _, statErr := os.Stat(filepath.Join(out, "README.md")); statErr != nil {
			return fmt.Errorf("generate scorecard: %w: %s", err, strings.TrimSpace(stderr.String()))
		}
	}
	return nil
}

// CheckInvariant validates freshness, semantic catalog structure and scorecard critical state.
func CheckInvariant(root string) (InvariantResult, error) {
	fresh, err := CheckFresh(root)
	if err != nil {
		return InvariantResult{}, err
	}
	inv := InvariantResult{Freshness: fresh, SemanticValid: true, CriticalClean: true, FamilyCoverage: map[string]float64{}}
	cat, err := Load(root)
	if err != nil {
		inv.SemanticValid = false
		inv.Detail = err.Error()
	} else if ds := Validate(cat); len(ds) > 0 {
		inv.SemanticValid = false
		b, _ := json.Marshal(ds)
		inv.Detail = string(b)
	}
	tmp, err := os.CreateTemp("", "fak-concept-score-*.json")
	if err != nil {
		return inv, err
	}
	name := tmp.Name()
	tmp.Close()
	defer os.Remove(name)
	script := filepath.Join(root, "tools", "concept_disambiguation_scorecard.py")
	python := "python3"
	if runtime.GOOS == "windows" {
		python = "python"
	}
	cmd := exec.Command(python, script, "--workspace", root, "--json")
	cmd.Dir = root
	out, runErr := cmd.Output()
	if runErr != nil {
		if _, ok := runErr.(*exec.ExitError); !ok {
			return inv, runErr
		}
	}
	var payload struct {
		OK     bool   `json:"ok"`
		Reason string `json:"reason"`
		Corpus struct {
			CoverageDebt int `json:"coverage_debt"`
			ClarityDebt  int `json:"clarity_defects"`
			Coverage     struct {
				CoveragePct float64 `json:"coverage_pct"`
				PerFamily   []struct {
					Family     string `json:"family"`
					Discovered int    `json:"discovered"`
					Covered    int    `json:"covered"`
				} `json:"per_family"`
			} `json:"coverage"`
		} `json:"corpus"`
	}
	if err := json.Unmarshal(out, &payload); err != nil {
		return inv, fmt.Errorf("decode scorecard: %w", err)
	}
	inv.ClarityDebt = payload.Corpus.ClarityDebt
	inv.CriticalClean = inv.SemanticValid && inv.ClarityDebt == 0
	inv.Coverage = payload.Corpus.Coverage.CoveragePct
	inv.CoverageDebt = payload.Corpus.CoverageDebt
	for _, f := range payload.Corpus.Coverage.PerFamily {
		if f.Discovered > 0 {
			inv.FamilyCoverage[f.Family] = 100 * float64(f.Covered) / float64(f.Discovered)
		}
	}
	if !payload.OK && inv.Detail == "" {
		inv.Detail = payload.Reason
	}
	return inv, nil
}

func git(root string, args ...string) ([]byte, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	b, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(string(b)))
	}
	return b, nil
}

func (r FreshnessResult) JSON() []byte { b, _ := json.Marshal(r); return b }
