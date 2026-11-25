package launchvisual

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// Report is the outcome of Verify: the SVGs checked, those that differ from their render,
// and the doc violations.
type Report struct {
	Checked    []string
	Drift      []string
	Violations []Violation
}

// Verify renders every chart from the data file and index read through read, then
// compares each committed SVG byte for byte and runs DocCheck.
func Verify(read ReadFunc, dataRel, indexRel string) (Report, error) {
	rendered, cf, ix, err := Render(read, dataRel, indexRel)
	if err != nil {
		return Report{}, err
	}
	var rep Report
	for _, r := range rendered {
		rep.Checked = append(rep.Checked, r.Rel)
		got, err := read(r.Rel)
		if err != nil || !bytes.Equal(got, r.SVG) {
			rep.Drift = append(rep.Drift, r.Rel)
		}
	}
	vs, err := DocCheck(read, cf, ix)
	if err != nil {
		return rep, err
	}
	rep.Violations = vs
	return rep, nil
}

// Err folds a report into one error, or nil when the report is clean.
func (r Report) Err() error {
	var parts []string
	if len(r.Drift) > 0 {
		parts = append(parts, fmt.Sprintf("launch visual drift: %s differ from their render; run `go run ./cmd/fak launch-visual` and stage the SVGs", strings.Join(r.Drift, ", ")))
	}
	if len(r.Violations) > 0 {
		lines := make([]string, 0, len(r.Violations))
		for _, v := range r.Violations {
			lines = append(lines, fmt.Sprintf("%s:%d: %s: %s", v.File, v.Line, v.Token, v.Reason))
		}
		parts = append(parts, fmt.Sprintf("launch doc check: %d violation(s)\n  %s", len(lines), strings.Join(lines, "\n  ")))
	}
	if len(parts) == 0 {
		return nil
	}
	return errors.New(strings.Join(parts, "\n"))
}

// Generate writes the charts next to the data file (write mode), or verifies the committed
// SVGs and the docs (check mode). It returns the absolute paths it wrote or checked.
func Generate(dataPath, indexPath string, check bool) ([]string, error) {
	absData, err := filepath.Abs(dataPath)
	if err != nil {
		return nil, err
	}
	absIndex, err := filepath.Abs(indexPath)
	if err != nil {
		return nil, err
	}
	root, err := FindRoot(filepath.Dir(absData))
	if err != nil {
		return nil, err
	}
	dataRel, err := relSlash(root, absData)
	if err != nil {
		return nil, err
	}
	read, indexRel, err := indexReader(DiskReader(root), root, absIndex)
	if err != nil {
		return nil, err
	}
	if !check {
		rendered, _, _, err := Render(read, dataRel, indexRel)
		if err != nil {
			return nil, err
		}
		var written []string
		for _, r := range rendered {
			abs := filepath.Join(root, filepath.FromSlash(r.Rel))
			if err := os.WriteFile(abs, r.SVG, 0o644); err != nil {
				return written, err
			}
			written = append(written, abs)
		}
		return written, nil
	}
	rep, err := Verify(read, dataRel, indexRel)
	if err != nil {
		return nil, err
	}
	paths := make([]string, 0, len(rep.Checked))
	for _, rel := range rep.Checked {
		paths = append(paths, filepath.Join(root, filepath.FromSlash(rel)))
	}
	return paths, rep.Err()
}

// FindRoot returns the nearest ancestor of dir that holds a go.mod.
func FindRoot(dir string) (string, error) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	for d := abs; ; d = filepath.Dir(d) {
		if _, err := os.Stat(filepath.Join(d, "go.mod")); err == nil {
			return d, nil
		}
		if filepath.Dir(d) == d {
			return "", fmt.Errorf("no go.mod above %s", dir)
		}
	}
}

// DiskReader reads repo-relative paths from the working tree under root.
func DiskReader(root string) ReadFunc {
	return func(rel string) ([]byte, error) {
		return os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
	}
}

// GitTreeReader reads repo-relative paths from a git tree: the staged index when treeish is
// empty (written out with write-tree, so the scored tree is named), otherwise treeish itself.
// It returns the tree that was read so a refusal can name it.
func GitTreeReader(root, treeish string) (ReadFunc, string, error) {
	if treeish == "" {
		out, err := exec.Command("git", "-C", root, "write-tree").Output()
		if err != nil {
			return nil, "", fmt.Errorf("git write-tree: %w", err)
		}
		treeish = strings.TrimSpace(string(out))
	}
	read := func(rel string) ([]byte, error) {
		return exec.Command("git", "-C", root, "show", treeish+":"+rel).Output()
	}
	return read, treeish, nil
}

// indexReader serves the index from its repo-relative path when it lies under root. An index
// outside the repo (a scratch copy under test) is read from disk and served at DefaultIndex,
// so the checks run unchanged against it.
func indexReader(read ReadFunc, root, absIndex string) (ReadFunc, string, error) {
	rel, err := relSlash(root, absIndex)
	if err == nil {
		return read, rel, nil
	}
	b, err := os.ReadFile(absIndex)
	if err != nil {
		return nil, "", err
	}
	return func(p string) ([]byte, error) {
		if p == DefaultIndex {
			return b, nil
		}
		return read(p)
	}, DefaultIndex, nil
}

func relSlash(root, abs string) (string, error) {
	rel, err := filepath.Rel(root, abs)
	if err != nil {
		return "", err
	}
	rel = filepath.ToSlash(rel)
	if rel == ".." || strings.HasPrefix(rel, "../") {
		return "", fmt.Errorf("%s is outside the repo root %s", abs, root)
	}
	return rel, nil
}
