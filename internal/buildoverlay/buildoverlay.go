// Package buildoverlay isolates Go commands from unrelated untracked files in
// a shared, peer-dirty checkout.
package buildoverlay

import (
	"encoding/json"
	"fmt"
	"go/parser"
	"go/token"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// Overlay is the JSON shape consumed by `go build -overlay`.
type Overlay struct {
	Replace map[string]string `json:"Replace"`
}

// SelectMaskedFiles decides which untracked Go files to hide and which must be
// retained because they are explicit or sit in a modified package directory.
func SelectMaskedFiles(untracked, mine []string, modifiedDirs map[string]bool) (masked, kept, staleMine []string) {
	untrackedSet := make(map[string]bool, len(untracked))
	for _, f := range untracked {
		untrackedSet[Slash(f)] = true
	}
	mineSet := make(map[string]bool, len(mine))
	for _, m := range mine {
		m = Slash(m)
		if mineSet[m] {
			continue
		}
		mineSet[m] = true
		if !untrackedSet[m] {
			staleMine = append(staleMine, m)
		}
	}
	for _, f := range untracked {
		f = Slash(f)
		if !strings.HasSuffix(f, ".go") || mineSet[f] {
			continue
		}
		if modifiedDirs[path.Dir(f)] {
			kept = append(kept, f)
			continue
		}
		masked = append(masked, f)
	}
	sort.Strings(masked)
	sort.Strings(kept)
	sort.Strings(staleMine)
	return masked, kept, staleMine
}

// Build constructs an overlay that makes each masked file absent.
func Build(root string, masked []string) Overlay {
	replace := make(map[string]string, len(masked))
	for _, rel := range masked {
		abs := filepath.Join(root, filepath.FromSlash(rel))
		replace[filepath.Clean(abs)] = ""
	}
	return Overlay{Replace: replace}
}

// Write serializes an overlay file for the Go command.
func Write(path string, overlay Overlay) error {
	b, err := json.Marshal(overlay)
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o600)
}

// UntrackedGoFiles returns untracked, non-ignored Go source files.
func UntrackedGoFiles(root string) ([]string, error) {
	cmd := windowgate.Command("git", "-C", root, "ls-files", "--others", "--exclude-standard", "--", "*.go")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	var files []string
	for _, ln := range strings.Split(string(out), "\n") {
		ln = strings.TrimSpace(ln)
		if ln != "" {
			files = append(files, Slash(ln))
		}
	}
	sort.Strings(files)
	return files, nil
}

// ModifiedDirs returns package directories containing tracked worktree edits.
func ModifiedDirs(root string) (map[string]bool, error) {
	cmd := windowgate.Command("git", "-C", root, "diff", "--name-only", "--diff-filter=ACMR")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	dirs := map[string]bool{}
	for _, ln := range strings.Split(string(out), "\n") {
		ln = strings.TrimSpace(ln)
		if strings.HasSuffix(strings.ToLower(ln), ".go") {
			dirs[filepath.ToSlash(filepath.Dir(filepath.FromSlash(ln)))] = true
		}
	}
	return dirs, nil
}

// LoadBearingUntrackedFiles returns untracked Go files that are imported,
// directly or transitively, by tracked Go source. These files must remain
// visible when an overlay masks unrelated peer work.
func LoadBearingUntrackedFiles(root string, untracked []string) ([]string, error) {
	modulePath, err := ModulePath(filepath.Join(root, "go.mod"))
	if err != nil {
		return nil, err
	}
	untrackedByDir := map[string][]string{}
	for _, name := range untracked {
		name = Slash(filepath.Clean(name))
		if strings.HasSuffix(name, ".go") {
			dir := path.Dir(name)
			untrackedByDir[dir] = append(untrackedByDir[dir], name)
		}
	}
	if len(untrackedByDir) == 0 {
		return nil, nil
	}
	cmd := windowgate.Command("git", "ls-files", "--", "*.go")
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	queue := localImportDirs(root, modulePath, strings.Fields(string(out)))
	seen := map[string]bool{}
	var kept []string
	for len(queue) > 0 {
		dir := path.Clean(queue[0])
		queue = queue[1:]
		if seen[dir] {
			continue
		}
		seen[dir] = true
		files := untrackedByDir[dir]
		if len(files) == 0 {
			continue
		}
		kept = append(kept, files...)
		queue = append(queue, localImportDirs(root, modulePath, files)...)
	}
	sort.Strings(kept)
	return kept, nil
}

func localImportDirs(root, modulePath string, files []string) []string {
	prefix := strings.TrimSuffix(modulePath, "/") + "/"
	seen := map[string]bool{}
	var dirs []string
	for _, name := range files {
		parsed, err := parser.ParseFile(token.NewFileSet(), filepath.Join(root, filepath.FromSlash(name)), nil, parser.ImportsOnly)
		if err != nil {
			continue
		}
		for _, spec := range parsed.Imports {
			importPath, err := strconv.Unquote(spec.Path.Value)
			if err != nil || !strings.HasPrefix(importPath, prefix) {
				continue
			}
			dir := path.Clean(strings.TrimPrefix(importPath, prefix))
			if dir != "." && !seen[dir] {
				seen[dir] = true
				dirs = append(dirs, dir)
			}
		}
	}
	return dirs
}

// ModulePath parses the module directive from go.mod.
func ModulePath(goMod string) (string, error) {
	b, err := os.ReadFile(goMod)
	if err != nil {
		return "", err
	}
	for _, ln := range strings.Split(string(b), "\n") {
		fields := strings.Fields(ln)
		if len(fields) == 2 && fields[0] == "module" {
			return strings.TrimSpace(fields[1]), nil
		}
	}
	return "", fmt.Errorf("module directive not found in %s", goMod)
}

// Slash normalizes repository paths across native and Git separators.
func Slash(path string) string { return strings.ReplaceAll(filepath.ToSlash(path), "\\", "/") }
