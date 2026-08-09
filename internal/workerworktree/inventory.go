package workerworktree

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const inventorySchema = "fak-worker-worktree-intent/1"

type Intent struct {
	Schema  string   `json:"schema"`
	Path    string   `json:"path"`
	BaseSHA string   `json:"base_sha"`
	Message string   `json:"message,omitempty"`
	Paths   []string `json:"paths,omitempty"`
}

type InventoryRow struct {
	Path          string   `json:"path"`
	BaseSHA       string   `json:"base_sha,omitempty"`
	HeadSHA       string   `json:"head_sha,omitempty"`
	DirtyPaths    []string `json:"dirty_paths,omitempty"`
	State         string   `json:"state"`
	NeedsOperator bool     `json:"needs_operator,omitempty"`
	Reason        string   `json:"reason,omitempty"`
	LandArgv      []string `json:"land_argv,omitempty"`
}

func intentDir(wtPath string) string {
	return filepath.Join(filepath.Dir(wtPath), ".fak-worker-intents")
}
func intentPath(wtPath string) string {
	return filepath.Join(intentDir(wtPath), filepath.Base(wtPath)+".json")
}
func messagePath(wtPath string) string {
	return filepath.Join(intentDir(wtPath), filepath.Base(wtPath)+".message")
}

// SaveIntent records operator-supplied land metadata outside the managed worktree.
// It cannot become part of the worker diff and is safe from git clean/reset.
func SaveIntent(wtPath, baseSHA, message string, paths []string) error {
	wtPath = filepath.Clean(strings.TrimSpace(wtPath))
	if wtPath == "." || wtPath == "" {
		return fmt.Errorf("worktree path is required")
	}
	clean := make([]string, 0, len(paths))
	seen := map[string]bool{}
	for _, p := range paths {
		p = strings.Trim(strings.ReplaceAll(p, "\\", "/"), "/")
		if p != "" && !seen[p] {
			seen[p] = true
			clean = append(clean, p)
		}
	}
	sort.Strings(clean)
	in := Intent{Schema: inventorySchema, Path: wtPath, BaseSHA: strings.TrimSpace(baseSHA), Message: strings.TrimSpace(message), Paths: clean}
	if err := os.MkdirAll(intentDir(wtPath), 0o700); err != nil {
		return err
	}
	b, err := json.MarshalIndent(in, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if err := os.WriteFile(intentPath(wtPath), b, 0o600); err != nil {
		return err
	}
	if in.Message != "" {
		return os.WriteFile(messagePath(wtPath), []byte(in.Message+"\n"), 0o600)
	}
	_ = os.Remove(messagePath(wtPath))
	return nil
}

func loadIntent(wtPath string) (Intent, error) {
	b, err := os.ReadFile(intentPath(wtPath))
	if err != nil {
		return Intent{}, err
	}
	var in Intent
	if err := json.Unmarshal(b, &in); err != nil {
		return Intent{}, err
	}
	if in.Schema != inventorySchema || !samePath(in.Path, wtPath) {
		return Intent{}, fmt.Errorf("invalid worker intent")
	}
	return in, nil
}

// Inventory inspects managed worktrees without writing refs, indexes, or worktree bytes.
func Inventory(root string, git GitRunner) ([]InventoryRow, error) {
	if git == nil {
		git = defaultGit
	}
	_, paths := Count(root, git)
	rows := make([]InventoryRow, 0, len(paths))
	for _, wt := range paths {
		wt = filepath.Clean(wt)
		row := InventoryRow{Path: wt, State: "NEEDS_OPERATOR", NeedsOperator: true}
		in, err := loadIntent(wt)
		if err != nil {
			row.Reason = "missing or invalid prepare metadata"
			rows = append(rows, row)
			continue
		}
		row.BaseSHA = in.BaseSHA
		if code, out := run(git, wt, []string{"rev-parse", "HEAD"}); code == 0 {
			row.HeadSHA = strings.TrimSpace(out)
		}
		code, out := run(git, wt, []string{"status", "--porcelain=v1", "--untracked-files=all"})
		if code != 0 {
			row.Reason = "cannot inspect worktree status"
			rows = append(rows, row)
			continue
		}
		for _, line := range strings.Split(strings.TrimRight(out, "\r\n"), "\n") {
			line = strings.TrimSuffix(line, "\r")
			if len(line) < 4 {
				continue
			}
			name := strings.TrimSpace(line[3:])
			if i := strings.LastIndex(name, " -> "); i >= 0 {
				name = name[i+4:]
			}
			name = strings.Trim(name, `"`)
			row.DirtyPaths = append(row.DirtyPaths, strings.ReplaceAll(name, "\\", "/"))
		}
		sort.Strings(row.DirtyPaths)
		if len(row.DirtyPaths) == 0 {
			row.State = "CLEAN"
			row.NeedsOperator = false
			row.Reason = ""
			rows = append(rows, row)
			continue
		}
		if in.BaseSHA == "" || in.Message == "" || len(in.Paths) == 0 {
			row.Reason = "prepare metadata lacks base SHA, message, or intended paths"
			rows = append(rows, row)
			continue
		}
		allowed := map[string]bool{}
		for _, p := range in.Paths {
			allowed[p] = true
		}
		ambiguous := false
		for _, p := range row.DirtyPaths {
			if !allowed[p] {
				ambiguous = true
			}
		}
		if ambiguous {
			row.Reason = "dirty paths exceed explicit intended paths"
			rows = append(rows, row)
			continue
		}
		row.State, row.NeedsOperator, row.Reason = "LAND_READY", false, ""
		row.LandArgv = []string{"fak", "worktree", "worker", "land", "--worktree", wt, "--base-sha", in.BaseSHA, "--msg-file", messagePath(wt)}
		for _, p := range in.Paths {
			row.LandArgv = append(row.LandArgv, "--paths", p)
		}
		rows = append(rows, row)
	}
	return rows, nil
}
