package treestatus

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// PathEntry describes one dirty working-tree or staged file.
type PathEntry struct {
	Path     string `json:"path"`
	Status   string `json:"status"` // "M ", " M", "A ", "??", "D ", etc.
	Staged   bool   `json:"staged"`
	Conflict bool   `json:"conflict,omitempty"`
	Lane     string `json:"lane,omitempty"`
	Owned    bool   `json:"owned"`
}

// Options configures the tree status inspection.
type Options struct {
	Lane string   `json:"lane,omitempty"`
	Mine []string `json:"mine,omitempty"`
}

// Report is the structured working tree status report (#10920).
type Report struct {
	OK                bool                `json:"ok"`
	Branch            string              `json:"branch"`
	Head              string              `json:"head"`
	MergeInProgress   bool                `json:"merge_in_progress"`
	LockFiles         []string            `json:"lock_files,omitempty"`
	HasConflicts      bool                `json:"has_conflicts"`
	ConflictPaths     []string            `json:"conflict_paths,omitempty"`
	TotalDirty        int                 `json:"total_dirty"`
	OwnedCount        int                 `json:"owned_count"`
	PeerWIPCount      int                 `json:"peer_wip_count"`
	UnclassifiedCount int                 `json:"unclassified_count"`
	OwnedPaths        []PathEntry         `json:"owned_paths,omitempty"`
	PeerWIPPaths      []PathEntry         `json:"peer_wip_paths,omitempty"`
	UnclassifiedPaths []PathEntry         `json:"unclassified_paths,omitempty"`
	LaneGroups        map[string][]string `json:"lane_groups,omitempty"`
	ElapsedMS         int64               `json:"elapsed_ms"`
}

// Collect inspects the working tree and produces a structured Report in <20ms.
func Collect(root string, opts Options) (*Report, error) {
	start := time.Now()
	root = resolveRoot(root)

	rep := &Report{
		OK:         true,
		LaneGroups: make(map[string][]string),
	}

	// 1. Resolve branch and HEAD
	branch, _ := gitOutput(root, "rev-parse", "--abbrev-ref", "HEAD")
	rep.Branch = strings.TrimSpace(branch)
	head, _ := gitOutput(root, "rev-parse", "--short", "HEAD")
	rep.Head = strings.TrimSpace(head)

	// 2. Inspect merge and lock state
	gitDir, _ := gitOutput(root, "rev-parse", "--git-dir")
	gitDir = strings.TrimSpace(gitDir)
	if gitDir == "" {
		gitDir = filepath.Join(root, ".git")
	} else if !filepath.IsAbs(gitDir) {
		gitDir = filepath.Join(root, gitDir)
	}

	if _, err := os.Stat(filepath.Join(gitDir, "MERGE_HEAD")); err == nil {
		rep.MergeInProgress = true
		rep.OK = false
	}

	for _, lockName := range []string{"index.lock", "fak-commit.lock"} {
		lockPath := filepath.Join(gitDir, lockName)
		if _, err := os.Stat(lockPath); err == nil {
			rep.LockFiles = append(rep.LockFiles, lockName)
		}
	}

	// 3. Load lane taxonomy from dos.toml
	laneMap := loadLaneTaxonomy(root)

	// 4. Query porcelain status
	statusOut, err := gitOutput(root, "status", "--porcelain=v1", "-z", "--no-renames")
	if err != nil {
		return nil, fmt.Errorf("treestatus: git status failed: %w", err)
	}

	entries := parsePorcelainEntries(statusOut)
	rep.TotalDirty = len(entries)

	// Normalize mine paths
	mineMap := make(map[string]bool)
	minePrefixes := make([]string, 0, len(opts.Mine))
	for _, m := range opts.Mine {
		clean := filepath.ToSlash(filepath.Clean(filepath.FromSlash(strings.TrimSpace(m))))
		if clean != "" && clean != "." {
			mineMap[clean] = true
			minePrefixes = append(minePrefixes, strings.TrimSuffix(clean, "/")+"/")
		}
	}

	for _, e := range entries {
		lane := matchLane(e.Path, laneMap)
		e.Lane = lane
		if lane != "" {
			rep.LaneGroups[lane] = append(rep.LaneGroups[lane], e.Path)
		}

		if e.Conflict {
			rep.HasConflicts = true
			rep.ConflictPaths = append(rep.ConflictPaths, e.Path)
			rep.OK = false
		}

		isOwned := false
		if len(opts.Mine) > 0 {
			if mineMap[e.Path] {
				isOwned = true
			} else {
				for _, pfx := range minePrefixes {
					if strings.HasPrefix(e.Path, pfx) {
						isOwned = true
						break
					}
				}
			}
		} else if opts.Lane != "" {
			isOwned = (lane != "" && lane == opts.Lane)
		}

		e.Owned = isOwned
		if isOwned {
			rep.OwnedPaths = append(rep.OwnedPaths, e)
			rep.OwnedCount++
		} else if opts.Lane != "" || len(opts.Mine) > 0 {
			rep.PeerWIPPaths = append(rep.PeerWIPPaths, e)
			rep.PeerWIPCount++
		} else {
			rep.UnclassifiedPaths = append(rep.UnclassifiedPaths, e)
			rep.UnclassifiedCount++
		}
	}

	rep.ElapsedMS = time.Since(start).Milliseconds()
	return rep, nil
}

func parsePorcelainEntries(out string) []PathEntry {
	if len(out) == 0 {
		return nil
	}
	parts := strings.Split(out, "\x00")
	var entries []PathEntry
	for _, p := range parts {
		if len(p) < 4 {
			continue
		}
		status := p[:2]
		path := filepath.ToSlash(strings.TrimSpace(p[3:]))
		if path == "" {
			continue
		}

		staged := status[0] != ' ' && status[0] != '?'
		conflict := status[0] == 'U' || status[1] == 'U' || status == "AA" || status == "DD"

		entries = append(entries, PathEntry{
			Path:     path,
			Status:   status,
			Staged:   staged,
			Conflict: conflict,
		})
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Path < entries[j].Path
	})
	return entries
}

func matchLane(path string, laneMap map[string][]string) string {
	for lane, globs := range laneMap {
		for _, g := range globs {
			if matchGlob(path, g) {
				return lane
			}
		}
	}
	return ""
}

func matchGlob(p, pattern string) bool {
	p = filepath.ToSlash(p)
	pattern = filepath.ToSlash(pattern)

	if strings.HasSuffix(pattern, "/**") {
		pfx := strings.TrimSuffix(pattern, "/**")
		return p == pfx || strings.HasPrefix(p, pfx+"/")
	}
	if strings.HasSuffix(pattern, "/*") {
		dir := filepath.Dir(p)
		expected := strings.TrimSuffix(pattern, "/*")
		return filepath.ToSlash(dir) == expected
	}
	if matched, err := filepath.Match(pattern, p); err == nil && matched {
		return true
	}
	return p == pattern
}

func loadLaneTaxonomy(root string) map[string][]string {
	data, err := os.ReadFile(filepath.Join(root, "dos.toml"))
	if err != nil {
		return nil
	}

	lines := strings.Split(string(data), "\n")
	lanes := make(map[string][]string)
	inLanes := false

	for _, line := range lines {
		trim := strings.TrimSpace(line)
		if strings.HasPrefix(trim, "[lanes.trees]") || strings.HasPrefix(trim, "[lanes]") {
			inLanes = true
			continue
		}
		if strings.HasPrefix(trim, "[") && inLanes {
			inLanes = false
			continue
		}
		if !inLanes || strings.HasPrefix(trim, "#") || !strings.Contains(trim, "=") {
			continue
		}

		parts := strings.SplitN(trim, "=", 2)
		lane := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(parts[1])
		if idx := strings.Index(val, "#"); idx >= 0 {
			val = strings.TrimSpace(val[:idx])
		}
		val = strings.Trim(val, "[] ")

		var globs []string
		for _, item := range strings.Split(val, ",") {
			item = strings.Trim(strings.TrimSpace(item), `"`)
			if item != "" {
				globs = append(globs, item)
			}
		}
		if len(globs) > 0 {
			lanes[lane] = globs
		}
	}
	return lanes
}

func gitOutput(root string, args ...string) (string, error) {
	cmdArgs := append([]string{"-C", root}, args...)
	cmd := exec.Command("git", cmdArgs...)
	windowgate.ConfigureBackgroundCommand(cmd)
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	err := cmd.Run()
	return stdout.String(), err
}

func resolveRoot(root string) string {
	if root != "" && root != "." {
		if _, err := os.Stat(filepath.Join(root, "go.mod")); err == nil {
			return root
		}
	}
	dir, err := os.Getwd()
	if err != nil {
		return root
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return root
}
