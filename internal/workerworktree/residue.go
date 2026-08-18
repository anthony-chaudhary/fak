package workerworktree

import (
	"archive/zip"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// ResidueItem is an unregistered worker directory classified for guarded collection.
type ResidueItem struct {
	Path     string `json:"path"`
	AgeSec   int64  `json:"age_sec"`
	Entries  int    `json:"entries"`
	Bytes    int64  `json:"bytes"`
	Eligible bool   `json:"eligible"`
	Reason   string `json:"reason"`
	Archive  string `json:"archive,omitempty"`
	Removed  bool   `json:"removed,omitempty"`
}

// ResidueOptions supplies deterministic roots and time for planning and tests.
type ResidueOptions struct {
	Repo       string
	BaseDir    string
	ArchiveDir string
	Now        time.Time
	AgeFloor   time.Duration
}

// CollectUnregisteredResidue classifies worker-root directories that no longer belong to repo's git worktree registry.
func CollectUnregisteredResidue(repo string, opts ResidueOptions) ([]ResidueItem, error) {
	if opts.Now.IsZero() {
		opts.Now = time.Now()
	}
	if opts.BaseDir == "" {
		opts.BaseDir = defaultWorkerBaseDir()
	}
	base, err := filepath.Abs(opts.BaseDir)
	if err != nil {
		return nil, err
	}
	registered, err := registeredWorktreePaths(repo)
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(base)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var out []ResidueItem
	for _, entry := range entries {
		if !entry.IsDir() || !IsWorkerWorktree(entry.Name()) {
			continue
		}
		path := filepath.Join(base, entry.Name())
		if registered[pathKey(path)] {
			continue
		}
		item := ResidueItem{Path: filepath.ToSlash(path)}
		latest, count, bytes, scanErr := residueStats(path)
		item.Entries, item.Bytes = count, bytes
		if scanErr != nil {
			item.Reason = "kept: unreadable residue: " + scanErr.Error()
			out = append(out, item)
			continue
		}
		item.AgeSec = int64(opts.Now.Sub(latest).Seconds())
		if item.AgeSec < 0 {
			item.AgeSec = 0
		}
		if _, err := os.Lstat(filepath.Join(path, ".git")); err == nil {
			item.Reason = "kept: foreign or ambiguous git checkout"
			out = append(out, item)
			continue
		} else if !os.IsNotExist(err) {
			item.Reason = "kept: cannot inspect git marker: " + err.Error()
			out = append(out, item)
			continue
		}
		if residueSidecarExists(base, entry.Name()) {
			item.Reason = "kept: owner or intent sidecar remains"
			out = append(out, item)
			continue
		}
		if opts.Now.Sub(latest) < opts.AgeFloor {
			item.Reason = fmt.Sprintf("kept: age %s is below floor %s", opts.Now.Sub(latest).Round(time.Second), opts.AgeFloor)
			out = append(out, item)
			continue
		}
		item.Eligible = true
		item.Reason = "eligible: unregistered, sidecar-free, past age floor, and not a git checkout"
		out = append(out, item)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out, nil
}

// ApplyUnregisteredResidue archives every eligible non-empty directory before verified removal.
func ApplyUnregisteredResidue(items []ResidueItem, opts ResidueOptions) ([]ResidueItem, error) {
	if opts.Now.IsZero() {
		opts.Now = time.Now()
	}
	if opts.ArchiveDir == "" {
		base := os.Getenv("LOCALAPPDATA")
		if base == "" {
			base = os.TempDir()
		}
		opts.ArchiveDir = filepath.Join(base, "Fleet", "watchdog", "worktree-archive", opts.Now.Format("2006-01-02"), "unregistered")
	}
	var first error
	for i := range items {
		if !items[i].Eligible {
			continue
		}
		path, err := filepath.Abs(filepath.FromSlash(items[i].Path))
		if err != nil {
			if first == nil {
				first = err
			}
			items[i].Reason = "failed: " + err.Error()
			continue
		}
		if opts.BaseDir != "" {
			base, _ := filepath.Abs(opts.BaseDir)
			rel, relErr := filepath.Rel(base, path)
			if relErr != nil || rel == "." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) || filepath.IsAbs(rel) {
				err = fmt.Errorf("path escapes worker root")
				if first == nil {
					first = err
				}
				items[i].Reason = "failed: " + err.Error()
				continue
			}
		}
		latest, entries, bytes, checkErr := residueStats(path)
		if checkErr != nil {
			if first == nil {
				first = checkErr
			}
			items[i].Reason = "failed: source revalidation: " + checkErr.Error()
			continue
		}
		items[i].Entries, items[i].Bytes = entries, bytes
		if opts.AgeFloor > 0 && opts.Now.Sub(latest) < opts.AgeFloor {
			items[i].Eligible = false
			items[i].Reason = "kept: source changed after planning"
			continue
		}
		if _, markerErr := os.Lstat(filepath.Join(path, ".git")); markerErr == nil || !os.IsNotExist(markerErr) {
			items[i].Eligible = false
			items[i].Reason = "kept: git marker appeared after planning"
			continue
		}
		if opts.BaseDir != "" && residueSidecarExists(opts.BaseDir, filepath.Base(path)) {
			items[i].Eligible = false
			items[i].Reason = "kept: owner or intent sidecar appeared after planning"
			continue
		}
		if opts.Repo != "" {
			registered, regErr := registeredWorktreePaths(opts.Repo)
			if regErr != nil {
				if first == nil {
					first = regErr
				}
				items[i].Reason = "failed: registry revalidation: " + regErr.Error()
				continue
			}
			if registered[pathKey(path)] {
				items[i].Eligible = false
				items[i].Reason = "kept: registered after planning"
				continue
			}
		}
		if items[i].Entries > 0 {
			if err = os.MkdirAll(opts.ArchiveDir, 0700); err == nil {
				archive := filepath.Join(opts.ArchiveDir, filepath.Base(path)+".zip")
				err = zipDirectory(path, archive)
				if err == nil {
					if st, e := os.Stat(archive); e != nil || st.Size() == 0 {
						err = fmt.Errorf("archive verification failed")
					} else {
						items[i].Archive = filepath.ToSlash(archive)
					}
				}
			}
			if err != nil {
				if first == nil {
					first = err
				}
				items[i].Reason = "failed: archive: " + err.Error()
				continue
			}
		}
		if err = os.RemoveAll(path); err == nil {
			if _, e := os.Stat(path); !os.IsNotExist(e) {
				err = fmt.Errorf("source still exists after removal")
			}
		}
		if err != nil {
			if first == nil {
				first = err
			}
			items[i].Reason = "failed: remove: " + err.Error()
			continue
		}
		items[i].Removed = true
		items[i].Reason = "removed: absence verified"

		if items[i].Entries > 0 {
			items[i].Reason = "removed: archive and absence verified"
		}
	}
	return items, first
}

func registeredWorktreePaths(repo string) (map[string]bool, error) {
	cmd := exec.Command("git", "worktree", "list", "--porcelain")
	cmd.Dir = repo
	configureDispatchHelperCommand(cmd)
	b, err := cmd.CombinedOutput()
	out := string(b)
	if err != nil {
		return nil, err
	}
	m := map[string]bool{}
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, "worktree ") {
			m[pathKey(strings.TrimPrefix(line, "worktree "))] = true
		}
	}
	return m, nil
}

func defaultWorkerBaseDir() string {
	base := os.Getenv("LOCALAPPDATA")
	if base == "" {
		base = os.TempDir()
	}
	return filepath.Join(base, "Fleet", "worker-worktrees")
}
func pathKey(p string) string { a, _ := filepath.Abs(p); return strings.ToLower(filepath.Clean(a)) }
func residueSidecarExists(base, name string) bool {
	for _, d := range []string{ownerStateDir, ".fak-worker-intents"} {
		if _, e := os.Stat(filepath.Join(base, d, name+".json")); e == nil || !os.IsNotExist(e) {
			return true
		}
	}
	return false
}
func residueStats(root string) (time.Time, int, int64, error) {
	st, err := os.Stat(root)
	if err != nil {
		return time.Time{}, 0, 0, err
	}
	latest := st.ModTime()
	count := 0
	var bytes int64
	err = filepath.Walk(root, func(_ string, info os.FileInfo, e error) error {
		if e != nil {
			return e
		}
		if info.ModTime().After(latest) {
			latest = info.ModTime()
		}
		if !info.IsDir() {
			count++
			bytes += info.Size()
		}
		return nil
	})
	return latest, count, bytes, err
}
func zipDirectory(root, dst string) error {
	tmp := dst + ".tmp"
	_ = os.Remove(tmp)
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	zw := zip.NewWriter(f)
	walkErr := filepath.Walk(root, func(path string, info os.FileInfo, e error) error {
		if e != nil {
			return e
		}
		if info.IsDir() {
			return nil
		}
		rel, e := filepath.Rel(root, path)
		if e != nil {
			return e
		}
		h, e := zip.FileInfoHeader(info)
		if e != nil {
			return e
		}
		h.Name = filepath.ToSlash(rel)
		h.Method = zip.Deflate
		w, e := zw.CreateHeader(h)
		if e != nil {
			return e
		}
		src, e := os.Open(path)
		if e != nil {
			return e
		}
		_, copyErr := io.Copy(w, src)
		closeErr := src.Close()
		if copyErr != nil {
			return copyErr
		}
		return closeErr
	})
	closeErr := zw.Close()
	fileErr := f.Close()
	if walkErr != nil {
		_ = os.Remove(tmp)
		return walkErr
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return closeErr
	}
	if fileErr != nil {
		_ = os.Remove(tmp)
		return fileErr
	}
	if err = os.Rename(tmp, dst); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	receipt, _ := json.Marshal(map[string]any{"schema": "fak-worker-residue-archive/1", "source": filepath.ToSlash(root), "archive": filepath.ToSlash(dst)})
	return os.WriteFile(dst+".json", append(receipt, '\n'), 0600)
}
