package gitindex

import (
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sync"
	"sync/atomic"
)

// dirStats caches lstat data gathered by enumerating each directory that
// holds tracked entries once, instead of one lstat per entry. On Windows a
// directory read (FindFirstFile/FindNextFile) returns size, mtime and
// attributes for every child in bulk — the same trick Git for Windows'
// core.fscache uses — so 30k per-file lstat syscalls become ~3k directory
// reads. Lookups that miss (case-folded names, unreadable dirs) fall back to
// os.Lstat, so the cache can only save work, never change a verdict.
type dirStats map[string]map[string]fs.FileInfo

func (d dirStats) lstat(rel, full string) (fs.FileInfo, error) {
	dir, name := path.Split(rel)
	if m, ok := d[dir]; ok {
		if fi, ok := m[name]; ok {
			return fi, nil
		}
	}
	return os.Lstat(full)
}

func statDirs(root string, idx *Index, workers int) dirStats {
	var dirs []string
	seen := map[string]bool{}
	for i := range idx.Entries {
		e := &idx.Entries[i]
		if e.Stage != 0 || e.AssumeValid || e.SkipWorktree {
			continue
		}
		dir, _ := path.Split(e.Path)
		if !seen[dir] {
			seen[dir] = true
			dirs = append(dirs, dir)
		}
	}
	maps := make([]map[string]fs.FileInfo, len(dirs))
	parallel(len(dirs), workers, func(i int) {
		ents, err := os.ReadDir(filepath.Join(root, filepath.FromSlash(dirs[i])))
		if err != nil {
			return
		}
		m := make(map[string]fs.FileInfo, len(ents))
		for _, de := range ents {
			if fi, err := de.Info(); err == nil {
				m[de.Name()] = fi
			}
		}
		maps[i] = m
	})
	out := make(dirStats, len(dirs))
	for i, d := range dirs {
		if maps[i] != nil {
			out[d] = maps[i]
		}
	}
	return out
}

// parallel runs fn(0..n-1) across workers goroutines.
func parallel(n, workers int, fn func(int)) {
	if workers > n {
		workers = n
	}
	var next atomic.Int64
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1) - 1)
				if i >= n {
					return
				}
				fn(i)
			}
		}()
	}
	wg.Wait()
}
