package devcmd

import (
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
)

// gocacheFile is one cache entry: Go records a cache entry's last use as its mtime.
type gocacheFile struct {
	path  string
	size  int64
	mtime int64
}

// RunGoCacheTrim bounds a Go build cache to a byte ceiling without wiping it. Above the
// ceiling the least-recently-used files (oldest mtime first) are removed until the cache
// is at or under it, so the newest entries survive. A removed entry is a cache miss for
// Go, which rebuilds it, so a trim costs time and never changes output.
func RunGoCacheTrim(stdout, stderr io.Writer, argv []string) int {
	const usage = "usage: fak-dev gocache-trim <cache-dir> <ceiling-bytes>"
	if len(argv) != 2 {
		fmt.Fprintln(stderr, usage)
		return 2
	}
	dir := argv[0]
	ceiling, err := strconv.ParseInt(argv[1], 10, 64)
	if err != nil || ceiling < 0 {
		fmt.Fprintf(stderr, "fak-dev gocache-trim: ceiling %q is not a non-negative byte count\n", argv[1])
		return 2
	}
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		fmt.Fprintf(stdout, "go build cache: %s does not exist, nothing to bound\n", dir)
		return 0
	}

	entries, total, err := scanGoCache(dir)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev gocache-trim: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "go build cache before bound: %d bytes (ceiling %d)\n", total, ceiling)
	if total <= ceiling {
		return 0
	}

	sort.Slice(entries, func(i, j int) bool {
		if entries[i].mtime != entries[j].mtime {
			return entries[i].mtime < entries[j].mtime
		}
		return entries[i].path < entries[j].path
	})
	removed := 0
	for _, e := range entries {
		if total <= ceiling {
			break
		}
		if err := os.Remove(e.path); err != nil && !os.IsNotExist(err) {
			fmt.Fprintf(stderr, "fak-dev gocache-trim: remove %s: %v\n", e.path, err)
			return 1
		}
		total -= e.size
		removed++
	}

	_, afterBytes, err := scanGoCache(dir)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev gocache-trim: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "::warning::go build cache trimmed: removed %d least-recently-used files, now %d bytes (ceiling %d)\n", removed, afterBytes, ceiling)
	if afterBytes > ceiling {
		fmt.Fprintf(stderr, "go build cache is still %d bytes after trimming, above the %d-byte ceiling\n", afterBytes, ceiling)
		return 1
	}
	return 0
}

// scanGoCache lists every regular file under dir with its size and mtime, and their total.
func scanGoCache(dir string) ([]gocacheFile, int64, error) {
	var entries []gocacheFile
	var total int64
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return nil
		}
		entries = append(entries, gocacheFile{path: path, size: info.Size(), mtime: info.ModTime().UnixNano()})
		total += info.Size()
		return nil
	})
	return entries, total, err
}
