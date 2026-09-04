// Package fleetreap provides bounded retention and footprint measurement for
// per-session fleet artifacts. Callers choose the directory and glob; the helper
// never follows directories or removes a file outside that explicit set.
package fleetreap

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Footprint captures cumulative disk usage, total file count, and oldest timestamp
// metrics across an evaluated set of matching session artifacts.
type Footprint struct {
	Files     int           `json:"files"`
	Bytes     int64         `json:"bytes"`
	Oldest    time.Time     `json:"oldest,omitempty"`
	OldestAge time.Duration `json:"-"`
}

// Result summarizes before-and-after footprint measurements alongside the count
// of deleted files from a completed pruning pass.
type Result struct {
	Before  Footprint
	After   Footprint
	Removed int
}

type fileRow struct {
	path string
	info os.FileInfo
}

// MeasureFootprint scans regular files in dir matching pattern and calculates their
// aggregate byte size, count, and age relative to reference timestamp now.
func MeasureFootprint(dir, pattern string, now time.Time) (Footprint, error) {
	rows, err := matchingFiles(dir, pattern)
	if err != nil {
		return Footprint{}, err
	}
	return footprint(rows, now), nil
}

// ReapByAgeCount keeps every matching regular file newer than maxAge, then caps
// survivors to maxCount newest files. Non-positive limits disable that dimension.
func ReapByAgeCount(dir, pattern string, maxAge time.Duration, maxCount int, now time.Time) (Result, error) {
	rows, err := matchingFiles(dir, pattern)
	if err != nil {
		return Result{}, err
	}
	res := Result{Before: footprint(rows, now)}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].info.ModTime().Equal(rows[j].info.ModTime()) {
			return rows[i].path > rows[j].path
		}
		return rows[i].info.ModTime().After(rows[j].info.ModTime())
	})
	keep := make([]fileRow, 0, len(rows))
	for _, row := range rows {
		if maxAge > 0 && now.Sub(row.info.ModTime()) > maxAge {
			if err := os.Remove(row.path); err != nil && !os.IsNotExist(err) {
				return res, err
			}
			res.Removed++
			continue
		}
		keep = append(keep, row)
	}
	if maxCount > 0 && len(keep) > maxCount {
		for _, row := range keep[maxCount:] {
			if err := os.Remove(row.path); err != nil && !os.IsNotExist(err) {
				return res, err
			}
			res.Removed++
		}
	}
	res.After, err = MeasureFootprint(dir, pattern, now)
	return res, err
}

// ReapByDeadOwner removes files whose basename contains a decimal pid and whose
// owner is no longer alive. Files without exactly one parseable pid are spared.
func ReapByDeadOwner(dir, pattern string, alive func(pid int) bool, now time.Time) (Result, error) {
	rows, err := matchingFiles(dir, pattern)
	if err != nil {
		return Result{}, err
	}
	res := Result{Before: footprint(rows, now)}
	for _, row := range rows {
		pid, ok := ownerPID(filepath.Base(row.path))
		if !ok || (alive != nil && alive(pid)) {
			continue
		}
		if err := os.Remove(row.path); err != nil && !os.IsNotExist(err) {
			return res, err
		}
		res.Removed++
	}
	res.After, err = MeasureFootprint(dir, pattern, now)
	return res, err
}

func matchingFiles(dir, pattern string) ([]fileRow, error) {
	if strings.TrimSpace(dir) == "" || strings.TrimSpace(pattern) == "" {
		return nil, fmt.Errorf("fleetreap: dir and pattern are required")
	}
	paths, err := filepath.Glob(filepath.Join(dir, pattern))
	if err != nil {
		return nil, err
	}
	rows := make([]fileRow, 0, len(paths))
	for _, path := range paths {
		info, err := os.Lstat(path)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, err
		}
		if !info.Mode().IsRegular() {
			continue
		}
		rows = append(rows, fileRow{path: path, info: info})
	}
	return rows, nil
}

func footprint(rows []fileRow, now time.Time) Footprint {
	var out Footprint
	for _, row := range rows {
		out.Files++
		out.Bytes += row.info.Size()
		if out.Oldest.IsZero() || row.info.ModTime().Before(out.Oldest) {
			out.Oldest = row.info.ModTime()
		}
	}
	if !out.Oldest.IsZero() && now.After(out.Oldest) {
		out.OldestAge = now.Sub(out.Oldest)
	}
	return out
}

func ownerPID(name string) (int, bool) {
	var found []int
	for _, field := range strings.FieldsFunc(name, func(r rune) bool { return r < '0' || r > '9' }) {
		n, err := strconv.Atoi(field)
		if err == nil && n > 0 {
			found = append(found, n)
		}
	}
	if len(found) != 1 {
		return 0, false
	}
	return found[0], true
}
