// Package guardaudit bounds repo-local guard audit journals only after an
// independently verified logvault mirror proves their bytes are durable.
package guardaudit

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"time"
)

const (
	DefaultMaxAge   = 7 * 24 * time.Hour
	DefaultMaxFiles = 1500
)

type Candidate struct {
	Path    string    `json:"path"`
	RelPath string    `json:"rel_path"`
	Size    int64     `json:"size"`
	ModTime time.Time `json:"mod_time"`
	SHA256  string    `json:"sha256"`
	Reason  string    `json:"reason"`
}

type Report struct {
	Root                  string      `json:"root"`
	Vault                 string      `json:"vault"`
	MaxAge                string      `json:"max_age"`
	MaxFiles              int         `json:"max_files"`
	Scanned               int         `json:"scanned"`
	Mirrored              int         `json:"mirrored"`
	Unmirrored            int         `json:"unmirrored"`
	Candidates            []Candidate `json:"candidates,omitempty"`
	GuardAuditPruned      int         `json:"guard_audit_pruned"`
	GuardAuditPrunedBytes int64       `json:"guard_audit_pruned_bytes"`
}

type fileInfo struct {
	path, rel string
	info      os.FileInfo
}

// Plan selects age-expired files and then the oldest files above maxFiles.
// A file is eligible only when its current source hash equals a digest returned
// by logvault's independently verified manifest+mirror read-back.
func Plan(repoRoot, vault string, now time.Time, maxAge time.Duration, maxFiles int, witnessed map[string]string) (Report, error) {
	root := filepath.Join(repoRoot, ".dispatch-runs", "guard-audit")
	rep := Report{Root: root, Vault: vault, MaxAge: maxAge.String(), MaxFiles: maxFiles}
	entries, err := os.ReadDir(root)
	if os.IsNotExist(err) {
		return rep, nil
	}
	if err != nil {
		return rep, err
	}
	var files []fileInfo
	for _, ent := range entries {
		if ent.IsDir() || filepath.Ext(ent.Name()) != ".jsonl" {
			continue
		}
		info, err := ent.Info()
		if err != nil {
			return rep, err
		}
		rel := filepath.ToSlash(filepath.Join("guard-audit", ent.Name()))
		files = append(files, fileInfo{filepath.Join(root, ent.Name()), rel, info})
	}
	sort.Slice(files, func(i, j int) bool {
		if files[i].info.ModTime().Equal(files[j].info.ModTime()) {
			return files[i].rel < files[j].rel
		}
		return files[i].info.ModTime().Before(files[j].info.ModTime())
	})
	rep.Scanned = len(files)
	selected := make(map[string]string)
	if maxAge > 0 {
		cutoff := now.Add(-maxAge)
		for _, f := range files {
			if f.info.ModTime().Before(cutoff) {
				selected[f.path] = "age"
			}
		}
	}
	if maxFiles >= 0 && len(files) > maxFiles {
		for _, f := range files[:len(files)-maxFiles] {
			if _, ok := selected[f.path]; !ok {
				selected[f.path] = "count"
			}
		}
	}
	for _, f := range files {
		expected, ok := witnessed[f.rel]
		if ok {
			rep.Mirrored++
		} else {
			rep.Unmirrored++
			continue
		}
		reason, selected := selected[f.path]
		if !selected {
			continue
		}
		got, err := hashPath(f.path)
		if err != nil {
			return rep, err
		}
		if got != expected {
			// A source changed after capture: retain it until a later capture.
			rep.Unmirrored++
			rep.Mirrored--
			continue
		}
		rep.Candidates = append(rep.Candidates, Candidate{f.path, f.rel, f.info.Size(), f.info.ModTime(), got, reason})
	}
	return rep, nil
}

// Apply re-hashes each candidate immediately before removing it. A changed file
// fails closed and remains on disk.
func Apply(rep *Report) error {
	for _, c := range rep.Candidates {
		got, err := hashPath(c.Path)
		if err != nil {
			return err
		}
		if got != c.SHA256 {
			return fmt.Errorf("guard audit changed after plan: %s", c.Path)
		}
		if err := os.Remove(c.Path); err != nil {
			return err
		}
		rep.GuardAuditPruned++
		rep.GuardAuditPrunedBytes += c.Size
	}
	return nil
}

func hashPath(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
