package safecommit

import (
	"context"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gitgate"
)

var datedNoteRE = regexp.MustCompile(`20\d\d-\d\d-\d\d`)

type noteIndexEntry struct {
	Path  string
	Base  string
	Date  string
	Title string
}

func autoIndexDatedNotes(ctx context.Context, run Runner, dir string, paths []string) ([]string, bool, error) {
	notes := newDatedNotesInPathset(ctx, run, dir, paths)
	if len(notes) == 0 {
		return paths, false, nil
	}
	indexPath := repoAbs(dir, "INDEX.md")
	indexBytes, err := os.ReadFile(indexPath)
	if err != nil {
		return paths, false, nil
	}
	indexText := string(indexBytes)
	var missing []noteIndexEntry
	for _, note := range notes {
		if strings.Contains(indexText, note.Base) {
			continue
		}
		missing = append(missing, note)
	}
	if len(missing) == 0 {
		return paths, false, nil
	}
	if !containsPath(paths, "INDEX.md") && !indexClean(ctx, run, dir) {
		return paths, false, nil
	}
	updated := insertNoteIndexEntries(indexText, missing)
	if updated == indexText {
		return paths, false, nil
	}
	if err := os.WriteFile(indexPath, []byte(updated), fileMode(indexPath)); err != nil {
		return paths, false, err
	}
	return appendMissingPath(paths, "INDEX.md"), true, nil
}

func newDatedNotesInPathset(ctx context.Context, run Runner, dir string, paths []string) []noteIndexEntry {
	var notes []noteIndexEntry
	for _, p := range paths {
		clean, ok := gitgate.CleanRepoPath(p)
		if !ok || !isDatedNotePath(clean) || pathExistsAtHEAD(ctx, run, dir, clean) {
			continue
		}
		if note, ok := readNoteIndexEntry(dir, clean); ok {
			notes = append(notes, note)
		}
	}
	sort.Slice(notes, func(i, j int) bool {
		if notes[i].Date != notes[j].Date {
			return notes[i].Date > notes[j].Date
		}
		return notes[i].Base < notes[j].Base
	})
	return notes
}

func isDatedNotePath(p string) bool {
	if !strings.HasPrefix(p, "docs/notes/") || !strings.HasSuffix(p, ".md") {
		return false
	}
	base := pathBase(p)
	return base != "README.md" && (datedNoteRE.MatchString(base) || strings.HasPrefix(base, "PLAN-"))
}

func pathExistsAtHEAD(ctx context.Context, run Runner, dir, p string) bool {
	_, code, err := run(ctx, dir, "cat-file", "-e", "HEAD:"+p)
	return err == nil && code == 0
}

func readNoteIndexEntry(dir, p string) (noteIndexEntry, bool) {
	data, err := os.ReadFile(repoAbs(dir, p))
	if err != nil {
		return noteIndexEntry{}, false
	}
	base := pathBase(p)
	return noteIndexEntry{
		Path:  p,
		Base:  base,
		Date:  noteDate(base),
		Title: noteTitle(string(data), base),
	}, true
}

func noteDate(base string) string {
	if m := datedNoteRE.FindString(base); m != "" {
		return m
	}
	return ""
}

func noteTitle(body, base string) string {
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "#") {
			title := strings.TrimSpace(strings.TrimLeft(line, "#"))
			if title != "" {
				return title
			}
		}
	}
	stem := strings.TrimSuffix(base, filepath.Ext(base))
	date := noteDate(base)
	if date != "" {
		stem = strings.Trim(strings.ReplaceAll(stem, date, ""), "-_ ")
	}
	stem = strings.ReplaceAll(stem, "_", " ")
	stem = strings.ReplaceAll(stem, "-", " ")
	stem = strings.Join(strings.Fields(stem), " ")
	if stem == "" {
		stem = strings.TrimSuffix(base, filepath.Ext(base))
	}
	if date != "" {
		return stem + " (" + date + ")"
	}
	return stem
}

func indexClean(ctx context.Context, run Runner, dir string) bool {
	out, code, err := run(ctx, dir, "status", "--porcelain", "--", "INDEX.md")
	return err == nil && code == 0 && strings.TrimSpace(out) == ""
}

func insertNoteIndexEntries(index string, entries []noteIndexEntry) string {
	sort.SliceStable(entries, func(i, j int) bool {
		if entries[i].Date != entries[j].Date {
			return entries[i].Date > entries[j].Date
		}
		return entries[i].Base < entries[j].Base
	})
	lines := strings.SplitAfter(index, "\n")
	insertAt := len(lines)
	inNotes := false
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "## ") {
			if strings.HasPrefix(trimmed, "## Notes & research") {
				inNotes = true
				continue
			}
			if inNotes {
				insertAt = i
				break
			}
		}
		if inNotes && strings.HasPrefix(trimmed, "- ") {
			insertAt = i
			break
		}
	}
	var b strings.Builder
	for _, entry := range entries {
		b.WriteString("- [")
		b.WriteString(entry.Title)
		b.WriteString("](")
		b.WriteString(entry.Path)
		b.WriteString(") -- auto-indexed dated note.\n")
	}
	if insertAt >= len(lines) {
		if !strings.HasSuffix(index, "\n") && index != "" {
			return index + "\n" + b.String()
		}
		return index + b.String()
	}
	out := strings.Join(lines[:insertAt], "")
	out += b.String()
	out += strings.Join(lines[insertAt:], "")
	return out
}

func repoAbs(dir, rel string) string {
	if dir == "" {
		return filepath.FromSlash(rel)
	}
	return filepath.Join(dir, filepath.FromSlash(rel))
}

func pathBase(p string) string {
	if i := strings.LastIndex(p, "/"); i >= 0 {
		return p[i+1:]
	}
	return p
}

func fileMode(path string) os.FileMode {
	if st, err := os.Stat(path); err == nil {
		return st.Mode()
	}
	return 0o644
}

func containsPath(paths []string, want string) bool {
	for _, p := range paths {
		if p == want {
			return true
		}
	}
	return false
}

func appendMissingPath(paths []string, p string) []string {
	if containsPath(paths, p) {
		return paths
	}
	out := append([]string(nil), paths...)
	out = append(out, p)
	return out
}
