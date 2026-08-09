package memoryindex

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// UnfiledHeading is the section [Apply] appends recovered pointer rows under. A
// dedicated, self-identifying section rather than a silent append at EOF: a row
// a machine wrote into a hand-curated index is exactly the thing a reviewer must
// be able to find and re-file, and burying it in whichever section happened to be
// last makes the index look hand-made when it is not.
const UnfiledHeading = "## Unfiled (added by `fak memory index --write` — re-file these)"

// Changes is what [Apply] did. Empty fields mean the index already agreed with
// the files.
type Changes struct {
	// Added names the memory files that gained a pointer row.
	Added []string `json:"added,omitempty"`
	// Removed names the dead link targets whose rows were deleted.
	Removed []string `json:"removed,omitempty"`
	// Tiers names the index files rewritten on disk.
	Tiers []string `json:"tiers,omitempty"`
}

// Any reports whether anything was written.
func (c Changes) Any() bool { return len(c.Added) > 0 || len(c.Removed) > 0 }

// Apply reconciles the INDEX toward the FILES. It never edits, renames or
// deletes a memory file: the index is the derived artefact and the memories are
// the source of truth, so the repair only ever runs in that direction.
//
// It therefore fixes exactly two of the seven findings — a memory with no
// pointer row gains one, and a row pointing at nothing loses it. The other five
// (a name/filename disagreement, a contended slug, a missing required field, an
// unrecognised type, a dangling link) need a decision this package must not make,
// so they SURVIVE the write and the returned Report still reports drift. That is
// the point: a reconciler that could zero its own findings by writing would be a
// laundering machine rather than a gate.
//
// The returned Report is the state AFTER the write, so a caller can print it and
// exit on it directly.
func Apply(dir string, opt Options) (Changes, Report, error) {
	before, ok := Load(dir)
	if !ok {
		return Changes{}, Report{Schema: Schema, Dir: dir, Counts: zeroCounts()},
			fmt.Errorf("no memory index at %s", filepath.Join(dir, IndexName))
	}
	rep := Reconcile(before, opt)

	var ch Changes
	// ---- drop rows whose target is gone, tier by tier ----------------------
	dead := map[string]bool{}
	for _, f := range rep.Findings {
		if f.Kind == KindIndexLineNoFile {
			dead[f.Subject] = true
			ch.Removed = append(ch.Removed, f.Subject)
		}
	}
	sort.Strings(ch.Removed)
	byTier := map[string]map[int]bool{}
	for _, r := range before.Rows {
		if !dead[r.Target] {
			continue
		}
		if byTier[r.Tier] == nil {
			byTier[r.Tier] = map[int]bool{}
		}
		byTier[r.Tier][r.Line] = true
	}
	tiers := make([]string, 0, len(byTier))
	for t := range byTier {
		tiers = append(tiers, t)
	}
	sort.Strings(tiers)
	for _, tier := range tiers {
		p := filepath.Join(dir, tier)
		raw, err := os.ReadFile(p)
		if err != nil {
			return ch, rep, fmt.Errorf("read index tier %s: %w", tier, err)
		}
		next, changed := dropLines(string(raw), byTier[tier])
		if !changed {
			continue
		}
		if err := writeFile(p, next); err != nil {
			return ch, rep, err
		}
		ch.Tiers = append(ch.Tiers, tier)
	}

	// ---- add a pointer row for every unindexed memory ----------------------
	front := map[string]Frontmatter{}
	for _, f := range before.Files {
		front[f.Name] = f.Front
	}
	var missing []string
	for _, f := range rep.Findings {
		if f.Kind == KindMissingFromIndex {
			missing = append(missing, f.Subject)
		}
	}
	sort.Strings(missing)
	if len(missing) > 0 {
		p := filepath.Join(dir, IndexName)
		raw, err := os.ReadFile(p)
		if err != nil {
			return ch, rep, fmt.Errorf("read index %s: %w", IndexName, err)
		}
		text := string(raw)
		rows := make([]string, 0, len(missing))
		for _, name := range missing {
			rows = append(rows, RenderRow(name, front[name], separatorOf(text)))
		}
		if err := writeFile(p, appendUnfiled(text, rows)); err != nil {
			return ch, rep, err
		}
		ch.Added = append(ch.Added, missing...)
		if !contains(ch.Tiers, IndexName) {
			ch.Tiers = append(ch.Tiers, IndexName)
		}
	}
	sort.Strings(ch.Tiers)

	if !ch.Any() {
		return ch, rep, nil
	}
	after, ok := Load(dir)
	if !ok {
		return ch, rep, fmt.Errorf("index at %s became unreadable after the write", dir)
	}
	return ch, Reconcile(after, opt), nil
}

// RenderRow renders one pointer line in the store's grammar:
// `- [Title](file.md)<sep>hook`. The title is the filename slug read back as
// prose and the hook is the file's own `description:`, so the recovered row
// carries the memory's self-description rather than a placeholder — and when
// there is no description, it says so IN THE ROW instead of quietly rendering a
// pointer with nothing behind it.
func RenderRow(name string, fm Frontmatter, sep string) string {
	title := Humanize(strings.TrimSuffix(name, ".md"))
	hook := strings.TrimSpace(fm.Description)
	if hook == "" {
		hook = "(no description: in this file's frontmatter — write one)"
	}
	return fmt.Sprintf("- [%s](%s)%s%s", title, name, sep, hook)
}

// Humanize turns a filename slug into title text: separators become spaces. It
// does not capitalize — the slugs in a real store already carry their own casing
// and dates, and "fixing" them would misquote the file.
func Humanize(stem string) string {
	s := strings.NewReplacer("-", " ", "_", " ").Replace(stem)
	return strings.Join(strings.Fields(s), " ")
}

// separatorOf reports the title/hook separator the index already uses, so a
// written row does not visibly differ from a hand-written one. Stores in the wild
// use both an em dash and a plain hyphen; whichever this index prefers wins, and
// an index with no rows yet gets the em dash.
func separatorOf(text string) string {
	em := strings.Count(text, ") — ")
	hy := strings.Count(text, ") - ")
	if hy > em {
		return " - "
	}
	return " — "
}

// appendUnfiled puts rows under UnfiledHeading, reusing the section if a previous
// run already made one so repeated writes do not stack duplicate headings.
func appendUnfiled(text string, rows []string) string {
	nl := "\n"
	if strings.Contains(text, "\r\n") {
		nl = "\r\n"
	}
	body := strings.TrimRight(text, "\r\n")
	if strings.Contains(body, UnfiledHeading) {
		return body + nl + strings.Join(rows, nl) + nl
	}
	return body + nl + nl + UnfiledHeading + nl + strings.Join(rows, nl) + nl
}

// dropLines removes the given 1-based line numbers, preserving the file's line
// endings.
func dropLines(text string, drop map[int]bool) (string, bool) {
	nl := "\n"
	if strings.Contains(text, "\r\n") {
		nl = "\r\n"
	}
	lines := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	out := make([]string, 0, len(lines))
	changed := false
	for i, line := range lines {
		if drop[i+1] {
			changed = true
			continue
		}
		out = append(out, line)
	}
	return strings.Join(out, nl), changed
}

func writeFile(p, text string) error {
	if err := os.WriteFile(p, []byte(text), 0o644); err != nil {
		return fmt.Errorf("write %s: %w", filepath.Base(p), err)
	}
	return nil
}

func contains(ss []string, s string) bool {
	for _, x := range ss {
		if x == s {
			return true
		}
	}
	return false
}
