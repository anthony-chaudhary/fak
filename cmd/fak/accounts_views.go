package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// accounts_views.go — the GENERATED-VIEW projection half of `fak accounts`. registry.json
// is the single source of truth; the dos roster (~/.claude/accounts.yaml), the job roster,
// and each seat's own settings.json are projections of it that `sync` (and `add`'s final
// step) rewrite. Every write here is atomic (temp + rename) so a reader never sees a
// half-written roster. Split out of accounts.go so the projection targets and their
// crash-safe write shape live in one place.

// syncViews projects the canonical registry (at registryPath) into the named roster views and
// writes them atomically, refreshing identities from disk first so emitted emails are current.
// It returns the number of views written and a process exit code (0 on success). Shared by the
// `sync` verb and `add`'s final step so both regenerate views identically.
func syncViews(stdout, stderr io.Writer, registryPath, dosView, jobView string) (int, int) {
	reg, err := accounts.LoadRegistry(registryPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 0, 1
	}
	reg = reg.Refresh()
	wrote := 0
	for _, t := range viewTargets(dosView, jobView) {
		text, err := reg.RenderView(t.view)
		if err != nil {
			fmt.Fprintf(stderr, "fak accounts: %v\n", err)
			return wrote, 1
		}
		if err := writeViewFile(t.path, text); err != nil {
			fmt.Fprintf(stderr, "fak accounts: write %s: %v\n", t.path, err)
			return wrote, 1
		}
		fmt.Fprintf(stdout, "synced %s view -> %s\n", t.view, t.path)
		wrote++
	}
	// Project the registry's per-account settings defaults (defaults.settings) into every active
	// account's own settings.json — the in-tree replacement for the external csync chore, so a
	// `sync` leaves the whole roster's bypass/permission defaults consistent, not just the roster
	// view files. A registry with no defaults.settings block is a clean no-op.
	if code := projectSettingsForHomes(stdout, stderr, reg, reg.Homes); code != 0 {
		return wrote, code
	}
	return wrote, 0
}

// projectSettingsForHomes deep-merges the registry's defaults.settings block into each home's
// own settings.json (via the atomic writeSettingsFile) and prints the per-account report csync
// used to: one "updated"/"ok (no change)" line per acted-on seat and a trailing count. A
// registry with no defaults.settings block prints one note and returns 0 (nothing to project).
// It returns a process exit code (0 on success, 1 on a write failure). Shared by the `sync`
// verb and `add`'s final step so both seed settings.json identically.
func projectSettingsForHomes(stdout, stderr io.Writer, reg accounts.Registry, homes []accounts.Home) int {
	results, ok, err := reg.ProjectSettings(homes, writeSettingsFile)
	if !ok {
		fmt.Fprintln(stdout, "settings: registry has no defaults.settings block — nothing to project")
		return 0
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: project settings: %v\n", err)
		return 1
	}
	changed := 0
	for _, r := range results {
		switch {
		case r.Skipped != "":
			fmt.Fprintf(stdout, "  settings %-24s skipped (%s)\n", r.Name, r.Skipped)
		case r.Changed:
			fmt.Fprintf(stdout, "  settings %-24s updated -> %s\n", r.Name, r.Path)
			changed++
		default:
			fmt.Fprintf(stdout, "  settings %-24s ok (no change)\n", r.Name)
		}
	}
	fmt.Fprintf(stdout, "settings: %d account(s) changed\n", changed)
	return 0
}

// viewTarget pairs a view name with its on-disk path.
type viewTarget struct {
	view accounts.ViewName
	path string
}

// viewTargets returns the view destinations to sync/check: the dos roster always (it has a
// default path), and the job roster only when a path was given (it lives in a separate repo,
// so it is opt-in via --job-view / FAK_JOB_ROSTER).
func viewTargets(dosPath, jobPath string) []viewTarget {
	var out []viewTarget
	if dosPath != "" {
		out = append(out, viewTarget{accounts.ViewDos, dosPath})
	}
	if jobPath != "" {
		out = append(out, viewTarget{accounts.ViewJob, jobPath})
	}
	return out
}

// writeViewFile writes a generated view atomically (temp + rename) so a reader never sees a
// half-written roster.
func writeViewFile(path, text string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".view-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.WriteString(text); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

// writeSettingsFile writes an account's settings.json atomically (temp + rename), creating the
// config dir if absent so a brand-new seat's file lands. It is the []byte sibling of
// writeViewFile — same crash-safe shape, a distinct temp prefix — and is the writeFn the
// settings projection is handed.
func writeSettingsFile(path string, b []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".settings-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(b); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

// defaultDosView is the dos roster's conventional path (~/.claude/accounts.yaml).
func defaultDosView(home string) string {
	if home == "" {
		return ""
	}
	return filepath.Join(home, ".claude", "accounts.yaml")
}
