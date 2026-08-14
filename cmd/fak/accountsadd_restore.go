package main

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// accountsadd_restore.go — the `fak accounts restore` seam, lifted out of accounts_add.go
// under the god-file growth gate (>1500 lines). It carries the whole un-tombstone concern: the
// archived (.DELETED-<date>) rename-back path, the in-place reactivation of a seat tombstoned
// without --archive, and the tombstone-name parsing both share. Pure relocation — the add and
// remove verbs that produce the tombstones stay in accounts_add.go.

// restoreParams carries the resolved flags for `fak accounts restore`.
type restoreParams struct {
	name         string
	registryPath string
	dosView      string
	jobView      string
	noSync       bool
}

// runAccountsRestore reverses the reversible half of `remove --archive`: an archived
// tombstone named <seat>.DELETED-<date> becomes the live <seat> again, its config dir is
// renamed back, tombstone policy fields are cleared, rehome references are repaired, and the
// generated views are synced. It is intentionally narrow: non-archived tombstones and missing
// archive dirs are refused instead of guessed, so this stays an audited inverse rather than a
// registry surgery escape hatch.
func runAccountsRestore(stdout, stderr io.Writer, p restoreParams) int {
	if p.name == "" {
		fmt.Fprintln(stderr, "usage: fak accounts restore --name <archived-or-live-seat>")
		return 2
	}
	reg, ok := loadRegistryOrErr(stderr, p.registryPath)
	if !ok {
		return 1
	}
	idx, liveName, liveDir, ok := findArchivedRestoreTarget(reg, p.name)
	if !ok {
		if h, exists := homeByName(reg, p.name); exists && h.Active() {
			fmt.Fprintf(stdout, "%q is already active\n", p.name)
			return 0
		}
		// No archived (.DELETED) tombstone matched. A plainly-removed seat — one
		// tombstoned by `remove` WITHOUT `--archive`, so its name + dir never moved —
		// is the other reversible shape. Reactivate it in place: no dir rename, just
		// clear the tombstone policy fields `remove` set. This makes `restore` the
		// true inverse of BOTH `remove` shapes rather than only `remove --archive`.
		if inIdx, inOK := inPlaceRestoreTarget(reg, p.name); inOK {
			return restoreInPlace(stdout, stderr, p, reg, inIdx)
		}
		fmt.Fprintf(stderr, "fak accounts: no archived tombstone found for %q\n", p.name)
		return 1
	}
	oldName, oldDir := reg.Homes[idx].Name, reg.Homes[idx].Dir
	for i, h := range reg.Homes {
		if i != idx && h.Name == liveName {
			fmt.Fprintf(stderr, "fak accounts: cannot restore %q: live target name already exists: %s\n", oldName, liveName)
			return 1
		}
	}
	if oldDir == "" {
		fmt.Fprintf(stderr, "fak accounts: cannot restore %q: archived registry entry has no dir\n", oldName)
		return 1
	}
	if _, err := os.Stat(oldDir); err != nil {
		fmt.Fprintf(stderr, "fak accounts: cannot restore %q: archive dir missing: %s\n", oldName, oldDir)
		return 1
	}
	if _, err := os.Stat(liveDir); err == nil {
		fmt.Fprintf(stderr, "fak accounts: cannot restore %q: live target dir already exists: %s\n", oldName, liveDir)
		return 1
	}

	reg.Homes[idx].Name = liveName
	reg.Homes[idx].Dir = liveDir
	reg.Homes[idx].Status = ""
	reg.Homes[idx].RehomeTo = ""
	reg.Homes[idx].Terminal = false
	reg.Homes[idx].TombstonedAt = ""
	reg.Homes[idx].TombstoneReason = ""
	reg.Homes[idx].Enabled = nil
	repaired := 0
	for i := range reg.Homes {
		if reg.Homes[i].RehomeTo == oldName {
			reg.Homes[i].RehomeTo = liveName
			repaired++
		}
	}
	if err := reg.Validate(); err != nil {
		fmt.Fprintf(stderr, "fak accounts: restore would make registry invalid: %v\n", err)
		return 1
	}

	if err := os.Rename(oldDir, liveDir); err != nil {
		fmt.Fprintf(stderr, "fak accounts: restore rename %s -> %s: %v\n", oldDir, liveDir, err)
		return 1
	}
	reg.Homes[idx].Identity = accounts.DeriveIdentity(liveDir)
	if !saveAccountsRegistry(stderr, p.registryPath, reg) {
		return 1
	}
	fmt.Fprintf(stdout, "restored dir: %s -> %s\n", oldDir, liveDir)
	fmt.Fprintf(stdout, "registry: restored %s -> %s\n", oldName, liveName)
	if repaired > 0 {
		fmt.Fprintf(stdout, "registry: repaired %d rehome reference(s) from %s -> %s\n", repaired, oldName, liveName)
	}
	if code := syncViewsUnlessNoSync(stdout, stderr, p.registryPath, p.dosView, p.jobView, p.noSync); code != 0 {
		return code
	}
	fmt.Fprintf(stdout, "restored account %q (was %q; dir renamed, active in registry + views)\n", liveName, oldName)
	return 0
}

// inPlaceRestoreTarget finds a seat tombstoned in place — by `remove` WITHOUT `--archive`,
// so its name and dir were left untouched (no `.DELETED-<date>` suffix). Such a seat has no
// archive to rename back; reactivating it is purely a registry field clear. Returns the home
// index and ok=true only for an exact-name match that is currently tombstoned and carries a
// plain (non-archived) name, so this never shadows the archived-restore path.
func inPlaceRestoreTarget(reg accounts.Registry, name string) (idx int, ok bool) {
	for i, h := range reg.Homes {
		if h.Name != name {
			continue
		}
		if h.Active() {
			return -1, false // handled by the already-active branch in the caller
		}
		if _, isArchived := stripDeletedSuffix(h.Name); isArchived {
			return -1, false // an archived tombstone is the findArchivedRestoreTarget path
		}
		return i, true
	}
	return -1, false
}

// restoreInPlace un-tombstones a plainly-removed seat: it clears exactly the fields
// `runAccountsRemove` set (Status, RehomeTo, TombstonedAt, TombstoneReason, Enabled), leaving
// the name and dir alone (they never moved). Roles that `remove` moved to the rehome target
// are NOT auto-reclaimed — that is a deliberate operator choice (a silent role/default flip is
// a surprise, and Validate forbids two seats claiming one role), so it prints a reminder.
func restoreInPlace(stdout, stderr io.Writer, p restoreParams, reg accounts.Registry, idx int) int {
	oldRehome := reg.Homes[idx].RehomeTo
	reg.Homes[idx].Status = ""
	reg.Homes[idx].RehomeTo = ""
	reg.Homes[idx].Terminal = false
	reg.Homes[idx].TombstonedAt = ""
	reg.Homes[idx].TombstoneReason = ""
	reg.Homes[idx].Enabled = nil
	// Re-derive identity from the (unchanged) dir so a login that changed while the seat was
	// tombstoned is reflected — matches the archived path's post-rename DeriveIdentity.
	if reg.Homes[idx].Dir != "" {
		reg.Homes[idx].Identity = accounts.DeriveIdentity(reg.Homes[idx].Dir)
	}
	if err := reg.Validate(); err != nil {
		fmt.Fprintf(stderr, "fak accounts: restore would make registry invalid: %v\n", err)
		return 1
	}
	if !saveAccountsRegistry(stderr, p.registryPath, reg) {
		return 1
	}
	fmt.Fprintf(stdout, "registry: restored %s in place (dir unchanged: %s)\n", reg.Homes[idx].Name, reg.Homes[idx].Dir)
	if oldRehome != "" {
		fmt.Fprintf(stdout, "note: roles moved to %q on removal are not auto-restored; use `fak accounts set-role`/`set-default` to reclaim\n", oldRehome)
	}
	if code := syncViewsUnlessNoSync(stdout, stderr, p.registryPath, p.dosView, p.jobView, p.noSync); code != 0 {
		return code
	}
	fmt.Fprintf(stdout, "restored account %q in place (tombstone cleared; active in registry + views)\n", reg.Homes[idx].Name)
	return 0
}

func findArchivedRestoreTarget(reg accounts.Registry, name string) (idx int, liveName, liveDir string, ok bool) {
	var matches []int
	for i, h := range reg.Homes {
		if h.Name == name {
			if h.Active() {
				return -1, "", "", false
			}
			base, baseOK := stripDeletedSuffix(h.Name)
			dir, dirOK := stripDeletedSuffix(h.Dir)
			if baseOK && dirOK {
				return i, base, dir, true
			}
			return -1, "", "", false
		}
	}
	for i, h := range reg.Homes {
		if h.Active() {
			continue
		}
		base, baseOK := stripDeletedSuffix(h.Name)
		if baseOK && base == name {
			matches = append(matches, i)
		}
	}
	if len(matches) != 1 {
		return -1, "", "", false
	}
	h := reg.Homes[matches[0]]
	base, baseOK := stripDeletedSuffix(h.Name)
	dir, dirOK := stripDeletedSuffix(h.Dir)
	if !baseOK || !dirOK {
		return -1, "", "", false
	}
	return matches[0], base, dir, true
}

func stripDeletedSuffix(s string) (string, bool) {
	const marker = ".DELETED-"
	idx := strings.LastIndex(s, marker)
	if idx < 0 {
		return "", false
	}
	date := s[idx+len(marker):]
	if len(date) != len("2006-01-02") || date[4] != '-' || date[7] != '-' {
		return "", false
	}
	return s[:idx], true
}
