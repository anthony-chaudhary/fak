package main

// `fak accounts backup` / `fak accounts restore-credential` — the operator surface over the
// content-addressed credential backup store (#3987). `backup` snapshots every live seat's
// credential blobs before a /login can overwrite them (and prunes the store); `restore-credential`
// brings a prior blob back. Both resolve seat dirs from the canonical registry so the store keys
// (seat names) match the roster the rest of `fak accounts` speaks.

import (
	"fmt"
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

type backupParams struct {
	name         string // limit to this seat; empty = every live seat
	list         bool   // list stored snapshots instead of taking a new one
	keep         int    // prune to at most this many snapshots per file per seat
	registryPath string
	homeDir      string
	asJSON       bool
}

type restoreCredParams struct {
	name         string // seat whose credential to restore (required)
	at           string // snapshot selector: timestamp OR sha prefix; empty = newest
	file         string // which blob (.credentials.json | .claude.json | .oauth-token)
	registryPath string
	homeDir      string
	asJSON       bool
}

// seatDirs resolves the (name -> dir) map to operate on: a single --name when given (error if it
// is not a known live seat with a dir), else every active seat that has a config dir.
func seatDirs(reg accounts.Registry, only string) (map[string]string, error) {
	out := map[string]string{}
	for _, h := range reg.Homes {
		if !h.Active() || h.Dir == "" {
			continue
		}
		if only != "" && h.Name != only {
			continue
		}
		out[h.Name] = h.Dir
	}
	if only != "" && len(out) == 0 {
		return nil, fmt.Errorf("seat %q is not a live seat with a config dir", only)
	}
	return out, nil
}

func runAccountsBackup(stdout, stderr io.Writer, p backupParams) int {
	if !requireAccountsHome(stderr, p.homeDir) {
		return 1
	}
	reg, err := loadOrDiscover(p.registryPath, p.homeDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	root := accounts.BackupRoot(p.homeDir)

	// --list is a read: show what is recoverable for the named seat (a name is required to know
	// which store dir to read).
	if p.list {
		if p.name == "" {
			fmt.Fprintln(stderr, "fak accounts backup --list: pass --name <seat> to list its snapshots")
			return 2
		}
		snaps, err := accounts.ListCredentialBackups(root, p.name)
		if err != nil {
			fmt.Fprintf(stderr, "fak accounts: %v\n", err)
			return 1
		}
		if p.asJSON {
			stdout.Write(mustJSON(map[string]any{"seat": p.name, "snapshots": snaps}))
			fmt.Fprintln(stdout)
			return 0
		}
		if len(snaps) == 0 {
			fmt.Fprintf(stdout, "no credential backups for seat %q\n", p.name)
			return 0
		}
		fmt.Fprintf(stdout, "credential backups for %q (newest first):\n", p.name)
		for _, s := range snaps {
			fmt.Fprintf(stdout, "  %s  %s  %s  (%d bytes)\n", s.Stamp, s.SHA, s.File, s.Size)
		}
		return 0
	}

	dirs, err := seatDirs(reg, p.name)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	now := time.Now()
	type seatResult struct {
		Seat   string                      `json:"seat"`
		Taken  []accounts.CredentialBackup `json:"taken"`
		Pruned int                         `json:"pruned"`
		Err    string                      `json:"error,omitempty"`
	}
	var results []seatResult
	totalTaken := 0
	for seat, dir := range dirs {
		r := seatResult{Seat: seat}
		taken, berr := accounts.BackupSeatCredentials(root, seat, dir, now)
		if berr != nil {
			r.Err = berr.Error()
			results = append(results, r)
			fmt.Fprintf(stderr, "fak accounts: backup %q: %v\n", seat, berr)
			continue
		}
		r.Taken = taken
		totalTaken += len(taken)
		if pruned, perr := accounts.PruneCredentialBackups(root, seat, p.keep); perr == nil {
			r.Pruned = pruned
		}
		results = append(results, r)
	}
	if p.asJSON {
		stdout.Write(mustJSON(map[string]any{"root": root, "seats": results, "total_taken": totalTaken}))
		fmt.Fprintln(stdout)
		return 0
	}
	if len(dirs) == 0 {
		fmt.Fprintln(stdout, "no live seats to back up")
		return 0
	}
	for _, r := range results {
		if r.Err != "" {
			continue
		}
		if len(r.Taken) == 0 {
			fmt.Fprintf(stdout, "%s: up to date (nothing new to snapshot)\n", r.Seat)
			continue
		}
		files := make([]string, 0, len(r.Taken))
		for _, b := range r.Taken {
			files = append(files, b.File)
		}
		fmt.Fprintf(stdout, "%s: snapshotted %d blob(s): %v\n", r.Seat, len(r.Taken), files)
	}
	fmt.Fprintf(stdout, "backed up %d new blob(s) into %s\n", totalTaken, root)
	return 0
}

func runAccountsRestoreCredential(stdout, stderr io.Writer, p restoreCredParams) int {
	if p.name == "" {
		fmt.Fprintln(stderr, "usage: fak accounts restore-credential --name <seat> [--at <stamp|sha>] [--file <blob>]")
		return 2
	}
	if !requireAccountsHome(stderr, p.homeDir) {
		return 1
	}
	reg, err := loadOrDiscover(p.registryPath, p.homeDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	dirs, err := seatDirs(reg, p.name)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	dir := dirs[p.name]
	root := accounts.BackupRoot(p.homeDir)
	restored, err := accounts.RestoreCredential(root, p.name, dir, p.file, p.at, time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	if p.asJSON {
		stdout.Write(mustJSON(map[string]any{"seat": p.name, "dir": dir, "restored": restored}))
		fmt.Fprintln(stdout)
		return 0
	}
	fmt.Fprintf(stdout, "restored %s for %q from snapshot %s-%s into %s (the overwritten blob was backed up first)\n",
		restored.File, p.name, restored.Stamp, restored.SHA, dir)
	return 0
}
