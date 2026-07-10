package accounts

// Credential backup/restore — the durable safety net for #3987: "no login should be able to
// lose an account." A `/login` (especially an in-harness one) overwrites a seat's
// .credentials.json / .claude.json in place; if it overwrites the WRONG dir, the previous
// account's credential is gone with no recovery path. This file gives every credential-overwriting
// accounts op a pre-image snapshot and a first-class restore verb.
//
// The store is content-addressed and lives under the gitignored home tree (never the repo): each
// snapshot is <root>/<seat>/<stamp>-<sha12>-<file>. Content addressing means an unchanged blob is
// snapshotted once no matter how many times the hook fires, and the sha in the name is a tamper-
// evident witness that a restored blob is exactly the bytes that were captured. Backups hold LIVE
// OAuth tokens, so the store deliberately stays under $HOME/.claude/backups — a subdir of the
// default seat's own config dir (never a ~/.claude-* sibling, so the config-home scanner never
// mistakes it for a seat), and never a git-tracked path.

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// credentialBackupFiles are the per-seat blobs a login/adopt/rehome/remove can overwrite and that
// the store snapshots. .claude.json carries the account METADATA (oauthAccount); the other two are
// the live credential the account is actually served by. A missing file in a dir is simply skipped.
var credentialBackupFiles = []string{".credentials.json", ".claude.json", ".oauth-token"}

// backupStampLayout is the sortable UTC timestamp token stamped into each snapshot name. It has no
// '-', so the name splits cleanly into (stamp, sha12, file) even when file is ".oauth-token".
const backupStampLayout = "20060102T150405Z"

// BackupRoot is the credential backup store root for a home: <home>/.claude/backups. It is nested
// under the default seat's own config dir on purpose — a ~/.claude-* SIBLING would be scanned as a
// candidate config home, whereas a subdir of ~/.claude is never enumerated by discovery. It is
// always under the gitignored home tree, so live tokens never reach a repo path.
func BackupRoot(home string) string { return filepath.Join(home, ".claude", "backups") }

// CredentialBackup identifies one stored snapshot blob.
type CredentialBackup struct {
	Seat    string    `json:"seat"`     // roster/seat label the snapshot belongs to
	File    string    `json:"file"`     // original file name (.credentials.json / .claude.json / .oauth-token)
	Stamp   string    `json:"stamp"`    // UTC timestamp token (sortable, backupStampLayout)
	SHA     string    `json:"sha"`      // content sha256, first 12 hex — the content address
	Path    string    `json:"path"`     // absolute path in the store
	Size    int64     `json:"size"`     // blob size in bytes
	ModTime time.Time `json:"mod_time"` // store-file mtime (when the snapshot was written)
}

// sha12 returns the first 12 hex chars of the sha256 of b — the content address of a blob.
func sha12(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])[:12]
}

// backupName renders the store filename for a snapshot: <stamp>-<sha12>-<file>. Because stamp and
// sha12 carry no '-', a reader recovers all three fields with a 3-way SplitN on '-'.
func backupName(stamp, sha, file string) string {
	return stamp + "-" + sha + "-" + file
}

// parseBackupName recovers (stamp, sha, file) from a store filename, or ok=false if it is not a
// snapshot this package wrote (an unrecognized file in the store is ignored, never guessed at).
func parseBackupName(name string) (stamp, sha, file string, ok bool) {
	parts := strings.SplitN(name, "-", 3)
	if len(parts) != 3 || parts[0] == "" || parts[1] == "" || parts[2] == "" {
		return "", "", "", false
	}
	return parts[0], parts[1], parts[2], true
}

// BackupSeatCredentials snapshots every present, non-empty credential blob in `dir` into the store
// under `seat`, stamped `now`. It is idempotent by content: a blob whose sha already has a snapshot
// under this seat is not re-copied, so a hook that fires on every op does not churn the store with
// duplicates. Missing, empty, or directory entries are skipped. A dir with nothing to back up is
// not an error — it returns an empty slice.
func BackupSeatCredentials(root, seat, dir string, now time.Time) ([]CredentialBackup, error) {
	if root == "" || seat == "" || dir == "" {
		return nil, fmt.Errorf("backup: root, seat, and dir are all required")
	}
	seatDir := filepath.Join(root, seat)
	// Index shas already stored for this seat so content-identical blobs are not re-copied.
	existing := map[string]bool{}
	if prior, err := ListCredentialBackups(root, seat); err == nil {
		for _, b := range prior {
			existing[b.File+"@"+b.SHA] = true
		}
	}
	stamp := now.UTC().Format(backupStampLayout)
	var out []CredentialBackup
	for _, file := range credentialBackupFiles {
		src := filepath.Join(dir, file)
		info, err := os.Stat(src)
		if err != nil || info.IsDir() || info.Size() == 0 {
			continue // absent / empty / a dir — nothing meaningful to snapshot
		}
		b, err := os.ReadFile(src)
		if err != nil || len(b) == 0 {
			continue
		}
		sha := sha12(b)
		if existing[file+"@"+sha] {
			continue // this exact blob is already captured for this seat
		}
		if err := os.MkdirAll(seatDir, 0o700); err != nil {
			return out, fmt.Errorf("backup: mkdir store %s: %w", seatDir, err)
		}
		dst := filepath.Join(seatDir, backupName(stamp, sha, file))
		if err := writeFile0600(dst, b); err != nil {
			return out, fmt.Errorf("backup: write %s: %w", dst, err)
		}
		st, _ := os.Stat(dst)
		cb := CredentialBackup{Seat: seat, File: file, Stamp: stamp, SHA: sha, Path: dst, Size: int64(len(b))}
		if st != nil {
			cb.ModTime = st.ModTime()
		}
		out = append(out, cb)
		existing[file+"@"+sha] = true
	}
	return out, nil
}

// SnapshotBeforeOverwrite is the backup-on-write hook: call it immediately BEFORE an accounts op
// overwrites `dir`'s credentials (adopt --force reconcile, rehome, remove, an in-harness login
// wrapper). It captures the CURRENT (pre-image) blobs so the prior account is always recoverable.
// It is a thin, self-documenting alias for BackupSeatCredentials so call sites read as intent, and
// so callers can treat a backup miss as a non-fatal warning without coupling to the store API.
func SnapshotBeforeOverwrite(root, seat, dir string, now time.Time) ([]CredentialBackup, error) {
	return BackupSeatCredentials(root, seat, dir, now)
}

// ListCredentialBackups returns the snapshots stored for `seat`, newest first (by stamp, then sha
// for a stable order within one stamp). A store with no dir for the seat is an empty result, not an
// error — nothing has been backed up yet is a valid state.
func ListCredentialBackups(root, seat string) ([]CredentialBackup, error) {
	seatDir := filepath.Join(root, seat)
	ents, err := os.ReadDir(seatDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("backup: read store %s: %w", seatDir, err)
	}
	var out []CredentialBackup
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		stamp, sha, file, ok := parseBackupName(e.Name())
		if !ok {
			continue
		}
		cb := CredentialBackup{Seat: seat, File: file, Stamp: stamp, SHA: sha, Path: filepath.Join(seatDir, e.Name())}
		if info, ierr := e.Info(); ierr == nil {
			cb.Size = info.Size()
			cb.ModTime = info.ModTime()
		}
		out = append(out, cb)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Stamp != out[j].Stamp {
			return out[i].Stamp > out[j].Stamp // newest stamp first
		}
		return out[i].SHA < out[j].SHA
	})
	return out, nil
}

// PruneCredentialBackups keeps the newest `keep` snapshots PER FILE for `seat` and deletes older
// ones, so the store is bounded without ever dropping the most recent recoverable blob of each
// kind. keep <= 0 is a no-op (never prune to empty by accident). Returns the number removed.
func PruneCredentialBackups(root, seat string, keep int) (int, error) {
	if keep <= 0 {
		return 0, nil
	}
	all, err := ListCredentialBackups(root, seat)
	if err != nil {
		return 0, err
	}
	perFile := map[string]int{}
	removed := 0
	for _, b := range all { // all is newest-first, so the first `keep` per file are the survivors
		perFile[b.File]++
		if perFile[b.File] <= keep {
			continue
		}
		if err := os.Remove(b.Path); err != nil && !os.IsNotExist(err) {
			return removed, fmt.Errorf("backup: prune %s: %w", b.Path, err)
		}
		removed++
	}
	return removed, nil
}

// RestoreCredential restores `seat`'s `file` from the store back into `dir`. `at` selects which
// snapshot: "" picks the newest; otherwise the newest snapshot whose Stamp OR SHA has `at` as a
// prefix (so an operator can name a timestamp or a content address). Before writing, it snapshots
// the CURRENT blob (stamped `now`) so a restore is itself reversible — restoring never destroys the
// state it replaces. The write is atomic (temp + rename) at 0o600, matching how live creds are
// written. It verifies the restored bytes hash back to the snapshot's content address before
// returning, so a corrupted store file surfaces as an error, not a silently-wrong credential.
func RestoreCredential(root, seat, dir, file, at string, now time.Time) (CredentialBackup, error) {
	if file == "" {
		file = ".credentials.json" // the live credential is the default target
	}
	all, err := ListCredentialBackups(root, seat)
	if err != nil {
		return CredentialBackup{}, err
	}
	var chosen *CredentialBackup
	for i := range all {
		if all[i].File != file {
			continue
		}
		if at == "" || strings.HasPrefix(all[i].Stamp, at) || strings.HasPrefix(all[i].SHA, at) {
			chosen = &all[i]
			break // all is newest-first, so the first match is the newest match
		}
	}
	if chosen == nil {
		if at == "" {
			return CredentialBackup{}, fmt.Errorf("no %s backup found for seat %q", file, seat)
		}
		return CredentialBackup{}, fmt.Errorf("no %s backup for seat %q matching %q", file, seat, at)
	}
	b, err := os.ReadFile(chosen.Path)
	if err != nil {
		return CredentialBackup{}, fmt.Errorf("restore: read snapshot %s: %w", chosen.Path, err)
	}
	if got := sha12(b); got != chosen.SHA {
		return CredentialBackup{}, fmt.Errorf("restore: snapshot %s corrupt: content %s != address %s", chosen.Path, got, chosen.SHA)
	}
	// Snapshot the current blob first so the restore is reversible, then write the chosen bytes.
	if _, berr := BackupSeatCredentials(root, seat, dir, now); berr != nil {
		return CredentialBackup{}, fmt.Errorf("restore: back up current blob before overwrite: %w", berr)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return CredentialBackup{}, fmt.Errorf("restore: mkdir target %s: %w", dir, err)
	}
	if err := writeFile0600(filepath.Join(dir, file), b); err != nil {
		return CredentialBackup{}, fmt.Errorf("restore: write %s: %w", filepath.Join(dir, file), err)
	}
	return *chosen, nil
}

// writeFile0600 writes b to path atomically (temp in the same dir + rename) at 0o600 — the mode
// live credentials are written with. The temp-then-rename keeps a reader from ever seeing a
// half-written credential, and a crash mid-write leaves the prior file intact.
func writeFile0600(path string, b []byte) error {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".credbak-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(b); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return err
	}
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return err
	}
	if err := os.Rename(tmpName, path); err != nil {
		os.Remove(tmpName)
		return err
	}
	return nil
}
