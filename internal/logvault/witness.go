package logvault

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

// WitnessedFiles returns the current, independently re-hashed mirror state for
// one source. The manifest chain and head anchor are verified before any state
// is returned; every returned digest was then re-derived from the mirror bytes.
// It shares the vault writer lock so capture/GC cannot race the read-back.
func (v *Vault) WitnessedFiles(sourceID string) (map[string]string, error) {
	if err := os.MkdirAll(v.Dir, 0o755); err != nil {
		return nil, err
	}
	lock, err := os.OpenFile(filepath.Join(v.Dir, LockName), os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, err
	}
	defer lock.Close()
	if err := flock.TryLock(lock); err != nil {
		if errors.Is(err, flock.ErrLockBusy) {
			return nil, fmt.Errorf("logvault: another writer holds %s", LockName)
		}
		return nil, err
	}
	defer flock.Unlock(lock)
	return v.witnessedFilesLocked(sourceID)
}

func (v *Vault) witnessedFilesLocked(sourceID string) (map[string]string, error) {
	manPath := filepath.Join(v.Dir, ManifestName)
	if _, err := VerifyManifest(manPath); err != nil {
		return nil, err
	}
	rows, err := ReadManifestRows(manPath)
	if err != nil {
		return nil, err
	}
	if a, ok, err := readAnchor(v.Dir); err != nil {
		return nil, fmt.Errorf("head anchor unreadable: %w", err)
	} else if !ok && len(rows) > 0 {
		return nil, fmt.Errorf("head anchor missing")
	} else if ok {
		if uint64(len(rows)) < a.Seq {
			return nil, fmt.Errorf("manifest truncated: anchor head seq %d, manifest tail seq %d", a.Seq, len(rows))
		}
		if a.Seq == 0 || rows[a.Seq-1].Hash != a.Hash {
			return nil, fmt.Errorf("manifest disagrees with head anchor")
		}
	}

	out := make(map[string]string)
	for key, state := range replayStates(rows) {
		srcID, rel, ok := cutStateKey(key)
		if !ok || srcID != sourceID {
			continue
		}
		got, err := hashFile(v.mirrorPath(srcID, rel))
		if err != nil {
			return nil, fmt.Errorf("mirror %s unreadable: %w", rel, err)
		}
		if got != state.SHA256 {
			return nil, fmt.Errorf("mirror %s hash mismatch vs manifest", rel)
		}
		out[filepath.ToSlash(rel)] = got
	}
	return out, nil
}

func cutStateKey(key string) (sourceID, rel string, ok bool) {
	for i := 0; i < len(key); i++ {
		if key[i] == 0 {
			return key[:i], key[i+1:], true
		}
	}
	return "", "", false
}
