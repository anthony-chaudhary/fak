//go:build !windows

package gateway

import (
	"errors"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

func acquireControlJournalOwnership(path, resolvedPath string) (*controlJournalOwnership, error) {
	pathLock, err := os.OpenFile(path+".writer.lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, fmt.Errorf("control ingress: open writer path lock: %w", err)
	}
	if err := flock.TryLock(pathLock); err != nil {
		_ = pathLock.Close()
		if errors.Is(err, flock.ErrLockBusy) {
			return nil, errControlJournalOwned
		}
		return nil, fmt.Errorf("control ingress: lock writer path: %w", err)
	}

	journal, err := os.OpenFile(resolvedPath, os.O_RDWR|os.O_APPEND, 0)
	if err != nil {
		_ = flock.Unlock(pathLock)
		_ = pathLock.Close()
		return nil, fmt.Errorf("control ingress: open journal for ownership: %w", err)
	}
	if err := flock.TryLock(journal); err != nil {
		_ = journal.Close()
		_ = flock.Unlock(pathLock)
		_ = pathLock.Close()
		if errors.Is(err, flock.ErrLockBusy) {
			return nil, errControlJournalOwned
		}
		return nil, fmt.Errorf("control ingress: lock journal identity: %w", err)
	}
	info, err := journal.Stat()
	if err != nil {
		_ = flock.Unlock(journal)
		_ = journal.Close()
		_ = flock.Unlock(pathLock)
		_ = pathLock.Close()
		return nil, fmt.Errorf("control ingress: stat owned journal: %w", err)
	}

	return &controlJournalOwnership{
		file: journal,
		info: info,
		release: func() error {
			journalErr := flock.Unlock(journal)
			pathErr := flock.Unlock(pathLock)
			closeErr := pathLock.Close()
			if journalErr != nil {
				return journalErr
			}
			if pathErr != nil {
				return pathErr
			}
			return closeErr
		},
	}, nil
}
