//go:build !windows

package ctxmmu

import (
	"fmt"
	"os"
	"path/filepath"
)

func syncQuarantineLedgerCreation(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return fmt.Errorf("open quarantine ledger dir for sync: %w", err)
	}
	defer d.Close()
	if err := d.Sync(); err != nil {
		return fmt.Errorf("sync quarantine ledger dir: %w", err)
	}
	return nil
}

func replaceQuarantineLedgerFile(tmpName, path string) error {
	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("replace quarantine ledger snapshot: %w", err)
	}
	return syncQuarantineLedgerCreation(filepath.Dir(path))
}
