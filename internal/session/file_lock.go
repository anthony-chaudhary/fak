package session

import (
	"fmt"
	"os"
	"path/filepath"
	"time"
)

const descriptorLockTimeout = 10 * time.Second

// lockDescriptorFile serializes the complete read-modify-publish transaction.
// The sibling lock file is deliberately stable across atomic registry replacement;
// kernel locks are released automatically when a process exits.
func lockDescriptorFile(path string) (func(), error) {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create session descriptor dir: %w", err)
	}
	f, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, fmt.Errorf("open session descriptor lock: %w", err)
	}
	if err := lockFile(f, descriptorLockTimeout); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("lock session descriptor file: %w", err)
	}
	return func() {
		_ = unlockFile(f)
		_ = f.Close()
	}, nil
}

func cleanupDescriptorTemps(path string) error {
	matches, err := filepath.Glob(filepath.Join(filepath.Dir(path), ".session-descriptors-*.tmp"))
	if err != nil {
		return fmt.Errorf("list stale session descriptor temp files: %w", err)
	}
	for _, name := range matches {
		if err := os.Remove(name); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("remove stale session descriptor temp file: %w", err)
		}
	}
	return nil
}
