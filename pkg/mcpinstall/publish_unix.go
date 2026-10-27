//go:build !windows

package mcpinstall

import (
	"errors"
	"os"
	"path/filepath"
)

const (
	platformGuard      = "unsupported-no-strong-config-guard"
	platformDurability = "artifact-file-and-directory-sync;config-update-unsupported"
)

func openGuardedConfig(string) (*os.File, error) {
	return nil, errors.New("mcpinstall: strong config append/rollback guard is supported only on Windows")
}

func configIdentity(f *os.File) (string, error) {
	// Preparation remains observational on unsupported hosts. This marker grants
	// no inode protection: public Install/Rollback fail before writing anything.
	if _, err := f.Stat(); err != nil {
		return "", err
	}
	return "unsupported", nil
}

func publishNew(tmp, path string) error {
	if err := os.Link(tmp, path); err != nil {
		return err
	}
	if err := os.Remove(tmp); err != nil {
		return err
	}
	return syncPublished(path)
}

func syncPublished(path string) error {
	f, err := os.Open(filepath.Dir(path))
	if err != nil {
		return err
	}
	defer f.Close()
	return f.Sync()
}
