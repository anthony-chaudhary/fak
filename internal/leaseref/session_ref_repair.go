package leaseref

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ReapMalformedSessionRefs removes only syntactically malformed loose refs in
// refs/fak/locks/session-*. A valid 40- or 64-hex object ID is retained even if
// its object is missing: that is object-database damage and needs operator
// investigation, not cache cleanup.
//
// Git's update-ref normally writes through a sibling .lock and atomically
// renames it. Process or host crashes can nevertheless leave a torn loose ref
// (observed as 41 NUL bytes); Git then aborts `log --all`, `reflog --all`, and
// fresh-session discovery with "fatal: bad object". This narrow startup repair
// restores Git traversal without touching ordinary refs or valid lease refs.
func (s *Store) ReapMalformedSessionRefs(ctx context.Context) ([]string, error) {
	out, code, err := s.run(ctx, s.dir, "rev-parse", "--path-format=absolute", "--git-common-dir")
	if err != nil {
		return nil, fmt.Errorf("leaseref: resolve git common dir: %w", err)
	}
	if code != 0 {
		return nil, fmt.Errorf("leaseref: rev-parse --git-common-dir exited %d", code)
	}
	common := strings.TrimSpace(out)
	if common == "" {
		return nil, fmt.Errorf("leaseref: rev-parse --git-common-dir produced no path")
	}
	root := filepath.Join(common, "refs", "fak", "locks")
	entries, err := os.ReadDir(root)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("leaseref: read session ref dir: %w", err)
	}

	var reaped []string
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasPrefix(name, "session-") || strings.HasSuffix(name, ".lock") {
			continue
		}
		path := filepath.Join(root, name)
		malformed, repairErr := repairMalformedLooseSessionRef(path)
		if repairErr != nil {
			return reaped, fmt.Errorf("leaseref: repair %s: %w", filepath.ToSlash(filepath.Join(refPrefix, name)), repairErr)
		}
		if !malformed {
			continue
		}
		reaped = append(reaped, filepath.ToSlash(filepath.Join(refPrefix, name)))
	}
	return reaped, nil
}

func validLooseObjectID(s string) bool {
	if len(s) != 40 && len(s) != 64 {
		return false
	}
	for _, r := range s {
		if !((r >= '0' && r <= '9') || (r >= 'a' && r <= 'f') || (r >= 'A' && r <= 'F')) {
			return false
		}
	}
	return true
}

// repairMalformedLooseSessionRef takes Git's sibling .lock before its final
// read and removal. A concurrent update-ref therefore either wins first (and
// we inspect its complete value) or observes our lock and retries/fails; we
// never unlink a valid ref that replaced the malformed file after inspection.
func repairMalformedLooseSessionRef(path string) (bool, error) {
	lockPath := path + ".lock"
	lock, err := os.OpenFile(lockPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if os.IsExist(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if closeErr := lock.Close(); closeErr != nil {
		_ = os.Remove(lockPath)
		return false, closeErr
	}
	defer os.Remove(lockPath)

	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if validLooseObjectID(strings.TrimSpace(string(data))) {
		return false, nil
	}
	if err := os.Remove(path); err != nil {
		return false, err
	}
	return true, nil
}
