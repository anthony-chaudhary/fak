package leaseref

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ReapMalformedLooseRefs removes only syntactically unusable loose refs under
// refs/fak/locks/ - the ONE lease namespace every fak side ref shares
// (session-, intent-, contract-, epoch-, resolve-, issue-, and the bare-id
// lease). It is the startup repair for the "fatal: bad object" wedge.
//
// A loose ref file whose content is not a full object id is unresolvable, and
// Git aborts EVERY --all traversal that touches it: `log --all`, `reflog --all`,
// `rev-list --all`, `rev-list --parents --all`, and fresh-session discovery all
// die with `fatal: bad object refs/fak/locks/<name>`. Those traversals are on
// the landing, sync and resume paths, so ONE torn byte wedges the whole
// repository's git work - not merely the lease it names. A process or host
// crash mid-`update-ref` leaves such a file (observed as 41 NUL bytes); a short
// or non-hex value is the same defect written by a different path.
//
// Two classes are deliberately RETAINED, holding the invariant "touch neither
// ordinary refs nor valid lease refs":
//
//   - a full 40- or 64-hex object id, even when its object is MISSING: that is
//     object-database damage and needs operator investigation, not cache cleanup;
//   - a symbolic ref ("ref: refs/..."): Git resolves it, so it does not wedge a
//     traversal, and it is a live lease a startup repair must not silently drop.
//
// Scope is the lease namespace only, and that is the point: Git's ordinary refs
// (refs/heads, refs/tags, refs/remotes) are never touched, so a torn branch ref
// is never silently deleted by a startup repair. Widening to the whole
// refs/fak/locks/ namespace (from the earlier session-* only) matters because
// the wedge is a property of the NAMESPACE, not of any one ref kind: a torn
// `contract-<ticket>` wedges `--all` exactly as a torn `session-<id>` does.
func (s *Store) ReapMalformedLooseRefs(ctx context.Context) ([]string, error) {
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
	// Derive the on-disk lease dir from refPrefix, so the one spelling of the
	// namespace lives in one place.
	root := filepath.Join(common, filepath.FromSlash(strings.TrimPrefix(strings.TrimSuffix(refPrefix, "/"), "/")))
	entries, err := os.ReadDir(root)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("leaseref: read lease-ref dir: %w", err)
	}

	var reaped []string
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || strings.HasSuffix(name, ".lock") {
			continue
		}
		path := filepath.Join(root, name)
		malformed, repairErr := repairMalformedLooseRef(path)
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

// isSymbolicLooseRef reports whether the ref file's content is a symref
// ("ref: <target>") rather than an object id. Git resolves a symref on every
// traversal, so a symref - even a dangling one - never produces the "bad
// object" wedge, and stays a live lease this repair must not drop.
func isSymbolicLooseRef(s string) bool {
	return strings.HasPrefix(strings.TrimSpace(s), "ref: ")
}

// usableLooseRefContent is the ONE authority for "can Git resolve this loose ref
// file?" - shared by every path that maintains refs/fak/locks/ (the startup
// repair below, which deletes; and QuarantineMalformedLockRefsInDir, which
// renames aside), so the two can never drift on what counts as usable.
//
// Usable means either a full object id (40- or 64-hex) or a symref. A full id
// whose object happens to be MISSING is still usable: Git can resolve the ref
// itself, so it does not wedge `--all`, and a missing object is object-database
// damage needing operator investigation rather than ref cleanup. A symref
// resolves by following its target, so it does not wedge either.
//
// Anything else is unusable: Git cannot resolve the ref, and every `--all`
// traversal that touches it aborts with `fatal: bad object <ref>`.
func usableLooseRefContent(content string) bool {
	content = strings.TrimSpace(content)
	// validObjectID is the package's existing authority for a resolvable object
	// id, and it rejects the all-zeros value Git writes as an uninitialized
	// placeholder - the classic torn ref - so an all-zero file is UNUSABLE.
	return validObjectID(content) || isSymbolicLooseRef(content)
}

// repairMalformedLooseRef takes Git's sibling .lock before its final read and
// removal. A concurrent update-ref therefore either wins first (and we inspect
// its complete value) or observes our lock and retries/fails; we never unlink a
// valid ref that replaced the malformed file after inspection.
func repairMalformedLooseRef(path string) (bool, error) {
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
	content := strings.TrimSpace(string(data))
	if usableLooseRefContent(content) {
		return false, nil
	}
	if err := os.Remove(path); err != nil {
		return false, err
	}
	return true, nil
}
