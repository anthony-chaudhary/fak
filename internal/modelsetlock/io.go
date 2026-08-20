package modelsetlock

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
)

const maxLockBytes = 4 << 20

// ParseJSON strictly reads one canonical lock, verifies its schema and digest,
// and rejects semantically valid but non-canonical bytes.
func ParseJSON(raw []byte) (Lock, error) {
	if len(raw) > maxLockBytes {
		return Lock{}, lockError(failure(CodeMalformed, "$", "lock exceeds the 4 MiB admission limit", "regenerate a bounded model-set lock"))
	}
	var lock Lock
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&lock); err != nil {
		return Lock{}, lockError(failure(CodeMalformed, "$", "lock is not one strict JSON document", "discard the malformed lock and resolve again"))
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return Lock{}, lockError(failure(CodeMalformed, "$", "lock contains trailing JSON data", "keep exactly one canonical lock document"))
	}
	canonical := canonicalizeLock(lock)
	if err := validateLock(canonical); err != nil {
		return Lock{}, err
	}
	canonicalJSON, err := CanonicalJSON(canonical)
	if err != nil {
		return Lock{}, err
	}
	if !bytes.Equal(raw, canonicalJSON) {
		return Lock{}, lockError(failure(CodeNonCanonical, "$", "lock bytes are not the canonical wire representation", "rewrite the lock through WriteFile"))
	}
	return canonical, nil
}

// ReadFile independently parses and verifies a lock from disk.
func ReadFile(path string) (Lock, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Lock{}, lockError(failure(CodeIO, path, err.Error(), "make the lock readable and retry"))
	}
	return ParseJSON(raw)
}

// WriteFile validates before mutation, then fsyncs and atomically renames a
// same-directory temporary file so a failed write preserves the prior lock.
func WriteFile(path string, lock Lock) error {
	raw, err := CanonicalJSON(lock)
	if err != nil {
		return err
	}
	if current, readErr := os.ReadFile(path); readErr == nil && bytes.Equal(current, raw) {
		return nil
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return lockError(failure(CodeIO, path, err.Error(), "make the lock directory writable and retry"))
	}
	tmp, err := os.CreateTemp(dir, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return lockError(failure(CodeIO, path, err.Error(), "make the lock directory writable and retry"))
	}
	tmpName := tmp.Name()
	keepTemp := true
	defer func() {
		_ = tmp.Close()
		if keepTemp {
			_ = os.Remove(tmpName)
		}
	}()
	if err := tmp.Chmod(0o644); err != nil {
		return lockError(failure(CodeIO, path, err.Error(), "permit writing the temporary lock and retry"))
	}
	if _, err := tmp.Write(raw); err != nil {
		return lockError(failure(CodeIO, path, err.Error(), "free disk space and retry the lock write"))
	}
	if err := tmp.Sync(); err != nil {
		return lockError(failure(CodeIO, path, err.Error(), "repair storage durability and retry"))
	}
	if err := tmp.Close(); err != nil {
		return lockError(failure(CodeIO, path, err.Error(), "repair storage durability and retry"))
	}
	if err := os.Rename(tmpName, path); err != nil {
		return lockError(failure(CodeIO, path, err.Error(), "make the destination replaceable and retry; the prior lock is unchanged"))
	}
	keepTemp = false
	return nil
}
