//go:build linux || darwin

package researcharm

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

const leaseStoreSchema = "fak.researcharm.lease-store/v1"
const maxLeaseStoreBytes = 1 << 20
const maxStoredLeases = 1024

type leaseSnapshot struct {
	Schema         string      `json:"schema"`
	AdmissionReady *bool       `json:"admission_ready"`
	Leases         []LeaseInfo `json:"leases"`
}

type leaseStore struct {
	path          string
	initialized   bool
	lock          *os.File
	renameFile    func(string, string) error
	syncDirectory func(string) error
}

// NewDurableCoordinator restores acknowledged leases from a private store.
// The caller must create the containing directory with mode 0700. A fresh
// store denies inference until its first exclusive lease is durably acquired.
func NewDurableCoordinator(max int, path string) (*Coordinator, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	dir := filepath.Dir(absolute)
	resolved, err := filepath.EvalSymlinks(dir)
	if err != nil {
		return nil, fmt.Errorf("researcharm: store directory must exist without symlinks")
	}
	if err := privatePath(dir, true); err != nil {
		return nil, err
	}
	absolute = filepath.Join(resolved, filepath.Base(absolute))
	lockPath := absolute + ".lock"
	lock, err := os.OpenFile(lockPath, os.O_RDWR|os.O_CREATE|os.O_EXCL|syscall.O_NOFOLLOW, 0600)
	created := err == nil
	if errors.Is(err, os.ErrExist) {
		lock, err = os.OpenFile(lockPath, os.O_RDWR|syscall.O_NOFOLLOW, 0)
	}
	if err != nil {
		return nil, fmt.Errorf("researcharm: open store lock: %w", err)
	}
	store := &leaseStore{path: absolute, lock: lock, renameFile: os.Rename, syncDirectory: syncLeaseDirectory}
	success := false
	defer func() {
		if !success {
			_ = store.close()
		}
	}()
	if err := privateFile(lock); err != nil {
		return nil, err
	}
	if err := flock.TryLock(lock); err != nil {
		return nil, fmt.Errorf("researcharm: store already owned: %w", err)
	}
	c := NewCoordinator(max)
	c.store = store
	if err := privatePath(absolute, false); err != nil {
		if !created || !errors.Is(err, os.ErrNotExist) {
			return nil, err
		}
		if _, err := store.save(c.leases, false); err != nil {
			return nil, err
		}
	} else {
		f, err := os.OpenFile(absolute, os.O_RDONLY|syscall.O_NOFOLLOW, 0)
		if err != nil {
			return nil, err
		}
		data, readErr := io.ReadAll(io.LimitReader(f, maxLeaseStoreBytes+1))
		statErr := privateFile(f)
		closeErr := f.Close()
		if err := errors.Join(readErr, statErr, closeErr); err != nil {
			return nil, err
		}
		if len(data) > maxLeaseStoreBytes {
			return nil, fmt.Errorf("researcharm: lease store exceeds size limit")
		}
		snapshot, err := decodeLeaseSnapshot(data)
		if err != nil {
			return nil, err
		}
		c.admissionReady = *snapshot.AdmissionReady
		now := time.Now()
		for i := range snapshot.Leases {
			lease := snapshot.Leases[i]
			if now.Before(lease.ExpiresAt) {
				c.leases[lease.ID] = &lease
			}
		}
	}
	store.initialized = true
	success = true
	return c, nil
}

func privatePath(path string, directory bool) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if directory && !info.IsDir() || !directory && !info.Mode().IsRegular() {
		return fmt.Errorf("researcharm: nonregular private store path")
	}
	return privateInfo(info, directory)
}
func privateFile(f *os.File) error {
	info, err := f.Stat()
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("researcharm: nonregular store file")
	}
	return privateInfo(info, false)
}
func privateInfo(info os.FileInfo, directory bool) error {
	mode := os.FileMode(0600)
	if directory {
		mode = 0700
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || stat.Uid != uint32(os.Geteuid()) || info.Mode().Perm() != mode || info.Mode()&(os.ModeSetuid|os.ModeSetgid|os.ModeSticky) != 0 {
		return fmt.Errorf("researcharm: store requires current owner and private mode %04o", mode)
	}
	return nil
}

func decodeLeaseSnapshot(data []byte) (leaseSnapshot, error) {
	var snapshot leaseSnapshot
	if err := rejectDuplicateFields(json.NewDecoder(bytes.NewReader(data))); err != nil {
		return snapshot, err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&snapshot); err != nil {
		return snapshot, err
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return snapshot, fmt.Errorf("researcharm: trailing store data")
	}
	if snapshot.Schema != leaseStoreSchema || snapshot.AdmissionReady == nil || snapshot.Leases == nil || len(snapshot.Leases) > maxStoredLeases {
		return snapshot, fmt.Errorf("researcharm: invalid lease store schema or bounds")
	}
	ids, arms, tokens := map[string]bool{}, map[string]bool{}, map[string]bool{}
	exclusiveArm := ""
	for _, l := range snapshot.Leases {
		if !strings.HasPrefix(l.ID, "lease-") || !validLeaseHex(strings.TrimPrefix(l.ID, "lease-"), 16) || !validLeaseHex(l.Token, 32) || strings.TrimSpace(l.ArmID) == "" || l.HolderPID < 0 || l.Concurrency < 0 || (l.Mode != LeaseModeShared && l.Mode != LeaseModeExclusive) || l.CreatedAt.IsZero() || !l.ExpiresAt.After(l.CreatedAt) || ids[l.ID] || arms[l.ArmID] || tokens[l.Token] {
			return snapshot, fmt.Errorf("researcharm: invalid or duplicate stored lease")
		}
		ids[l.ID], arms[l.ArmID], tokens[l.Token] = true, true, true
		if l.Mode == LeaseModeExclusive {
			exclusiveArm = l.ArmID
		}
	}
	if !*snapshot.AdmissionReady && len(snapshot.Leases) > 0 || exclusiveArm != "" && len(snapshot.Leases) > 1 {
		return snapshot, fmt.Errorf("researcharm: inconsistent stored lease admission")
	}
	return snapshot, nil
}
func validLeaseHex(value string, length int) bool {
	if len(value) != length || value != strings.ToLower(value) {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}
func rejectDuplicateFields(d *json.Decoder) error {
	token, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	if delim == '{' {
		seen := map[string]bool{}
		for d.More() {
			key, err := d.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			if !ok || name != strings.ToLower(name) || seen[strings.ToLower(name)] {
				return fmt.Errorf("researcharm: duplicate JSON field")
			}
			seen[name] = true
			if err := rejectDuplicateFields(d); err != nil {
				return err
			}
		}
	} else if delim == '[' {
		for d.More() {
			if err := rejectDuplicateFields(d); err != nil {
				return err
			}
		}
	} else {
		return fmt.Errorf("researcharm: invalid JSON delimiter")
	}
	_, err = d.Token()
	return err
}

// save returns uncertainty once replacement was attempted. No mutation is
// acknowledged before the replacement and parent directory are both synced.
func (s *leaseStore) save(leases map[string]*LeaseInfo, ready bool) (bool, error) {
	if err := privatePath(filepath.Dir(s.path), true); err != nil {
		return false, err
	}
	if err := privatePath(s.path, false); err != nil {
		if s.initialized && errors.Is(err, os.ErrNotExist) {
			return true, fmt.Errorf("researcharm: initialized store snapshot missing")
		}
		if !errors.Is(err, os.ErrNotExist) {
			return false, err
		}
	}
	if len(leases) > maxStoredLeases {
		return false, fmt.Errorf("researcharm: too many stored leases")
	}
	snapshot := leaseSnapshot{Schema: leaseStoreSchema, AdmissionReady: &ready, Leases: make([]LeaseInfo, 0, len(leases))}
	for _, lease := range leases {
		snapshot.Leases = append(snapshot.Leases, *lease)
	}
	sort.Slice(snapshot.Leases, func(i, j int) bool { return snapshot.Leases[i].ID < snapshot.Leases[j].ID })
	data, err := json.Marshal(snapshot)
	if err != nil {
		return false, err
	}
	if len(data) > maxLeaseStoreBytes {
		return false, fmt.Errorf("researcharm: lease store exceeds size limit")
	}
	if _, err := decodeLeaseSnapshot(data); err != nil {
		return false, err
	}
	temp, err := os.CreateTemp(filepath.Dir(s.path), ".arm-leases-*")
	if err != nil {
		return false, err
	}
	defer os.Remove(temp.Name())
	_, writeErr := temp.Write(data)
	syncErr := temp.Sync()
	closeErr := temp.Close()
	if err := errors.Join(writeErr, syncErr, closeErr); err != nil {
		return false, err
	}
	if err := s.renameFile(temp.Name(), s.path); err != nil {
		return true, err
	}
	if err := s.syncDirectory(filepath.Dir(s.path)); err != nil {
		return true, err
	}
	return false, nil
}
func syncLeaseDirectory(dir string) error {
	f, err := os.Open(dir)
	if err != nil {
		return err
	}
	return errors.Join(f.Sync(), f.Close())
}
func (s *leaseStore) close() error { return errors.Join(flock.Unlock(s.lock), s.lock.Close()) }
