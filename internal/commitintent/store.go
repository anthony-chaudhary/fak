package commitintent

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"time"
)

const (
	StoreDirName   = "commit-intents"
	StoreQueueFile = "queue.json"
	StoreLockFile  = "queue.lock"
)

type Store struct {
	Dir         string
	Now         func() time.Time
	LockTimeout time.Duration
	RetryDelay  time.Duration
}

func DefaultQueueDir(repoRoot string) string {
	if repoRoot == "" {
		return filepath.Join(".fak", StoreDirName)
	}
	return filepath.Join(repoRoot, ".fak", StoreDirName)
}

func (s Store) Load() (Queue, error) {
	dir := s.dir()
	raw, err := os.ReadFile(filepath.Join(dir, StoreQueueFile))
	if errors.Is(err, fs.ErrNotExist) {
		return NewQueue(), nil
	}
	if err != nil {
		return Queue{}, err
	}
	return ParseQueue(raw)
}

func (s Store) Submit(intent Intent) (Queue, SubmitRecord, error) {
	var next Queue
	var rec SubmitRecord
	err := s.withLock(func() error {
		q, err := s.Load()
		if err != nil {
			return err
		}
		next, rec, err = Submit(q, s.now(), intent)
		if err != nil {
			return err
		}
		return s.save(next)
	})
	return next, rec, err
}

func (s Store) Drain(currentBaseSHA string, limit int) (DrainPlan, error) {
	q, err := s.Load()
	if err != nil {
		return DrainPlan{}, err
	}
	return Drain(q.Records, currentBaseSHA, limit), nil
}

func MarshalQueue(q Queue) ([]byte, error) {
	if stringsTrim(q.Schema) == "" {
		q.Schema = QueueSchema
	}
	if err := ValidateQueue(q); err != nil {
		return nil, err
	}
	b, err := json.MarshalIndent(q, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}

func ParseQueue(data []byte) (Queue, error) {
	var q Queue
	if err := json.Unmarshal(data, &q); err != nil {
		return Queue{}, err
	}
	if err := ValidateQueue(q); err != nil {
		return Queue{}, err
	}
	for i := range q.Records {
		q.Records[i].Intent, _ = NormalizeIntent(q.Records[i].Intent)
	}
	if q.NextSequence <= nextSequence(q.Records) {
		q.NextSequence = nextSequence(q.Records)
	}
	return q, nil
}

func ValidateQueue(q Queue) error {
	if q.Schema == "" {
		return fieldError("schema", ErrMissingField, "queue schema is required")
	}
	if q.Schema != QueueSchema {
		return fieldError("schema", ErrInvalidField, q.Schema)
	}
	if q.NextSequence <= 0 {
		return fieldError("next_sequence", ErrInvalidField, "must be positive")
	}
	for _, rec := range q.Records {
		if err := ValidateRecord(rec); err != nil {
			return err
		}
	}
	return nil
}

func (s Store) save(q Queue) error {
	dir := s.dir()
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	b, err := MarshalQueue(q)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, StoreQueueFile), b, 0o600)
}

func (s Store) withLock(fn func() error) error {
	dir := s.dir()
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	lockPath := filepath.Join(dir, StoreLockFile)
	timeout := s.LockTimeout
	if timeout <= 0 {
		timeout = 5 * time.Second
	}
	delay := s.RetryDelay
	if delay <= 0 {
		delay = 10 * time.Millisecond
	}
	deadline := time.Now().Add(timeout)
	for {
		f, err := os.OpenFile(lockPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
		if err == nil {
			_, _ = fmt.Fprintf(f, "%d\n", os.Getpid())
			closeErr := f.Close()
			defer os.Remove(lockPath)
			if closeErr != nil {
				return closeErr
			}
			return fn()
		}
		if !isBusyLock(err, lockPath) {
			return err
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("commitintent: queue lock busy: %w", err)
		}
		time.Sleep(delay)
	}
}

func isBusyLock(err error, path string) bool {
	if errors.Is(err, fs.ErrExist) {
		return true
	}
	if errors.Is(err, fs.ErrPermission) {
		if _, statErr := os.Stat(path); statErr == nil {
			return true
		}
	}
	return false
}

func (s Store) now() time.Time {
	if s.Now != nil {
		return s.Now()
	}
	return time.Now()
}

func (s Store) dir() string {
	if s.Dir != "" {
		return s.Dir
	}
	return DefaultQueueDir("")
}

func stringsTrim(s string) string {
	for len(s) > 0 && (s[0] == ' ' || s[0] == '\t' || s[0] == '\n' || s[0] == '\r') {
		s = s[1:]
	}
	for len(s) > 0 {
		last := s[len(s)-1]
		if last != ' ' && last != '\t' && last != '\n' && last != '\r' {
			break
		}
		s = s[:len(s)-1]
	}
	return s
}
