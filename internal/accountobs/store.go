package accountobs

import (
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Store persists the latest coalesced provider quota observation per admission key.
type Store struct{ Dir string }

type Record struct {
	Key        string            `json:"key"`
	UpdatedAt  time.Time         `json:"updated_at"`
	LastStatus int               `json:"last_status"`
	Headers    map[string]string `json:"headers"`
}

func (s Store) Observe(key string, status int, h http.Header, at time.Time) error {
	if strings.TrimSpace(key) == "" {
		key = "default"
	}
	if at.IsZero() {
		at = time.Now().UTC()
	}
	rec, err := s.Load(key)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if !rec.UpdatedAt.IsZero() && at.Before(rec.UpdatedAt) {
		return nil
	}
	if rec.Headers == nil {
		rec.Headers = map[string]string{}
	}
	for k, values := range h {
		lk := strings.ToLower(k)
		if !hasObservedPrefix(lk) || len(values) == 0 {
			continue
		}
		rec.Headers[lk] = strings.TrimSpace(values[0])
	}
	rec.Key, rec.UpdatedAt, rec.LastStatus = key, at.UTC(), status
	if err := os.MkdirAll(s.Dir, 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(s.Dir, ".quota-*.tmp")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if err = tmp.Chmod(0o600); err == nil {
		_, err = tmp.Write(append(data, '\n'))
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(name, s.path(key))
}

func (s Store) Load(key string) (Record, error) {
	data, err := os.ReadFile(s.path(key))
	if err != nil {
		return Record{}, err
	}
	var rec Record
	if err := json.Unmarshal(data, &rec); err != nil {
		return Record{}, err
	}
	return rec, nil
}

func (s Store) path(key string) string {
	safe := strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_' {
			return r
		}
		return '-'
	}, key)
	return filepath.Join(s.Dir, safe+".json")
}

func hasObservedPrefix(name string) bool {
	for _, p := range observedPrefixes {
		if strings.HasPrefix(name, p) {
			return true
		}
	}
	return false
}
