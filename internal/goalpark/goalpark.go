// Package goalpark persists long provider Retry-After waits outside a worker's
// active context budget and arbitrates exactly-once resume after the reset.
package goalpark

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const Schema = "fak.goal-park.v1"
const LongWaitFloor = time.Hour

var (
	ErrNotDue   = errors.New("goalpark: earliest legal resume time has not arrived")
	ErrClaimed  = errors.New("goalpark: resume already claimed")
	ErrNotFound = errors.New("goalpark: parked goal not found")
)

type Record struct {
	Schema      string   `json:"schema"`
	Goal        string   `json:"goal"`
	Lane        string   `json:"lane,omitempty"`
	Reason      string   `json:"reason"`
	ParkedUntil int64    `json:"parked_until"`
	ParkedAt    int64    `json:"parked_at"`
	Account     string   `json:"account,omitempty"`
	Pool        string   `json:"pool,omitempty"`
	Lease       string   `json:"lease,omitempty"`
	Witness     string   `json:"witness_requirement,omitempty"`
	Command     []string `json:"command"`
	ClaimedAt   int64    `json:"claimed_at,omitempty"`
	ClaimedBy   string   `json:"claimed_by,omitempty"`
	NextAction  string   `json:"next_legal_action"`
}

type Store struct{ Dir string }

func (s Store) path(goal string) string {
	sum := sha256.Sum256([]byte(strings.TrimSpace(goal)))
	return filepath.Join(s.Dir, hex.EncodeToString(sum[:12])+".json")
}

func (s Store) Park(r Record) error {
	if strings.TrimSpace(r.Goal) == "" || r.ParkedUntil <= r.ParkedAt || len(r.Command) == 0 {
		return errors.New("goalpark: invalid park record")
	}
	r.Schema = Schema
	r.ClaimedAt = 0
	r.ClaimedBy = ""
	r.NextAction = "wait until parked_until; then supervisor atomically claims and resumes the same command"
	if err := os.MkdirAll(s.Dir, 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	tmp, err := os.CreateTemp(s.Dir, "park-*.tmp")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if _, err = tmp.Write(b); err == nil {
		err = tmp.Sync()
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(name, s.path(r.Goal))
}

// List returns every readable parked record. Malformed siblings are skipped so
// one torn/foreign file cannot hide the rest of the supervisor queue.
func (s Store) List() ([]Record, error) {
	entries, err := os.ReadDir(s.Dir)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	out := make([]Record, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		b, readErr := os.ReadFile(filepath.Join(s.Dir, entry.Name()))
		if readErr != nil {
			continue
		}
		var r Record
		if json.Unmarshal(b, &r) == nil && r.Schema == Schema {
			out = append(out, r)
		}
	}
	return out, nil
}

func (s Store) Load(goal string) (Record, error) {
	b, err := os.ReadFile(s.path(goal))
	if errors.Is(err, os.ErrNotExist) {
		return Record{}, ErrNotFound
	}
	if err != nil {
		return Record{}, err
	}
	var r Record
	if err = json.Unmarshal(b, &r); err != nil {
		return Record{}, err
	}
	if r.Schema != Schema {
		return Record{}, errors.New("goalpark: unsupported schema")
	}
	return r, nil
}

// ClaimDue creates an exclusive claim sidecar before updating the public record.
// Across process restarts and concurrent supervisors exactly one caller can win.
func (s Store) ClaimDue(goal, supervisor string, now time.Time) (Record, error) {
	r, err := s.Load(goal)
	if err != nil {
		return Record{}, err
	}
	if r.ClaimedAt != 0 {
		return r, ErrClaimed
	}
	if now.Unix() < r.ParkedUntil {
		return r, ErrNotDue
	}
	claim := s.path(goal) + ".claim"
	f, err := os.OpenFile(claim, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if errors.Is(err, os.ErrExist) {
		return r, ErrClaimed
	}
	if err != nil {
		return r, err
	}
	fmt.Fprintf(f, "%s %d\n", supervisor, now.Unix())
	if err = f.Close(); err != nil {
		return r, err
	}
	r.ClaimedAt = now.Unix()
	r.ClaimedBy = supervisor
	r.NextAction = "resume claimed; launch command exactly once and retain lease/witness contract"
	b, _ := json.MarshalIndent(r, "", "  ")
	b = append(b, '\n')
	if err = os.WriteFile(s.path(goal), b, 0o600); err != nil {
		return r, err
	}
	return r, nil
}

func ParseRetryAfter(value string, now time.Time) (time.Time, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return time.Time{}, false
	}
	if seconds, err := strconv.ParseInt(value, 10, 64); err == nil && seconds >= 0 {
		return now.Add(time.Duration(seconds) * time.Second), true
	}
	when, err := http.ParseTime(value)
	return when, err == nil && !when.Before(now)
}

// RecordLongRetry persists only a provider 429 whose legal wait is at least one
// hour. Other 429 classes remain on their existing transient/quarantine paths.
func (s Store) RecordLongRetry(status int, h http.Header, now time.Time, r Record) (bool, error) {
	if status != http.StatusTooManyRequests {
		return false, nil
	}
	until, ok := ParseRetryAfter(h.Get("Retry-After"), now)
	if !ok || until.Sub(now) < LongWaitFloor {
		return false, nil
	}
	r.Reason = "LONG_RETRY_AFTER"
	r.ParkedAt = now.Unix()
	r.ParkedUntil = until.Unix()
	return true, s.Park(r)
}
