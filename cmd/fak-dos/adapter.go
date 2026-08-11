// Package main implements the public fak adapter for writable DOS queue entries.
package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const relativeLog = ".dos/decisions/host.jsonl"

// Row is a host-authored decision projected alongside native DOS rows.
type Row struct {
	Key          string         `json:"key"`
	Action       string         `json:"action"`
	Severity     string         `json:"severity"`
	Payload      map[string]any `json:"payload,omitempty"`
	Kind         string         `json:"kind"`
	ResolverKind string         `json:"resolver_kind"`
	SourcePath   string         `json:"source_path"`
	CreatedAt    string         `json:"created_at"`
}

type event struct {
	Op  string `json:"op"`
	Row *Row   `json:"row,omitempty"`
	Key string `json:"key"`
}

func LogPath(workspace string) string {
	return filepath.Join(workspace, filepath.FromSlash(relativeLog))
}

func validate(key, action, severity string) error {
	if strings.TrimSpace(key) == "" {
		return errors.New("--key is required")
	}
	if strings.TrimSpace(action) == "" {
		return errors.New("--action is required")
	}
	if strings.TrimSpace(severity) == "" {
		return errors.New("--severity is required")
	}
	if strings.ContainsAny(key, "\r\n") {
		return errors.New("--key must be one line")
	}
	return nil
}

// Add writes an idempotent keyed decision. Repeating the same write returns the existing row;
// reusing a key for different content is rejected.
func Add(workspace, key, action, severity string, payload map[string]any, now time.Time) (Row, bool, error) {
	if err := validate(key, action, severity); err != nil {
		return Row{}, false, err
	}
	var out Row
	created := false
	err := withLock(workspace, func() error {
		rows, err := Read(workspace)
		if err != nil {
			return err
		}
		for _, r := range rows {
			if r.Key != key {
				continue
			}
			if r.Action != action || r.Severity != severity || !jsonEqual(r.Payload, payload) {
				return fmt.Errorf("decision key %q already exists with different action, severity, or payload", key)
			}
			out = r
			return nil
		}
		out = Row{Key: key, Action: action, Severity: severity, Payload: payload, Kind: "HOST_QUEUE_ITEM", ResolverKind: "HUMAN", SourcePath: LogPath(workspace), CreatedAt: now.UTC().Format(time.RFC3339Nano)}
		if err := appendEvent(workspace, event{Op: "add", Key: key, Row: &out}); err != nil {
			return err
		}
		created = true
		return nil
	})
	return out, created, err
}

// Remove idempotently removes a host decision and returns whether a live row existed.
func Remove(workspace, key string) (bool, error) {
	if strings.TrimSpace(key) == "" {
		return false, errors.New("--key is required")
	}
	removed := false
	err := withLock(workspace, func() error {
		rows, err := Read(workspace)
		if err != nil {
			return err
		}
		for _, r := range rows {
			if r.Key == key {
				removed = true
				break
			}
		}
		if !removed {
			return nil
		}
		return appendEvent(workspace, event{Op: "remove", Key: key})
	})
	return removed, err
}

// Read folds the append-only host log into current rows.
func Read(workspace string) ([]Row, error) {
	f, err := os.Open(LogPath(workspace))
	if errors.Is(err, os.ErrNotExist) {
		return []Row{}, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	live := map[string]Row{}
	order := []string{}
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 64*1024), 1024*1024)
	line := 0
	for s.Scan() {
		line++
		var e event
		if err := json.Unmarshal(s.Bytes(), &e); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", LogPath(workspace), line, err)
		}
		switch e.Op {
		case "add":
			if e.Row == nil || e.Key == "" || e.Row.Key != e.Key {
				return nil, fmt.Errorf("%s:%d: invalid add event", LogPath(workspace), line)
			}
			if _, ok := live[e.Key]; !ok {
				order = append(order, e.Key)
			}
			live[e.Key] = *e.Row
		case "remove":
			delete(live, e.Key)
		default:
			return nil, fmt.Errorf("%s:%d: unknown operation %q", LogPath(workspace), line, e.Op)
		}
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	out := make([]Row, 0, len(live))
	for _, k := range order {
		if r, ok := live[k]; ok {
			out = append(out, r)
		}
	}
	return out, nil
}

func appendEvent(workspace string, e event) error {
	path := LogPath(workspace)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := json.Marshal(e)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	if _, err = f.Write(b); err == nil {
		err = f.Sync()
	}
	cerr := f.Close()
	if err == nil {
		err = cerr
	}
	return err
}

func withLock(workspace string, fn func() error) error {
	lock := LogPath(workspace) + ".lock"
	if err := os.MkdirAll(filepath.Dir(lock), 0o755); err != nil {
		return err
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		err := os.Mkdir(lock, 0o700)
		if err == nil {
			defer os.Remove(lock)
			return fn()
		}
		if !errors.Is(err, os.ErrExist) {
			return err
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("timed out acquiring %s", lock)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func jsonEqual(a, b map[string]any) bool {
	x, _ := json.Marshal(a)
	y, _ := json.Marshal(b)
	return string(x) == string(y)
}
