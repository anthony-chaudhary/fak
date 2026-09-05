package agent

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// DefaultSessionCheckpointDir is the default workspace directory for session checkpoints.
const DefaultSessionCheckpointDir = ".fak/sessions"

// SessionCheckpoint represents the durable state of an agent session at a turn boundary.
type SessionCheckpoint struct {
	SessionID string    `json:"session_id"`
	CWD       string    `json:"cwd"`
	Task      string    `json:"task"`
	Model     string    `json:"model"`
	Provider  string    `json:"provider,omitempty"`
	BaseURL   string    `json:"base_url,omitempty"`
	Messages  []Message `json:"messages"`
	Turn      int       `json:"turn"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
	Status    string    `json:"status"`
}

// SaveSessionCheckpoint writes a session checkpoint to a JSON file in dir named <session_id>.json.
func SaveSessionCheckpoint(dir string, cp SessionCheckpoint) error {
	if strings.TrimSpace(cp.SessionID) == "" {
		return errors.New("session checkpoint: session_id is required")
	}
	if strings.TrimSpace(dir) == "" {
		dir = DefaultSessionCheckpointDir
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("session checkpoint: create dir %s: %w", dir, err)
	}
	if cp.CreatedAt.IsZero() {
		cp.CreatedAt = time.Now().UTC()
	}
	cp.UpdatedAt = time.Now().UTC()
	if cp.Status == "" {
		cp.Status = "active"
	}
	if cp.CWD == "" {
		if cwd, err := os.Getwd(); err == nil {
			cp.CWD = cwd
		}
	}

	filename := cp.SessionID
	if !strings.HasSuffix(filename, ".json") {
		filename += ".json"
	}
	targetPath := filepath.Join(dir, filename)
	data, err := json.MarshalIndent(cp, "", "  ")
	if err != nil {
		return fmt.Errorf("session checkpoint: marshal %s: %w", cp.SessionID, err)
	}
	if err := os.WriteFile(targetPath, data, 0o644); err != nil {
		return fmt.Errorf("session checkpoint: write %s: %w", targetPath, err)
	}
	return nil
}

// LoadSessionCheckpoint reads a session checkpoint from either an explicit file path or
// a session ID resolved within defaultDir (or DefaultSessionCheckpointDir if empty).
func LoadSessionCheckpoint(pathOrID string, defaultDir string) (*SessionCheckpoint, error) {
	pathOrID = strings.TrimSpace(pathOrID)
	if pathOrID == "" {
		return nil, errors.New("session checkpoint: path or session ID is required")
	}
	if defaultDir == "" {
		defaultDir = DefaultSessionCheckpointDir
	}

	targetPath := pathOrID
	fi, err := os.Stat(targetPath)
	if err != nil || fi.IsDir() {
		cand1 := filepath.Join(defaultDir, pathOrID)
		if fi1, err1 := os.Stat(cand1); err1 == nil && !fi1.IsDir() {
			targetPath = cand1
		} else {
			cand2 := filepath.Join(defaultDir, pathOrID+".json")
			if fi2, err2 := os.Stat(cand2); err2 == nil && !fi2.IsDir() {
				targetPath = cand2
			} else {
				return nil, fmt.Errorf("session checkpoint not found for %q (checked %s, %s, %s)", pathOrID, pathOrID, cand1, cand2)
			}
		}
	}

	data, err := os.ReadFile(targetPath)
	if err != nil {
		return nil, fmt.Errorf("session checkpoint: read %s: %w", targetPath, err)
	}
	var cp SessionCheckpoint
	if err := json.Unmarshal(data, &cp); err != nil {
		return nil, fmt.Errorf("session checkpoint: unmarshal %s: %w", targetPath, err)
	}
	if cp.SessionID == "" {
		base := filepath.Base(targetPath)
		cp.SessionID = strings.TrimSuffix(base, ".json")
	}
	return &cp, nil
}
