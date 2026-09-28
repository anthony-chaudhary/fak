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

// SessionCheckpointVersion is the on-disk layout version SaveSessionCheckpoint stamps.
// LoadSessionCheckpoint refuses any other non-zero version instead of guessing at a
// layout this build does not know. An absent (zero) version is a checkpoint written
// before the field existed, which has version 1's layout.
const SessionCheckpointVersion = 1

// ErrUnsupportedSessionCheckpointVersion is returned (wrapped) by LoadSessionCheckpoint
// for a checkpoint whose version this build cannot read.
var ErrUnsupportedSessionCheckpointVersion = errors.New("session checkpoint: unsupported version")

// SessionCheckpoint represents the durable state of an agent session at a turn boundary.
type SessionCheckpoint struct {
	Version   int       `json:"version"`
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

// SaveSessionCheckpoint writes a session checkpoint to a JSON file in dir named
// <session_id>.json, stamped with SessionCheckpointVersion. The write is crash-safe: the
// file is replaced atomically and fsynced (publishSessionCheckpoint), so a crash or a
// failed write leaves the previous checkpoint loadable.
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
	cp.Version = SessionCheckpointVersion
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
	if err := publishSessionCheckpoint(targetPath, data); err != nil {
		return fmt.Errorf("session checkpoint: write %s: %w", targetPath, err)
	}
	return nil
}

// LoadSessionCheckpoint reads a session checkpoint from either an explicit file path or
// a session ID resolved within defaultDir (or DefaultSessionCheckpointDir if empty). A
// checkpoint with an unknown version is refused with ErrUnsupportedSessionCheckpointVersion.
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
	// Check the version before decoding the rest, so a future layout is refused by version
	// rather than by whatever field-shape error it would trip first.
	var head struct {
		Version int `json:"version"`
	}
	if err := json.Unmarshal(data, &head); err != nil {
		return nil, fmt.Errorf("session checkpoint: unmarshal %s: %w", targetPath, err)
	}
	if head.Version != 0 && head.Version != SessionCheckpointVersion {
		return nil, fmt.Errorf("%w: %s is version %d, this build reads version %d",
			ErrUnsupportedSessionCheckpointVersion, targetPath, head.Version, SessionCheckpointVersion)
	}
	var cp SessionCheckpoint
	if err := json.Unmarshal(data, &cp); err != nil {
		return nil, fmt.Errorf("session checkpoint: unmarshal %s: %w", targetPath, err)
	}
	cp.Version = SessionCheckpointVersion
	if cp.SessionID == "" {
		base := filepath.Base(targetPath)
		cp.SessionID = strings.TrimSuffix(base, ".json")
	}
	return &cp, nil
}
