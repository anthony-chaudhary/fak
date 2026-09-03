package sessionmine

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

const (
	SchemaV1    = "fak-sessionmine-index/1"
	SchemaV2    = "fak-sessionmine-index/2"
	indexSchema = SchemaV2
)

// IndexState is the privacy-safe checkpoint for recurring history scans. Keys are
// one-way source fingerprints; raw paths and transcript content are never stored.
type IndexState struct {
	Schema        string                 `json:"schema"`
	Files         map[string]IndexedFile `json:"files"`
	Seen          map[string]bool        `json:"seen_candidates,omitempty"`
	UpdatedAt     string                 `json:"updated_at"`
	Lineage       map[string]string      `json:"lineage,omitempty"`
	RetentionDays int                    `json:"retention_days,omitempty"`
	OutcomeStats  map[string]int         `json:"outcome_stats,omitempty"`
}

type IndexedFile struct {
	Provider  string  `json:"provider"`
	Size      int64   `json:"size"`
	ModUnix   int64   `json:"mod_unix_nano"`
	Session   Session `json:"session"`
	Malformed int     `json:"malformed_rows,omitempty"`
}

type IndexedResult struct {
	Report        Report      `json:"report"`
	NewCandidates []Candidate `json:"new_candidates"`
	ReusedFiles   int         `json:"reused_files"`
	ParsedFiles   int         `json:"parsed_files"`
}

// MineIndexed incrementally updates a durable index and emits only candidates
// that crossed the configured threshold for the first time.
func MineIndexed(opts Options, statePath string) (IndexedResult, error) {
	if statePath == "" {
		return IndexedResult{}, errors.New("index path is required")
	}
	state, err := LoadIndex(statePath)
	if err != nil {
		return IndexedResult{}, err
	}
	next := IndexState{Schema: indexSchema, Files: map[string]IndexedFile{}, Seen: state.Seen}
	if next.Seen == nil {
		next.Seen = map[string]bool{}
	}
	result := IndexedResult{}
	var sessions []Session
	malformed := 0
	roots := []struct{ provider, root string }{{"codex", opts.CodexRoot}, {"claude", opts.ClaudeRoot}}
	for _, src := range roots {
		if src.root == "" {
			continue
		}
		walkErr := filepath.WalkDir(src.root, func(path string, d os.DirEntry, walkErr error) error {
			if walkErr != nil || d.IsDir() || filepath.Ext(path) != ".jsonl" {
				return nil
			}
			info, e := d.Info()
			if e != nil {
				return nil
			}
			if !opts.Since.IsZero() && info.ModTime().Before(opts.Since) {
				return nil
			}
			key := sourceFingerprint(src.provider, path)
			cached, ok := state.Files[key]
			if ok && cached.Provider == src.provider && cached.Size == info.Size() && cached.ModUnix == info.ModTime().UnixNano() {
				next.Files[key] = cached
				sessions = append(sessions, cached.Session)
				malformed += cached.Malformed
				result.ReusedFiles++
				return nil
			}
			session, bad, e := parseFile(path, src.provider)
			if e != nil {
				return nil
			}
			if session.ToolCalls+session.UserTurns+session.AssistantTurns == 0 {
				return nil
			}
			indexed := IndexedFile{Provider: src.provider, Size: info.Size(), ModUnix: info.ModTime().UnixNano(), Session: session, Malformed: bad}
			next.Files[key] = indexed
			sessions = append(sessions, session)
			malformed += bad
			result.ParsedFiles++
			return nil
		})
		if walkErr != nil {
			return IndexedResult{}, walkErr
		}
	}
	report := reportFromSessions(opts, sessions, len(next.Files), malformed)
	result.Report = report
	for _, candidate := range report.Candidates {
		fp := candidateFingerprint(candidate)
		if !next.Seen[fp] {
			result.NewCandidates = append(result.NewCandidates, candidate)
		}
		next.Seen[fp] = true
	}
	next.UpdatedAt = historyWatermark(sessions)
	if err := writeIndexAtomic(statePath, next); err != nil {
		return IndexedResult{}, err
	}
	return result, nil
}

func historyWatermark(sessions []Session) string {
	watermark := "all"
	for _, session := range sessions {
		if session.EndedAt != "" && (watermark == "all" || session.EndedAt > watermark) {
			watermark = session.EndedAt
		}
	}
	return watermark
}

func LoadIndex(path string) (IndexState, error) {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return IndexState{Schema: indexSchema, Files: map[string]IndexedFile{}, Seen: map[string]bool{}}, nil
	}
	if err != nil {
		return IndexState{}, err
	}
	var state IndexState
	if err := json.Unmarshal(data, &state); err != nil {
		return IndexState{}, fmt.Errorf("decode index: %w", err)
	}
	switch state.Schema {
	case SchemaV1:
		// Backward-compatible V1 load: upgrade in memory to V2 structure
		state.Schema = SchemaV2
		if state.Lineage == nil {
			state.Lineage = make(map[string]string)
		}
		if state.OutcomeStats == nil {
			state.OutcomeStats = make(map[string]int)
		}
	case SchemaV2:
		// Current schema
	default:
		return IndexState{}, fmt.Errorf("unsupported index schema %q", state.Schema)
	}
	if state.Files == nil {
		state.Files = map[string]IndexedFile{}
	}
	if state.Seen == nil {
		state.Seen = map[string]bool{}
	}
	return state, nil
}

func writeIndexAtomic(path string, state IndexState) error {
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	tmp, err := os.CreateTemp(filepath.Dir(path), ".sessionmine-index-*")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if err := tmp.Chmod(0600); err != nil {
		tmp.Close()
		return err
	}
	if _, err = tmp.Write(data); err == nil {
		err = tmp.Sync()
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(name, path)
}

func sourceFingerprint(provider, path string) string {
	sum := sha256.Sum256([]byte(provider + "\x00" + filepath.Clean(path)))
	return hex.EncodeToString(sum[:16])
}
func candidateFingerprint(c Candidate) string {
	sum := sha256.Sum256([]byte(c.Fingerprint + "\x00" + c.SuggestedLeaf))
	return hex.EncodeToString(sum[:16])
}
