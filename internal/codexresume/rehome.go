package codexresume

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const BindingSchemaVersion = 1

var (
	ErrBindingNotFound   = errors.New("codex thread binding not found")
	ErrRolloutNotFound   = errors.New("codex rollout not found")
	ErrRolloutAmbiguous  = errors.New("codex rollout query is ambiguous")
	ErrDestinationExists = errors.New("codex rollout destination differs")
)

// ThreadBinding records only credential-free facts needed to rehome one thread.
// AccountKeyDigest is derived from a stable account key supplied by the caller;
// fak never needs to read OAuth tokens or auth.json.
type ThreadBinding struct {
	SchemaVersion       int       `json:"schema_version"`
	ThreadID            string    `json:"thread_id"`
	CanonicalHome       string    `json:"canonical_home"`
	AccountKeyDigest    string    `json:"account_key_digest"`
	RelativeRolloutPath string    `json:"relative_rollout_path"`
	ObservedAt          time.Time `json:"observed_at"`
}

// NewThreadBinding validates and constructs a credential-free binding.
func NewThreadBinding(threadID, home, accountKey, rolloutPath string, observedAt time.Time) (ThreadBinding, error) {
	if strings.TrimSpace(threadID) == "" {
		return ThreadBinding{}, errors.New("thread id is required")
	}
	canonicalHome, err := canonicalHomePath(home)
	if err != nil {
		return ThreadBinding{}, err
	}
	rel, err := rolloutRelativePath(canonicalHome, rolloutPath)
	if err != nil {
		return ThreadBinding{}, err
	}
	if observedAt.IsZero() {
		observedAt = time.Now().UTC()
	}
	return ThreadBinding{
		SchemaVersion:       BindingSchemaVersion,
		ThreadID:            threadID,
		CanonicalHome:       canonicalHome,
		AccountKeyDigest:    AccountKeyDigest(accountKey),
		RelativeRolloutPath: filepath.ToSlash(rel),
		ObservedAt:          observedAt.UTC(),
	}, nil
}

// AccountKeyDigest derives the only account identity persisted by this package.
func AccountKeyDigest(accountKey string) string {
	if accountKey == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(accountKey))
	return "sha256:" + hex.EncodeToString(sum[:])
}

// BindingStore persists one JSON record per thread beneath a caller-selected root.
type BindingStore struct {
	Root string
}

func (s BindingStore) Save(binding ThreadBinding) error {
	if err := validateBinding(binding); err != nil {
		return err
	}
	path, err := s.bindingPath(binding.ThreadID)
	if err != nil {
		return err
	}
	payload, err := json.MarshalIndent(binding, "", "  ")
	if err != nil {
		return err
	}
	payload = append(payload, '\n')
	return atomicReplaceFile(path, payload, 0o600)
}

func (s BindingStore) Load(threadID string) (ThreadBinding, error) {
	path, err := s.bindingPath(threadID)
	if err != nil {
		return ThreadBinding{}, err
	}
	payload, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return ThreadBinding{}, ErrBindingNotFound
	}
	if err != nil {
		return ThreadBinding{}, err
	}
	var binding ThreadBinding
	if err := json.Unmarshal(payload, &binding); err != nil {
		return ThreadBinding{}, fmt.Errorf("decode thread binding: %w", err)
	}
	if err := validateBinding(binding); err != nil {
		return ThreadBinding{}, err
	}
	if binding.ThreadID != threadID {
		return ThreadBinding{}, errors.New("thread binding id does not match file name")
	}
	return binding, nil
}

func (s BindingStore) bindingPath(threadID string) (string, error) {
	if threadID == "" || filepath.Base(threadID) != threadID || strings.ContainsAny(threadID, `/\\`) {
		return "", errors.New("invalid thread id")
	}
	if strings.TrimSpace(s.Root) == "" {
		return "", errors.New("binding store root is required")
	}
	return filepath.Join(s.Root, threadID+".json"), nil
}

func validateBinding(binding ThreadBinding) error {
	if binding.SchemaVersion != BindingSchemaVersion {
		return fmt.Errorf("unsupported binding schema version %d", binding.SchemaVersion)
	}
	if binding.ThreadID == "" || binding.CanonicalHome == "" || binding.RelativeRolloutPath == "" || binding.ObservedAt.IsZero() {
		return errors.New("incomplete thread binding")
	}
	if _, err := rolloutRelativePath(binding.CanonicalHome, filepath.Join(binding.CanonicalHome, filepath.FromSlash(binding.RelativeRolloutPath))); err != nil {
		return err
	}
	return nil
}

// RolloutMatch identifies one rollout without exposing any authentication state.
type RolloutMatch struct {
	ThreadID     string
	Path         string
	RelativePath string
}

// AmbiguousRolloutError reports every credential-free candidate for a prefix.
type AmbiguousRolloutError struct {
	Query   string
	Matches []RolloutMatch
}

func (e *AmbiguousRolloutError) Error() string {
	return fmt.Sprintf("%v %q (%d matches)", ErrRolloutAmbiguous, e.Query, len(e.Matches))
}

func (e *AmbiguousRolloutError) Unwrap() error { return ErrRolloutAmbiguous }

// FindRollout searches only sessions/YYYY/MM/DD rollout JSONL files. Exact IDs
// win; otherwise the query must be a unique thread-ID prefix.
func FindRollout(home, query string) (RolloutMatch, error) {
	canonicalHome, err := canonicalHomePath(home)
	if err != nil {
		return RolloutMatch{}, err
	}
	query = strings.TrimSpace(query)
	if query == "" {
		return RolloutMatch{}, ErrRolloutNotFound
	}
	root := filepath.Join(canonicalHome, "sessions")
	var exact, prefixes []RolloutMatch
	err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			if errors.Is(walkErr, fs.ErrNotExist) {
				return nil
			}
			return walkErr
		}
		if entry.IsDir() {
			if path == root {
				return nil
			}
			rel, err := filepath.Rel(root, path)
			if err != nil {
				return err
			}
			parts := splitPath(rel)
			if len(parts) > 3 || !validDatePathPrefix(parts) {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(entry.Name(), ".jsonl") {
			return nil
		}
		rel, err := filepath.Rel(canonicalHome, path)
		if err != nil || !validRolloutRelativePath(rel) {
			return nil
		}
		threadID, err := rolloutThreadID(path)
		if err != nil {
			return nil
		}
		match := RolloutMatch{ThreadID: threadID, Path: path, RelativePath: filepath.ToSlash(rel)}
		if threadID == query {
			exact = append(exact, match)
		} else if strings.HasPrefix(threadID, query) {
			prefixes = append(prefixes, match)
		}
		return nil
	})
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return RolloutMatch{}, err
	}
	matches := prefixes
	if len(exact) > 0 {
		matches = exact
	}
	sort.Slice(matches, func(i, j int) bool { return matches[i].Path < matches[j].Path })
	switch len(matches) {
	case 0:
		return RolloutMatch{}, ErrRolloutNotFound
	case 1:
		return matches[0], nil
	default:
		return RolloutMatch{}, &AmbiguousRolloutError{Query: query, Matches: matches}
	}
}

func rolloutThreadID(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		var envelope struct {
			Type    string `json:"type"`
			Payload struct {
				ID string `json:"id"`
			} `json:"payload"`
		}
		if json.Unmarshal(scanner.Bytes(), &envelope) == nil && envelope.Type == "session_meta" && envelope.Payload.ID != "" {
			return envelope.Payload.ID, nil
		}
	}
	if err := scanner.Err(); err != nil {
		return "", err
	}
	return "", errors.New("rollout has no session_meta id")
}

type RehomeClass string

const (
	RehomeSameHomeSameAccount           RehomeClass = "same_home_same_account"
	RehomeDifferentHomeSameAccount      RehomeClass = "different_home_same_account"
	RehomeSameHomeDifferentAccount      RehomeClass = "same_home_different_account"
	RehomeDifferentHomeDifferentAccount RehomeClass = "different_home_different_account"
	RehomeUnknown                       RehomeClass = "unknown"
	RehomeUnbound                       RehomeClass = "unbound"
	RehomeAmbiguous                     RehomeClass = "ambiguous"
)

// ClassifyRehome compares a recorded binding with a caller-selected target.
// Set ambiguous when ownership resolution produced more than one candidate.
func ClassifyRehome(binding *ThreadBinding, targetHome, targetAccountKey string, ambiguous bool) RehomeClass {
	if ambiguous {
		return RehomeAmbiguous
	}
	if binding == nil {
		return RehomeUnbound
	}
	targetCanonical, err := canonicalHomePath(targetHome)
	if err != nil || targetAccountKey == "" || binding.CanonicalHome == "" || binding.AccountKeyDigest == "" {
		return RehomeUnknown
	}
	sameHome := samePath(binding.CanonicalHome, targetCanonical)
	sameAccount := binding.AccountKeyDigest == AccountKeyDigest(targetAccountKey)
	switch {
	case sameHome && sameAccount:
		return RehomeSameHomeSameAccount
	case !sameHome && sameAccount:
		return RehomeDifferentHomeSameAccount
	case sameHome && !sameAccount:
		return RehomeSameHomeDifferentAccount
	default:
		return RehomeDifferentHomeDifferentAccount
	}
}

type CopyResult struct {
	Path       string
	Idempotent bool
}

// CopyRollout atomically installs only the selected rollout at the same relative
// sessions path in targetHome. It never traverses or copies auth/global files.
func CopyRollout(sourceHome, targetHome, sourcePath string) (CopyResult, error) {
	sourceCanonical, err := canonicalHomePath(sourceHome)
	if err != nil {
		return CopyResult{}, err
	}
	targetCanonical, err := canonicalHomePath(targetHome)
	if err != nil {
		return CopyResult{}, err
	}
	rel, err := rolloutRelativePath(sourceCanonical, sourcePath)
	if err != nil {
		return CopyResult{}, err
	}
	destination := filepath.Join(targetCanonical, rel)
	if samePath(sourcePath, destination) {
		return CopyResult{Path: destination, Idempotent: true}, nil
	}
	if err := os.MkdirAll(filepath.Dir(destination), 0o700); err != nil {
		return CopyResult{}, err
	}
	if identical, exists, err := filesIdentical(sourcePath, destination); err != nil {
		return CopyResult{}, err
	} else if exists {
		if identical {
			return CopyResult{Path: destination, Idempotent: true}, nil
		}
		return CopyResult{}, ErrDestinationExists
	}

	source, err := os.Open(sourcePath)
	if err != nil {
		return CopyResult{}, err
	}
	defer source.Close()
	info, err := source.Stat()
	if err != nil {
		return CopyResult{}, err
	}
	temp, err := os.CreateTemp(filepath.Dir(destination), ".rollout-copy-*")
	if err != nil {
		return CopyResult{}, err
	}
	tempPath := temp.Name()
	defer os.Remove(tempPath)
	ok := false
	defer func() {
		if !ok {
			temp.Close()
		}
	}()
	if _, err := io.Copy(temp, source); err != nil {
		return CopyResult{}, err
	}
	if err := temp.Chmod(info.Mode().Perm()); err != nil {
		return CopyResult{}, err
	}
	if err := temp.Sync(); err != nil {
		return CopyResult{}, err
	}
	if err := temp.Close(); err != nil {
		return CopyResult{}, err
	}
	ok = true
	if err := os.Link(tempPath, destination); err != nil {
		if !errors.Is(err, fs.ErrExist) {
			return CopyResult{}, err
		}
		identical, _, compareErr := filesIdentical(sourcePath, destination)
		if compareErr != nil {
			return CopyResult{}, compareErr
		}
		if identical {
			return CopyResult{Path: destination, Idempotent: true}, nil
		}
		return CopyResult{}, ErrDestinationExists
	}
	return CopyResult{Path: destination}, nil
}

func atomicReplaceFile(path string, payload []byte, mode fs.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	temp, err := os.CreateTemp(filepath.Dir(path), ".binding-*")
	if err != nil {
		return err
	}
	tempPath := temp.Name()
	defer os.Remove(tempPath)
	if err := temp.Chmod(mode); err != nil {
		temp.Close()
		return err
	}
	if _, err := temp.Write(payload); err != nil {
		temp.Close()
		return err
	}
	if err := temp.Sync(); err != nil {
		temp.Close()
		return err
	}
	if err := temp.Close(); err != nil {
		return err
	}
	return os.Rename(tempPath, path)
}

func filesIdentical(a, b string) (identical, bExists bool, err error) {
	left, err := os.ReadFile(a)
	if err != nil {
		return false, false, err
	}
	right, err := os.ReadFile(b)
	if errors.Is(err, fs.ErrNotExist) {
		return false, false, nil
	}
	if err != nil {
		return false, false, err
	}
	return bytes.Equal(left, right), true, nil
}

func canonicalHomePath(path string) (string, error) {
	if strings.TrimSpace(path) == "" {
		return "", errors.New("CODEX_HOME is required")
	}
	absolute, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	absolute = filepath.Clean(absolute)
	if resolved, err := filepath.EvalSymlinks(absolute); err == nil {
		absolute = resolved
	}
	return absolute, nil
}

func rolloutRelativePath(home, rolloutPath string) (string, error) {
	absolute, err := filepath.Abs(rolloutPath)
	if err != nil {
		return "", err
	}
	absolute = filepath.Clean(absolute)
	if resolved, resolveErr := filepath.EvalSymlinks(absolute); resolveErr == nil {
		absolute = resolved
	}
	rel, err := filepath.Rel(home, absolute)
	if err != nil || !validRolloutRelativePath(rel) {
		return "", errors.New("rollout must be beneath CODEX_HOME/sessions/YYYY/MM/DD")
	}
	return rel, nil
}

func validRolloutRelativePath(rel string) bool {
	parts := splitPath(rel)
	return len(parts) == 5 && parts[0] == "sessions" && validDatePathPrefix(parts[1:4]) && strings.HasSuffix(parts[4], ".jsonl")
}

func validDatePathPrefix(parts []string) bool {
	widths := []int{4, 2, 2}
	for i, part := range parts {
		if i >= len(widths) || len(part) != widths[i] {
			return false
		}
		for _, r := range part {
			if r < '0' || r > '9' {
				return false
			}
		}
	}
	return true
}

func splitPath(path string) []string {
	path = filepath.ToSlash(filepath.Clean(path))
	if path == "." || path == "" {
		return nil
	}
	return strings.Split(path, "/")
}

func samePath(a, b string) bool {
	left, errLeft := filepath.Abs(a)
	right, errRight := filepath.Abs(b)
	if errLeft != nil || errRight != nil {
		return false
	}
	left = filepath.Clean(left)
	right = filepath.Clean(right)
	if resolved, err := filepath.EvalSymlinks(left); err == nil {
		left = resolved
	}
	if resolved, err := filepath.EvalSymlinks(right); err == nil {
		right = resolved
	}
	return left == right
}
