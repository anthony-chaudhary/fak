package studymonitor

import (
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
	"runtime"
	"sort"
	"strings"
	"sync"
)

const (
	SelfInventorySchema      = "fak-study-self-inventory/1"
	DefaultSelfInventoryPath = "docs/research/inventory/anthony-chaudhary-fak.json"
)

// SelfInventory records normalized committed paths and blob identities. It deliberately
// excludes its own path, making an explicit refresh stable after the refreshed artifact is
// committed. SourceRoot is derived only from the entries and never from a checkout pathname.
type SelfInventory struct {
	Schema         string               `json:"schema"`
	Repository     string               `json:"repository"`
	Authority      string               `json:"authority"`
	ContentRoot    string               `json:"content_root"`
	TrackedFiles   int                  `json:"tracked_files"`
	TrackedBytes   int64                `json:"tracked_bytes"`
	ExcludedPaths  []string             `json:"excluded_paths"`
	Entries        []SelfInventoryEntry `json:"entries"`
	RefreshWitness SelfRefreshWitness   `json:"refresh_witness"`
}

type SelfInventoryEntry struct {
	Path           string   `json:"path"`
	ContentSHA256  string   `json:"content_sha256"`
	Bytes          int64    `json:"bytes"`
	Classification string   `json:"classification"`
	SourceClasses  []string `json:"source_classes,omitempty"`
}

// SelfRefreshWitness makes the fail -> explicit refresh -> pass contract inspectable in the
// persisted artifact; deterministic tests execute this exact transition.
type SelfRefreshWitness struct {
	MutationVerdict string `json:"mutation_verdict"`
	Repair          string `json:"repair"`
	FreshVerdict    string `json:"fresh_verdict"`
}

type SelfInventoryDriftKind string

const (
	SelfDriftManifestMissing SelfInventoryDriftKind = "manifest_missing"
	SelfDriftManifestInvalid SelfInventoryDriftKind = "manifest_invalid"
	SelfDriftPathAdded       SelfInventoryDriftKind = "path_added"
	SelfDriftPathRemoved     SelfInventoryDriftKind = "path_removed"
	SelfDriftContentChanged  SelfInventoryDriftKind = "content_changed"
	SelfDriftClassChanged    SelfInventoryDriftKind = "classification_changed"
)

// SelfInventoryDrift is a typed and falsifiable explanation of a stale manifest.
type SelfInventoryDrift struct {
	Kind     SelfInventoryDriftKind `json:"kind"`
	Path     string                 `json:"path,omitempty"`
	Expected string                 `json:"expected,omitempty"`
	Actual   string                 `json:"actual,omitempty"`
}

type SelfInventoryVerification struct {
	OK           bool                 `json:"ok"`
	ExpectedRoot string               `json:"expected_root,omitempty"`
	ActualRoot   string               `json:"actual_root"`
	Drift        []SelfInventoryDrift `json:"drift"`
}

// BuildSelfInventory reads a materialized committed tree. Callers that start from a live
// checkout must first resolve and extract a Git object with internal/committedtree; the builder
// performs no Git discovery and therefore cannot accidentally mix in peer WIP or .git data.
func BuildSelfInventory(committedRoot, repository, manifestPath string) (SelfInventory, error) {
	if strings.TrimSpace(committedRoot) == "" {
		return SelfInventory{}, errors.New("committed root is required")
	}
	manifestPath, err := normalizeSelfPath(manifestPath)
	if err != nil {
		return SelfInventory{}, fmt.Errorf("manifest path: %w", err)
	}
	type pendingEntry struct {
		path  string
		rel   string
		class inventoryFileClass
	}
	var pending []pendingEntry
	err = filepath.WalkDir(committedRoot, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if path == committedRoot {
			return nil
		}
		rel, err := filepath.Rel(committedRoot, path)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		if entry.IsDir() {
			if rel == ".git" || rel == "worktrees" {
				return filepath.SkipDir
			}
			return nil
		}
		if rel == manifestPath {
			return nil
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return nil
		}
		class := classifyInventoryFile(rel)
		pending = append(pending, pendingEntry{path: path, rel: rel, class: class})
		return nil
	})
	if err != nil {
		return SelfInventory{}, err
	}
	entries := make([]SelfInventoryEntry, len(pending))
	readErrs := make([]error, len(pending))
	jobs := make(chan int)
	workers := min(runtime.GOMAXPROCS(0), 8, len(pending))
	if workers < 1 {
		workers = 1
	}
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				item := pending[i]
				data, err := os.ReadFile(item.path)
				if err != nil {
					readErrs[i] = err
					continue
				}
				digest := sha256.Sum256(data)
				entries[i] = SelfInventoryEntry{
					Path: item.rel, ContentSHA256: "sha256:" + hex.EncodeToString(digest[:]), Bytes: int64(len(data)),
					Classification: selfInventoryClassification(item.class), SourceClasses: item.class.SourceClasses,
				}
			}
		}()
	}
	for i := range pending {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	for i, err := range readErrs {
		if err != nil {
			return SelfInventory{}, fmt.Errorf("read %s: %w", pending[i].rel, err)
		}
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Path < entries[j].Path })
	var total int64
	for _, entry := range entries {
		total += entry.Bytes
	}
	return SelfInventory{
		Schema: SelfInventorySchema, Repository: strings.TrimSpace(repository), Authority: "committed_git_tree",
		ContentRoot: selfInventoryContentRoot(entries), TrackedFiles: len(entries), TrackedBytes: total,
		ExcludedPaths: []string{manifestPath}, Entries: entries,
		RefreshWitness: SelfRefreshWitness{
			MutationVerdict: "typed changed-path drift", Repair: "fak study-inventory --self --refresh", FreshVerdict: "verified",
		},
	}, nil
}

func ReadSelfInventory(r io.Reader) (SelfInventory, error) {
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	var manifest SelfInventory
	if err := dec.Decode(&manifest); err != nil {
		return SelfInventory{}, err
	}
	if manifest.Schema != SelfInventorySchema {
		return SelfInventory{}, fmt.Errorf("schema must be %q", SelfInventorySchema)
	}
	if manifest.ContentRoot == "" {
		return SelfInventory{}, errors.New("content_root is required")
	}
	return manifest, nil
}

func WriteSelfInventory(w io.Writer, manifest SelfInventory) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(manifest)
}

// VerifySelfInventory compares a committed materialization with its persisted manifest and
// reports every changed path using a closed drift vocabulary.
func VerifySelfInventory(committedRoot, manifestPath, repository string) (SelfInventoryVerification, error) {
	actual, err := BuildSelfInventory(committedRoot, repository, manifestPath)
	if err != nil {
		return SelfInventoryVerification{}, err
	}
	result := SelfInventoryVerification{ActualRoot: actual.ContentRoot, Drift: []SelfInventoryDrift{}}
	path := filepath.Join(committedRoot, filepath.FromSlash(manifestPath))
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		result.Drift = append(result.Drift, SelfInventoryDrift{Kind: SelfDriftManifestMissing, Path: manifestPath})
		return result, nil
	}
	if err != nil {
		return result, err
	}
	expected, parseErr := ReadSelfInventory(f)
	_ = f.Close()
	if parseErr != nil {
		result.Drift = append(result.Drift, SelfInventoryDrift{Kind: SelfDriftManifestInvalid, Path: manifestPath, Expected: SelfInventorySchema, Actual: parseErr.Error()})
		return result, nil
	}
	result.ExpectedRoot = expected.ContentRoot
	result.Drift = compareSelfInventoryEntries(expected.Entries, actual.Entries)
	if len(result.Drift) == 0 && expected.ContentRoot != actual.ContentRoot {
		result.Drift = append(result.Drift, SelfInventoryDrift{Kind: SelfDriftManifestInvalid, Path: manifestPath, Expected: expected.ContentRoot, Actual: actual.ContentRoot})
	}
	result.OK = len(result.Drift) == 0
	return result, nil
}

func compareSelfInventoryEntries(expected, actual []SelfInventoryEntry) []SelfInventoryDrift {
	want := make(map[string]SelfInventoryEntry, len(expected))
	got := make(map[string]SelfInventoryEntry, len(actual))
	for _, entry := range expected {
		want[entry.Path] = entry
	}
	for _, entry := range actual {
		got[entry.Path] = entry
	}
	paths := make(map[string]bool, len(want)+len(got))
	for path := range want {
		paths[path] = true
	}
	for path := range got {
		paths[path] = true
	}
	ordered := make([]string, 0, len(paths))
	for path := range paths {
		ordered = append(ordered, path)
	}
	sort.Strings(ordered)
	var drift []SelfInventoryDrift
	for _, path := range ordered {
		w, wok := want[path]
		g, gok := got[path]
		switch {
		case !wok:
			drift = append(drift, SelfInventoryDrift{Kind: SelfDriftPathAdded, Path: path, Actual: g.ContentSHA256})
		case !gok:
			drift = append(drift, SelfInventoryDrift{Kind: SelfDriftPathRemoved, Path: path, Expected: w.ContentSHA256})
		case w.ContentSHA256 != g.ContentSHA256 || w.Bytes != g.Bytes:
			drift = append(drift, SelfInventoryDrift{Kind: SelfDriftContentChanged, Path: path, Expected: w.ContentSHA256, Actual: g.ContentSHA256})
		case w.Classification != g.Classification || !equalStrings(w.SourceClasses, g.SourceClasses):
			drift = append(drift, SelfInventoryDrift{Kind: SelfDriftClassChanged, Path: path, Expected: w.Classification, Actual: g.Classification})
		}
	}
	return drift
}

func selfInventoryContentRoot(entries []SelfInventoryEntry) string {
	h := sha256.New()
	for _, entry := range entries {
		data, _ := json.Marshal(entry)
		_, _ = h.Write(data)
		_, _ = h.Write([]byte{'\n'})
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}

func selfInventoryClassification(file inventoryFileClass) string {
	switch {
	case file.Test:
		return "test_fixture"
	case file.Doc:
		return "documentation"
	case file.Runtime:
		return "runtime_source"
	case file.TextLike:
		return "structured_text"
	default:
		return "artifact"
	}
}

func normalizeSelfPath(path string) (string, error) {
	path = filepath.ToSlash(filepath.Clean(strings.TrimSpace(path)))
	if path == "." || path == "" || filepath.IsAbs(path) || path == ".." || strings.HasPrefix(path, "../") {
		return "", errors.New("must be a repository-relative path")
	}
	return path, nil
}

func equalStrings(a, b []string) bool {
	return bytes.Equal([]byte(strings.Join(a, "\x00")), []byte(strings.Join(b, "\x00")))
}
