package trajectory

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
	"time"
)

// AuditSnapshotSchema versions the private, replayable audit input corpus.
const AuditSnapshotSchema = "fak-trajectory-audit-corpus/1"

const auditSnapshotManifestName = "manifest.json"

// AuditSnapshotError is a typed, fail-closed capture or replay refusal.
type AuditSnapshotError struct {
	Code   string
	Detail string
}

func (e *AuditSnapshotError) Error() string {
	return fmt.Sprintf("trajectory audit snapshot: %s: %s", e.Code, e.Detail)
}

// AuditSnapshotRefusalCode extracts the stable refusal code for CLI reporting.
func AuditSnapshotRefusalCode(err error) string {
	var typed *AuditSnapshotError
	if errors.As(err, &typed) {
		return typed.Code
	}
	return "SNAPSHOT_IO"
}

type AuditSnapshotSelection struct {
	SinceNanoseconds int64 `json:"since_nanoseconds"`
	UserContainsSet  bool  `json:"user_contains_set"`
}

type AuditSnapshotSource struct {
	Name                 string `json:"name"`
	RootLabel            string `json:"root_label"`
	RootPresent          bool   `json:"root_present"`
	FilesDiscovered      int    `json:"files_discovered"`
	FilesSelected        int    `json:"files_selected"`
	FixtureFilesExcluded int    `json:"fixture_files_excluded"`
}

type AuditSnapshotFile struct {
	Source       string `json:"source"`
	RelativePath string `json:"relative_path"`
	Bytes        int64  `json:"bytes"`
	SHA256       string `json:"sha256"`
}

// AuditSnapshotManifest is content-free: it records identity and selection,
// never transcript payloads, topic literals, or absolute source roots.
type AuditSnapshotManifest struct {
	Schema               string                 `json:"schema"`
	AuditSchema          string                 `json:"audit_schema"`
	CapturedAtUTC        string                 `json:"captured_at_utc"`
	CorpusDigest         string                 `json:"corpus_digest"`
	CapturedOutputDigest string                 `json:"captured_output_digest"`
	Selection            AuditSnapshotSelection `json:"selection"`
	Sources              []AuditSnapshotSource  `json:"sources"`
	Files                []AuditSnapshotFile    `json:"files"`
}

// CaptureAuditSnapshot selects live inputs, captures their exact bytes into a
// new private directory, and audits the captured copy. Publication is atomic;
// an existing target is never replaced.
func CaptureAuditSnapshot(target string, opts AuditOptions) (AuditSnapshotManifest, AuditResult, error) {
	rawTarget := strings.TrimSpace(target)
	if rawTarget == "" {
		return AuditSnapshotManifest{}, AuditResult{}, snapshotError("SNAPSHOT_TARGET_INVALID", "snapshot target must be an explicit directory")
	}
	target, err := filepath.Abs(filepath.Clean(rawTarget))
	if err != nil {
		return AuditSnapshotManifest{}, AuditResult{}, snapshotError("SNAPSHOT_TARGET_INVALID", "snapshot target must be an explicit directory")
	}
	if _, err := os.Lstat(target); err == nil {
		return AuditSnapshotManifest{}, AuditResult{}, snapshotError("SNAPSHOT_TARGET_EXISTS", "target already exists")
	} else if !os.IsNotExist(err) {
		return AuditSnapshotManifest{}, AuditResult{}, snapshotError("SNAPSHOT_IO", "inspect target: "+err.Error())
	}
	parent := filepath.Dir(target)
	if info, err := os.Stat(parent); err != nil || !info.IsDir() {
		return AuditSnapshotManifest{}, AuditResult{}, snapshotError("SNAPSHOT_TARGET_INVALID", "target parent must already exist")
	}

	lockPath := target + ".capture-lock"
	lock, err := os.OpenFile(lockPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return AuditSnapshotManifest{}, AuditResult{}, snapshotError("SNAPSHOT_TARGET_BUSY", "another capture owns the target")
	}
	_ = lock.Close()
	defer os.Remove(lockPath)

	temp, err := os.MkdirTemp(parent, "."+filepath.Base(target)+".tmp-")
	if err != nil {
		return AuditSnapshotManifest{}, AuditResult{}, snapshotError("SNAPSHOT_IO", "create private staging directory: "+err.Error())
	}
	complete := false
	defer func() {
		if !complete {
			_ = os.RemoveAll(temp)
		}
	}()
	if err := os.Chmod(temp, 0o700); err != nil {
		return AuditSnapshotManifest{}, AuditResult{}, snapshotError("SNAPSHOT_IO", "restrict staging directory: "+err.Error())
	}

	if opts.Now.IsZero() {
		opts.Now = time.Now().UTC()
	}
	if len(opts.Sources) == 0 {
		opts.Sources = DefaultAuditSources()
	}
	manifest := AuditSnapshotManifest{
		Schema: AuditSnapshotSchema, AuditSchema: AuditSchema,
		CapturedAtUTC: opts.Now.UTC().Format(time.RFC3339Nano),
		Selection:     AuditSnapshotSelection{SinceNanoseconds: int64(opts.Since), UserContainsSet: opts.UserContains != ""},
	}
	for _, source := range opts.Sources {
		meta, files, err := captureAuditSnapshotSource(temp, source, opts)
		if err != nil {
			return AuditSnapshotManifest{}, AuditResult{}, err
		}
		manifest.Sources = append(manifest.Sources, meta)
		manifest.Files = append(manifest.Files, files...)
	}
	sortSnapshotManifest(&manifest)
	manifest.CorpusDigest, err = auditSnapshotCorpusDigest(manifest)
	if err != nil {
		return AuditSnapshotManifest{}, AuditResult{}, err
	}
	result, err := runAuditSnapshotCorpus(temp, manifest)
	if err != nil {
		return AuditSnapshotManifest{}, AuditResult{}, err
	}
	manifest.CapturedOutputDigest, err = auditSnapshotOutputDigest(result)
	if err != nil {
		return AuditSnapshotManifest{}, AuditResult{}, err
	}
	if err := writeAuditSnapshotManifest(temp, manifest); err != nil {
		return AuditSnapshotManifest{}, AuditResult{}, err
	}
	if err := restrictAuditSnapshotTree(temp); err != nil {
		return AuditSnapshotManifest{}, AuditResult{}, err
	}
	verifiedManifest, verifiedResult, err := replayAuditSnapshot(temp, nil)
	if err != nil {
		return AuditSnapshotManifest{}, AuditResult{}, err
	}
	if verifiedManifest.CorpusDigest != manifest.CorpusDigest {
		return AuditSnapshotManifest{}, AuditResult{}, snapshotError("SNAPSHOT_CORPUS_CHANGED", "staged corpus identity changed before publication")
	}
	if _, err := os.Lstat(target); err == nil {
		return AuditSnapshotManifest{}, AuditResult{}, snapshotError("SNAPSHOT_TARGET_EXISTS", "target appeared before publication")
	} else if !os.IsNotExist(err) {
		return AuditSnapshotManifest{}, AuditResult{}, snapshotError("SNAPSHOT_IO", "recheck target: "+err.Error())
	}
	if err := os.Rename(temp, target); err != nil {
		return AuditSnapshotManifest{}, AuditResult{}, snapshotError("SNAPSHOT_PUBLISH_FAILED", "publish complete snapshot: "+err.Error())
	}
	complete = true
	return verifiedManifest, verifiedResult, nil
}

// ReplayAuditSnapshot verifies all snapshot metadata and bytes before parsing,
// audits without wall-clock selection, then verifies the inputs again.
func ReplayAuditSnapshot(root string) (AuditSnapshotManifest, AuditResult, error) {
	return replayAuditSnapshot(root, nil)
}

func replayAuditSnapshot(root string, beforeFinalVerify func()) (AuditSnapshotManifest, AuditResult, error) {
	manifest, err := verifyAuditSnapshot(root)
	if err != nil {
		return AuditSnapshotManifest{}, AuditResult{}, err
	}
	result, err := runAuditSnapshotCorpus(root, manifest)
	if err != nil {
		return AuditSnapshotManifest{}, AuditResult{}, err
	}
	if beforeFinalVerify != nil {
		beforeFinalVerify()
	}
	verified, err := verifyAuditSnapshot(root)
	if err != nil {
		return AuditSnapshotManifest{}, AuditResult{}, err
	}
	if verified.CorpusDigest != manifest.CorpusDigest {
		return AuditSnapshotManifest{}, AuditResult{}, snapshotError("SNAPSHOT_CORPUS_CHANGED", "manifest changed during replay")
	}
	digest, err := auditSnapshotOutputDigest(result)
	if err != nil {
		return AuditSnapshotManifest{}, AuditResult{}, err
	}
	if digest != manifest.CapturedOutputDigest {
		return AuditSnapshotManifest{}, AuditResult{}, snapshotError("SNAPSHOT_OUTPUT_CHANGED", "current audit output does not match the captured audit schema")
	}
	return manifest, result, nil
}

func captureAuditSnapshotSource(root string, source AuditSource, opts AuditOptions) (AuditSnapshotSource, []AuditSnapshotFile, error) {
	if source.Name != AuditSourceClaude && source.Name != AuditSourceCodex {
		return AuditSnapshotSource{}, nil, snapshotError("SNAPSHOT_SOURCE_INVALID", "unsupported source "+source.Name)
	}
	if !safeSnapshotLabel(source.RootLabel) {
		return AuditSnapshotSource{}, nil, snapshotError("SNAPSHOT_ROOT_LABEL_INVALID", "root label must be relative and content-free")
	}
	meta := AuditSnapshotSource{Name: source.Name, RootLabel: source.RootLabel}
	destinationRoot := filepath.Join(root, filepath.FromSlash(snapshotSourcePath(source.Name)))
	if err := os.MkdirAll(destinationRoot, 0o700); err != nil {
		return meta, nil, snapshotError("SNAPSHOT_IO", "create source directory: "+err.Error())
	}
	info, err := os.Stat(source.Root)
	if err != nil {
		if os.IsNotExist(err) {
			return meta, nil, nil
		}
		return meta, nil, snapshotError("SNAPSHOT_IO", "stat source root: "+err.Error())
	}
	if !info.IsDir() {
		return meta, nil, snapshotError("SNAPSHOT_SOURCE_INVALID", "source root is not a directory")
	}
	meta.RootPresent = true
	var paths []string
	err = filepath.WalkDir(source.Root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if !entry.IsDir() && strings.EqualFold(filepath.Ext(entry.Name()), ".jsonl") {
			if entry.Type()&os.ModeSymlink != 0 {
				return snapshotError("SNAPSHOT_PATH_INVALID", "symlink transcript is outside the captured-byte contract")
			}
			paths = append(paths, path)
		}
		return nil
	})
	if err != nil {
		return meta, nil, snapshotWrap("SNAPSHOT_IO", "discover source transcripts", err)
	}
	sort.Strings(paths)
	meta.FilesDiscovered = len(paths)
	var files []AuditSnapshotFile
	for _, path := range paths {
		stat, err := os.Stat(path)
		if err != nil || !stat.Mode().IsRegular() {
			return meta, nil, snapshotError("SNAPSHOT_PATH_INVALID", "selected transcript is not a regular file")
		}
		if opts.Since > 0 && stat.ModTime().Before(opts.Now.Add(-opts.Since)) {
			continue
		}
		rel, err := filepath.Rel(source.Root, path)
		if err != nil {
			return meta, nil, snapshotWrap("SNAPSHOT_PATH_INVALID", "relativize source transcript", err)
		}
		rel, err = cleanSnapshotRelative(filepath.ToSlash(rel))
		if err != nil {
			return meta, nil, err
		}
		if opts.UserContains != "" {
			matched, err := auditFileUserContains(path, opts.UserContains)
			if err != nil {
				return meta, nil, snapshotWrap("SNAPSHOT_IO", "apply topic selection", err)
			}
			if !matched {
				continue
			}
		}
		if source.Name == AuditSourceClaude && auditIsClaudePytestFixture(path, rel) {
			meta.FixtureFilesExcluded++
			continue
		}
		payload, err := os.ReadFile(path)
		if err != nil {
			return meta, nil, snapshotWrap("SNAPSHOT_IO", "read selected transcript", err)
		}
		destination := filepath.Join(destinationRoot, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(destination), 0o700); err != nil {
			return meta, nil, snapshotWrap("SNAPSHOT_IO", "create transcript parent", err)
		}
		file, err := os.OpenFile(destination, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if err != nil {
			return meta, nil, snapshotWrap("SNAPSHOT_IO", "create captured transcript", err)
		}
		_, writeErr := file.Write(payload)
		closeErr := file.Close()
		if writeErr != nil || closeErr != nil {
			return meta, nil, snapshotError("SNAPSHOT_IO", "write captured transcript")
		}
		sum := sha256.Sum256(payload)
		files = append(files, AuditSnapshotFile{Source: source.Name, RelativePath: rel, Bytes: int64(len(payload)), SHA256: hex.EncodeToString(sum[:])})
		meta.FilesSelected++
	}
	return meta, files, nil
}

func verifyAuditSnapshot(root string) (AuditSnapshotManifest, error) {
	root, err := filepath.Abs(filepath.Clean(root))
	if err != nil {
		return AuditSnapshotManifest{}, snapshotError("SNAPSHOT_PATH_INVALID", "resolve snapshot root")
	}
	info, err := os.Lstat(root)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return AuditSnapshotManifest{}, snapshotError("SNAPSHOT_PATH_INVALID", "snapshot root must be a directory, not a symlink")
	}
	if runtime.GOOS != "windows" && info.Mode().Perm()&0o077 != 0 {
		return AuditSnapshotManifest{}, snapshotError("SNAPSHOT_PERMISSION_INSECURE", "snapshot root permits group or other access")
	}
	manifestPath := filepath.Join(root, auditSnapshotManifestName)
	file, err := os.Open(manifestPath)
	if err != nil {
		code := "SNAPSHOT_MANIFEST_MISSING"
		if !os.IsNotExist(err) {
			code = "SNAPSHOT_IO"
		}
		return AuditSnapshotManifest{}, snapshotWrap(code, "open manifest", err)
	}
	payload, readErr := io.ReadAll(io.LimitReader(file, 16*1024*1024+1))
	closeErr := file.Close()
	if readErr != nil || closeErr != nil || len(payload) > 16*1024*1024 {
		return AuditSnapshotManifest{}, snapshotError("SNAPSHOT_MANIFEST_MALFORMED", "manifest is unreadable or too large")
	}
	var manifest AuditSnapshotManifest
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&manifest); err != nil {
		return AuditSnapshotManifest{}, snapshotWrap("SNAPSHOT_MANIFEST_MALFORMED", "decode manifest", err)
	}
	if manifest.Schema != AuditSnapshotSchema || manifest.AuditSchema != AuditSchema {
		return AuditSnapshotManifest{}, snapshotError("SNAPSHOT_SCHEMA_INCOMPATIBLE", "snapshot or audit schema has no reader")
	}
	if _, err := time.Parse(time.RFC3339Nano, manifest.CapturedAtUTC); err != nil {
		return AuditSnapshotManifest{}, snapshotError("SNAPSHOT_MANIFEST_MALFORMED", "capture UTC is invalid")
	}
	if len(manifest.Sources) != 2 {
		return AuditSnapshotManifest{}, snapshotError("SNAPSHOT_MANIFEST_MALFORMED", "manifest must name both supported sources")
	}
	seenSources := map[string]bool{}
	for _, source := range manifest.Sources {
		if source.Name != AuditSourceClaude && source.Name != AuditSourceCodex || seenSources[source.Name] || !safeSnapshotLabel(source.RootLabel) {
			return AuditSnapshotManifest{}, snapshotError("SNAPSHOT_MANIFEST_MALFORMED", "source identity or root label is invalid")
		}
		seenSources[source.Name] = true
	}
	expected := map[string]AuditSnapshotFile{}
	for _, entry := range manifest.Files {
		if !seenSources[entry.Source] {
			return AuditSnapshotManifest{}, snapshotError("SNAPSHOT_MANIFEST_MALFORMED", "file names an unknown source")
		}
		rel, err := cleanSnapshotRelative(entry.RelativePath)
		if err != nil || rel != entry.RelativePath {
			return AuditSnapshotManifest{}, snapshotError("SNAPSHOT_PATH_INVALID", "manifest contains a non-canonical source path")
		}
		key := filepath.ToSlash(filepath.Join(snapshotSourcePath(entry.Source), rel))
		if _, exists := expected[key]; exists {
			return AuditSnapshotManifest{}, snapshotError("SNAPSHOT_MANIFEST_MALFORMED", "manifest contains a duplicate file")
		}
		expected[key] = entry
	}
	actual := map[string]bool{}
	err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		if rel == "." {
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return snapshotError("SNAPSHOT_PATH_INVALID", "snapshot contains a symlink")
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if runtime.GOOS != "windows" && info.Mode().Perm()&0o077 != 0 {
			return snapshotError("SNAPSHOT_PERMISSION_INSECURE", "snapshot entry permits group or other access")
		}
		if entry.IsDir() {
			return nil
		}
		if !info.Mode().IsRegular() {
			return snapshotError("SNAPSHOT_PATH_INVALID", "snapshot contains a non-regular file")
		}
		if rel == auditSnapshotManifestName {
			return nil
		}
		actual[rel] = true
		if _, ok := expected[rel]; !ok {
			return snapshotError("SNAPSHOT_FILE_EXTRA", "snapshot contains an undeclared file")
		}
		return nil
	})
	if err != nil {
		var typed *AuditSnapshotError
		if errors.As(err, &typed) {
			return AuditSnapshotManifest{}, typed
		}
		return AuditSnapshotManifest{}, snapshotWrap("SNAPSHOT_IO", "walk snapshot", err)
	}
	for rel, entry := range expected {
		if !actual[rel] {
			return AuditSnapshotManifest{}, snapshotError("SNAPSHOT_FILE_MISSING", "snapshot is missing a declared file")
		}
		path := filepath.Join(root, filepath.FromSlash(rel))
		payload, err := os.ReadFile(path)
		if err != nil {
			return AuditSnapshotManifest{}, snapshotWrap("SNAPSHOT_IO", "read captured transcript", err)
		}
		sum := sha256.Sum256(payload)
		if int64(len(payload)) != entry.Bytes || hex.EncodeToString(sum[:]) != entry.SHA256 {
			return AuditSnapshotManifest{}, snapshotError("SNAPSHOT_FILE_CHANGED", "captured transcript length or hash changed")
		}
	}
	wantDigest, err := auditSnapshotCorpusDigest(manifest)
	if err != nil {
		return AuditSnapshotManifest{}, err
	}
	if wantDigest != manifest.CorpusDigest {
		return AuditSnapshotManifest{}, snapshotError("SNAPSHOT_CORPUS_CHANGED", "manifest corpus digest does not match its selection and files")
	}
	return manifest, nil
}

func runAuditSnapshotCorpus(root string, manifest AuditSnapshotManifest) (AuditResult, error) {
	sources := make([]AuditSource, 0, len(manifest.Sources))
	for _, source := range manifest.Sources {
		sources = append(sources, AuditSource{
			Name:      source.Name,
			Root:      filepath.Join(root, filepath.FromSlash(snapshotSourcePath(source.Name))),
			RootLabel: source.RootLabel + " [snapshot " + manifest.CorpusDigest + "]",
		})
	}
	result, err := RunAudit(AuditOptions{Sources: sources, Since: 0})
	if err != nil {
		return AuditResult{}, snapshotWrap("SNAPSHOT_AUDIT_REFUSED", "audit captured corpus", err)
	}
	var totalBytes int64
	for _, file := range manifest.Files {
		totalBytes += file.Bytes
	}
	result.Corpus = &AuditCorpusRow{Schema: AuditSchema, Kind: "corpus", CorpusSchema: AuditSnapshotSchema, Digest: manifest.CorpusDigest, Verified: true, Files: len(manifest.Files), Bytes: totalBytes}
	applyAuditSnapshotCoverage(&result, manifest)
	return result, nil
}

func applyAuditSnapshotCoverage(result *AuditResult, manifest AuditSnapshotManifest) {
	result.Summary.FilesDiscovered = 0
	result.Summary.FilesMatched = 0
	result.Summary.FixtureFilesExcluded = 0
	for i := range result.Denominators {
		for _, source := range manifest.Sources {
			if result.Denominators[i].Source != source.Name {
				continue
			}
			result.Denominators[i].RootPresent = source.RootPresent
			result.Denominators[i].FilesDiscovered = source.FilesDiscovered
			result.Denominators[i].FixtureFilesExcluded = source.FixtureFilesExcluded
			if manifest.Selection.UserContainsSet {
				result.Denominators[i].FilesMatched = source.FilesSelected
			}
			result.Summary.FilesDiscovered += source.FilesDiscovered
			result.Summary.FilesMatched += result.Denominators[i].FilesMatched
			result.Summary.FixtureFilesExcluded += source.FixtureFilesExcluded
		}
	}
}

func auditSnapshotCorpusDigest(manifest AuditSnapshotManifest) (string, error) {
	identity := struct {
		Schema      string                 `json:"schema"`
		AuditSchema string                 `json:"audit_schema"`
		Selection   AuditSnapshotSelection `json:"selection"`
		Sources     []AuditSnapshotSource  `json:"sources"`
		Files       []AuditSnapshotFile    `json:"files"`
	}{manifest.Schema, manifest.AuditSchema, manifest.Selection, manifest.Sources, manifest.Files}
	payload, err := json.Marshal(identity)
	if err != nil {
		return "", snapshotWrap("SNAPSHOT_MANIFEST_MALFORMED", "encode corpus identity", err)
	}
	sum := sha256.Sum256(payload)
	return hex.EncodeToString(sum[:]), nil
}

func auditSnapshotOutputDigest(result AuditResult) (string, error) {
	var jsonl, markdown bytes.Buffer
	if err := WriteAuditJSONL(&jsonl, result); err != nil {
		return "", snapshotWrap("SNAPSHOT_OUTPUT_CHANGED", "render JSONL", err)
	}
	if err := WriteAuditMarkdown(&markdown, result); err != nil {
		return "", snapshotWrap("SNAPSHOT_OUTPUT_CHANGED", "render markdown", err)
	}
	h := sha256.New()
	_, _ = h.Write([]byte("jsonl\x00"))
	_, _ = h.Write(jsonl.Bytes())
	_, _ = h.Write([]byte("markdown\x00"))
	_, _ = h.Write(markdown.Bytes())
	return hex.EncodeToString(h.Sum(nil)), nil
}

func writeAuditSnapshotManifest(root string, manifest AuditSnapshotManifest) error {
	payload, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return snapshotWrap("SNAPSHOT_IO", "encode manifest", err)
	}
	payload = append(payload, '\n')
	path := filepath.Join(root, auditSnapshotManifestName)
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return snapshotWrap("SNAPSHOT_IO", "create manifest", err)
	}
	_, writeErr := file.Write(payload)
	closeErr := file.Close()
	if writeErr != nil || closeErr != nil {
		return snapshotError("SNAPSHOT_IO", "write manifest")
	}
	return nil
}

func restrictAuditSnapshotTree(root string) error {
	return filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		mode := fs.FileMode(0o600)
		if entry.IsDir() {
			mode = 0o700
		}
		if err := os.Chmod(path, mode); err != nil {
			return snapshotWrap("SNAPSHOT_IO", "restrict snapshot entry", err)
		}
		return nil
	})
}

func sortSnapshotManifest(manifest *AuditSnapshotManifest) {
	sort.Slice(manifest.Sources, func(i, j int) bool { return manifest.Sources[i].Name < manifest.Sources[j].Name })
	sort.Slice(manifest.Files, func(i, j int) bool {
		if manifest.Files[i].Source != manifest.Files[j].Source {
			return manifest.Files[i].Source < manifest.Files[j].Source
		}
		return manifest.Files[i].RelativePath < manifest.Files[j].RelativePath
	})
}

func snapshotSourcePath(source string) string {
	if source == AuditSourceClaude {
		return "claude/projects"
	}
	return "codex/sessions"
}

func cleanSnapshotRelative(value string) (string, error) {
	if value == "" || strings.Contains(value, "\\") || filepath.IsAbs(value) || strings.HasPrefix(value, "/") {
		return "", snapshotError("SNAPSHOT_PATH_INVALID", "source path must be a nonempty slash-relative path")
	}
	clean := filepath.ToSlash(filepath.Clean(filepath.FromSlash(value)))
	if clean == "." || clean == ".." || strings.HasPrefix(clean, "../") || clean != value {
		return "", snapshotError("SNAPSHOT_PATH_INVALID", "source path escapes its declared root")
	}
	return clean, nil
}

func safeSnapshotLabel(value string) bool {
	if value == "" || filepath.IsAbs(value) || strings.Contains(value, "\\") {
		return false
	}
	clean := filepath.ToSlash(filepath.Clean(filepath.FromSlash(value)))
	return clean == value && clean != "." && clean != ".." && !strings.HasPrefix(clean, "../")
}

func snapshotError(code, detail string) error {
	return &AuditSnapshotError{Code: code, Detail: detail}
}

func snapshotWrap(code, detail string, err error) error {
	if err == nil {
		return snapshotError(code, detail)
	}
	var typed *AuditSnapshotError
	if errors.As(err, &typed) {
		return typed
	}
	return snapshotError(code, detail+": "+err.Error())
}
