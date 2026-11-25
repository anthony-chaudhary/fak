package inklingmetal

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io/fs"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"testing"
)

const (
	provenanceSchema    = "fak.inklingmetal.vendor.provenance.v1"
	provenanceVendorDir = "third_party/inklingmetal"
	provenanceManifest  = provenanceVendorDir + "/PINNED-SOURCE.json"
	provenanceRuntime   = provenanceVendorDir + "/runtime"

	statusIdentical = "identical"
	statusModified  = "modified"
	statusFakAdded  = "fak-added"

	// A local clone of the upstream repository holding the pinned commit. Setting
	// it makes the test compare every recorded upstream digest with the real object.
	provenanceUpstreamEnv = "FAK_INKLINGMETAL_UPSTREAM"
	// With the clone set, a non-empty value rewrites the manifest from the tree.
	provenanceRefreshEnv = "FAK_INKLINGMETAL_REFRESH"
)

// Field names follow third_party/flashmla/PINNED-SOURCE.json so one reader shape
// covers both vendored trees.
type provenanceFile struct {
	Path           string `json:"path"`
	Bytes          int64  `json:"bytes"`
	SHA256         string `json:"sha256"`
	Status         string `json:"status"`
	UpstreamPath   string `json:"upstream_path,omitempty"`
	UpstreamSHA256 string `json:"upstream_sha256,omitempty"`
	LocalChange    string `json:"local_change,omitempty"`
}

type provenanceCounts struct {
	Identical int `json:"identical"`
	Modified  int `json:"modified"`
	FakAdded  int `json:"fak_added"`
}

type provenanceUpstreamRuntime struct {
	Files                  int            `json:"files"`
	Imported               int            `json:"imported"`
	NotImported            int            `json:"not_imported"`
	NotImportedByDirectory map[string]int `json:"not_imported_by_directory"`
}

type provenanceDoc struct {
	Schema             string                    `json:"schema"`
	Origin             string                    `json:"origin"`
	UpstreamRepository string                    `json:"upstream_repository"`
	UpstreamCommit     string                    `json:"upstream_commit"`
	License            string                    `json:"license"`
	UpstreamRoot       string                    `json:"upstream_root"`
	VendorRoot         string                    `json:"vendor_root"`
	Counts             provenanceCounts          `json:"counts"`
	UpstreamRuntime    provenanceUpstreamRuntime `json:"upstream_runtime"`
	Notices            []provenanceFile          `json:"notices"`
	Files              []provenanceFile          `json:"files"`
}

func provenanceRepoRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, "go.mod")); err != nil {
		t.Fatalf("repository root not found from the package directory: %v", err)
	}
	return root
}

func provenanceDigest(t *testing.T, raw []byte) (int64, string) {
	t.Helper()
	// A converting checkout materializes CRLF; the pinned bytes are the LF blobs.
	if runtime.GOOS == "windows" {
		raw = bytes.ReplaceAll(raw, []byte("\r\n"), []byte("\n"))
	}
	sum := sha256.Sum256(raw)
	return int64(len(raw)), hex.EncodeToString(sum[:])
}

func provenanceFileDigest(t *testing.T, root, relative string) (int64, string) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(relative)))
	if err != nil {
		t.Fatalf("%s: %v", relative, err)
	}
	return provenanceDigest(t, raw)
}

func provenanceTree(t *testing.T, root string) []string {
	t.Helper()
	var files []string
	base := filepath.Join(root, filepath.FromSlash(provenanceRuntime))
	err := filepath.WalkDir(base, func(p string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		relative, err := filepath.Rel(root, p)
		if err != nil {
			return err
		}
		relative = filepath.ToSlash(relative)
		if !entry.Type().IsRegular() {
			t.Errorf("%s: vendored entries must be regular files", relative)
			return nil
		}
		files = append(files, relative)
		return nil
	})
	if err != nil {
		t.Fatalf("walk %s: %v", provenanceRuntime, err)
	}
	sort.Strings(files)
	return files
}

func provenanceLoad(t *testing.T, root string) provenanceDoc {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(provenanceManifest)))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	var doc provenanceDoc
	if err := decoder.Decode(&doc); err != nil {
		t.Fatalf("decode %s: %v", provenanceManifest, err)
	}
	return doc
}

var provenanceHex = regexp.MustCompile(`^[0-9a-f]+$`)

func provenanceIsHex(value string, length int) bool {
	return len(value) == length && provenanceHex.MatchString(value)
}

const provenanceRefreshHint = "; refresh with " + provenanceUpstreamEnv + "=<splash clone> " + provenanceRefreshEnv + "=1 go test ./internal/inklingmetal -run Provenance -count=1"

func provenanceCheckEntry(t *testing.T, root string, entry provenanceFile) {
	t.Helper()
	if !provenanceIsHex(entry.SHA256, 64) {
		t.Errorf("%s: sha256 is not a 64-digit lowercase hex digest", entry.Path)
		return
	}
	size, digest := provenanceFileDigest(t, root, entry.Path)
	if size != entry.Bytes || digest != entry.SHA256 {
		t.Errorf("%s: content differs from the manifest (manifest %d bytes %s, tree %d bytes %s)%s",
			entry.Path, entry.Bytes, entry.SHA256, size, digest, provenanceRefreshHint)
	}
	switch entry.Status {
	case statusIdentical:
		if entry.UpstreamSHA256 != entry.SHA256 {
			t.Errorf("%s: identical file must record the same upstream_sha256", entry.Path)
		}
		if entry.LocalChange != "" {
			t.Errorf("%s: identical file must not describe a local change", entry.Path)
		}
	case statusModified:
		if !provenanceIsHex(entry.UpstreamSHA256, 64) || entry.UpstreamSHA256 == entry.SHA256 {
			t.Errorf("%s: modified file must record a different upstream_sha256", entry.Path)
		}
		if strings.TrimSpace(entry.LocalChange) == "" || strings.Contains(entry.LocalChange, "\n") {
			t.Errorf("%s: modified file must describe the local change in one line", entry.Path)
		}
	case statusFakAdded:
		if entry.UpstreamSHA256 != "" || entry.UpstreamPath != "" {
			t.Errorf("%s: fak-added file has no upstream object", entry.Path)
		}
	default:
		t.Errorf("%s: unknown status %q", entry.Path, entry.Status)
	}
	if entry.Status != statusFakAdded && entry.UpstreamPath == "" {
		t.Errorf("%s: missing upstream_path", entry.Path)
	}
}

// fak-test:justify why=contract when=changed:third_party/inklingmetal/**
// fak-test:runtime fast est=1s
func TestProvenanceManifestMatchesVendoredTree(t *testing.T) {
	root := provenanceRepoRoot(t)
	sparse := provenanceSparsePaths(root)
	if clone := os.Getenv(provenanceUpstreamEnv); clone != "" && os.Getenv(provenanceRefreshEnv) != "" {
		if len(sparse) > 0 {
			t.Fatalf("refresh needs a full checkout: %d vendored files are off disk", len(sparse))
		}
		provenanceRefresh(t, root, clone)
	}
	if len(sparse) > 0 {
		t.Logf("sparse checkout omits %d vendored files; only the files on disk are verified", len(sparse))
	}
	doc := provenanceLoad(t, root)

	if doc.Schema != provenanceSchema {
		t.Fatalf("schema = %q, want %q", doc.Schema, provenanceSchema)
	}
	if doc.Origin != "vendored" {
		t.Errorf("origin = %q, want vendored: this tree is not fak-authored", doc.Origin)
	}
	if !strings.HasPrefix(doc.UpstreamRepository, "https://") || !provenanceIsHex(doc.UpstreamCommit, 40) || doc.License == "" {
		t.Errorf("manifest must name the upstream repository, a full commit and a license")
	}
	if doc.VendorRoot != provenanceRuntime || doc.UpstreamRoot == "" {
		t.Errorf("vendor_root = %q upstream_root = %q", doc.VendorRoot, doc.UpstreamRoot)
	}

	recorded := map[string]bool{}
	var counts provenanceCounts
	for index, entry := range doc.Files {
		if recorded[entry.Path] {
			t.Errorf("%s: listed twice", entry.Path)
		}
		recorded[entry.Path] = true
		if index > 0 && doc.Files[index-1].Path >= entry.Path {
			t.Errorf("%s: files must be sorted by path", entry.Path)
		}
		if !strings.HasPrefix(entry.Path, provenanceRuntime+"/") || path.Clean(entry.Path) != entry.Path {
			t.Errorf("%s: outside %s", entry.Path, provenanceRuntime)
			continue
		}
		switch entry.Status {
		case statusIdentical:
			counts.Identical++
		case statusModified:
			counts.Modified++
		case statusFakAdded:
			counts.FakAdded++
		}
		if entry.Status != statusFakAdded {
			want := doc.UpstreamRoot + strings.TrimPrefix(entry.Path, provenanceRuntime)
			if entry.UpstreamPath != want {
				t.Errorf("%s: upstream_path = %q, want %q", entry.Path, entry.UpstreamPath, want)
			}
		}
	}

	present := map[string]bool{}
	for _, relative := range provenanceTree(t, root) {
		present[relative] = true
		if !recorded[relative] {
			t.Errorf("%s: in the vendored tree but not in the manifest%s", relative, provenanceRefreshHint)
		}
	}
	for _, entry := range doc.Files {
		if !present[entry.Path] && sparse[entry.Path] {
			continue
		}
		if !present[entry.Path] {
			t.Errorf("%s: in the manifest but not in the vendored tree%s", entry.Path, provenanceRefreshHint)
			continue
		}
		provenanceCheckEntry(t, root, entry)
	}

	if counts != doc.Counts {
		t.Errorf("counts = %+v, files give %+v", doc.Counts, counts)
	}
	upstream := doc.UpstreamRuntime
	if upstream.Imported != counts.Identical+counts.Modified {
		t.Errorf("upstream_runtime.imported = %d, files give %d", upstream.Imported, counts.Identical+counts.Modified)
	}
	if upstream.Files != upstream.Imported+upstream.NotImported {
		t.Errorf("upstream_runtime: files %d != imported %d + not_imported %d", upstream.Files, upstream.Imported, upstream.NotImported)
	}
	total := 0
	for _, count := range upstream.NotImportedByDirectory {
		total += count
	}
	if total != upstream.NotImported {
		t.Errorf("upstream_runtime.not_imported_by_directory sums to %d, want %d", total, upstream.NotImported)
	}

	if len(doc.Notices) == 0 {
		t.Errorf("manifest must pin the upstream license and notice files")
	}
	for _, entry := range doc.Notices {
		if path.Dir(entry.Path) != provenanceVendorDir {
			t.Errorf("%s: notices sit beside the manifest", entry.Path)
			continue
		}
		if sparse[entry.Path] {
			continue
		}
		if entry.Status != statusIdentical {
			t.Errorf("%s: upstream notices must stay byte-identical", entry.Path)
		}
		provenanceCheckEntry(t, root, entry)
	}

	if clone := os.Getenv(provenanceUpstreamEnv); clone != "" {
		provenanceCheckUpstream(t, doc, clone)
	}
}

// A sparse lane leaves tracked vendor sources off disk on purpose; that is not a
// removal. Files deleted from a full checkout carry no skip-worktree bit and still fail.
func provenanceSparsePaths(root string) map[string]bool {
	sparse := map[string]bool{}
	out, err := exec.Command("git", "-C", root, "ls-files", "-t", "--", provenanceVendorDir).Output()
	if err != nil {
		return sparse
	}
	for _, line := range strings.Split(string(out), "\n") {
		if name, ok := strings.CutPrefix(line, "S "); ok {
			sparse[name] = true
		}
	}
	return sparse
}

func provenanceGit(t *testing.T, clone string, args ...string) []byte {
	t.Helper()
	command := exec.Command("git", append([]string{"-C", clone}, args...)...)
	var stderr bytes.Buffer
	command.Stderr = &stderr
	out, err := command.Output()
	if err != nil {
		t.Fatalf("git %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return out
}

func provenanceUpstreamList(t *testing.T, clone, commit, upstreamRoot string) []string {
	t.Helper()
	out := provenanceGit(t, clone, "ls-tree", "-r", "-z", "--name-only", commit, "--", upstreamRoot)
	var names []string
	for _, name := range strings.Split(string(out), "\x00") {
		if name != "" {
			names = append(names, name)
		}
	}
	return names
}

func provenanceUpstreamDigest(t *testing.T, clone, commit, upstreamPath string) string {
	t.Helper()
	sum := sha256.Sum256(provenanceGit(t, clone, "cat-file", "blob", commit+":"+upstreamPath))
	return hex.EncodeToString(sum[:])
}

func provenanceCheckUpstream(t *testing.T, doc provenanceDoc, clone string) {
	t.Helper()
	imported := map[string]bool{}
	for _, entry := range append(append([]provenanceFile{}, doc.Files...), doc.Notices...) {
		if entry.Status == statusFakAdded {
			continue
		}
		imported[entry.UpstreamPath] = true
		if got := provenanceUpstreamDigest(t, clone, doc.UpstreamCommit, entry.UpstreamPath); got != entry.UpstreamSHA256 {
			t.Errorf("%s: upstream_sha256 = %s, upstream object is %s", entry.Path, entry.UpstreamSHA256, got)
		}
	}
	names := provenanceUpstreamList(t, clone, doc.UpstreamCommit, doc.UpstreamRoot)
	skipped := map[string]int{}
	for _, name := range names {
		if !imported[name] {
			skipped[path.Dir(name)]++
		}
	}
	if len(names) != doc.UpstreamRuntime.Files {
		t.Errorf("upstream_runtime.files = %d, upstream has %d", doc.UpstreamRuntime.Files, len(names))
	}
	for directory, count := range skipped {
		if doc.UpstreamRuntime.NotImportedByDirectory[directory] != count {
			t.Errorf("not_imported_by_directory[%s] = %d, upstream gives %d", directory, doc.UpstreamRuntime.NotImportedByDirectory[directory], count)
		}
	}
	if len(skipped) != len(doc.UpstreamRuntime.NotImportedByDirectory) {
		t.Errorf("not_imported_by_directory lists %d directories, upstream gives %d", len(doc.UpstreamRuntime.NotImportedByDirectory), len(skipped))
	}
}

// The pin, license and local-change descriptions are authored; everything else is derived.
func provenanceRefresh(t *testing.T, root, clone string) {
	t.Helper()
	doc := provenanceLoad(t, root)
	described := map[string]string{}
	previous := map[string]string{}
	for _, entry := range doc.Files {
		described[entry.Path] = entry.LocalChange
		previous[entry.Path] = entry.Status
	}
	upstreamNames := map[string]bool{}
	for _, name := range provenanceUpstreamList(t, clone, doc.UpstreamCommit, doc.UpstreamRoot) {
		upstreamNames[name] = true
	}

	doc.Files = nil
	doc.Counts = provenanceCounts{}
	var unowned []string
	imported := map[string]bool{}
	for _, relative := range provenanceTree(t, root) {
		entry := provenanceFile{Path: relative}
		entry.Bytes, entry.SHA256 = provenanceFileDigest(t, root, relative)
		upstreamPath := doc.UpstreamRoot + strings.TrimPrefix(relative, provenanceRuntime)
		switch {
		case !upstreamNames[upstreamPath]:
			// Absence upstream is not authorship: a pin bump that drops or renames
			// an imported file must not relabel the copy as fak's own.
			if previous[relative] != statusFakAdded {
				unowned = append(unowned, relative)
				continue
			}
			entry.Status = statusFakAdded
			doc.Counts.FakAdded++
		default:
			imported[upstreamPath] = true
			entry.UpstreamPath = upstreamPath
			entry.UpstreamSHA256 = provenanceUpstreamDigest(t, clone, doc.UpstreamCommit, upstreamPath)
			if entry.UpstreamSHA256 == entry.SHA256 {
				entry.Status = statusIdentical
				doc.Counts.Identical++
			} else {
				entry.Status = statusModified
				entry.LocalChange = described[relative]
				doc.Counts.Modified++
			}
		}
		doc.Files = append(doc.Files, entry)
	}

	if len(unowned) > 0 {
		t.Fatalf("refresh refused, manifest left unchanged: %d files have no upstream object at %s and are not recorded as %s: %s; "+
			"restore or remove a file upstream dropped, or add a %s entry by hand for a file fak wrote",
			len(unowned), doc.UpstreamCommit, statusFakAdded, strings.Join(unowned, ", "), statusFakAdded)
	}

	skipped := map[string]int{}
	for name := range upstreamNames {
		if !imported[name] {
			skipped[path.Dir(name)]++
		}
	}
	doc.UpstreamRuntime = provenanceUpstreamRuntime{
		Files:                  len(upstreamNames),
		Imported:               len(imported),
		NotImported:            len(upstreamNames) - len(imported),
		NotImportedByDirectory: skipped,
	}
	for index := range doc.Notices {
		entry := &doc.Notices[index]
		entry.Bytes, entry.SHA256 = provenanceFileDigest(t, root, entry.Path)
		entry.UpstreamSHA256 = provenanceUpstreamDigest(t, clone, doc.UpstreamCommit, entry.UpstreamPath)
		entry.Status = statusIdentical
		if entry.UpstreamSHA256 != entry.SHA256 {
			entry.Status = statusModified
		}
	}

	var out bytes.Buffer
	encoder := json.NewEncoder(&out)
	encoder.SetEscapeHTML(false)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(doc); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, filepath.FromSlash(provenanceManifest)), out.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("rewrote %s from %s", provenanceManifest, doc.UpstreamCommit)
}
