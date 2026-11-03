// Package skilldiscovery adapts OpenCode's remote skill catalog/staging subsystem.
// Source: anomalyco/opencode packages/core/src/skill/discovery.ts
// @4eb29a64f0054672950acf789f2b09487ebfbb20 (MIT; see LICENSE).
// Integrity pins, byte limits and immutable generations are FAK adaptations.
package skilldiscovery

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"
)

const (
	MaxIndexBytes  int64 = 256 << 10
	MaxFileBytes   int64 = 1 << 20
	MaxTotalBytes  int64 = 8 << 20
	MaxSkills            = 16
	MaxFiles             = 64
	maxRecordBytes int64 = 512 << 10
)

// Options requires an independently trusted index pin; Pull never derives trust
// by accepting the digest of the response it is about to download.
type Options struct {
	Source      string
	IndexSHA256 string
	CacheRoot   string
	// ValidateSkill parses an entrypoint without filesystem/network execution.
	// The CLI binds the real native parser and refuses external canonical imports.
	ValidateSkill func(name string, body []byte) error
}

// Index is a strict, integrity-bearing extension of OpenCode's catalog shape.
// OpenCode's name/version/files fields remain; sha256 and sizes bind every file.
type Index struct {
	Skills []CatalogSkill `json:"skills"`
}
type CatalogSkill struct {
	Name    string            `json:"name"`
	Version string            `json:"version"`
	Files   []string          `json:"files"`
	SHA256  map[string]string `json:"sha256"`
	Sizes   map[string]int64  `json:"sizes"`
}
type Receipt struct {
	Schema       string         `json:"schema"`
	SourceSHA256 string         `json:"source_sha256"`
	IndexSHA256  string         `json:"index_sha256"`
	SkillsRoot   string         `json:"skills_root"`
	Skills       []CatalogSkill `json:"skills"`
	CacheHit     bool           `json:"cache_hit"`
	Activated    bool           `json:"activated"`
}
type record struct {
	Schema       string `json:"schema"`
	SourceSHA256 string `json:"source_sha256"`
	IndexSHA256  string `json:"index_sha256"`
	Generation   string `json:"generation"`
	IndexBytes   []byte `json:"index_bytes"`
}

// Error contains only a bounded reason code, never a URL, response, credential
// or remotely supplied skill text.
type Error struct{ Code string }

func (e *Error) Error() string { return "skill pull: " + e.Code }
func fail(code string) error   { return &Error{Code: code} }
func digest(b []byte) string   { sum := sha256.Sum256(b); return hex.EncodeToString(sum[:]) }
func validDigest(s string) bool {
	if len(s) != 64 || strings.ToLower(s) != s {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}

// Pull fetches and publishes data only. It does not arm skills, change policy,
// install executable hooks, refresh credentials, or remove any old generation.
func Pull(ctx context.Context, opts Options) (Receipt, error) {
	client := newHTTPClient()
	defer client.CloseIdleConnections()
	return pull(ctx, opts, client)
}
func newHTTPClient() *http.Client {
	transport := &http.Transport{
		Proxy: nil, DisableCompression: true,
		DialContext:         (&net.Dialer{Timeout: 10 * time.Second}).DialContext,
		TLSHandshakeTimeout: 10 * time.Second, ResponseHeaderTimeout: 15 * time.Second,
	}
	return &http.Client{Transport: transport, Timeout: 30 * time.Second,
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error { return fail("REDIRECT_REFUSED") }}
}

type httpDoer interface {
	Do(*http.Request) (*http.Response, error)
}

func sourceURL(raw string) (*url.URL, error) {
	u, err := url.Parse(raw)
	if err != nil || u.Scheme != "https" || u.Hostname() == "" || u.User != nil ||
		u.RawQuery != "" || u.ForceQuery || u.Fragment != "" || u.Opaque != "" || u.RawPath != "" ||
		strings.ContainsAny(u.Path, "\\\x00\r\n") {
		return nil, fail("SOURCE_REFUSED")
	}
	for _, part := range strings.Split(u.Path, "/") {
		if part == "." || part == ".." {
			return nil, fail("SOURCE_REFUSED")
		}
	}
	if u.Path == "" {
		u.Path = "/"
	}
	if !strings.HasSuffix(u.Path, "/") {
		u.Path += "/"
	}
	u.RawPath = ""
	return u, nil
}
func fetch(ctx context.Context, c httpDoer, u *url.URL, limit int64) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, fail("SOURCE_REFUSED")
	}
	req.Header.Set("Accept-Encoding", "identity")
	res, err := c.Do(req)
	if err != nil {
		return nil, fail("FETCH_FAILED")
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return nil, fail("HTTP_STATUS_REFUSED")
	}
	if res.ContentLength > limit {
		return nil, fail("BYTE_LIMIT_EXCEEDED")
	}
	b, err := io.ReadAll(io.LimitReader(res.Body, limit+1))
	if err != nil {
		return nil, fail("FETCH_FAILED")
	}
	if int64(len(b)) > limit {
		return nil, fail("BYTE_LIMIT_EXCEEDED")
	}
	return b, nil
}
func safeSegment(s string) bool {
	if len(s) == 0 || len(s) > 128 || s == "." || s == ".." || strings.HasSuffix(s, ".") {
		return false
	}
	for _, c := range s {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_' || c == '.') {
			return false
		}
	}
	base := strings.ToUpper(strings.SplitN(s, ".", 2)[0])
	if base == "CON" || base == "PRN" || base == "AUX" || base == "NUL" {
		return false
	}
	if len(base) == 4 && (strings.HasPrefix(base, "COM") || strings.HasPrefix(base, "LPT")) && base[3] >= '1' && base[3] <= '9' {
		return false
	}
	return true
}
func safePath(s string) bool {
	if len(s) == 0 || len(s) > 512 {
		return false
	}
	parts := strings.Split(s, "/")
	if len(parts) > 8 {
		return false
	}
	for _, p := range parts {
		if !safeSegment(p) {
			return false
		}
	}
	return true
}
func decodeIndex(b []byte) (Index, error) {
	var index Index
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(&index); err != nil {
		return index, fail("INDEX_INVALID")
	}
	if err := d.Decode(new(any)); err != io.EOF {
		return index, fail("INDEX_INVALID")
	}
	if len(index.Skills) == 0 || len(index.Skills) > MaxSkills {
		return index, fail("INDEX_LIMIT_EXCEEDED")
	}
	names := map[string]bool{}
	count := 0
	var total int64
	for _, s := range index.Skills {
		key := strings.ToLower(s.Name)
		if !safeSegment(s.Name) || !safeSegment(s.Version) || names[key] {
			return index, fail("SKILL_ID_REFUSED")
		}
		names[key] = true
		if len(s.Files) == 0 || len(s.SHA256) != len(s.Files) || len(s.Sizes) != len(s.Files) {
			return index, fail("FILE_BINDING_REQUIRED")
		}
		seen := map[string]bool{}
		entry := false
		for _, f := range s.Files {
			key = strings.ToLower(f)
			if !safePath(f) || seen[key] {
				return index, fail("FILE_PATH_REFUSED")
			}
			seen[key] = true
			size, ok := s.Sizes[f]
			if !ok || size < 0 || size > MaxFileBytes || !validDigest(s.SHA256[f]) {
				return index, fail("FILE_BINDING_REQUIRED")
			}
			if f == "SKILL.md" {
				entry = true
				if size == 0 {
					return index, fail("ENTRYPOINT_INVALID")
				}
			}
			total += size
			count++
			if total > MaxTotalBytes || count > MaxFiles {
				return index, fail("INDEX_LIMIT_EXCEEDED")
			}
		}
		if !entry {
			return index, fail("ENTRYPOINT_REQUIRED")
		}
		for f := range seen {
			for p := path.Dir(f); p != "."; p = path.Dir(p) {
				if seen[p] {
					return index, fail("FILE_PATH_REFUSED")
				}
			}
		}
	}
	return index, nil
}
func rootDir(r *os.Root, name string) error {
	if err := r.Mkdir(name, 0700); err != nil && !errors.Is(err, os.ErrExist) {
		return fail("CACHE_WRITE_FAILED")
	}
	st, err := r.Lstat(name)
	if err != nil || !st.IsDir() || st.Mode()&os.ModeSymlink != 0 {
		return fail("CACHE_PATH_REFUSED")
	}
	return nil
}
func readBounded(r *os.Root, name string, limit int64) ([]byte, error) {
	st, err := r.Lstat(name)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, os.ErrNotExist
		}
		return nil, fail("CACHE_READ_FAILED")
	}
	if !st.Mode().IsRegular() || st.Size() > limit {
		return nil, fail("CACHE_PATH_REFUSED")
	}
	f, err := r.Open(name)
	if err != nil {
		return nil, fail("CACHE_READ_FAILED")
	}
	defer f.Close()
	actual, err := f.Stat()
	if err != nil || !actual.Mode().IsRegular() {
		return nil, fail("CACHE_PATH_REFUSED")
	}
	b, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil || int64(len(b)) > limit {
		return nil, fail("CACHE_READ_FAILED")
	}
	return b, nil
}
func writeNew(r *os.Root, name string, b []byte) error {
	f, err := r.OpenFile(name, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return fail("CACHE_WRITE_FAILED")
	}
	_, err = f.Write(b)
	if err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err != nil || closeErr != nil {
		return fail("CACHE_WRITE_FAILED")
	}
	return nil
}

func verifyGeneration(r *os.Root, gen string, index Index, validate func(string, []byte) error) error {
	if !strings.HasPrefix(gen, "objects/gen-") || len(strings.TrimPrefix(gen, "objects/gen-")) != 32 {
		return fail("CACHE_RECORD_INVALID")
	}
	if _, err := hex.DecodeString(strings.TrimPrefix(gen, "objects/gen-")); err != nil {
		return fail("CACHE_RECORD_INVALID")
	}
	st, err := r.Lstat(gen)
	if err != nil || !st.IsDir() || st.Mode()&os.ModeSymlink != 0 {
		return fail("CACHE_PATH_REFUSED")
	}
	dirs := map[string]bool{"skills": true}
	files := map[string]CatalogSkill{}
	for _, s := range index.Skills {
		dirs["skills/"+s.Name] = true
		for _, f := range s.Files {
			p := "skills/" + s.Name + "/" + f
			files[p] = s
			for d := path.Dir(p); d != "."; d = path.Dir(d) {
				dirs[d] = true
			}
		}
	}
	var walk func(string) error
	walk = func(dir string) error {
		st, err := r.Lstat(gen + "/" + dir)
		if err != nil || !st.IsDir() || st.Mode()&os.ModeSymlink != 0 {
			return fail("CACHE_PATH_REFUSED")
		}
		fd, err := r.Open(gen + "/" + dir)
		if err != nil {
			return fail("CACHE_READ_FAILED")
		}
		entries, readErr := fd.ReadDir(MaxFiles + MaxSkills + 1)
		closeErr := fd.Close()
		if readErr != nil && !errors.Is(readErr, io.EOF) || closeErr != nil {
			return fail("CACHE_READ_FAILED")
		}
		if len(entries) > MaxFiles+MaxSkills {
			return fail("CACHE_CONTENT_MISMATCH")
		}
		for _, entry := range entries {
			p := dir + "/" + entry.Name()
			if entry.Type()&os.ModeSymlink != 0 {
				return fail("CACHE_PATH_REFUSED")
			}
			if entry.IsDir() {
				if !dirs[p] {
					return fail("CACHE_CONTENT_MISMATCH")
				}
				if err := walk(p); err != nil {
					return err
				}
			} else if _, ok := files[p]; !ok {
				return fail("CACHE_CONTENT_MISMATCH")
			}
		}
		return nil
	}
	if err := walk("skills"); err != nil {
		return err
	}
	for p, s := range files {
		f := strings.TrimPrefix(p, "skills/"+s.Name+"/")
		b, err := readBounded(r, gen+"/"+p, s.Sizes[f])
		if err != nil || int64(len(b)) != s.Sizes[f] || digest(b) != s.SHA256[f] {
			return fail("CACHE_CONTENT_MISMATCH")
		}
		if f == "SKILL.md" && validate(s.Name, b) != nil {
			return fail("ENTRYPOINT_INVALID")
		}
	}
	return nil
}

func cached(r *os.Root, recordName, sourceHash, indexHash string, validate func(string, []byte) error) (record, Index, error) {
	var rec record
	var index Index
	b, err := readBounded(r, recordName, maxRecordBytes)
	if err != nil {
		return rec, index, err
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if d.Decode(&rec) != nil || d.Decode(new(any)) != io.EOF ||
		rec.Schema != "fak-skill-cache/1" || rec.SourceSHA256 != sourceHash || rec.IndexSHA256 != indexHash ||
		int64(len(rec.IndexBytes)) > MaxIndexBytes || digest(rec.IndexBytes) != indexHash {
		return rec, index, fail("CACHE_RECORD_INVALID")
	}
	index, err = decodeIndex(rec.IndexBytes)
	if err != nil {
		return rec, index, err
	}
	err = verifyGeneration(r, rec.Generation, index, validate)
	return rec, index, err
}
func receipt(root string, rec record, index Index, hit bool) Receipt {
	return Receipt{Schema: "fak-skill-pull/1", SourceSHA256: rec.SourceSHA256, IndexSHA256: rec.IndexSHA256,
		SkillsRoot: filepath.Join(root, filepath.FromSlash(rec.Generation), "skills"), Skills: index.Skills, CacheHit: hit, Activated: false}
}

func pull(ctx context.Context, opts Options, c httpDoer) (result Receipt, retErr error) {
	if !validDigest(opts.IndexSHA256) {
		return result, fail("TRUSTED_INDEX_PIN_REQUIRED")
	}
	if opts.ValidateSkill == nil {
		return result, fail("ENTRYPOINT_VALIDATOR_REQUIRED")
	}
	source, err := sourceURL(opts.Source)
	if err != nil {
		return result, err
	}
	abs, err := filepath.Abs(opts.CacheRoot)
	if err != nil || opts.CacheRoot == "" {
		return result, fail("CACHE_ROOT_REQUIRED")
	}
	real, err := filepath.EvalSymlinks(abs)
	if err != nil || real != abs {
		return result, fail("CACHE_ROOT_REFUSED")
	}
	st, err := os.Lstat(abs)
	if err != nil || !st.IsDir() || st.Mode()&os.ModeSymlink != 0 {
		return result, fail("CACHE_ROOT_REFUSED")
	}
	root, err := os.OpenRoot(abs)
	if err != nil {
		return result, fail("CACHE_ROOT_REFUSED")
	}
	defer root.Close()
	sourceHash := digest([]byte(source.String()))
	key := digest([]byte(sourceHash + ":" + opts.IndexSHA256))
	for _, d := range []string{"objects", "records"} {
		if err := rootDir(root, d); err != nil {
			return result, err
		}
	}
	recordName := "records/" + key + ".json"
	rec, index, err := cached(root, recordName, sourceHash, opts.IndexSHA256, opts.ValidateSkill)
	if err == nil {
		return receipt(abs, rec, index, true), nil
	}
	if !errors.Is(err, os.ErrNotExist) {
		return result, err
	}
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	indexURL := source.ResolveReference(&url.URL{Path: "index.json"})
	indexBytes, err := fetch(ctx, c, indexURL, MaxIndexBytes)
	if err != nil {
		return result, err
	}
	if digest(indexBytes) != opts.IndexSHA256 {
		return result, fail("INDEX_DIGEST_MISMATCH")
	}
	index, err = decodeIndex(indexBytes)
	if err != nil {
		return result, err
	}
	var token [16]byte
	if _, err := rand.Read(token[:]); err != nil {
		return result, fail("STAGING_CREATE_FAILED")
	}
	gen := "objects/gen-" + hex.EncodeToString(token[:])
	if err := root.Mkdir(gen, 0700); err != nil {
		return result, fail("STAGING_CREATE_FAILED")
	}
	published := false
	defer func() {
		if !published {
			if err := root.RemoveAll(gen); err != nil {
				retErr = errors.Join(retErr, fail("STAGING_CLEANUP_FAILED"))
			}
		}
	}()
	if err := root.Mkdir(gen+"/skills", 0700); err != nil {
		return result, fail("STAGING_CREATE_FAILED")
	}
	for _, s := range index.Skills {
		for _, f := range s.Files {
			u := source.ResolveReference(&url.URL{Path: s.Name + "/" + f})
			if u.Scheme != source.Scheme || u.Host != source.Host || !strings.HasPrefix(u.Path, source.Path) {
				return result, fail("SOURCE_REFUSED")
			}
			b, err := fetch(ctx, c, u, s.Sizes[f])
			if err != nil {
				return result, err
			}
			if int64(len(b)) != s.Sizes[f] || digest(b) != s.SHA256[f] {
				return result, fail("FILE_DIGEST_MISMATCH")
			}
			if f == "SKILL.md" && opts.ValidateSkill(s.Name, b) != nil {
				return result, fail("ENTRYPOINT_INVALID")
			}
			name := gen + "/skills/" + s.Name + "/" + f
			if err := root.MkdirAll(path.Dir(name), 0700); err != nil {
				return result, fail("STAGING_CREATE_FAILED")
			}
			if err := writeNew(root, name, b); err != nil {
				return result, err
			}
		}
	}
	rec = record{Schema: "fak-skill-cache/1", SourceSHA256: sourceHash, IndexSHA256: opts.IndexSHA256, Generation: gen, IndexBytes: indexBytes}
	recordBytes, err := json.Marshal(rec)
	if err != nil || int64(len(recordBytes)) > maxRecordBytes {
		return result, fail("CACHE_RECORD_INVALID")
	}
	if err := writeNew(root, gen+"/record.json", recordBytes); err != nil {
		return result, err
	}
	if err := verifyGeneration(root, gen, index, opts.ValidateSkill); err != nil {
		return result, err
	}
	if err := ctx.Err(); err != nil {
		return result, fail("FETCH_CANCELLED")
	}
	current, err := os.Stat(abs)
	if err != nil || !os.SameFile(st, current) {
		return result, fail("CACHE_ROOT_MOVED")
	}
	// Publishing this hard link is the single create-only visibility point.
	// A concurrent winner is verified, not overwritten. No old root is replaced.
	if err := root.Link(gen+"/record.json", recordName); err != nil {
		if !errors.Is(err, os.ErrExist) {
			return result, fail("CACHE_PUBLISH_FAILED")
		}
		winner, winningIndex, err := cached(root, recordName, sourceHash, opts.IndexSHA256, opts.ValidateSkill)
		if err != nil {
			return result, err
		}
		return receipt(abs, winner, winningIndex, true), nil
	}
	published = true
	return receipt(abs, rec, index, false), nil
}
