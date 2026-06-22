// Package hfhub resolves and downloads model files from the Hugging Face Hub
// via `hf://` URIs, with a local content cache, optional HF_TOKEN auth, and
// best-effort SHA256 verification against the Hub's LFS oid (the X-Linked-Etag
// the Hub stamps on every LFS object).
//
// URI grammar:
//
//	hf://<owner>/<repo>[@<revision>]/<path/to/file>
//
//	owner/repo   required — the model repo (e.g. mradermacher/Qwen2.5-1.5B-GGUF)
//	@revision    optional — a branch, tag, or commit; default "main"
//	path/to/file required — the file within the repo (e.g. model.Q8_0.gguf)
//
// The package is pure standard library (the `fak` binary ships with zero
// external dependencies), so it can be linked into cmd/fak without bloating the
// static build.
package hfhub

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// DefaultBaseURL is the Hugging Face Hub origin the resolve URLs are built on.
const DefaultBaseURL = "https://huggingface.co"

// Ref is a parsed hf:// reference to a single file in a Hub repo.
type Ref struct {
	Repo     string // "<owner>/<repo>"
	Revision string // branch/tag/commit; "main" when unspecified
	File     string // repo-relative file path
}

// ParseURI parses an hf:// URI into a Ref. It requires the owner/repo/file form
// so the boundary between the repo and the file is unambiguous; a bare
// single-segment canonical repo (e.g. hf://gpt2/...) is intentionally not
// accepted.
func ParseURI(uri string) (Ref, error) {
	rest, ok := strings.CutPrefix(uri, "hf://")
	if !ok {
		return Ref{}, fmt.Errorf("hfhub: %q is not an hf:// URI", uri)
	}
	rest = strings.TrimPrefix(rest, "/")
	parts := strings.Split(rest, "/")
	if len(parts) < 3 || parts[0] == "" || parts[1] == "" {
		return Ref{}, fmt.Errorf("hfhub: %q must be hf://<owner>/<repo>[@rev]/<file>", uri)
	}
	owner, repoAndRev := parts[0], parts[1]
	repoName, rev := repoAndRev, "main"
	if name, r, found := strings.Cut(repoAndRev, "@"); found {
		if name == "" || r == "" {
			return Ref{}, fmt.Errorf("hfhub: %q has an empty repo or revision around '@'", uri)
		}
		repoName, rev = name, r
	}
	file := strings.Join(parts[2:], "/")
	if file == "" || strings.HasSuffix(file, "/") {
		return Ref{}, fmt.Errorf("hfhub: %q names no file within the repo", uri)
	}
	return Ref{Repo: owner + "/" + repoName, Revision: rev, File: file}, nil
}

// ResolveURL builds the Hub download URL for the ref against base (e.g.
// DefaultBaseURL): {base}/{repo}/resolve/{revision}/{file}.
func (r Ref) ResolveURL(base string) string {
	base = strings.TrimRight(base, "/")
	return base + "/" + r.Repo + "/resolve/" + r.Revision + "/" + r.File
}

// cacheRel is the ref's path under the cache root: {repo}/{revision}/{file}.
func (r Ref) cacheRel() string {
	return filepath.Join(filepath.FromSlash(r.Repo), r.Revision, filepath.FromSlash(r.File))
}

// Client downloads Hub files into a local cache. The zero value is not usable;
// call NewClient.
type Client struct {
	BaseURL  string       // Hub origin; defaults to DefaultBaseURL
	HTTP     *http.Client // defaults to http.DefaultClient (follows redirects)
	Token    string       // bearer token for gated/private repos; from HF_TOKEN
	CacheDir string       // cache root; defaults to <user-cache>/fak-models/hub
}

// NewClient returns a Client wired from the environment: BaseURL = DefaultBaseURL,
// Token = $HF_TOKEN (or $HUGGING_FACE_HUB_TOKEN), CacheDir = $FAK_MODELS_DIR or
// <user-cache-dir>/fak-models/hub.
func NewClient() *Client {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		token = os.Getenv("HUGGING_FACE_HUB_TOKEN")
	}
	return &Client{
		BaseURL:  DefaultBaseURL,
		HTTP:     http.DefaultClient,
		Token:    token,
		CacheDir: defaultCacheDir(),
	}
}

func defaultCacheDir() string {
	if dir := os.Getenv("FAK_MODELS_DIR"); dir != "" {
		return filepath.Join(dir, "hub")
	}
	if dir, err := os.UserCacheDir(); err == nil {
		return filepath.Join(dir, "fak-models", "hub")
	}
	return filepath.Join(".", ".fak-models", "hub")
}

func (c *Client) base() string {
	if c.BaseURL != "" {
		return c.BaseURL
	}
	return DefaultBaseURL
}

func (c *Client) httpClient() *http.Client {
	if c.HTTP != nil {
		return c.HTTP
	}
	return http.DefaultClient
}

func (c *Client) authorize(req *http.Request) {
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
}

// CachePath is the absolute local path the ref caches to. It is returned by
// Download and can be checked ahead of time for a cache hit.
func (c *Client) CachePath(r Ref) string {
	return filepath.Join(c.CacheDir, r.cacheRel())
}

// Download fetches r into the cache and returns the local path. A non-empty
// cached file is returned without any network call (idempotent re-runs are
// free). When the Hub stamps an LFS sha256 (X-Linked-Etag), the downloaded
// bytes are verified against it and a mismatch is a hard error. progress, when
// non-nil, receives human-readable status lines.
func (c *Client) Download(ctx context.Context, r Ref, progress io.Writer) (string, error) {
	dst := c.CachePath(r)
	if fi, err := os.Stat(dst); err == nil && fi.Size() > 0 {
		logf(progress, "cache hit: %s", dst)
		return dst, nil
	}
	resolveURL := r.ResolveURL(c.base())
	want := c.linkedSHA(ctx, resolveURL) // best-effort; "" when the Hub stamps none

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, resolveURL, nil)
	if err != nil {
		return "", err
	}
	c.authorize(req)
	logf(progress, "GET %s", resolveURL)
	resp, err := c.httpClient().Do(req)
	if err != nil {
		return "", fmt.Errorf("hfhub: download %s: %w", r.File, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("hfhub: download %s: hub returned %s", r.File, resp.Status)
	}
	if want == "" {
		want = trimETag(resp.Header.Get("X-Linked-Etag"))
	}

	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return "", err
	}
	tmp, err := os.CreateTemp(filepath.Dir(dst), ".hfhub-*.part")
	if err != nil {
		return "", err
	}
	tmpName := tmp.Name()
	hasher := sha256.New()
	n, copyErr := io.Copy(io.MultiWriter(tmp, hasher), resp.Body)
	closeErr := tmp.Close()
	if copyErr != nil || closeErr != nil {
		os.Remove(tmpName)
		return "", errors.Join(copyErr, closeErr)
	}
	got := hex.EncodeToString(hasher.Sum(nil))
	if want != "" && !strings.EqualFold(want, got) {
		os.Remove(tmpName)
		return "", fmt.Errorf("hfhub: sha256 mismatch for %s: hub oid %s, got %s", r.File, want, got)
	}
	if err := os.Rename(tmpName, dst); err != nil {
		os.Remove(tmpName)
		return "", err
	}
	if want != "" {
		logf(progress, "downloaded %d bytes, sha256 verified: %s", n, dst)
	} else {
		logf(progress, "downloaded %d bytes (no hub oid to verify): %s", n, dst)
	}
	return dst, nil
}

// linkedSHA does a best-effort HEAD to read the Hub's LFS sha256 (X-Linked-Etag)
// without downloading the object. Any failure yields "" — verification is then
// skipped rather than blocking the download.
func (c *Client) linkedSHA(ctx context.Context, url string) string {
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, url, nil)
	if err != nil {
		return ""
	}
	c.authorize(req)
	resp, err := c.httpClient().Do(req)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	return trimETag(resp.Header.Get("X-Linked-Etag"))
}

// trimETag strips the surrounding quotes (and any weak-validator prefix) the Hub
// wraps the LFS oid in, returning a bare 64-hex sha256 or "".
func trimETag(etag string) string {
	etag = strings.TrimSpace(etag)
	etag = strings.TrimPrefix(etag, "W/")
	etag = strings.Trim(etag, `"`)
	if len(etag) == 64 && isHex(etag) {
		return etag
	}
	return ""
}

func isHex(s string) bool {
	_, err := hex.DecodeString(s)
	return err == nil
}

func logf(w io.Writer, format string, args ...any) {
	if w == nil {
		return
	}
	fmt.Fprintf(w, format+"\n", args...)
}

// (cache layout helpers live above; FetchURI/IsURI below are the CLI entry points.)

// FetchURI is the one-call convenience used by the CLI: parse, download, return
// the local path. base, when "", defaults to DefaultBaseURL via NewClient.
func FetchURI(ctx context.Context, uri string, progress io.Writer) (string, error) {
	ref, err := ParseURI(uri)
	if err != nil {
		return "", err
	}
	return NewClient().Download(ctx, ref, progress)
}

// IsURI reports whether s looks like an hf:// reference (used to branch a
// path-or-hf argument before touching the filesystem).
func IsURI(s string) bool {
	return strings.HasPrefix(s, "hf://")
}
