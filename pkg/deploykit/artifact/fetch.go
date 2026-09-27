package artifact

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/deploykit"
)

// Fetch downloads a prebuilt artifact. Its size and SHA-256 are checked before anything
// executes it; only matching bytes are smoked, and the smoke must attest Commit.
type Fetch struct {
	// URL is https, or http to a loopback host (the same rule `fak self-update` applies).
	URL string `json:"url"`
	// SHA256 is the pinned hex digest of the artifact bytes.
	SHA256 string `json:"sha256"`
	// Size is the pinned artifact length in bytes.
	Size int64 `json:"size"`
	// Commit is the full source commit the artifact must attest through `version --json`.
	Commit string `json:"commit"`
	// AppVersion, when set, is the app version the artifact must report. Empty accepts the
	// version the digest-verified bytes report.
	AppVersion string `json:"app_version,omitempty"`
	// Generation is the slot generation the candidate is stored under. It must be positive.
	Generation uint64 `json:"generation"`
	// Manifest is a signed release manifest envelope. Setting it fails closed with
	// ErrManifestNotWired until the selfupdate verifier is exported: a manifest-backed fetch is
	// never silently downgraded to a SHA-256-only check.
	Manifest *Manifest `json:"manifest,omitempty"`
}

// Manifest is the signed `fak self-update` manifest envelope: a canonical JSON payload and its
// base64 ed25519 signature.
type Manifest struct {
	Payload   json.RawMessage `json:"payload"`
	Signature string          `json:"signature"`
}

// Kind reports KindFetch.
func (Fetch) Kind() string { return KindFetch }

var httpClient = &http.Client{}

func (f Fetch) preflight(deploykit.Executor) error {
	if f.Manifest != nil {
		return fmt.Errorf("%w: the signed-manifest verifier is unexported in internal/selfupdate/cmd", ErrManifestNotWired)
	}
	if err := fetchURLAllowed(f.URL); err != nil {
		return err
	}
	if !validDigest(strings.ToLower(strings.TrimSpace(f.SHA256))) || f.Size < 1 {
		return fmt.Errorf("%w: pinned SHA-256 or size is invalid", ErrInvalidSource)
	}
	if !fullCommit(strings.TrimSpace(f.Commit)) {
		return fmt.Errorf("%w: commit %q is not a full 40-hex object ID", ErrInvalidSource, clip(f.Commit))
	}
	if f.Generation == 0 {
		return fmt.Errorf("%w: generation must be positive", ErrInvalidSource)
	}
	return nil
}

func (f Fetch) acquire(ctx context.Context, x deploykit.Executor, dir string) (Candidate, error) {
	path := filepath.Join(dir, candidateName(runtime.GOOS))
	want := strings.ToLower(strings.TrimSpace(f.SHA256))
	got, err := download(ctx, strings.TrimSpace(f.URL), path, f.Size)
	if err != nil {
		return Candidate{}, err
	}
	if got != want {
		return Candidate{}, fmt.Errorf("%w: got %s, want %s", ErrDigestMismatch, got, want)
	}
	commit := strings.ToLower(strings.TrimSpace(f.Commit))
	id, err := smoke(ctx, x, path, dir, commit)
	if err != nil {
		return Candidate{}, err
	}
	version := f.AppVersion
	if version == "" {
		version = id.AppVersion
	}
	return Candidate{
		Kind: KindFetch, Path: path, Generation: f.Generation, Commit: commit,
		Digest: want, Size: f.Size, AppVersion: version,
	}, nil
}

// download streams rawURL into a new file at path, reading at most size+1 bytes, and returns
// the SHA-256 of what it wrote. A response of any other length is refused.
func download(ctx context.Context, rawURL, path string, size int64) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return "", fmt.Errorf("%w: %v", ErrFetchFailed, err)
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("%w: %v", ErrFetchFailed, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("%w: HTTP %s", ErrFetchFailed, resp.Status)
	}
	if resp.ContentLength >= 0 && resp.ContentLength != size {
		return "", fmt.Errorf("%w: server announces %d bytes, want %d", ErrSizeMismatch, resp.ContentLength, size)
	}
	out, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o700)
	if err != nil {
		return "", fmt.Errorf("%w: %v", ErrFetchFailed, err)
	}
	h := sha256.New()
	n, err := io.Copy(io.MultiWriter(out, h), io.LimitReader(resp.Body, size+1))
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return "", fmt.Errorf("%w: %v", ErrFetchFailed, err)
	}
	if n != size {
		return "", fmt.Errorf("%w: got %d bytes, want %d", ErrSizeMismatch, n, size)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// fetchURLAllowed applies the `fak self-update` artifact URL rule: https, or plain http only to
// a loopback host.
func fetchURLAllowed(raw string) error {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || u.Host == "" {
		return fmt.Errorf("%w: URL %q is not absolute", ErrInvalidSource, clip(raw))
	}
	switch host := u.Hostname(); {
	case u.Scheme == "https":
	case u.Scheme == "http" && (host == "127.0.0.1" || host == "localhost" || host == "::1"):
	default:
		return fmt.Errorf("%w: URL %q must be https (plain http only to a loopback host)", ErrInvalidSource, clip(raw))
	}
	return nil
}
