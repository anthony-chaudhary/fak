package hfhub

import (
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ErrDigestMismatch reports that downloaded bytes failed integrity
// verification against the expected LFS sha256 or the Hub's git-blob-sha1
// ETag. The partial file has been removed and the destination was never
// published; callers can branch on it with errors.Is.
var ErrDigestMismatch = errors.New("hfhub: digest mismatch")

// maxDownloadAttempts bounds the transfer loop: every GET counts, including
// the single restart taken after a digest mismatch.
const maxDownloadAttempts = 5

// defaultDownloadBackoff is the first retry delay; each subsequent delay
// doubles (500ms, 1s, 2s, 4s across the remaining attempt slots).
const defaultDownloadBackoff = 500 * time.Millisecond

// partSuffix names the durable partial sibling of a cache entry. A stable
// name (not a random temp) is what makes an interrupted run recoverable: the
// next invocation finds dst+".part", rehashes its prefix, and continues with
// "Range: bytes=N-".
const partSuffix = ".part"

// hubStatusError carries the Hub response status so the retry loop can tell
// permanent refusals (other 4xx) apart from retryable ones (408/429/5xx).
type hubStatusError struct {
	status int
	text   string
}

func (e *hubStatusError) Error() string { return "hfhub: download: hub returned " + e.text }

func (e *hubStatusError) retryable() bool {
	return e.status == http.StatusRequestTimeout ||
		e.status == http.StatusTooManyRequests ||
		e.status >= http.StatusInternalServerError
}

// downloadFile fetches url into dst through the durable dst+".part" sibling,
// adapting the TensorFold pull_file transfer contract (Apache-2.0; source
// cited in the commit that introduced this seam):
//
//   - when the LFS sha256 (want) is known, a retained .part prefix is resumed
//     with "Range: bytes=N-" and rehashed into the same digest as the suffix;
//   - a 200 where a 206 was requested means the server ignored the range:
//     the transfer restarts from zero;
//   - transient failures retry up to maxDownloadAttempts with doubling
//     backoff; a digest mismatch removes the partial and restarts once, then
//     fails with ErrDigestMismatch;
//   - without a sha256, a stamped git-blob-sha1 ETag verifies the finished
//     bytes (sha1 over "blob <size>\x00" + bytes);
//   - dst is published (renamed) only after verification.
func (c *Client) downloadFile(ctx context.Context, url, dst, want string, progress io.Writer) error {
	part := dst + partSuffix
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	base := c.backoff()
	mismatchRestarted := false
	var lastErr error
	for attempt := 1; attempt <= maxDownloadAttempts; attempt++ {
		lastErr = c.attemptTransfer(ctx, url, part, dst, &want, progress)
		if lastErr == nil {
			return nil
		}
		retryable := false
		switch {
		case errors.Is(lastErr, ErrDigestMismatch):
			os.Remove(part)
			if !mismatchRestarted {
				mismatchRestarted = true
				retryable = true
			}
		default:
			var se *hubStatusError
			switch {
			case errors.As(lastErr, &se):
				retryable = se.retryable() && attempt < maxDownloadAttempts
			default:
				retryable = ctx.Err() == nil && attempt < maxDownloadAttempts
			}
		}
		if !retryable {
			return lastErr
		}
		if !errors.Is(lastErr, ErrDigestMismatch) {
			delay := base << uint(attempt-1)
			if c.backoffProbe != nil {
				c.backoffProbe(delay)
			}
			if !sleepThrough(ctx, delay) {
				return ctx.Err()
			}
		}
	}
	return lastErr
}

// attemptTransfer performs one GET of url (resuming from the retained part
// prefix when the LFS sha256 is known), verifies the complete object, and
// publishes it to dst. It returns nil on success, a wrapped ErrDigestMismatch
// when verification fails, or a transport/status error otherwise.
func (c *Client) attemptTransfer(ctx context.Context, url, part, dst string, want *string, progress io.Writer) error {
	offset := int64(0)
	if *want != "" {
		if fi, err := os.Stat(part); err == nil && fi.Size() > 0 {
			offset = fi.Size()
		}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	c.authorize(req)
	if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	}
	logf(progress, "GET %s", url)
	resp, err := c.httpClient().Do(req)
	if err != nil {
		return fmt.Errorf("hfhub: download: %w", err)
	}
	defer resp.Body.Close()

	// An encoded body would hash differently from the stored object; refuse
	// rather than verify (or resume) against the wrong bytes.
	if enc := resp.Header.Get("Content-Encoding"); enc != "" && !strings.EqualFold(enc, "identity") {
		return fmt.Errorf("hfhub: download: unsupported content encoding %q", enc)
	}

	resumed := false
	switch resp.StatusCode {
	case http.StatusPartialContent:
		if offset == 0 {
			// We requested the entire object. Without a digest, accepting an
			// unsolicited suffix could publish incomplete bytes as a cache hit.
			return errors.New("hfhub: download: unsolicited partial response to full-object request")
		}
		resumed = true
		logf(progress, "resuming from byte %d", offset)
	case http.StatusOK:
		if offset > 0 {
			logf(progress, "server ignored the range request; restarting from zero")
		}
		offset = 0
	case http.StatusRequestedRangeNotSatisfiable:
		// The retained prefix reaches or exceeds the object: drop it and let
		// the loop retry from zero.
		os.Remove(part)
		return errors.New("hfhub: download: retained partial is not resumable; restarting")
	default:
		return &hubStatusError{status: resp.StatusCode, text: resp.Status}
	}
	if *want == "" {
		*want = trimETag(resp.Header.Get("X-Linked-Etag"))
	}

	f, err := openPartForWrite(part, offset)
	if err != nil {
		return err
	}
	hasher := sha256.New()
	if resumed {
		// Feed the retained prefix into the same digest the suffix will fill,
		// so the final sum covers the whole object.
		if _, err := io.CopyN(hasher, f, offset); err != nil {
			f.Close()
			return fmt.Errorf("hfhub: rehash retained partial: %w", err)
		}
		if _, err := f.Seek(0, io.SeekEnd); err != nil {
			f.Close()
			return err
		}
	}
	n, copyErr := io.Copy(io.MultiWriter(f, hasher), resp.Body)
	closeErr := f.Close()
	if copyErr != nil || closeErr != nil {
		return errors.Join(fmt.Errorf("hfhub: download: %w", copyErr), closeErr)
	}

	verifiedAs := ""
	switch {
	case *want != "":
		got := hex.EncodeToString(hasher.Sum(nil))
		if !strings.EqualFold(*want, got) {
			return fmt.Errorf("hfhub: sha256 mismatch: hub oid %s, got %s: %w", *want, got, ErrDigestMismatch)
		}
		verifiedAs = "sha256 verified"
	default:
		if etag := bareHashETag(resp.Header.Get("ETag"), 40); etag != "" {
			blob, err := gitBlobSHA1File(part)
			if err != nil {
				return fmt.Errorf("hfhub: verify git blob sha1: %w", err)
			}
			if !strings.EqualFold(etag, blob) {
				return fmt.Errorf("hfhub: git blob sha1 mismatch: hub etag %s, got %s: %w", etag, blob, ErrDigestMismatch)
			}
			verifiedAs = "git blob sha1 verified"
		}
	}
	if err := os.Rename(part, dst); err != nil {
		os.Remove(part)
		return err
	}
	if verifiedAs == "" {
		verifiedAs = "no hub digest to verify"
	}
	logf(progress, "downloaded %d bytes (%s): %s", offset+n, verifiedAs, dst)
	return nil
}

// openPartForWrite opens the partial for appending at offset>0 (an existing
// retained prefix) or creates/truncates it at offset==0. If the prefix
// vanished between stat and open, it falls back to a fresh start rather than
// failing the attempt.
func openPartForWrite(part string, offset int64) (*os.File, error) {
	if offset > 0 {
		if f, err := os.OpenFile(part, os.O_RDWR, 0o644); err == nil {
			return f, nil
		}
	}
	return os.OpenFile(part, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
}

// gitBlobSHA1File computes the git blob identifier of the file at path:
// sha1 over the "blob <size>\x00" header followed by the raw bytes — the
// value git (and the Hub's ETag for non-LFS objects) stamps on the object.
func gitBlobSHA1File(path string) (string, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha1.New()
	if _, err := fmt.Fprintf(h, "blob %d\x00", fi.Size()); err != nil {
		return "", err
	}
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// sleepThrough waits d unless ctx is canceled first; it reports whether the
// full delay elapsed.
func sleepThrough(ctx context.Context, d time.Duration) bool {
	if d <= 0 {
		return true
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}
