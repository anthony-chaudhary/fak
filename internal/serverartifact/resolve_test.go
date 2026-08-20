package serverartifact_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/serverartifact"
)

func TestResolveReturnsDigestBoundCanonicalIdentity(t *testing.T) {
	data := []byte("fixture model bytes")
	path := writeFixture(t, "model.gguf", data)

	resolved, err := serverartifact.Resolve(context.Background(), serverartifact.Reference{
		Path:   path,
		SHA256: digest(data),
	})
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	t.Cleanup(func() { _ = resolved.Close() })
	identity := resolved.Identity()
	if !filepath.IsAbs(identity.CanonicalPath) {
		t.Fatalf("CanonicalPath = %q, want absolute", identity.CanonicalPath)
	}
	if identity.SizeBytes != int64(len(data)) {
		t.Fatalf("SizeBytes = %d, want %d", identity.SizeBytes, len(data))
	}
	if identity.DigestAlgorithm != serverartifact.DigestAlgorithmSHA256 || identity.Digest != digest(data) {
		t.Fatalf("digest identity = %s:%s, want sha256:%s", identity.DigestAlgorithm, identity.Digest, digest(data))
	}
	if err := resolved.VerifyUnchanged(context.Background()); err != nil {
		t.Fatalf("VerifyUnchanged() error = %v", err)
	}
	if serverartifact.DefaultMaxSizeBytes < 1<<40 {
		t.Fatalf("DefaultMaxSizeBytes = %d, want at least 1 TiB", serverartifact.DefaultMaxSizeBytes)
	}
}

func TestResolveRejectsUnacceptedArtifacts(t *testing.T) {
	data := []byte("model")
	validDigest := digest(data)
	t.Run("digest mismatch is explicit", func(t *testing.T) {
		path := writeFixture(t, "model.gguf", data)
		_, err := serverartifact.Resolve(context.Background(), serverartifact.Reference{Path: path, SHA256: digest([]byte("other"))})
		if !errors.Is(err, serverartifact.ErrDigestMismatch) {
			t.Fatalf("Resolve() error = %v, want ErrDigestMismatch", err)
		}
		var mismatch *serverartifact.DigestMismatchError
		if !errors.As(err, &mismatch) || mismatch.Actual != validDigest {
			t.Fatalf("Resolve() error = %#v, want typed mismatch with actual digest", err)
		}
	})
	t.Run("missing file", func(t *testing.T) {
		_, err := serverartifact.Resolve(context.Background(), serverartifact.Reference{Path: filepath.Join(t.TempDir(), "missing.gguf"), SHA256: validDigest})
		if !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("Resolve() error = %v, want os.ErrNotExist", err)
		}
	})
	t.Run("directory", func(t *testing.T) {
		_, err := serverartifact.Resolve(context.Background(), serverartifact.Reference{Path: t.TempDir(), SHA256: validDigest})
		if !errors.Is(err, serverartifact.ErrNotRegular) {
			t.Fatalf("Resolve() error = %v, want ErrNotRegular", err)
		}
	})
	t.Run("size limit", func(t *testing.T) {
		path := writeFixture(t, "model.gguf", data)
		_, err := serverartifact.Resolve(context.Background(), serverartifact.Reference{Path: path, SHA256: validDigest, MaxSizeBytes: int64(len(data) - 1)})
		if !errors.Is(err, serverartifact.ErrSizeLimit) {
			t.Fatalf("Resolve() error = %v, want ErrSizeLimit", err)
		}
		var limit *serverartifact.SizeLimitError
		if !errors.As(err, &limit) || limit.MaxBytes != int64(len(data)-1) {
			t.Fatalf("Resolve() error = %#v, want typed size limit", err)
		}
	})
	t.Run("canceled context", func(t *testing.T) {
		path := writeFixture(t, "model.gguf", data)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := serverartifact.Resolve(ctx, serverartifact.Reference{Path: path, SHA256: validDigest})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Resolve() error = %v, want context.Canceled", err)
		}
	})
	t.Run("invalid digest declaration", func(t *testing.T) {
		path := writeFixture(t, "model.gguf", data)
		_, err := serverartifact.Resolve(context.Background(), serverartifact.Reference{Path: path, SHA256: "not-a-digest"})
		if !errors.Is(err, serverartifact.ErrInvalidReference) {
			t.Fatalf("Resolve() error = %v, want ErrInvalidReference", err)
		}
	})
}

func TestResolveRejectsLinks(t *testing.T) {
	target := writeFixture(t, "target.gguf", []byte("model"))
	link := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.Symlink(target, link); err != nil {
		t.Skipf("symlink creation unavailable: %v", err)
	}
	_, err := serverartifact.Resolve(context.Background(), serverartifact.Reference{Path: link, SHA256: digest([]byte("model"))})
	if !errors.Is(err, serverartifact.ErrDisallowedLink) {
		t.Fatalf("Resolve() error = %v, want ErrDisallowedLink", err)
	}

	realDir := t.TempDir()
	ancestorLink := filepath.Join(t.TempDir(), "models")
	if err := os.Symlink(realDir, ancestorLink); err != nil {
		t.Skipf("directory symlink creation unavailable: %v", err)
	}
	linkedPath := filepath.Join(realDir, "linked.gguf")
	if err := os.WriteFile(linkedPath, []byte("linked model"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err = serverartifact.Resolve(context.Background(), serverartifact.Reference{
		Path: filepath.Join(ancestorLink, "linked.gguf"), SHA256: digest([]byte("linked model")),
	})
	if !errors.Is(err, serverartifact.ErrDisallowedLink) {
		t.Fatalf("Resolve() through linked ancestor error = %v, want ErrDisallowedLink", err)
	}
}

func TestVerifyUnchangedRejectsPostCheckMutation(t *testing.T) {
	t.Run("in-place content mutation", func(t *testing.T) {
		original := []byte("model-a")
		path := writeFixture(t, "model.gguf", original)
		resolved := mustResolve(t, path, original)
		if err := os.WriteFile(path, []byte("model-b"), 0o600); err != nil {
			t.Fatal(err)
		}
		future := time.Now().Add(2 * time.Second)
		if err := os.Chtimes(path, future, future); err != nil {
			t.Fatal(err)
		}
		if err := resolved.VerifyUnchanged(context.Background()); !errors.Is(err, serverartifact.ErrArtifactChanged) {
			t.Fatalf("VerifyUnchanged() error = %v, want ErrArtifactChanged", err)
		}
	})
	t.Run("path replacement", func(t *testing.T) {
		original := []byte("model-a")
		path := writeFixture(t, "model.gguf", original)
		resolved := mustResolve(t, path, original)
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, original, 0o600); err != nil {
			t.Fatal(err)
		}
		if err := resolved.VerifyUnchanged(context.Background()); !errors.Is(err, serverartifact.ErrArtifactChanged) {
			t.Fatalf("VerifyUnchanged() error = %v, want ErrArtifactChanged", err)
		}
	})
}

func mustResolve(t *testing.T, path string, data []byte) serverartifact.Resolution {
	t.Helper()
	resolved, err := serverartifact.Resolve(context.Background(), serverartifact.Reference{Path: path, SHA256: digest(data)})
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	t.Cleanup(func() { _ = resolved.Close() })
	return resolved
}

func writeFixture(t *testing.T, name string, data []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func digest(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
