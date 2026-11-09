package service

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/deploykit/swap"
)

// stageDocument publishes text as target atomically: it writes a temp file
// beside the target and activates it through the L3 swap facade, so a
// concurrent reader sees either the intact old or the intact new document,
// never a partial one. The temp file is consumed by the swap on success and
// removed on failure.
func stageDocument(target, text string) error {
	dir := filepath.Dir(target)
	tmp, err := os.CreateTemp(dir, "."+filepath.Base(target)+".tmp-*")
	if err != nil {
		return fmt.Errorf("stage document %s: %w", target, err)
	}
	name := tmp.Name()
	defer os.Remove(name) // no-op once the swap has consumed the file
	if _, err := tmp.WriteString(text); err != nil {
		tmp.Close()
		return fmt.Errorf("stage document %s: %w", target, err)
	}
	if err := tmp.Chmod(0o644); err != nil {
		tmp.Close()
		return fmt.Errorf("stage document %s: %w", target, err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("stage document %s: %w", target, err)
	}
	if err := swap.Swap(name, target); err != nil {
		return fmt.Errorf("activate document %s: %w", target, err)
	}
	return nil
}

// planStagedExecutable derives the content-addressed path for source's bytes:
// <dir>/fak-<sha256>.exe. Staging to a digest-named target makes the published
// executable immutable-safe — a republish of identical bytes lands on the same
// path, and any different content gets a different path.
func planStagedExecutable(dir, source string) (string, error) {
	digest, err := fileSHA256(source)
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "fak-"+hex.EncodeToString(digest[:])+".exe"), nil
}

// stageExecutable copies source to its content-addressed target in dir. When
// the target already exists with matching bytes it is kept as-is (only its
// permissions are re-applied); a mismatched existing target is refused rather
// than overwritten.
func stageExecutable(source, target string) error {
	if source == target {
		return fmt.Errorf("refusing to stage executable onto itself: %s", target)
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return fmt.Errorf("create staged executable directory: %w", err)
	}
	if targetDigest, err := fileSHA256(target); err == nil {
		sourceDigest, sourceErr := fileSHA256(source)
		if sourceErr != nil {
			return sourceErr
		}
		if targetDigest != sourceDigest {
			return fmt.Errorf("refusing to overwrite non-matching immutable staged executable: %s", target)
		}
		return os.Chmod(target, 0o755)
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	in, err := os.Open(source)
	if err != nil {
		return err
	}
	defer in.Close()
	tmp, err := os.CreateTemp(filepath.Dir(target), ".fak-service-*.exe")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name) // no-op once moveNoReplace has consumed the file
	if _, err = io.Copy(tmp, in); err == nil {
		err = tmp.Sync()
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return fmt.Errorf("copy staged executable: %w", err)
	}
	stagedDigest, err := fileSHA256(name)
	if err != nil {
		return err
	}
	contentTarget := filepath.Join(filepath.Dir(target), "fak-"+hex.EncodeToString(stagedDigest[:])+".exe")
	if !strings.EqualFold(filepath.Clean(target), filepath.Clean(contentTarget)) {
		return fmt.Errorf("staged executable changed while copying: content target %s does not match planned target %s", contentTarget, target)
	}
	if err := os.Chmod(name, 0o755); err != nil {
		return err
	}
	return moveNoReplace(name, target)
}

// moveNoReplace moves src onto dst, refusing to overwrite an existing dst. The
// Lstat guard plus the caller's content-addressed digest check make the move
// immutability-safe: a concurrent publisher can make this move fail, but it
// cannot land different bytes over a published target.
func moveNoReplace(src, dst string) error {
	if _, err := os.Lstat(dst); err == nil {
		return fmt.Errorf("refusing to replace existing staged file: %s", dst)
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return os.Rename(src, dst)
}

// fileSHA256 digests a whole file.
func fileSHA256(path string) ([sha256.Size]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return [sha256.Size]byte{}, err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return [sha256.Size]byte{}, err
	}
	var digest [sha256.Size]byte
	copy(digest[:], h.Sum(nil))
	return digest, nil
}
