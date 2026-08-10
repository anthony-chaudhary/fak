// Package committedtree materializes committed git trees without reading a
// shared checkout's dirty worktree or index.
package committedtree

import (
	"archive/tar"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const defaultTimeout = 2 * time.Minute

// Resolve returns the full object ID for ref in repo.
func Resolve(repo, ref string) (string, error) {
	cmd := windowgate.Command("git", "-C", repo, "rev-parse", ref)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

// Extract resolves no refs: object must already identify a commit or tree that
// git archive accepts. The caller owns the returned temporary directory.
func Extract(repo, object string) (string, error) {
	dir, err := os.MkdirTemp("", "fak-committed-tree-*")
	if err != nil {
		return "", err
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer cancel()
	if err := extract(ctx, repo, object, dir); err != nil {
		_ = os.RemoveAll(dir)
		return "", err
	}
	return dir, nil
}

func extract(ctx context.Context, repo, object, dir string) error {
	cmd := windowgate.CommandContext(ctx, "git", "-C", repo, "archive", "--format=tar", object)
	windowgate.ConfigureBackgroundCommand(cmd)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	var stderr strings.Builder
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("git archive start: %w", err)
	}
	untarErr := untar(stdout, dir)
	if untarErr != nil {
		_ = cmd.Process.Kill()
	}
	_, _ = io.Copy(io.Discard, stdout)
	waitErr := cmd.Wait()
	switch {
	case ctx.Err() != nil:
		return fmt.Errorf("git archive timed out after %s: %w", defaultTimeout, ctx.Err())
	case untarErr != nil:
		return fmt.Errorf("untar archive: %w", untarErr)
	case waitErr != nil:
		return fmt.Errorf("git archive: %w (%s)", waitErr, strings.TrimSpace(stderr.String()))
	default:
		return nil
	}
}

func untar(r io.Reader, dir string) error {
	tr := tar.NewReader(r)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		target, err := safeJoin(dir, hdr.Name)
		if err != nil {
			return err
		}
		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			f, err := os.OpenFile(target, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, os.FileMode(hdr.Mode)&0o777|0o600)
			if err != nil {
				return err
			}
			if _, err := io.Copy(f, tr); err != nil {
				_ = f.Close()
				return err
			}
			if err := f.Close(); err != nil {
				return err
			}
		}
	}
}

func safeJoin(root, name string) (string, error) {
	rel := filepath.Clean(filepath.FromSlash(name))
	if filepath.IsAbs(rel) || filepath.VolumeName(rel) != "" || rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return "", fmt.Errorf("archive entry %q escapes extraction root", name)
	}
	return filepath.Join(root, rel), nil
}
