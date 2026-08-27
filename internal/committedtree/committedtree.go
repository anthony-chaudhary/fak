// Package committedtree materializes committed git trees without reading a
// shared checkout's dirty worktree or index.
package committedtree

import (
	"archive/tar"
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
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
	regularPaths, untarErr := untarRegularPaths(stdout, dir)
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
	}
	if err := restoreRawBlobs(ctx, repo, object, dir, regularPaths); err != nil {
		return err
	}
	return nil
}

func untar(r io.Reader, dir string) error {
	_, err := untarRegularPaths(r, dir)
	return err
}

func untarRegularPaths(r io.Reader, dir string) ([]string, error) {
	tr := tar.NewReader(r)
	var regularPaths []string
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return regularPaths, nil
		}
		if err != nil {
			return nil, err
		}
		target, err := safeJoin(dir, hdr.Name)
		if err != nil {
			return nil, err
		}
		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return nil, err
			}
		case tar.TypeReg, tar.TypeRegA:
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return nil, err
			}
			f, err := os.OpenFile(target, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, os.FileMode(hdr.Mode)&0o777|0o600)
			if err != nil {
				return nil, err
			}
			if _, err := io.Copy(f, tr); err != nil {
				_ = f.Close()
				return nil, err
			}
			if err := f.Close(); err != nil {
				return nil, err
			}
			regularPaths = append(regularPaths, hdr.Name)
		}
	}
}

// restoreRawBlobs keeps git archive as the source of entry selection and file
// modes, but replaces regular-file payloads with bytes read directly from the
// object database. Git archive otherwise applies checkout conversions such as
// core.autocrlf on Windows, so its payload is not necessarily the Git blob.
func restoreRawBlobs(ctx context.Context, repo, object, dir string, paths []string) error {
	blobs, err := listBlobs(ctx, repo, object)
	if err != nil {
		return err
	}
	cmd := windowgate.CommandContext(ctx, "git", "-C", repo, "cat-file", "--batch")
	windowgate.ConfigureBackgroundCommand(cmd)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	var stderr strings.Builder
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("git cat-file start: %w", err)
	}
	abort := func() {
		_ = stdin.Close()
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	}
	br := bufio.NewReader(stdout)
	for _, path := range paths {
		oid, ok := blobs[path]
		if !ok {
			abort()
			return fmt.Errorf("git archive regular file %q has no blob", path)
		}
		if _, err := io.WriteString(stdin, oid+"\n"); err != nil {
			abort()
			return fmt.Errorf("git cat-file request %q: %w", path, err)
		}
		header, err := br.ReadString('\n')
		if err != nil {
			abort()
			return fmt.Errorf("git cat-file header %q: %w", path, err)
		}
		fields := strings.Fields(header)
		if len(fields) != 3 || fields[1] != "blob" {
			abort()
			return fmt.Errorf("git cat-file header %q: %q", path, strings.TrimSpace(header))
		}
		size, err := strconv.ParseInt(fields[2], 10, 64)
		if err != nil || size < 0 {
			abort()
			return fmt.Errorf("git cat-file size %q: %q", path, fields[2])
		}
		target, err := safeJoin(dir, path)
		if err != nil {
			abort()
			return err
		}
		f, err := os.OpenFile(target, os.O_WRONLY|os.O_TRUNC, 0)
		if err != nil {
			abort()
			return err
		}
		_, copyErr := io.CopyN(f, br, size)
		closeErr := f.Close()
		terminator, terminatorErr := br.ReadByte()
		if copyErr != nil {
			abort()
			return fmt.Errorf("git cat-file blob %q: %w", path, copyErr)
		}
		if closeErr != nil {
			abort()
			return closeErr
		}
		if terminatorErr != nil || terminator != '\n' {
			abort()
			return fmt.Errorf("git cat-file terminator %q: %w", path, terminatorErr)
		}
	}
	if err := stdin.Close(); err != nil {
		abort()
		return err
	}
	if err := cmd.Wait(); err != nil {
		if ctx.Err() != nil {
			return fmt.Errorf("git extraction timed out after %s: %w", defaultTimeout, ctx.Err())
		}
		return fmt.Errorf("git cat-file: %w (%s)", err, strings.TrimSpace(stderr.String()))
	}
	return nil
}

func listBlobs(ctx context.Context, repo, object string) (map[string]string, error) {
	cmd := windowgate.CommandContext(ctx, "git", "-C", repo, "ls-tree", "-r", "-z", "--full-tree", object)
	windowgate.ConfigureBackgroundCommand(cmd)
	var stderr strings.Builder
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		if ctx.Err() != nil {
			return nil, fmt.Errorf("git extraction timed out after %s: %w", defaultTimeout, ctx.Err())
		}
		return nil, fmt.Errorf("git ls-tree: %w (%s)", err, strings.TrimSpace(stderr.String()))
	}
	blobs := make(map[string]string)
	for len(out) > 0 {
		end := bytes.IndexByte(out, 0)
		if end < 0 {
			return nil, fmt.Errorf("git ls-tree: unterminated record")
		}
		record := out[:end]
		out = out[end+1:]
		parts := bytes.SplitN(record, []byte{'\t'}, 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("git ls-tree: malformed record %q", record)
		}
		fields := strings.Fields(string(parts[0]))
		if len(fields) != 3 {
			return nil, fmt.Errorf("git ls-tree: malformed metadata %q", parts[0])
		}
		if fields[1] == "blob" {
			blobs[string(parts[1])] = fields[2]
		}
	}
	return blobs, nil
}

func safeJoin(root, name string) (string, error) {
	rel := filepath.Clean(filepath.FromSlash(name))
	if filepath.IsAbs(rel) || filepath.VolumeName(rel) != "" || rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return "", fmt.Errorf("archive entry %q escapes extraction root", name)
	}
	return filepath.Join(root, rel), nil
}
