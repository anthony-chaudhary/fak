// Package committedtree materializes committed git trees without reading a
// shared checkout's dirty worktree or index.
package committedtree

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
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
// git ls-tree accepts. The caller owns the returned temporary directory.
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
	return extractWithCommand(ctx, repo, object, dir, windowgate.CommandContext)
}

type commandContextFunc func(context.Context, string, ...string) *exec.Cmd

type treeEntry struct {
	path    string
	oid     string
	mode    os.FileMode
	regular bool
}

func extractWithCommand(ctx context.Context, repo, object, dir string, command commandContextFunc) error {
	entries, err := listTree(ctx, repo, object, command)
	if err != nil {
		return err
	}
	if err := validateTreeEntries(dir, entries); err != nil {
		return err
	}
	return materializeRawBlobs(ctx, repo, dir, entries, command)
}

func listTree(ctx context.Context, repo, object string, command commandContextFunc) ([]treeEntry, error) {
	cmd := command(ctx, "git", "-C", repo, "ls-tree", "-r", "-z", "--full-tree", object)
	windowgate.ConfigureBackgroundCommand(cmd)
	var stderr strings.Builder
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		if ctx.Err() != nil {
			return nil, extractionTimeout(ctx)
		}
		return nil, fmt.Errorf("git ls-tree: %w (%s)", err, strings.TrimSpace(stderr.String()))
	}
	return parseTreeEntries(out)
}

func parseTreeEntries(out []byte) ([]treeEntry, error) {
	var entries []treeEntry
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
		mode, err := strconv.ParseUint(fields[0], 8, 32)
		if err != nil {
			return nil, fmt.Errorf("git ls-tree: malformed mode %q", fields[0])
		}
		entries = append(entries, treeEntry{
			path:    string(parts[1]),
			oid:     fields[2],
			mode:    os.FileMode(mode & 0o777),
			regular: fields[1] == "blob" && mode&0o170000 == 0o100000,
		})
	}
	return entries, nil
}

func validateTreeEntries(dir string, entries []treeEntry) error {
	for _, entry := range entries {
		if _, err := safeJoin(dir, entry.path); err != nil {
			return err
		}
	}
	return nil
}

// materializeRawBlobs consumes each regular file exactly once from one batch
// process. Tree modes choose the on-disk permissions; symlink blobs and gitlinks
// remain ignored, matching the previous archive extractor's safety boundary.
func materializeRawBlobs(ctx context.Context, repo, dir string, entries []treeEntry, command commandContextFunc) error {
	regularCount := 0
	for _, entry := range entries {
		if entry.regular {
			regularCount++
		}
	}
	if regularCount == 0 {
		return nil
	}

	cmd := command(ctx, "git", "-C", repo, "cat-file", "--batch")
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
	waited := false
	defer func() {
		if !waited {
			_ = stdin.Close()
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	}()
	commandError := func(err error) error {
		if ctx.Err() != nil {
			return extractionTimeout(ctx)
		}
		return err
	}

	br := bufio.NewReader(stdout)
	for _, entry := range entries {
		if !entry.regular {
			continue
		}
		if _, err := io.WriteString(stdin, entry.oid+"\n"); err != nil {
			return commandError(fmt.Errorf("git cat-file request %q: %w", entry.path, err))
		}
		header, err := br.ReadString('\n')
		if err != nil {
			return commandError(fmt.Errorf("git cat-file header %q: %w", entry.path, err))
		}
		fields := strings.Fields(header)
		if len(fields) != 3 || fields[1] != "blob" {
			return fmt.Errorf("git cat-file header %q: %q", entry.path, strings.TrimSpace(header))
		}
		size, err := strconv.ParseInt(fields[2], 10, 64)
		if err != nil || size < 0 {
			return fmt.Errorf("git cat-file size %q: %q", entry.path, fields[2])
		}
		target, err := safeJoin(dir, entry.path)
		if err != nil {
			return err
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		f, err := os.OpenFile(target, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, entry.mode.Perm()|0o600)
		if err != nil {
			return err
		}
		_, copyErr := io.CopyN(f, br, size)
		closeErr := f.Close()
		terminator, terminatorErr := br.ReadByte()
		if copyErr != nil {
			return commandError(fmt.Errorf("git cat-file blob %q: %w", entry.path, copyErr))
		}
		if closeErr != nil {
			return closeErr
		}
		if terminatorErr != nil || terminator != '\n' {
			return commandError(fmt.Errorf("git cat-file terminator %q: %w", entry.path, terminatorErr))
		}
	}
	if err := stdin.Close(); err != nil {
		return commandError(err)
	}
	waitErr := cmd.Wait()
	waited = true
	if waitErr != nil {
		if ctx.Err() != nil {
			return extractionTimeout(ctx)
		}
		return fmt.Errorf("git cat-file: %w (%s)", waitErr, strings.TrimSpace(stderr.String()))
	}
	return nil
}

func extractionTimeout(ctx context.Context) error {
	return fmt.Errorf("git extraction timed out after %s: %w", defaultTimeout, ctx.Err())
}

func safeJoin(root, name string) (string, error) {
	rel := filepath.Clean(filepath.FromSlash(name))
	if filepath.IsAbs(rel) || filepath.VolumeName(rel) != "" || rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return "", fmt.Errorf("archive entry %q escapes extraction root", name)
	}
	return filepath.Join(root, rel), nil
}
