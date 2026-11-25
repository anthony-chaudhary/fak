package committedtree

import (
	"context"
	"os"
	"path/filepath"
)

func extractPaths(ctx context.Context, repo, object, dir string, paths []string) error {
	return extractWithGitPaths(ctx, repo, object, dir, paths, gitCommand)
}

// extractWithGit materializes the whole tree of object into dir.
func extractWithGit(parentCtx context.Context, repo, object, dir string, command commandFactory) error {
	return extractWithGitPaths(parentCtx, repo, object, dir, nil, command)
}

// extractWithGitPaths writes the regular files under paths into dir. Symlinks
// and gitlinks are not materialized, and blob bytes are written as committed.
// Blobs are pipelined through one cat-file batch, so the cost is independent of
// per-object round trips.
func extractWithGitPaths(parentCtx context.Context, repo, object, dir string, paths []string, command commandFactory) error {
	entries, err := listTreeEntries(parentCtx, repo, object, paths, command, func(entry archiveEntry) error {
		if entry.gitMode == symlinkMode {
			return nil
		}
		_, err := safeJoin(dir, entry.path)
		return err
	})
	if err != nil {
		return err
	}
	files := make([]archiveEntry, 0, len(entries))
	for _, entry := range entries {
		if entry.gitMode != symlinkMode {
			files = append(files, entry)
		}
	}
	return forEachBlob(parentCtx, repo, files, command, func(i int, data []byte) error {
		return writeBlobFile(dir, files[i], data)
	})
}

func writeBlobFile(root string, entry archiveEntry, data []byte) error {
	target, err := safeJoin(root, entry.path)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return err
	}
	// Match git archive's regular-file modes while ensuring the owner can
	// complete the write under the caller's umask.
	perm := os.FileMode(0o644)
	if entry.gitMode == executableFileMode {
		perm = 0o755
	}
	f, err := os.OpenFile(target, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, perm|0o600)
	if err != nil {
		return err
	}
	_, writeErr := f.Write(data)
	closeErr := f.Close()
	if writeErr != nil {
		return writeErr
	}
	return closeErr
}
