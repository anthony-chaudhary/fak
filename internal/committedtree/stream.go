package committedtree

import "context"

// ForEachRegularFile visits the regular-file blobs of object in path order without
// writing a checkout. Symlinks and gitlinks are skipped, matching the files that
// Extract materializes as regular files, and each blob's bytes are exactly as
// committed. One blob is held at a time, and the extraction budget bounds the
// whole stream.
func ForEachRegularFile(repo, object string, visit func(path string, data []byte) error) error {
	ctx, cancel := operationContext(context.Background())
	defer cancel()
	listed, err := listTreeEntries(ctx, repo, object, nil, gitCommand, nil)
	if err != nil {
		return err
	}
	files := make([]archiveEntry, 0, len(listed))
	for _, entry := range listed {
		if entry.gitMode != symlinkMode {
			files = append(files, entry)
		}
	}
	return forEachBlob(ctx, repo, files, gitCommand, func(i int, data []byte) error {
		return visit(files[i].path, data)
	})
}
