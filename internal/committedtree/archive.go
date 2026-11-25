package committedtree

import (
	"archive/tar"
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
	"time"
)

// WriteTar streams the committed tree of object to w as a tar archive.
//
// Regular files and symlinks are written with the blob bytes exactly as
// committed. Gitlinks are skipped. Export-ignored paths are included, which is
// the one deliberate difference from git archive: gates that snapshot a commit
// must see the full committed tree, while the published module keeps git
// archive's stripped tree. No text or eol conversion runs, so the bytes never
// depend on the host's core.autocrlf, core.eol, or .gitattributes. Entries are
// sorted by path and carry zero ownership and a fixed modification time, so the
// bytes are deterministic.
//
// The package extraction budget bounds the call unless ctx already has a
// deadline, which then bounds it instead. BudgetError unwraps to
// context.DeadlineExceeded.
func WriteTar(ctx context.Context, repo, object string, w io.Writer) error {
	return writeTar(ctx, repo, object, nil, w, gitCommand)
}

// WriteTarPaths is WriteTar limited to the given path prefixes, each naming a
// file or a directory. Only the selected objects are read. At least one prefix
// is required.
func WriteTarPaths(ctx context.Context, repo, object string, w io.Writer, paths ...string) error {
	cleaned, err := normalizePathPrefixes(paths)
	if err != nil {
		return err
	}
	return writeTar(ctx, repo, object, cleaned, w, gitCommand)
}

// Tar entries carry the committed tree's zero time, so archives built from the
// same tree are byte-identical across machines and runs.
var tarEpoch = time.Unix(0, 0)

// Permission bits match git archive's choice for tree modes: group-writable
// files, and executables that are also group-executable.
const (
	regularTarMode    = 0o664
	executableTarMode = 0o775
	symlinkTarMode    = 0o777
)

type archiveEntry struct {
	gitMode string
	oid     string
	path    string
}

func writeTar(parentCtx context.Context, repo, object string, paths []string, w io.Writer, command commandFactory) error {
	ctx, cancel := operationContext(parentCtx)
	defer cancel()

	entries, err := listTreeEntries(ctx, repo, object, paths, command, nil)
	if err != nil {
		return err
	}
	tw := tar.NewWriter(w)
	err = forEachBlob(ctx, repo, entries, command, func(i int, data []byte) error {
		return writeTarEntry(tw, entries[i], data)
	})
	if err != nil {
		return err
	}
	return tw.Close()
}

func writeTarEntry(tw *tar.Writer, entry archiveEntry, data []byte) error {
	hdr := &tar.Header{Name: entry.path, ModTime: tarEpoch}
	if entry.gitMode == symlinkMode {
		hdr.Typeflag = tar.TypeSymlink
		hdr.Linkname = string(data)
		hdr.Mode = symlinkTarMode
		return tw.WriteHeader(hdr)
	}
	hdr.Typeflag = tar.TypeReg
	hdr.Mode = regularTarMode
	if entry.gitMode == executableFileMode {
		hdr.Mode = executableTarMode
	}
	hdr.Size = int64(len(data))
	if err := tw.WriteHeader(hdr); err != nil {
		return err
	}
	_, err := tw.Write(data)
	return err
}

// listTreeEntries lists the regular files and symlinks of object under paths
// (all of object when paths is empty), sorted by path. check, when non-nil, sees
// each entry while the listing is still streaming, so a rejected entry stops
// the git process before the rest of the tree is listed.
func listTreeEntries(parentCtx context.Context, repo, object string, paths []string, command commandFactory, check func(archiveEntry) error) ([]archiveEntry, error) {
	ctx, cancel := context.WithCancel(parentCtx)
	defer cancel()

	args := []string{"ls-tree", "-r", "-z", "--full-tree", object}
	if len(paths) > 0 {
		args = append(args, "--")
		args = append(args, paths...)
	}
	cmd := command(ctx, repo, args...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("git ls-tree stdout: %w", err)
	}
	var stderr cappedBuffer
	cmd.Stderr = &stderr
	proc := &ownedGitCommand{cmd: cmd, stdout: stdout}
	if err := proc.start(ctx); err != nil {
		_ = stdout.Close()
		return nil, commandError(ctx, "git ls-tree start", err)
	}

	entries, readErr := readTreeListing(ctx, bufio.NewReaderSize(stdout, gitPipeBufferSize), check)
	if readErr != nil {
		if ctx.Err() != nil {
			readErr = extractionStopError(ctx)
		}
		proc.finish(true)
		return nil, readErr
	}
	waitErr := proc.wait()
	if ctx.Err() != nil {
		return nil, extractionStopError(ctx)
	}
	if waitErr != nil {
		return nil, fmt.Errorf("git ls-tree: %w (%s)", waitErr, stderr.String())
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].path < entries[j].path })
	return entries, nil
}

func readTreeListing(ctx context.Context, r *bufio.Reader, check func(archiveEntry) error) ([]archiveEntry, error) {
	var entries []archiveEntry
	for {
		record, err := r.ReadSlice(0)
		if err == io.EOF && len(record) == 0 {
			return entries, nil
		}
		if err != nil {
			if errors.Is(err, bufio.ErrBufferFull) {
				return nil, fmt.Errorf("git ls-tree record exceeds %d bytes", maxGitRecordSize)
			}
			return nil, commandError(ctx, "read git ls-tree", err)
		}
		entry, include, err := parseArchiveRecord(record[:len(record)-1])
		if err != nil {
			return nil, err
		}
		if !include {
			continue
		}
		if check != nil {
			if err := check(entry); err != nil {
				return nil, err
			}
		}
		entries = append(entries, entry)
	}
}

func parseArchiveRecord(record []byte) (archiveEntry, bool, error) {
	tab := bytes.IndexByte(record, '\t')
	if tab < 0 || tab == len(record)-1 {
		return archiveEntry{}, false, fmt.Errorf("malformed git ls-tree record")
	}
	header := strings.Fields(string(record[:tab]))
	if len(header) != 3 {
		return archiveEntry{}, false, fmt.Errorf("malformed git ls-tree header %q", record[:tab])
	}
	mode, objectType, oid := header[0], header[1], header[2]
	path := string(record[tab+1:])
	switch mode {
	case gitlinkMode:
		return archiveEntry{}, false, nil
	case regularFileMode, executableFileMode, symlinkMode:
		if objectType != "blob" || !validObjectID(oid) {
			return archiveEntry{}, false, fmt.Errorf("invalid git blob entry for %q", path)
		}
		return archiveEntry{gitMode: mode, oid: oid, path: path}, true, nil
	default:
		return archiveEntry{}, false, fmt.Errorf("unsupported git tree mode %q for %q", mode, path)
	}
}

// forEachBlob streams the contents of entries through one cat-file --batch
// process and calls visit in entry order. A writer goroutine feeds object IDs
// while this goroutine reads responses, so neither pipe can stall the other.
// Cancellation kills the process and closes its pipes, which unblocks both.
func forEachBlob(parentCtx context.Context, repo string, entries []archiveEntry, command commandFactory, visit func(int, []byte) error) error {
	if len(entries) == 0 {
		return nil
	}
	ctx, cancel := context.WithCancel(parentCtx)
	defer cancel()

	cmd := command(ctx, repo, "cat-file", "--batch")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("git cat-file stdin: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("git cat-file stdout: %w", err)
	}
	var stderr cappedBuffer
	cmd.Stderr = &stderr
	proc := &ownedGitCommand{cmd: cmd, stdin: stdin, stdout: stdout}
	if err := proc.start(ctx); err != nil {
		return commandError(ctx, "git cat-file start", err)
	}

	writeDone := make(chan error, 1)
	go func() {
		w := bufio.NewWriterSize(stdin, gitPipeBufferSize)
		var writeErr error
		for _, entry := range entries {
			if _, writeErr = w.WriteString(entry.oid + "\n"); writeErr != nil {
				break
			}
		}
		if writeErr == nil {
			writeErr = w.Flush()
		}
		if closeErr := stdin.Close(); writeErr == nil && closeErr != nil {
			writeErr = closeErr
		}
		writeDone <- writeErr
	}()

	reader := bufio.NewReaderSize(stdout, gitPipeBufferSize)
	readErr := func() error {
		for i, entry := range entries {
			data, err := readBatchObject(reader, entry.oid)
			if err != nil {
				return err
			}
			if err := visit(i, data); err != nil {
				return err
			}
		}
		return nil
	}()
	if readErr != nil {
		if ctx.Err() != nil {
			readErr = extractionStopError(ctx)
		}
		cancel()
		proc.finish(true)
		<-writeDone
		return readErr
	}

	waitErr := proc.wait()
	writeErr := <-writeDone
	if ctx.Err() != nil {
		return extractionStopError(ctx)
	}
	if waitErr != nil {
		return fmt.Errorf("git cat-file: %w (%s)", waitErr, stderr.String())
	}
	if writeErr != nil {
		return commandError(ctx, "write git cat-file request", writeErr)
	}
	return nil
}

// readBatchObject reads one "<oid> blob <size>" response from cat-file --batch.
func readBatchObject(r *bufio.Reader, oid string) ([]byte, error) {
	header, err := r.ReadSlice('\n')
	if err != nil {
		if errors.Is(err, bufio.ErrBufferFull) {
			return nil, fmt.Errorf("git cat-file header exceeds %d bytes", maxGitRecordSize)
		}
		return nil, err
	}
	fields := strings.Fields(string(header))
	if len(fields) == 2 && fields[1] == "missing" {
		return nil, fmt.Errorf("git object %s is missing", oid)
	}
	if len(fields) != 3 || fields[0] != oid || fields[1] != "blob" {
		return nil, fmt.Errorf("unexpected git cat-file header %q for %s", strings.TrimSpace(string(header)), oid)
	}
	size, err := strconv.ParseInt(fields[2], 10, 64)
	if err != nil || size < 0 {
		return nil, fmt.Errorf("invalid git blob size %q for %s", fields[2], oid)
	}
	data := make([]byte, size)
	if _, err := io.ReadFull(r, data); err != nil {
		return nil, err
	}
	terminator, err := r.ReadByte()
	if err != nil {
		return nil, err
	}
	if terminator != '\n' {
		return nil, fmt.Errorf("git cat-file blob %s has invalid terminator", oid)
	}
	return data, nil
}
