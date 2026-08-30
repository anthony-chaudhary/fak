// Package committedtree materializes committed git trees without reading a
// shared checkout's dirty worktree or index.
package committedtree

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	defaultTimeout    = 2 * time.Minute
	gitPipeBufferSize = 64 * 1024
	maxGitRecordSize  = gitPipeBufferSize
	maxGitStderrBytes = 64 * 1024
	// 256 SHA-256 object requests remain well below the 64 KiB pipe buffer.
	gitBatchWindow     = 256
	regularFileMode    = "100644"
	executableFileMode = "100755"
	symlinkMode        = "120000"
	gitlinkMode        = "160000"
)

type commandFactory func(context.Context, string, ...string) *exec.Cmd

type treeEntry struct {
	mode os.FileMode
	oid  string
	path string
}

// ownedGitCommand binds one Git subprocess and its pipes to a cancellation
// watcher. On Windows the job owns the launcher and every descendant; closing
// stdout as well as the job makes a blocked reader return even when a child
// inherited the launcher's pipe handle.
type ownedGitCommand struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser
	job    *windowgate.JobObject

	started   bool
	waited    bool
	abortOnce sync.Once
	stop      chan struct{}
	done      chan struct{}
}

// Resolve returns the full object ID for ref in repo.
func Resolve(repo, ref string) (string, error) {
	cmd := gitCommand(context.Background(), repo, "rev-parse", ref)
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

// Extract resolves no refs: object must already identify a commit or tree. The
// caller owns the returned temporary directory.
func Extract(repo, object string) (string, error) {
	return extractTemp(repo, object, "")
}

func extractTemp(repo, object, tempParent string) (string, error) {
	dir, err := os.MkdirTemp(tempParent, "fak-committed-tree-*")
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

func gitCommand(ctx context.Context, repo string, args ...string) *exec.Cmd {
	gitArgs := make([]string, 0, len(args)+2)
	gitArgs = append(gitArgs, "-C", repo)
	gitArgs = append(gitArgs, args...)
	return windowgate.CommandContext(ctx, "git", gitArgs...)
}

func (p *ownedGitCommand) start(ctx context.Context) error {
	job, err := windowgate.StartInNewJob(p.cmd)
	if err != nil {
		return err
	}
	p.job = job
	p.started = true
	p.stop = make(chan struct{})
	p.done = make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			p.abortNow()
		case <-p.stop:
		}
		close(p.done)
	}()
	return nil
}

func (p *ownedGitCommand) abortNow() {
	if p == nil {
		return
	}
	p.abortOnce.Do(func() {
		// Closing both ends owned by the parent is what unblocks an in-flight
		// WriteString, ReadSlice, or CopyN. Closing the Windows job then reaps
		// launcher descendants that still hold their inherited pipe handles.
		if p.stdin != nil {
			_ = p.stdin.Close()
		}
		if p.stdout != nil {
			_ = p.stdout.Close()
		}
		_ = p.job.Close()
		if p.cmd != nil && p.cmd.Process != nil {
			_ = p.cmd.Process.Kill()
		}
	})
}

func (p *ownedGitCommand) stopWatcher() {
	if p == nil || p.stop == nil {
		return
	}
	select {
	case <-p.done:
		return
	default:
	}
	close(p.stop)
	<-p.done
}

func (p *ownedGitCommand) wait() error {
	if p == nil || !p.started || p.waited {
		return nil
	}
	err := p.cmd.Wait()
	p.waited = true
	p.stopWatcher()
	_ = p.job.Close()
	return err
}

func (p *ownedGitCommand) finish(failed bool) {
	if p == nil {
		return
	}
	if failed {
		p.abortNow()
	}
	if p.started && !p.waited {
		_ = p.cmd.Wait()
		p.waited = true
	}
	p.stopWatcher()
	_ = p.job.Close()
}

func extract(ctx context.Context, repo, object, dir string) error {
	return extractWithGit(ctx, repo, object, dir, gitCommand)
}

// extractWithGit streams one tree listing into one long-lived cat-file batch.
// It keeps only a bounded request window in memory, removing a pipe round trip
// per blob while retaining the raw-object, path-safety, and cleanup contracts.
func extractWithGit(ctx context.Context, repo, object, dir string, command commandFactory) (retErr error) {
	lsCmd := command(ctx, repo, "ls-tree", "-r", "-z", "--full-tree", object)
	lsOut, err := lsCmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("git ls-tree stdout: %w", err)
	}
	var lsStderr cappedBuffer
	lsCmd.Stderr = &lsStderr
	lsProcess := &ownedGitCommand{cmd: lsCmd, stdout: lsOut}
	if err := lsProcess.start(ctx); err != nil {
		_ = lsOut.Close()
		return commandError(ctx, "git ls-tree start", err)
	}
	var catProcess *ownedGitCommand
	defer func() {
		failed := retErr != nil
		if catProcess != nil {
			catProcess.finish(failed)
		}
		lsProcess.finish(failed)
	}()

	catCmd := command(ctx, repo, "cat-file", "--batch")
	catIn, err := catCmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("git cat-file stdin: %w", err)
	}
	catProcess = &ownedGitCommand{cmd: catCmd, stdin: catIn}
	catOut, err := catCmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("git cat-file stdout: %w", err)
	}
	catProcess.stdout = catOut
	var catStderr cappedBuffer
	catCmd.Stderr = &catStderr
	if err := catProcess.start(ctx); err != nil {
		return commandError(ctx, "git cat-file start", err)
	}

	treeReader := bufio.NewReaderSize(lsOut, gitPipeBufferSize)
	objectReader := bufio.NewReaderSize(catOut, gitPipeBufferSize)
	pending := make([]treeEntry, 0, gitBatchWindow)
	flushPending := func() error {
		for _, entry := range pending {
			if err := writeBlobFile(objectReader, dir, entry); err != nil {
				return commandError(ctx, "read git cat-file", err)
			}
		}
		pending = pending[:0]
		return nil
	}
	for {
		record, readErr := treeReader.ReadSlice(0)
		if readErr == io.EOF && len(record) == 0 {
			break
		}
		if readErr != nil {
			if errors.Is(readErr, bufio.ErrBufferFull) {
				return fmt.Errorf("git ls-tree record exceeds %d bytes", maxGitRecordSize)
			}
			return commandError(ctx, "read git ls-tree", readErr)
		}

		entry, include, err := parseTreeEntry(record[:len(record)-1])
		if err != nil {
			return err
		}
		if !include {
			continue
		}
		if _, err := safeJoin(dir, entry.path); err != nil {
			return err
		}
		if _, err := io.WriteString(catIn, entry.oid+"\n"); err != nil {
			return commandError(ctx, "write git cat-file request", err)
		}
		pending = append(pending, entry)
		if len(pending) == cap(pending) {
			if err := flushPending(); err != nil {
				return err
			}
		}
	}
	if err := flushPending(); err != nil {
		return err
	}

	lsWaitErr := lsProcess.wait()
	if err := catIn.Close(); err != nil {
		return commandError(ctx, "close git cat-file input", err)
	}
	catWaitErr := catProcess.wait()

	if ctx.Err() != nil {
		return extractionStopError(ctx)
	}
	if lsWaitErr != nil {
		return fmt.Errorf("git ls-tree: %w (%s)", lsWaitErr, lsStderr.String())
	}
	if catWaitErr != nil {
		return fmt.Errorf("git cat-file: %w (%s)", catWaitErr, catStderr.String())
	}
	return nil
}

func parseTreeEntry(record []byte) (treeEntry, bool, error) {
	tab := bytes.IndexByte(record, '\t')
	if tab < 0 || tab == len(record)-1 {
		return treeEntry{}, false, fmt.Errorf("malformed git ls-tree record")
	}
	header := strings.Fields(string(record[:tab]))
	if len(header) != 3 {
		return treeEntry{}, false, fmt.Errorf("malformed git ls-tree header %q", record[:tab])
	}
	mode, objectType, oid := header[0], header[1], header[2]
	path := string(record[tab+1:])
	switch mode {
	case symlinkMode, gitlinkMode:
		return treeEntry{}, false, nil
	case regularFileMode, executableFileMode:
		if objectType != "blob" || !validObjectID(oid) {
			return treeEntry{}, false, fmt.Errorf("invalid git blob entry for %q", path)
		}
		perm := os.FileMode(0o644)
		if mode == executableFileMode {
			perm = 0o755
		}
		return treeEntry{mode: perm, oid: oid, path: path}, true, nil
	default:
		return treeEntry{}, false, fmt.Errorf("unsupported git tree mode %q for %q", mode, path)
	}
}

func writeBlobFile(r *bufio.Reader, root string, entry treeEntry) error {
	target, err := safeJoin(root, entry.path)
	if err != nil {
		return err
	}
	header, err := r.ReadSlice('\n')
	if err != nil {
		if errors.Is(err, bufio.ErrBufferFull) {
			return fmt.Errorf("git cat-file header exceeds %d bytes", maxGitRecordSize)
		}
		return err
	}
	fields := strings.Fields(string(header))
	if len(fields) == 2 && fields[1] == "missing" {
		return fmt.Errorf("git object %s is missing", entry.oid)
	}
	if len(fields) != 3 || fields[0] != entry.oid || fields[1] != "blob" {
		return fmt.Errorf("unexpected git cat-file header %q for %s", strings.TrimSpace(string(header)), entry.oid)
	}
	size, err := strconv.ParseInt(fields[2], 10, 64)
	if err != nil || size < 0 {
		return fmt.Errorf("invalid git blob size %q for %s", fields[2], entry.oid)
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return err
	}
	// Match git archive's regular-file modes while ensuring the owner can
	// complete the write under the caller's umask.
	f, err := os.OpenFile(target, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, entry.mode|0o600)
	if err != nil {
		return err
	}
	_, copyErr := io.CopyN(f, r, size)
	closeErr := f.Close()
	if copyErr != nil {
		return copyErr
	}
	if closeErr != nil {
		return closeErr
	}
	terminator, err := r.ReadByte()
	if err != nil {
		return err
	}
	if terminator != '\n' {
		return fmt.Errorf("git cat-file blob %s has invalid terminator", entry.oid)
	}
	return nil
}

func safeJoin(root, name string) (string, error) {
	if name == "" {
		return "", fmt.Errorf("git tree entry has an empty path")
	}
	rel := filepath.Clean(filepath.FromSlash(name))
	if rel == "." || filepath.IsAbs(rel) || filepath.VolumeName(rel) != "" || rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return "", fmt.Errorf("git tree entry %q escapes extraction root", name)
	}
	return filepath.Join(root, rel), nil
}

func validObjectID(oid string) bool {
	if len(oid) != 40 && len(oid) != 64 {
		return false
	}
	for _, c := range oid {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			return false
		}
	}
	return true
}

func commandError(ctx context.Context, stage string, err error) error {
	if ctx.Err() != nil {
		return extractionStopError(ctx)
	}
	return fmt.Errorf("%s: %w", stage, err)
}

func extractionStopError(ctx context.Context) error {
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return fmt.Errorf("committed tree extraction timed out after %s: %w", defaultTimeout, ctx.Err())
	}
	return fmt.Errorf("committed tree extraction canceled: %w", ctx.Err())
}

// cappedBuffer keeps subprocess diagnostics useful without making stderr an
// unbounded memory side channel during a bounded extraction.
type cappedBuffer struct {
	b []byte
}

func (b *cappedBuffer) Write(p []byte) (int, error) {
	n := len(p)
	remaining := maxGitStderrBytes - len(b.b)
	if remaining > 0 {
		if len(p) > remaining {
			p = p[:remaining]
		}
		b.b = append(b.b, p...)
	}
	return n, nil
}

func (b *cappedBuffer) String() string {
	return strings.TrimSpace(string(b.b))
}
