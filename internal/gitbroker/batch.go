package gitbroker

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

var (
	// batchDeadline bounds one round trip to a warm process. A warm
	// `cat-file --batch` answers in well under a millisecond, so this is not a
	// performance knob — it is the tripwire that turns "the pool wedged" into
	// "the caller spawned", and it is generous because a false trip costs the
	// pool permanently. A var so tests can shrink it.
	batchDeadline = 5 * time.Second

	// batchReapGrace is how long Close waits for a child to leave on its own
	// after its stdin is closed, before killing it.
	batchReapGrace = 2 * time.Second
)

// errPoolUnavailable is the internal "no answer from the pool" signal. It never
// reaches a caller of Run: every path that produces it falls back to spawning,
// which is the point.
var errPoolUnavailable = errors.New("gitbroker: batch pool unavailable")

// batchProc is one long-lived `git cat-file --batch` (payload) or
// `--batch-check` (header only) process for one repository.
//
// ONE PROCESS, ONE MUTEX, ONE OUTSTANDING QUERY. The batch protocol is a
// request/response stream with no request IDs, so the Nth reply belongs to the
// Nth request and to nothing else. Serializing is not a simplification, it is
// the protocol: two concurrent callers on one stream would each read the
// other's object, and that is the one failure mode this package must not have.
//
// DEATH IS PERMANENT. Any transport-level failure — a short read, a header that
// does not parse, a missing record terminator, a wedged round trip — leaves the
// stream at an unknown offset, and an unknown offset means the next answer
// could be the previous object. So the process is killed and the pool stays
// dead for the life of the Exec; every later read spawns. A pool that might be
// desynchronized is worse than no pool at all.
type batchProc struct {
	dir     string
	payload bool

	mu   sync.Mutex
	cmd  *exec.Cmd
	in   io.WriteCloser
	out  *bufio.Reader
	dead bool

	// killedOnClose records that close had to fall back to the kill backstop
	// because the child did not leave on stdin EOF. Only a test reads it, but
	// the distinction is the whole reaping argument: EOF is what reaps a pool
	// whose parent died without calling anything, and if the child stops
	// honouring it, orphans return and only this flag would say so.
	killedOnClose bool
}

// query asks the warm process about rev, starting it on first use.
//
// It returns ErrNotFound when git says the key resolves to nothing — a real
// answer that leaves the stream in sync, so the pool survives it. Every other
// error means the pool failed and the caller should spawn.
func (p *batchProc) query(ctx context.Context, rev string) (Info, []byte, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.dead {
		return Info{}, nil, errPoolUnavailable
	}
	if p.cmd == nil {
		if err := p.start(); err != nil {
			// Could not start git at all: never try again on this Exec, and let
			// the caller spawn (which will fail the same way, loudly, in the
			// caller's own error vocabulary rather than this package's).
			p.dead = true
			return Info{}, nil, errPoolUnavailable
		}
	}

	// Snapshot the pipes so the round-trip goroutine never touches p's fields:
	// a timeout kills the process while that goroutine may still be blocked in
	// a read, and it must be able to unwind without racing the reaper.
	in, out, payload := p.in, p.out, p.payload

	type reply struct {
		info Info
		data []byte
		err  error
	}
	ch := make(chan reply, 1)
	go func() {
		info, data, err := batchRoundTrip(in, out, rev, payload)
		ch <- reply{info, data, err}
	}()

	timer := time.NewTimer(batchDeadline)
	defer timer.Stop()

	var cancelled <-chan struct{}
	if ctx != nil {
		cancelled = ctx.Done()
	}

	select {
	case r := <-ch:
		if r.err != nil && !errors.Is(r.err, ErrNotFound) {
			p.kill()
			return Info{}, nil, errPoolUnavailable
		}
		return r.info, r.data, r.err
	case <-timer.C:
		p.kill()
		return Info{}, nil, errPoolUnavailable
	case <-cancelled:
		// The stream is mid-record with no way to know where; the process
		// cannot be trusted again even though the caller, not the pool, is the
		// one that gave up.
		p.kill()
		return Info{}, nil, errPoolUnavailable
	}
}

// start launches the warm process. Deliberately exec.Command and not
// exec.CommandContext: the process's lifetime is the pool's, not any one
// caller's, and binding it to the first caller's context would kill the pool
// the moment that caller finished.
func (p *batchProc) start() error {
	mode := "--batch-check"
	if p.payload {
		mode = "--batch"
	}
	cmd := exec.Command("git", "-C", p.dir, "cat-file", mode)
	// The pool exists to remove spawns from background automation on a Windows
	// fleet; the one spawn it does make must still not flash a console.
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Stderr = io.Discard

	in, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		_ = in.Close()
		return err
	}
	if err := cmd.Start(); err != nil {
		_ = in.Close()
		_ = out.Close()
		return err
	}
	gitProcesses.Add(1)
	p.cmd, p.in, p.out = cmd, in, bufio.NewReader(out)
	return nil
}

// kill tears the process down and marks the pool permanently dead. Callers hold
// p.mu.
func (p *batchProc) kill() {
	p.dead = true
	cmd, in := p.cmd, p.in
	p.cmd, p.in, p.out = nil, nil, nil
	if cmd == nil || cmd.Process == nil {
		return
	}
	// Close stdin HERE rather than leaving it to the Wait below, because on
	// Windows killing the process this package started does not necessarily
	// reap the git that is actually serving the stream. The `git` every
	// non-Git-Bash shell resolves first on PATH (`C:\Program Files\Git\cmd\
	// git.exe`, ~46 KB) is a launcher that re-execs the real `git.exe` (~4 MB,
	// under mingw64\bin) as a CHILD holding the inherited pipes and the
	// repository as its working directory. TerminateProcess reaches the
	// launcher alone, so the real `cat-file --batch` outlives the kill —
	// measured still alive more than two seconds after it, pinning the repo
	// directory. Stdin EOF is the one signal that reaches the process actually
	// reading the pipe, whichever one that is, and it is the same mechanism
	// close() and the parent-exit path already rely on.
	if in != nil {
		_ = in.Close()
	}
	_ = cmd.Process.Kill()
	// Reap the child so it does not linger as a zombie. Wait may race a
	// still-blocked read of the stdout pipe and lose buffered bytes; that is
	// fine, this process's answers are being discarded by construction.
	go func() { _ = cmd.Wait() }()
}

// close reaps the process the way it is meant to go: by closing its stdin.
// `git cat-file --batch` exits on EOF, which is also exactly what happens
// without any help when the PARENT process exits and the OS closes the write
// end of the pipe. The kill is the backstop for a child that ignores it.
func (p *batchProc) close() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.dead = true
	cmd, in := p.cmd, p.in
	p.cmd, p.in, p.out = nil, nil, nil
	if cmd == nil {
		return nil
	}
	if in != nil {
		_ = in.Close()
	}

	done := make(chan struct{})
	go func() { _ = cmd.Wait(); close(done) }()
	select {
	case <-done:
		return nil
	case <-time.After(batchReapGrace):
		p.killedOnClose = true
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		<-done
		return nil
	}
}

// batchRoundTrip writes one key and reads exactly one record back.
//
//	--batch-check   <oid> SP <type> SP <size> LF
//	--batch         <oid> SP <type> SP <size> LF <payload> LF
//
// The trailing LF after the payload is part of the record, not part of the
// object, and it MUST be consumed: leaving it would shift every later header by
// one byte.
func batchRoundTrip(in io.Writer, out *bufio.Reader, rev string, payload bool) (Info, []byte, error) {
	if _, err := io.WriteString(in, rev+"\n"); err != nil {
		return Info{}, nil, err
	}
	header, err := out.ReadString('\n')
	if err != nil {
		return Info{}, nil, err
	}
	info, err := parseBatchHeader(header)
	if err != nil {
		// A missing/ambiguous key is answered with the header alone, so the
		// stream is still aligned and the pool survives. Any other header
		// problem is a desync and the caller kills the pool.
		return Info{}, nil, err
	}
	if !payload {
		return info, nil, nil
	}
	data := make([]byte, info.Size)
	if _, err := io.ReadFull(out, data); err != nil {
		return Info{}, nil, err
	}
	if b, err := out.ReadByte(); err != nil {
		return Info{}, nil, err
	} else if b != '\n' {
		return Info{}, nil, errors.New("gitbroker: cat-file record not terminated")
	}
	return info, data, nil
}

// parseBatchHeader decodes one record header. A two-field header is git saying
// the key did not resolve (`<key> missing`, `<key> ambiguous`), which is
// ErrNotFound and not a transport failure.
func parseBatchHeader(header string) (Info, error) {
	fields := strings.Fields(header)
	if len(fields) == 2 {
		return Info{}, fmt.Errorf("%w: %s", ErrNotFound, strings.TrimSpace(header))
	}
	if len(fields) != 3 {
		return Info{}, fmt.Errorf("gitbroker: unparseable cat-file header %q", strings.TrimSpace(header))
	}
	size, err := strconv.ParseInt(fields[2], 10, 64)
	if err != nil || size < 0 {
		return Info{}, fmt.Errorf("gitbroker: unparseable cat-file size in %q", strings.TrimSpace(header))
	}
	return Info{OID: fields[0], Type: fields[1], Size: size}, nil
}

// parseBatchReply decodes the whole output of a one-shot `cat-file --batch`
// spawn — the fallback path. It shares parseBatchHeader with the warm path on
// purpose: the fallback must not be able to disagree with the pool about what a
// record means, only about how it was obtained.
func parseBatchReply(b []byte, payload bool) (Info, []byte, error) {
	nl := bytes.IndexByte(b, '\n')
	if nl < 0 {
		return Info{}, nil, errors.New("gitbroker: truncated cat-file record (no header)")
	}
	info, err := parseBatchHeader(string(b[:nl]))
	if err != nil {
		return Info{}, nil, err
	}
	if !payload {
		return info, nil, nil
	}
	rest := b[nl+1:]
	if int64(len(rest)) < info.Size {
		return Info{}, nil, fmt.Errorf("gitbroker: truncated cat-file payload (%d of %d bytes)", len(rest), info.Size)
	}
	// Copy: rest aliases the command's whole output buffer and the payload
	// outlives it.
	data := make([]byte, info.Size)
	copy(data, rest[:info.Size])
	return info, data, nil
}
