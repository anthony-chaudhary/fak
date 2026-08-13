// Package gitbroker is the one place Go code in this tree reaches git through.
//
// RUNG 2 OF EPIC #5619 (#5621) — the spine, and deliberately the part with no
// daemon, no socket, and no cache in it. Two things live here:
//
//  1. ONE runner interface (Git) with one spawning implementation (Exec). The
//     near-duplicate git exec helpers scattered through cmd/fak — gitOut,
//     runGitOutput, gitOutput, gitRunner — become thin shims over it and keep
//     their exact signatures, so no call site moves. Their differences were
//     never about *what* they ran, only about where the working directory came
//     from, what happened to stderr, and whether a non-zero exit was an error
//     or a returned code. Invocation carries those four axes so each helper's
//     observable behaviour is preserved rather than silently normalized.
//
//  2. A warm `git cat-file --batch` / `--batch-check` pool behind that seam.
//     Object reads are the batchable half of this tree's git traffic: 200 x
//     `git cat-file -t <oid>` costs 6,408 ms across 200 processes, while one
//     `--batch-check` fed the same 200 OIDs on stdin costs 82 ms in one. The
//     pool is per repo, started on first use, and reaped on exit.
//
// THE POOL ANSWERS ONLY WHEN IT CAN ANSWER BYTE-IDENTICALLY. This rung is a
// refactor and must not change what any existing call site sees. So the fast
// path is deliberately narrow: it recognizes a small closed set of object-read
// argument shapes, and if the batch backend does not return a clean record for
// one — missing object, ambiguous key, a pretty-printed tree whose format is
// not the raw payload, a dead or wedged pool, anything unrecognized — the call
// falls through and spawns git exactly as it did before this package existed.
// Failure is never a different answer; it is only a slower one.
//
// AN ORPHANED `git cat-file` IS WORSE THAN THE CHURN THIS FIXES. Reaping is not
// best-effort. Close closes the child's stdin, which is what `cat-file --batch`
// exits on, and kills it if it does not. The same EOF is what reaps a pool
// whose parent died without calling anything: the OS closes the write end of
// the pipe on process exit, so the child sees EOF and leaves. TestPoolIsReaped
// proves that against a real subprocess rather than asserting it.
package gitbroker

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// Invocation is one git command to run. Its fields are exactly the axes the
// helpers this package replaces disagreed on — nothing here is a preference,
// each field exists because dropping it would change some caller's behaviour.
type Invocation struct {
	// Args is the git argument vector without the leading "git".
	Args []string

	// Dir is the repository to run against.
	Dir string

	// DirAsFlag passes Dir as `git -C <Dir>` instead of setting the child
	// process's working directory. The two are NOT interchangeable: -C moves
	// git's idea of the repository while leaving the child's cwd alone, so a
	// relative path elsewhere in Args still resolves against the caller's cwd.
	// gitOut used -C; gitOutput and gitRunner set cmd.Dir. Both are preserved.
	DirAsFlag bool

	// Env is appended to the environment childEnv builds — the parent's, minus
	// git's own steering variables. gitRunner's GIT_OPTIONAL_LOCKS=0 is the only
	// production user: its burst-time reads run exactly when peers are
	// committing, and a plain read that opportunistically refreshes the index
	// would contend with them on .git/index.lock. Because it is applied last, a
	// caller can still steer deliberately — GIT_INDEX_FILE, a one-off
	// GIT_CONFIG_* — where the parent process no longer can.
	Env []string

	// Stdin, when non-empty, is fed to git on standard input.
	Stdin string

	// Combined folds stderr into Stdout, i.e. CombinedOutput semantics.
	Combined bool

	// DiscardStderr drops stderr instead of capturing it. gitOut discarded it,
	// and a caller that never sees stderr must keep never seeing it.
	DiscardStderr bool
}

// Output is one finished git command.
type Output struct {
	Stdout []byte
	Stderr []byte
	// ExitCode is git's exit status, or -1 when git could not be started at
	// all. The distinction is load-bearing for gitRunner, whose contract is
	// "error only when git could not be executed" — a non-zero exit is data.
	ExitCode int
}

// Info is what `git cat-file --batch-check` reports about one object: the
// resolved (and, for a `^{type}` key, peeled) OID, its type, and the byte size
// git declared for its payload.
type Info struct {
	OID  string
	Type string
	Size int64
}

// ErrNotFound reports that git resolved the key to nothing — a real answer
// about the repository, not a failure of this package. `cat-file --batch` and
// `--batch-check` both spell it as a two-field record (`<key> missing`, or
// `<key> ambiguous`), which is why it is detected rather than inferred.
var ErrNotFound = errors.New("gitbroker: object not found")

// Git is the ONE interface every Go git call in this tree reaches git through.
//
// Run is the general seam and always spawns unless the invocation is one of the
// object-read shapes the warm pool can answer byte-identically. ObjectInfo and
// ObjectBytes are the explicit object-read surface: they are the pool's real
// API, and Run's fast path is implemented in terms of them, so there is exactly
// one code path from "a caller wants an object" to "the batch backend answers".
type Git interface {
	Run(ctx context.Context, inv Invocation) (Output, error)
	ObjectInfo(ctx context.Context, dir, rev string) (Info, error)
	ObjectBytes(ctx context.Context, dir, rev string) (Info, []byte, error)
	Close() error
}

// Exec is the spawning implementation of Git with the warm pool in front of
// object reads. The zero value is usable; New and Shared exist so callers do
// not have to reason about that.
type Exec struct {
	mu     sync.Mutex
	pools  map[string]*batchProc
	closed bool
}

// New returns an Exec with its own pool. Tests want this; production wants
// Shared, because a pool that is not shared across a process's call sites saves
// nothing.
func New() *Exec { return &Exec{} }

var (
	sharedOnce sync.Once
	sharedExec *Exec
)

// Shared is the process-wide Exec. One process, one warm pool per repository —
// which is the whole saving, since the cost this package exists to remove is
// paid per process created.
func Shared() *Exec {
	sharedOnce.Do(func() { sharedExec = New() })
	return sharedExec
}

// gitProcesses counts every git process this package has created — one-shot
// spawns and warm pool starts alike. It is the honest denominator for the claim
// #5621 is built on, and it is in-process rather than sampled so a test can
// assert the collapse instead of a benchmark having to allege it.
var gitProcesses atomic.Int64

// Spawns reports how many git processes this package has created since start.
// The warm pool's whole value is that this number stops tracking the number of
// object reads; TestWarmPoolCollapsesObjectReadsToOneProcess pins the ratio.
func Spawns() int64 { return gitProcesses.Load() }

// CloseAll reaps the shared Exec's pools. A process that never calls it is
// still safe — see the package doc on stdin EOF — but a long-lived process that
// is done with git should not keep a `cat-file --batch` warm for nothing.
func CloseAll() error { return Shared().Close() }

// Close reaps every pool this Exec started. It is idempotent, and an Exec that
// has been closed spawns for everything rather than silently restarting a pool
// the caller asked to be rid of.
func (e *Exec) Close() error {
	e.mu.Lock()
	procs := make([]*batchProc, 0, len(e.pools))
	for _, p := range e.pools {
		procs = append(procs, p)
	}
	e.pools = nil
	e.closed = true
	e.mu.Unlock()

	var first error
	for _, p := range procs {
		if err := p.close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}

// Run executes one git invocation, through the warm pool when — and only when —
// the pool can produce the identical bytes, and by spawning otherwise.
func (e *Exec) Run(ctx context.Context, inv Invocation) (Output, error) {
	if out, ok := e.viaPool(ctx, inv); ok {
		return out, nil
	}
	return spawn(ctx, inv)
}

// ObjectInfo answers `cat-file -t`/`-s` and the `<rev>^{type}` peel from the
// warm `--batch-check` pool, falling back to one `--batch-check` spawn when no
// pool is usable. ErrNotFound means git resolved the key to nothing.
func (e *Exec) ObjectInfo(ctx context.Context, dir, rev string) (Info, error) {
	info, _, err := e.object(ctx, dir, rev, false)
	return info, err
}

// ObjectBytes answers `cat-file -p` from the warm `--batch` pool, returning the
// object's raw payload. Raw is what `--batch` emits and what `-p` prints for
// every type except a tree, whose pretty form is not its payload — see
// poolServiceableRead for why a tree is never served from here.
func (e *Exec) ObjectBytes(ctx context.Context, dir, rev string) (Info, []byte, error) {
	return e.object(ctx, dir, rev, true)
}

// object is the single path from an object question to an answer: warm pool
// first, one-shot spawn if the pool is unavailable or wedged. A pool failure is
// never surfaced to the caller as an error — that is the fallback guarantee.
func (e *Exec) object(ctx context.Context, dir, rev string, payload bool) (Info, []byte, error) {
	if !safeBatchKey(rev) {
		return Info{}, nil, errors.New("gitbroker: unusable object key " + rev)
	}
	if p := e.pool(dir, payload); p != nil {
		info, data, err := p.query(ctx, rev)
		switch {
		case err == nil:
			return info, data, nil
		case errors.Is(err, ErrNotFound):
			// A real answer about the repo. Spawning would only say it slower.
			return Info{}, nil, err
		}
		// Anything else — the pool never started, died, wedged, or lost stream
		// sync — is a POOL problem, not a repository answer. Fall through.
	}
	return spawnObject(ctx, dir, rev, payload)
}

// pool returns the warm process for (dir, mode), starting the bookkeeping entry
// lazily. It returns nil once the Exec is closed, or when dir is empty: the
// pool runs with `-C dir`, so a caller that did not name a repository cannot be
// served without guessing which one it meant.
func (e *Exec) pool(dir string, payload bool) *batchProc {
	if strings.TrimSpace(dir) == "" {
		return nil
	}
	key := dir + "\x00"
	if payload {
		key += "batch"
	} else {
		key += "batch-check"
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	if e.pools == nil {
		e.pools = map[string]*batchProc{}
	}
	if p, ok := e.pools[key]; ok {
		return p
	}
	p := &batchProc{dir: dir, payload: payload}
	e.pools[key] = p
	return p
}

// viaPool is the narrow fast path: it recognizes an object read that the pool
// can answer byte-identically, and reports ok=false for everything else. Every
// false here is a spawn, which is exactly the behaviour that shipped before
// this package, so a miss costs latency and nothing else.
func (e *Exec) viaPool(ctx context.Context, inv Invocation) (Output, bool) {
	rev, kind, ok := objectReadShape(inv)
	if !ok {
		return Output{}, false
	}

	if kind == readPrint {
		info, data, err := e.ObjectBytes(ctx, inv.Dir, rev)
		if err != nil || !poolServiceableRead(info.Type) {
			return Output{}, false
		}
		return Output{Stdout: data}, true
	}

	info, err := e.ObjectInfo(ctx, inv.Dir, rev)
	if err != nil {
		return Output{}, false
	}
	switch kind {
	case readType:
		return Output{Stdout: []byte(info.Type + "\n")}, true
	case readSize:
		return Output{Stdout: []byte(itoa(info.Size) + "\n")}, true
	case readVerify:
		return Output{Stdout: []byte(info.OID + "\n")}, true
	}
	return Output{}, false
}

// The object-read shapes the pool will serve. Deliberately a closed set: an
// unrecognized shape spawns, and adding one means proving the bytes match.
const (
	readType = iota + 1
	readSize
	readPrint
	readVerify
)

// objectReadShape classifies an invocation as one of the servable object reads,
// or refuses it.
//
// The refusals are the interesting part. Stdin, extra environment, combined
// stderr and an unnamed directory each mean the pool would be answering a
// subtly different question than the spawn would: the pool runs `-C dir` with
// the brokered environment childEnv builds and a clean stderr, so any
// invocation that tunes those is left to spawn rather than approximated.
func objectReadShape(inv Invocation) (rev string, kind int, ok bool) {
	if inv.Dir == "" || inv.Stdin != "" || len(inv.Env) > 0 || inv.Combined {
		return "", 0, false
	}
	a := inv.Args
	if len(a) == 3 && a[0] == "cat-file" {
		switch a[1] {
		case "-t":
			kind = readType
		case "-s":
			kind = readSize
		case "-p":
			kind = readPrint
		}
		if kind != 0 && safeBatchKey(a[2]) {
			return a[2], kind, true
		}
		return "", 0, false
	}
	// `rev-parse --verify [--quiet] <rev>` and nothing else: --verify is what
	// makes rev-parse print exactly one resolved OID, which is exactly the OID
	// field of a clean --batch-check record. Any other rev-parse flag changes
	// the output shape, so it spawns.
	if len(a) >= 3 && a[0] == "rev-parse" && a[1] == "--verify" {
		rest := a[2:]
		if len(rest) == 2 && rest[0] == "--quiet" {
			rest = rest[1:]
		}
		if len(rest) == 1 && safeBatchKey(rest[0]) {
			return rest[0], readVerify, true
		}
	}
	return "", 0, false
}

// poolServiceableRead reports whether `cat-file -p` on this object type prints
// the raw payload `--batch` hands back. It does for a blob, a commit and a tag;
// it does NOT for a tree, whose pretty form is a rendered listing and not the
// binary payload at all. TestPoolMatchesSpawnPerObjectType pins this against
// real git rather than trusting the claim.
func poolServiceableRead(typ string) bool {
	return typ == "blob" || typ == "commit" || typ == "tag"
}

// safeBatchKey refuses a key the pool must not put on a batch process's stdin.
//
// A newline would inject a second record and desynchronize the stream, so every
// later answer would be off by one — silent wrong bytes, the one failure mode
// this package must not have. A leading '-' would be read as a flag by the
// spawn fallback. A cwd-relative path ("HEAD:./x") resolves against the child's
// working directory, which the pool sets with -C and the caller may not share.
func safeBatchKey(rev string) bool {
	if rev == "" || strings.HasPrefix(rev, "-") || strings.Contains(rev, "./") {
		return false
	}
	return !strings.ContainsAny(rev, "\r\n\t ")
}

// spawn runs one git process. This is the pre-refactor path, preserved
// field-for-field: it is both what every non-object call still does and what
// every pool failure falls back to, and that identity is what makes the
// fallback guarantee byte-exact rather than aspirational.
func spawn(ctx context.Context, inv Invocation) (Output, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	args := inv.Args
	if inv.DirAsFlag && inv.Dir != "" {
		args = append([]string{"-C", inv.Dir}, args...)
	}
	cmd := exec.CommandContext(ctx, "git", args...)
	// Every spawn here runs under background automation on a Windows fleet;
	// without the no-window hook each one flashes a console, which the
	// DESKTOP_POPUP_REGRESSION gate refuses.
	windowgate.ConfigureBackgroundCommand(cmd)
	if !inv.DirAsFlag {
		cmd.Dir = inv.Dir
	}
	// Never the parent environment wholesale: an inherited GIT_DIR, GIT_CONFIG_*
	// or GIT_SSH_COMMAND would re-aim or re-configure an operation the broker
	// resolved, and a missing GIT_TERMINAL_PROMPT=0 turns a credential-needing
	// call into a headless hang. See childEnv.
	cmd.Env = childEnv(os.Environ(), inv.Env)
	if inv.Stdin != "" {
		cmd.Stdin = strings.NewReader(inv.Stdin)
	}

	var out, errb bytes.Buffer
	cmd.Stdout = &out
	switch {
	case inv.Combined:
		cmd.Stderr = &out
	case inv.DiscardStderr:
		cmd.Stderr = io.Discard
	default:
		cmd.Stderr = &errb
	}

	err := cmd.Run()
	var startErr *exec.Error
	if !errors.As(err, &startErr) {
		gitProcesses.Add(1)
	}

	res := Output{Stdout: out.Bytes(), Stderr: errb.Bytes()}
	var ee *exec.ExitError
	switch {
	case err == nil:
		res.ExitCode = 0
	case errors.As(err, &ee):
		res.ExitCode = ee.ExitCode()
		// exec.Cmd.Output populates ExitError.Stderr when it owns the stderr
		// buffer; callers that used to go through Output (runGitOutput) can
		// read it off the error, so keep it populated here too.
		if ee.Stderr == nil && !inv.Combined && !inv.DiscardStderr {
			ee.Stderr = res.Stderr
		}
	default:
		res.ExitCode = -1
	}
	return res, err
}

// spawnObject is the one-shot object read the pool falls back to. One
// `cat-file --batch` spawn yields type, size and payload together, so it is the
// cheapest honest single-object read git offers — and it parses the identical
// record format, so the fallback cannot disagree with the pool about shape.
func spawnObject(ctx context.Context, dir, rev string, payload bool) (Info, []byte, error) {
	mode := "--batch-check"
	if payload {
		mode = "--batch"
	}
	out, err := spawn(ctx, Invocation{
		Dir: dir, DirAsFlag: dir != "",
		Args:  []string{"cat-file", mode},
		Stdin: rev + "\n",
	})
	if err != nil {
		return Info{}, nil, err
	}
	return parseBatchReply(out.Stdout, payload)
}

// itoa keeps the size formatting in one place; strconv would be the same, this
// just avoids an import used exactly once.
func itoa(n int64) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	var buf [24]byte
	i := len(buf)
	for n != 0 {
		i--
		d := n % 10
		if d < 0 {
			d = -d
		}
		buf[i] = byte('0' + d)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
