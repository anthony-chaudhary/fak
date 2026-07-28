package dispatchtick

// Worker shell rack (#3405): reuse warm worker shells across dispatch ticks
// instead of cold-spawning one per tick. Each cold spawn on Windows creates a
// ConPTY with its own conhost/OpenConsole process + pipes, so per-console cost
// scales linearly with fleet size (root cause of the #3153 whole-machine stall;
// no registry knob is a substitute for reducing spawns). This file is the
// reuse policy only — bounded checkout/return with health-checked reuse and
// max-idle retirement — behind the WarmShell/ShellSpawnFunc seam; nothing here
// touches the OS. The first live wiring is the Windows host-probe seam
// (cmd/fak dispatchRunHostProbe), which used to burn one `powershell` process
// per preflight probe and now runs every probe of a tick as a TaskShell task on
// one racked shell. The long-lived ConPTY worker spawner
// (cmd/fak spawnDispatchIssueWorker) is a separate follow-on: a worker process
// is bound to one issue, so it is a pooling candidate only once a worker can
// outlive its task.
//
// Semantics:
//   - Checkout(key) hands back an idle warm shell for the key when one is
//     healthy and fresh; only a miss cold-spawns. Expired (max-idle) and
//     unhealthy idle shells are retired (closed) at checkout, never reused.
//   - The rack retains at most Cap shells (idle + checked-out racked). A
//     checkout while the rack is full still cold-spawns — dispatch must not
//     queue behind a full rack — but the overflow shell is unracked: Return
//     closes it instead of retaining it, so the warm set stays bounded.
//   - Return puts a healthy racked shell back on its key's warm stack (LIFO,
//     warmest first) and closes an unhealthy one, freeing its slot.

import (
	"errors"
	"sync"
	"time"
)

// WarmShell is the rack's view of one warm worker shell/process. Healthy
// controls reuse at checkout and retention at return; Close releases the OS
// resources (process tree, console, pipes) and must be safe to call once.
type WarmShell interface {
	Healthy() bool
	Close() error
}

// ShellSpawnFunc cold-spawns a new worker shell for a rack key. The key names
// the reuse domain (e.g. backend+lane): shells are only ever reused within the
// key they were spawned for.
type ShellSpawnFunc func(key string) (WarmShell, error)

// TaskShell is a warm shell that can RUN a short task (a probe script, a
// command line) and hand back its stdout. A bare WarmShell is only a lifetime
// cache; TaskShell is what turns the rack into a reuse SPINE: N tasks ride one
// shell, so N console creations collapse into one.
type TaskShell interface {
	WarmShell
	RunTask(task string) ([]byte, error)
}

// ErrShellNotTaskCapable is returned by RunTask when the rack's spawner hands
// back a shell that cannot run tasks. It is a wiring defect, not a runtime
// condition: the caller should fall back to its own one-shot spawn path rather
// than fail the tick.
var ErrShellNotTaskCapable = errors.New("shell rack: shell cannot run tasks")

// ShellRackStats counts rack traffic; every field is monotonic.
type ShellRackStats struct {
	ColdSpawns       int // spawner invocations (racked misses + overflow)
	WarmReuses       int // checkouts served by an idle warm shell
	UnhealthyRetired int // shells closed because Healthy() went false
	IdleRetired      int // shells closed because they sat idle past MaxIdle
	OverflowSpawns   int // cold spawns issued while the rack was full
	OverflowClosed   int // overflow shells closed at Return (never retained)
	TasksRun         int // tasks executed through RunTask (reused + cold)
}

// SpawnsAvoided is the reuse dividend: how many process/console creations the
// rack removed for the tasks it ran. Without a rack each task is its own spawn,
// so the unpooled cost is TasksRun; the pooled cost is ColdSpawns. This is the
// before/after churn axis #3405 is measured on.
func (s ShellRackStats) SpawnsAvoided() int {
	if s.TasksRun <= s.ColdSpawns {
		return 0
	}
	return s.TasksRun - s.ColdSpawns
}

// ShellLease is one checked-out shell plus the bookkeeping Return needs.
type ShellLease struct {
	Shell  WarmShell
	Key    string
	Warm   bool // served from the idle warm set (no spawn)
	Racked bool // false: overflow spawn over the cap; Return closes it
}

type idleShell struct {
	shell WarmShell
	since time.Time
}

// ShellRack is a bounded, health-checked set of warm worker shells keyed by
// reuse domain. Safe for concurrent use.
type ShellRack struct {
	mu      sync.Mutex
	cap     int
	maxIdle time.Duration
	spawn   ShellSpawnFunc
	now     func() time.Time
	idle    map[string][]idleShell
	live    int // retained shells: idle + checked-out racked (excludes overflow)
	stats   ShellRackStats
	closed  bool
}

// NewShellRack builds a rack retaining at most capacity shells. maxIdle <= 0
// disables idle expiry (health checks still apply).
func NewShellRack(capacity int, maxIdle time.Duration, spawn ShellSpawnFunc) (*ShellRack, error) {
	if capacity < 1 {
		return nil, errors.New("shell rack capacity must be >= 1")
	}
	if spawn == nil {
		return nil, errors.New("shell rack needs a spawn func")
	}
	return &ShellRack{
		cap:     capacity,
		maxIdle: maxIdle,
		spawn:   spawn,
		now:     time.Now,
		idle:    map[string][]idleShell{},
	}, nil
}

// Checkout hands back a warm shell for key, cold-spawning only on miss. Stale
// and unhealthy idle shells for the key are retired (closed) first, so a
// reused shell is always fresh and healthy at the moment of checkout.
func (p *ShellRack) Checkout(key string) (ShellLease, error) {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return ShellLease{}, errors.New("shell rack is closed")
	}
	if shell, ok := p.takeWarmLocked(key); ok {
		p.stats.WarmReuses++
		p.mu.Unlock()
		return ShellLease{Shell: shell, Key: key, Warm: true, Racked: true}, nil
	}
	racked := p.live < p.cap
	if racked {
		p.live++ // reserve the slot before dropping the lock to spawn
	}
	p.mu.Unlock()

	shell, err := p.spawn(key)
	p.mu.Lock()
	if err != nil {
		if racked {
			p.live-- // release the reserved slot; no shell was produced
		}
		p.mu.Unlock()
		return ShellLease{}, err
	}
	p.stats.ColdSpawns++
	if !racked {
		p.stats.OverflowSpawns++
	}
	p.mu.Unlock()
	return ShellLease{Shell: shell, Key: key, Racked: racked}, nil
}

// RunTask is the reuse spine's one-call entry point: check a shell out for key,
// run task on it, put it back. The SECOND and later calls for the same key ride
// the shell the first call spawned, so a caller that used to spawn one process
// per task now spawns one process per key. Callers that cannot tolerate a rack
// failure should treat any error as "take my own one-shot path": RunTask never
// retains a shell that failed its task (Return closes an unhealthy one), so a
// broken shell is retired instead of being handed to the next task.
func (p *ShellRack) RunTask(key, task string) ([]byte, error) {
	lease, err := p.Checkout(key)
	if err != nil {
		return nil, err
	}
	shell, ok := lease.Shell.(TaskShell)
	if !ok {
		p.Return(lease)
		return nil, ErrShellNotTaskCapable
	}
	out, runErr := shell.RunTask(task)
	p.mu.Lock()
	p.stats.TasksRun++
	p.mu.Unlock()
	p.Return(lease)
	if runErr != nil {
		return nil, runErr
	}
	return out, nil
}

// takeWarmLocked pops the warmest healthy, unexpired idle shell for key,
// closing every stale or unhealthy shell it skips over. Caller holds p.mu.
func (p *ShellRack) takeWarmLocked(key string) (WarmShell, bool) {
	stack := p.idle[key]
	now := p.now()
	for len(stack) > 0 {
		top := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if p.maxIdle > 0 && now.Sub(top.since) > p.maxIdle {
			p.stats.IdleRetired++
			p.live--
			_ = top.shell.Close()
			continue
		}
		if !top.shell.Healthy() {
			p.stats.UnhealthyRetired++
			p.live--
			_ = top.shell.Close()
			continue
		}
		p.storeIdleLocked(key, stack)
		return top.shell, true
	}
	p.storeIdleLocked(key, stack)
	return nil, false
}

func (p *ShellRack) storeIdleLocked(key string, stack []idleShell) {
	if len(stack) == 0 {
		delete(p.idle, key)
		return
	}
	p.idle[key] = stack
}

// Return gives a lease back. An overflow (unracked) shell is closed outright;
// an unhealthy racked shell is closed and its slot freed; a healthy racked
// shell rejoins its key's warm stack. Safe on a zero-value lease.
func (p *ShellRack) Return(l ShellLease) {
	if l.Shell == nil {
		return
	}
	p.mu.Lock()
	if !l.Racked || p.closed {
		if !l.Racked {
			p.stats.OverflowClosed++
		} else {
			p.live--
		}
		p.mu.Unlock()
		_ = l.Shell.Close()
		return
	}
	if !l.Shell.Healthy() {
		p.stats.UnhealthyRetired++
		p.live--
		p.mu.Unlock()
		_ = l.Shell.Close()
		return
	}
	p.idle[l.Key] = append(p.idle[l.Key], idleShell{shell: l.Shell, since: p.now()})
	p.mu.Unlock()
}

// Prune closes every idle shell that has sat past MaxIdle, across all keys,
// and reports how many it retired. A maintenance tick can call this so stale
// consoles do not linger between checkouts.
func (p *ShellRack) Prune() int {
	p.mu.Lock()
	if p.maxIdle <= 0 {
		p.mu.Unlock()
		return 0
	}
	now := p.now()
	var doomed []WarmShell
	for key, stack := range p.idle {
		kept := stack[:0]
		for _, s := range stack {
			if now.Sub(s.since) > p.maxIdle {
				doomed = append(doomed, s.shell)
				continue
			}
			kept = append(kept, s)
		}
		p.storeIdleLocked(key, kept)
	}
	p.stats.IdleRetired += len(doomed)
	p.live -= len(doomed)
	p.mu.Unlock()
	for _, s := range doomed {
		_ = s.Close()
	}
	return len(doomed)
}

// Close shuts the rack: every idle shell is closed and later Checkouts refuse.
// Shells still checked out are the holders' to Return (which will close them).
func (p *ShellRack) Close() {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return
	}
	p.closed = true
	var doomed []WarmShell
	for _, stack := range p.idle {
		for _, s := range stack {
			doomed = append(doomed, s.shell)
		}
	}
	p.idle = map[string][]idleShell{}
	p.live -= len(doomed)
	p.mu.Unlock()
	for _, s := range doomed {
		_ = s.Close()
	}
}

// Stats returns a snapshot of the traffic counters.
func (p *ShellRack) Stats() ShellRackStats {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.stats
}

// IdleCount reports how many warm shells sit idle for key.
func (p *ShellRack) IdleCount(key string) int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.idle[key])
}

// LiveCount reports how many shells the rack currently retains
// (idle + checked-out racked; overflow shells are never counted).
func (p *ShellRack) LiveCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.live
}
