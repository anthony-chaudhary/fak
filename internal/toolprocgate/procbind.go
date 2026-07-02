package toolprocgate

import (
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

// Seam 6: the tool-call ↔ OS-process-tree binding. A kill or reap advice has
// teeth only when the supervisor knows which OS process tree the tool call
// spawned. The embedder binds the pid it launched (and only that pid) right
// after Spawn; when Tick acts on a kill/reap, the bound tree is terminated
// through the registered OSReaper in the same tick that fires the cancel
// lever and arms the revocation table. procguard already owns the OS-level
// termination (taskkill /T /F on Windows, SIGKILL on POSIX); this file is
// only the missing binding, not a second reaper.

// OSReaper terminates the process tree rooted at pid. The signature is
// exactly procguard.KillPID's, so the production wiring is a direct handoff;
// tests inject a recorder. ok reports the termination succeeded; detail is
// the reaper's message (recorded in the TickAction either way).
type OSReaper func(pid int) (ok bool, detail string)

// SetReaper registers the OS lever Tick pulls for bound pids. nil (the
// default) keeps the supervisor advice-only: kills still cancel and revoke,
// they just carry no OS teeth. Not concurrency-guarded against a running
// Tick — register at construction, before observations flow.
func (s *Supervisor) SetReaper(r OSReaper) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.reaper = r
}

// BindPID binds a spawned, still-live call to the OS process tree it
// launched. Rebinding overwrites (a helper that respawns under a new pid
// reports its current root); Exit, PruneTerminal, and an enforced kill all
// retire the binding. The self-pid is refused — a supervisor must never hold
// a lever aimed at its own process tree (procguard's Enact draws the same
// line).
func (s *Supervisor) BindPID(callID string, pid int) error {
	if pid <= 0 {
		return fmt.Errorf("toolprocgate: bind for call %s: invalid pid %d", callID, pid)
	}
	if pid == os.Getpid() {
		return fmt.Errorf("toolprocgate: bind for call %s: refusing self pid %d", callID, pid)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.spawned[callID] {
		return fmt.Errorf("toolprocgate: bind for unknown call %s", callID)
	}
	s.pids[callID] = pid
	return nil
}

// NewOSSupervisor is the production form of seam 6: a Supervisor whose kill
// and reap advice terminates the bound OS process tree via procguard.
func NewOSSupervisor(cfg toolproc.Config) *Supervisor {
	s := NewSupervisor(cfg)
	s.reaper = procguard.KillPID
	return s
}
