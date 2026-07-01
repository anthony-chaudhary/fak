package fleetmon

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// JanitorSchema tags the janitor's JSON payload.
const JanitorSchema = "fak-fleet-janitor/1"

// CommandClass buckets a child command so a stuck simple probe and a legitimate
// long-running test get different staleness thresholds.
type CommandClass string

const (
	CmdSimpleShell CommandClass = "simple-shell" // ls/cat/echo/pwd — should be instant
	CmdTest        CommandClass = "test"         // go test / make ci / ./test.ps1 — legitimately minutes
	CmdBroadScan   CommandClass = "broad-scan"   // find/rg/grep over a big tree
	CmdHook        CommandClass = "hook"         // a commit/pre-commit hook actuator
	CmdLongWitness CommandClass = "long-witness" // an intentionally long process (serve/bench/sleep)
	CmdMCP         CommandClass = "mcp"          // a DOS/persistent MCP server — always protected
	CmdOther       CommandClass = "other"        // unclassified
)

// JanitorPolicy is the per-class staleness ceiling. A child older than its class
// ceiling is stale; a zero ceiling disables staleness for that class (used for
// long-witness, which must never be reaped on age alone).
type JanitorPolicy struct {
	SimpleShell time.Duration
	Test        time.Duration
	BroadScan   time.Duration
	Hook        time.Duration
	Other       time.Duration
	LongWitness time.Duration
}

// DefaultJanitorPolicy returns the issue's default ceilings: 5m simple, 10m test.
func DefaultJanitorPolicy() JanitorPolicy {
	return JanitorPolicy{
		SimpleShell: 5 * time.Minute,
		Test:        10 * time.Minute,
		BroadScan:   5 * time.Minute,
		Hook:        2 * time.Minute,
		Other:       10 * time.Minute,
		LongWitness: 0, // never stale on age alone
	}
}

func (p JanitorPolicy) ceiling(c CommandClass) time.Duration {
	switch c {
	case CmdSimpleShell:
		return p.SimpleShell
	case CmdTest:
		return p.Test
	case CmdBroadScan:
		return p.BroadScan
	case CmdHook:
		return p.Hook
	case CmdLongWitness:
		return p.LongWitness
	case CmdMCP:
		return 0 // protected; never reaped on age
	default:
		return p.Other
	}
}

// WorkerRoot names one worker process the janitor must protect: its own PID is
// never a kill target, and its start time is the PID-reuse fence for its subtree.
type WorkerRoot struct {
	PID     int    `json:"pid"`
	Session string `json:"session,omitempty"`
	Issue   int    `json:"issue,omitempty"`
	Start   string `json:"start,omitempty"` // RFC3339 process start; the reuse fence
}

// ChildCommand is one classified child process tree under a worker root.
type ChildCommand struct {
	RootPID   int    `json:"root_pid"`   // the child subtree root — the tree-kill target
	PPID      int    `json:"ppid"`       //
	Name      string `json:"name"`       //
	Command   string `json:"command"`    // the child command line (trimmed)
	Class     CommandClass `json:"class"` //
	AgeSec    int    `json:"age_sec"`    //
	WorkerPID int    `json:"worker_pid"` // the owning worker root PID
	Session   string `json:"session,omitempty"`
	Issue     int    `json:"issue,omitempty"`
	TreePIDs  []int  `json:"tree_pids"`  // the child + all its descendants (for reporting)
	Stale     bool   `json:"stale"`      //
	Protected bool   `json:"protected"`  // an MCP/persistent server or a predates-worker process — never killed
	Reason    string `json:"reason"`     //
}

// JanitorInput is the injected snapshot the pure evaluator folds.
type JanitorInput struct {
	Procs             []procguard.Proc // a relations snapshot (PPID/Cmdline/AgeSec/Start)
	Workers           []WorkerRoot     // the worker root PIDs to protect + attribute children to
	Policy            JanitorPolicy    //
	ProtectedPatterns []string         // extra cmdline substrings that mark a protected process (default: MCP servers)
	Now               time.Time        // for the predates-worker fence
}

// JanitorResult is the machine-readable janitor payload.
type JanitorResult struct {
	Schema     string         `json:"schema"`
	Scanned    int            `json:"scanned"`
	Children   []ChildCommand `json:"children"`  // every child command under a worker root
	Stale      []ChildCommand `json:"stale"`     // the stale AND safe-to-terminate subset
	Protected  []ChildCommand `json:"protected"` // stale-by-age but protected (reported, never killed)
	NextAction string         `json:"next_action"`
}

// defaultProtectedCmd marks a process whose command line means "do not touch":
// a DOS MCP server, a generic MCP/stdio server, or a persistent `fak guard`/serve.
var defaultProtectedCmd = []string{"dos_mcp", "dos.mcp", "mcp.server", "mcp serve", "fak guard", "fak serve"}

// EvaluateJanitor builds the process forest under each worker root, classifies
// every descendant command, and returns the stale ones that are SAFE to
// terminate. Safety is layered:
//
//   - the worker root PID itself is never a target (kill the child, keep the worker);
//   - a DOS/persistent MCP server (by command-line pattern) is protected;
//   - an ancestor of any worker root is protected (killing it would take a worker down);
//   - a process whose start time PREDATES its claimed worker root is protected —
//     that is the PID-reuse case where an old process carries a stale ParentProcessId
//     pointing at a pid the OS has since recycled into a worker root.
//
// The evaluator does no I/O and never kills anything; the cmd layer calls
// procguard.KillPID(child.RootPID) (taskkill /T on Windows) for the --apply pass.
func EvaluateJanitor(in JanitorInput) JanitorResult {
	pol := in.Policy
	if pol == (JanitorPolicy{}) {
		pol = DefaultJanitorPolicy()
	}
	patterns := in.ProtectedPatterns
	if patterns == nil {
		patterns = defaultProtectedCmd
	}

	byPID := map[int]procguard.Proc{}
	children := map[int][]int{} // ppid -> child pids
	for _, p := range in.Procs {
		byPID[p.PID] = p
		if p.PPID != nil {
			children[*p.PPID] = append(children[*p.PPID], p.PID)
		}
	}
	workerByPID := map[int]WorkerRoot{}
	for _, w := range in.Workers {
		workerByPID[w.PID] = w
	}
	// ancestorsOfWorkers: every pid that is an ancestor of a worker root. Killing
	// one would take the worker down, so it is protected even if it looks stale.
	ancestors := map[int]bool{}
	for _, w := range in.Workers {
		for cur := byPID[w.PID].PPID; cur != nil; {
			pid := *cur
			if ancestors[pid] { // already walked (shared ancestry)
				break
			}
			ancestors[pid] = true
			cur = byPID[pid].PPID
		}
	}

	var all []ChildCommand
	seen := map[int]bool{}
	for _, w := range in.Workers {
		rootStart, hasRootStart := parseTS(w.Start)
		// BFS over the worker's descendants.
		queue := append([]int{}, children[w.PID]...)
		for len(queue) > 0 {
			pid := queue[0]
			queue = queue[1:]
			if seen[pid] || pid == w.PID {
				continue
			}
			seen[pid] = true
			queue = append(queue, children[pid]...)

			p := byPID[pid]
			cmd := strings.TrimSpace(p.Cmdline)
			class := classifyCommand(p.Name, cmd)
			age := 0
			if p.AgeSec != nil {
				age = *p.AgeSec
			}

			cc := ChildCommand{
				RootPID:   pid,
				PPID:      derefInt(p.PPID),
				Name:      p.Name,
				Command:   trimCmd(cmd),
				Class:     class,
				AgeSec:    age,
				WorkerPID: w.PID,
				Session:   w.Session,
				Issue:     w.Issue,
				TreePIDs:  subtree(pid, children),
			}

			// Protection layers (any one protects).
			predates := hasRootStart && childPredatesRoot(p, rootStart)
			switch {
			case workerByPID[pid].PID != 0:
				cc.Protected, cc.Reason = true, "is itself a worker root — never a kill target"
			case ancestors[pid]:
				cc.Protected, cc.Reason = true, "ancestor of a worker root — killing it would stop the worker"
			case class == CmdMCP || matchesAny(cmd, p.Name, patterns):
				cc.Protected, cc.Reason = true, "MCP/persistent server — protected"
			case predates:
				cc.Protected, cc.Reason = true, "process predates the worker root (stale ParentProcessId / PID reuse) — not this worker's child"
			}

			ceiling := pol.ceiling(class)
			cc.Stale = !cc.Protected && ceiling > 0 && age >= int(ceiling.Seconds())
			if cc.Stale {
				cc.Reason = fmt.Sprintf("%s child alive %s ≥ %s ceiling", class, roundDur(float64(age)), ceiling)
			} else if cc.Reason == "" {
				cc.Reason = fmt.Sprintf("%s child, age %s < %s ceiling", class, roundDur(float64(age)), ceilingLabel(ceiling))
			}
			all = append(all, cc)
		}
	}

	sort.SliceStable(all, func(i, j int) bool {
		if all[i].WorkerPID != all[j].WorkerPID {
			return all[i].WorkerPID < all[j].WorkerPID
		}
		return all[i].RootPID < all[j].RootPID
	})

	var stale, protected []ChildCommand
	for _, c := range all {
		switch {
		case c.Stale:
			stale = append(stale, c)
		case c.Protected && noteworthyProtected(c, pol):
			protected = append(protected, c)
		}
	}
	return JanitorResult{
		Schema:     JanitorSchema,
		Scanned:    len(in.Procs),
		Children:   all,
		Stale:      stale,
		Protected:  protected,
		NextAction: janitorNextAction(stale, protected),
	}
}

// classifyCommand buckets a child by process name and command line.
func classifyCommand(name, cmd string) CommandClass {
	hay := strings.ToLower(name + " " + cmd)
	switch {
	case containsAnyLower(hay, mcpNeedles):
		return CmdMCP
	case containsAnyLower(hay, longWitnessNeedles):
		return CmdLongWitness
	case testRE.MatchString(hay):
		return CmdTest
	case containsAnyLower(hay, hookNeedles):
		return CmdHook
	case containsAnyLower(hay, broadScanNeedles):
		return CmdBroadScan
	case isSimpleShell(name, cmd):
		return CmdSimpleShell
	default:
		return CmdOther
	}
}

var (
	testRE             = regexp.MustCompile(`\b(go test|go build|go vet|go run|make ci|make test|test\.ps1|pytest|npm (?:run )?test|cargo test|ctest)\b`)
	mcpNeedles         = []string{"dos_mcp", "dos.mcp", "mcp.server", "mcp serve", "fak guard", "fak serve", "modelcontextprotocol"}
	longWitnessNeedles = []string{"fak serve", "sleep ", "benchmark", "llama-bench", "llama-cli", "fak bench", "nohup"}
	hookNeedles        = []string{"fak hooks", "pre-commit", "commit-msg", "check_committed", "scrub_public"}
	broadScanNeedles   = []string{"find ", "rg ", "ripgrep", "grep -r", "grep -R", "select-string", "get-childitem -recurse", "dir /s"}
	simpleShellNames   = map[string]bool{
		"ls": true, "cat": true, "echo": true, "pwd": true, "head": true, "tail": true,
		"wc": true, "dir": true, "type": true, "whoami": true, "date": true, "env": true,
	}
	simpleShellNeedles = []string{"ls ", "cat ", "echo ", "pwd", "head ", "tail ", "wc ", "get-childitem", "get-content", "get-location"}
)

func isSimpleShell(name, cmd string) bool {
	if simpleShellNames[stemLower(name)] {
		return true
	}
	low := strings.ToLower(strings.TrimSpace(cmd))
	// A plain shell wrapping a simple probe (e.g. `bash -lc "ls -la"`).
	return containsAnyLower(low, simpleShellNeedles)
}

// childPredatesRoot reports whether a child's start time is before the worker
// root's start — the PID-reuse signature: the child's ParentProcessId points at
// a pid the OS recycled into the (younger) worker root, so the child is not
// actually this worker's descendant and must not be reaped as one.
func childPredatesRoot(child procguard.Proc, rootStart time.Time) bool {
	ct, ok := parseTS(child.Start)
	if !ok {
		return false // no start time -> cannot prove reuse; do not protect on this basis
	}
	return ct.Before(rootStart)
}

func subtree(root int, children map[int][]int) []int {
	var out []int
	seen := map[int]bool{root: true}
	queue := []int{root}
	for len(queue) > 0 {
		pid := queue[0]
		queue = queue[1:]
		out = append(out, pid)
		for _, c := range children[pid] {
			if !seen[c] {
				seen[c] = true
				queue = append(queue, c)
			}
		}
	}
	sort.Ints(out)
	return out
}

// noteworthyProtected reports whether a protected child is worth surfacing: it is
// older than the smallest reap floor, so — were it not protected — the janitor
// would have flagged it. A freshly spawned MCP server is protected but not noted;
// an hour-old one is. Classes whose own ceiling is 0 (MCP/long-witness) fall back
// to the simple-shell floor so age is still measured against something.
func noteworthyProtected(c ChildCommand, pol JanitorPolicy) bool {
	floor := pol.SimpleShell
	if floor <= 0 {
		floor = 5 * time.Minute
	}
	return c.AgeSec >= int(floor.Seconds())
}

func janitorNextAction(stale, protected []ChildCommand) string {
	if len(stale) == 0 {
		if len(protected) > 0 {
			return fmt.Sprintf("no safe-to-reap stale child; %d stale-but-protected (MCP/ancestor/pid-reuse) left alone", len(protected))
		}
		return "no stale child command; no action"
	}
	names := make([]string, 0, len(stale))
	for _, c := range stale {
		names = append(names, fmt.Sprintf("%s(pid %d)", c.Name, c.RootPID))
	}
	return fmt.Sprintf("re-run with --apply to terminate %d stale child tree(s): %s", len(stale), strings.Join(names, ", "))
}

// --- small helpers -------------------------------------------------------- //

func matchesAny(cmd, name string, patterns []string) bool {
	hay := strings.ToLower(name + " " + cmd)
	for _, p := range patterns {
		if p != "" && strings.Contains(hay, strings.ToLower(p)) {
			return true
		}
	}
	return false
}

func containsAnyLower(hay string, needles []string) bool {
	for _, n := range needles {
		if strings.Contains(hay, n) {
			return true
		}
	}
	return false
}

func stemLower(name string) string {
	n := strings.ToLower(strings.TrimSpace(name))
	return strings.TrimSuffix(n, ".exe")
}

func trimCmd(s string) string {
	s = collapseWS(s)
	const max = 200
	if len(s) > max {
		return s[:max] + "…"
	}
	return s
}

func ceilingLabel(d time.Duration) string {
	if d == 0 {
		return "∞"
	}
	return d.String()
}

func derefInt(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}
