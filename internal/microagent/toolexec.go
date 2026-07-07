package microagent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// ToolExec is the subprocess tool-action backend (#2014, epic #2000 M14): the
// first real isolation step above a goroutine. It runs each tool action as an
// os/exec SUBPROCESS with stdin/stdout/stderr capture and a per-action timeout,
// and — the reason it exists at all — it routes EVERY action through the
// in-process kernel floor (M18) BEFORE the subprocess runs. A denied action
// never reaches exec: adjudication is the gate, not a post-hoc audit.
//
// The process-tree kill on timeout/cancel is not re-implemented here. It reuses
// the ONE cross-platform reaper procguard already owns — Setpgid + kill(-pgid)
// on POSIX, the native process-tree walk on Windows — wired to exec.Cmd's Cancel
// hook via procguard.ConfigureProcessTreeCancel. A runaway action that forks
// grandchildren is killed TREE-WIDE, not just at the direct child, on both
// platforms.
//
// Generation intent: gen/second-next architectural exploration (#2014). This is
// an OPTION behind an explicit import boundary — nothing in the default
// serve/guard/dispatch path constructs a ToolExec; it is the seam microagent's
// doc.go named as the prerequisite ("this host must grow a subprocess ToolExec
// seam (#2003/#2014) before it can carry production agents"). Closing evidence
// for the generation frame:
//
//   - Promotion evidence: toolexec_test.go witnesses (a) the real registered
//     kernel floor DENYING an action so no subprocess is ever spawned, and
//     (b) a runaway subprocess that forks a grandchild being killed TREE-WIDE on
//     timeout (the grandchild never reaches its post-grace marker). Promote once
//     the microagent Host drives real agent loops whose tool actions are shell
//     commands (the #2001 RunArm extraction) and a density measurement (#2033)
//     confirms subprocess-per-action is the isolation level production needs.
//   - Demotion / retirement criteria: retire this backend if the goroutine floor
//     (or a cheaper in-process sandbox) proves sufficient isolation for the agent
//     loops the host carries — i.e. no action needs an OS process boundary — so
//     the per-action spawn cost buys nothing, OR if the isolation floor demands a
//     stronger boundary than a process tree (a container/VM per action), which a
//     bare os/exec backend cannot provide.
//   - Invalidating assumption: this minimal backend proceeds ONLY on a kernel
//     Allow — a Transform (redact-and-dispatch), RequireWitness, or Quarantine
//     verdict is treated as a refusal, not honored. If real agent actions need the
//     floor's arg-rewrite path (a secret redacted before exec), that assumption is
//     invalid and Run must grow a Transform arm that re-derives argv from the
//     rewritten args before spawning.
type ToolExec struct {
	floor KernelFloor
}

// KernelFloor is the in-process kernel adjudication seam ToolExec routes every
// action through before exec. It is satisfied by *kernel.Kernel (whose Decide
// folds the Adjudicator chain and returns the resolved Verdict); an explicit
// interface keeps the backend testable with an injected floor and keeps the
// import a type dependency, not a construction dependency.
type KernelFloor interface {
	Decide(ctx context.Context, c *abi.ToolCall) abi.Verdict
}

// DefaultActionTimeout bounds a single action when ToolAction.Timeout is unset.
const DefaultActionTimeout = 30 * time.Second

// waitDelay backstops exec.Cmd.Wait: after the process is killed, if a
// grandchild kept an inherited stdio pipe open, Wait would block on the copy
// goroutine forever. WaitDelay forces Wait to close the pipes and return. In the
// tree-kill success path the whole group dies and the pipes close at once, so
// Wait returns promptly and this delay never elapses.
const waitDelay = 10 * time.Second

// Structured refusals for the exec backend.
var (
	ErrNilFloor     = errors.New("microagent: NewToolExec requires the kernel floor (nil KernelFloor)")
	ErrNoProgram    = errors.New("microagent: ToolAction has no Path to exec")
	ErrActionDenied = errors.New("microagent: kernel floor refused the action before exec")
)

// ToolAction is one tool call to run as a subprocess. Tool + Args are the
// adjudication token the kernel floor decides on; Path + Argv are what actually
// executes once (and only if) the floor Allows it. Keeping them distinct is
// deliberate: the floor gates the logical CALL, and the caller is responsible
// for deriving a faithful argv from the same args — a mismatch is the caller's
// bug, not a hole the backend can close.
type ToolAction struct {
	Tool    string         // logical tool name the kernel floor adjudicates
	Args    map[string]any // decoded args the floor inspects (inlined into the call)
	Path    string         // the program to exec (a compiled tool/binary)
	Argv    []string       // program arguments (Path is argv[0] implicitly)
	Stdin   []byte         // fed to the subprocess stdin (nil => no stdin)
	Timeout time.Duration  // per-action timeout; <=0 => DefaultActionTimeout
}

// ToolResult is the captured outcome of one action.
type ToolResult struct {
	Verdict  abi.Verdict // the kernel-floor verdict (always set)
	Ran      bool        // true iff the subprocess was actually started
	Stdout   []byte      // captured stdout
	Stderr   []byte      // captured stderr
	ExitCode int         // process exit code (-1 if killed/never-exited)
	TimedOut bool        // the per-action deadline fired and the tree was killed
	Killed   bool        // the process tree was killed (timeout OR parent cancel)
}

// NewToolExec builds a subprocess backend over the in-process kernel floor.
func NewToolExec(floor KernelFloor) (*ToolExec, error) {
	if floor == nil {
		return nil, ErrNilFloor
	}
	return &ToolExec{floor: floor}, nil
}

// Run adjudicates the action through the kernel floor and, ONLY on a kernel
// Allow, runs it as a process-tree-killable subprocess. The return contract:
//
//   - denied by the floor            -> (result with Verdict, Ran=false), ErrActionDenied
//   - no program to exec / bad args  -> (result with Verdict), a wrapped error
//   - the subprocess ran (or timed   -> (result with the captured outcome), nil
//     out and was killed tree-wide)     — inspect Result.TimedOut / ExitCode
//
// A timeout is a SUCCESSFUL run that was reaped, not a Run error: the caller
// reads it off Result.TimedOut, so a runaway action is a normal, observable
// outcome rather than an exception.
func (t *ToolExec) Run(ctx context.Context, act ToolAction) (ToolResult, error) {
	call, err := toolCall(act)
	if err != nil {
		return ToolResult{}, err
	}

	// (M18) Route through the in-process kernel floor BEFORE exec. This is the
	// whole point of the backend: adjudication is the gate the subprocess is
	// behind, so a refusal costs zero processes.
	v := t.floor.Decide(ctx, call)
	res := ToolResult{Verdict: v}
	if v.Kind != abi.VerdictAllow {
		return res, fmt.Errorf("%w: %s (by %s)", ErrActionDenied, verdictReason(v), v.By)
	}
	if act.Path == "" {
		return res, ErrNoProgram
	}

	// Per-action timeout: the subprocess runs under a deadline-bounded context so
	// a runaway action is reaped without a caller watchdog.
	timeout := act.Timeout
	if timeout <= 0 {
		timeout = DefaultActionTimeout
	}
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(runCtx, act.Path, act.Argv...)
	// Reuse the ONE cross-platform process-tree reaper (POSIX Setpgid+kill(-pgid),
	// Windows native tree walk) — wired to the context-cancel hook, so a timeout or
	// a parent cancel kills the whole tree, grandchildren included.
	procguard.ConfigureProcessTreeCancel(cmd)
	cmd.WaitDelay = waitDelay

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if len(act.Stdin) > 0 {
		cmd.Stdin = bytes.NewReader(act.Stdin)
	}

	res.Ran = true
	runErr := cmd.Run()
	res.Stdout = stdout.Bytes()
	res.Stderr = stderr.Bytes()
	if cmd.ProcessState != nil {
		res.ExitCode = cmd.ProcessState.ExitCode()
	} else {
		res.ExitCode = -1
	}

	// Classify the outcome. A deadline fire means the tree was killed on timeout;
	// a parent-context cancel means the tree was killed on cancel. Either way the
	// non-nil runErr is the EXPECTED signal-kill, not a Run failure — it is
	// reported through Result, not returned.
	switch {
	case runCtx.Err() == context.DeadlineExceeded:
		res.TimedOut = true
		res.Killed = true
	case ctx.Err() != nil:
		res.Killed = true
	}
	_ = runErr
	return res, nil
}

// toolCall builds the adjudication envelope. Args are inlined (RefInline) so the
// floor decodes them without a Resolver round-trip; a nil Args map decodes as an
// empty object.
func toolCall(act ToolAction) (*abi.ToolCall, error) {
	inline := []byte("{}")
	if act.Args != nil {
		b, err := json.Marshal(act.Args)
		if err != nil {
			return nil, fmt.Errorf("microagent: marshal action args: %w", err)
		}
		inline = b
	}
	return &abi.ToolCall{
		Tool: act.Tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: inline, Len: int64(len(inline))},
	}, nil
}

// verdictReason renders a bounded, human-readable reason for a non-Allow verdict.
func verdictReason(v abi.Verdict) string {
	switch v.Kind {
	case abi.VerdictDeny:
		return abi.ReasonName(v.Reason)
	case abi.VerdictTransform:
		return "transform (arg-rewrite not honored by the exec backend)"
	case abi.VerdictRequireWitness:
		return "require-witness"
	case abi.VerdictQuarantine:
		return "quarantine"
	case abi.VerdictDefer:
		return "defer (no rung admitted)"
	case abi.VerdictIndeterminate:
		return "indeterminate"
	default:
		return "not-allowed"
	}
}
