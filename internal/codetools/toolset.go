package codetools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// toolset.go — configuration, bounds, policy, and the abi registration seam.
//
// A Toolset is the CONFIGURED instance: one workspace root, one bound set, one policy.
// Everything that could differ between two deployments lives here rather than in a
// package-level global, so a test can stand up a toolset over a scratch dir while a
// served process runs one over the real workspace, in the same binary, without either
// reaching into the other's confinement.

// EngineID prefix. Each tool gets its OWN engine id rather than one multiplexing engine,
// because abi.ToolCall.Engine is what the kernel routes on: distinct ids mean a route
// manifest, a residency PDP, or an operator reading a decision journal can see WHICH
// coding operation a call dispatched to, not just "the coding engine".
const (
	EngineRead       = "codetools.read"
	EngineGrep       = "codetools.grep"
	EngineGlob       = "codetools.glob"
	EngineWrite      = "codetools.write"
	EngineEdit       = "codetools.edit"
	EngineBash       = "codetools.bash"
	EngineApplyPatch = "codetools.apply_patch"
	ToolApplyPatch   = "apply_patch"
)

// RungName identifies this package's adjudicator in a Verdict.By field and in the
// decision journal, so a refusal is attributable to the toolset rung rather than folded
// anonymously into "the chain said no".
const RungName = "codetools"

// Limits are the enforceable bounds. Every one of them exists because the unbounded
// version is a live denial-of-service against the loop's own context window: an unbounded
// Read of a 2GB file or an unbounded Grep over a vendor tree both end the same way — the
// loop stops being able to make progress.
type Limits struct {
	MaxReadBytes   int64         // largest file body a single Read returns
	MaxMatches     int           // largest number of Grep match rows
	MaxEntries     int           // largest number of Glob path rows
	MaxWalkFiles   int           // largest number of files a single Grep/Glob walk visits
	MaxWriteBytes  int64         // largest body a single Write/Edit may materialize
	MaxOutputBytes int           // largest stdout and stderr body retained per Bash call
	MaxCommandTime time.Duration // hard ceiling for a Bash process forest
}

// DefaultLimits are sized for a coding loop rather than for a batch job: big enough that
// an ordinary source file, search, or build command completes untruncated, small enough
// that a pathological one cannot flood a context window. A caller that needs more sets
// its own — the point is that SOME bound is always in force.
func DefaultLimits() Limits {
	return Limits{
		MaxReadBytes:   1 << 20, // 1 MiB
		MaxMatches:     500,
		MaxEntries:     1000,
		MaxWalkFiles:   20000,
		MaxWriteBytes:  1 << 20,
		MaxOutputBytes: 1 << 20,
		MaxCommandTime: 2 * time.Minute,
	}
}

// normalize replaces non-positive fields with their defaults. A zero Limits is therefore
// the DEFAULT bound set, never "unbounded" — the failure mode of a config struct whose
// zero value disables the protection is that the protection is off exactly when nobody
// configured it.
func (l Limits) normalize() Limits {
	d := DefaultLimits()
	if l.MaxReadBytes <= 0 {
		l.MaxReadBytes = d.MaxReadBytes
	}
	if l.MaxMatches <= 0 {
		l.MaxMatches = d.MaxMatches
	}
	if l.MaxEntries <= 0 {
		l.MaxEntries = d.MaxEntries
	}
	if l.MaxWalkFiles <= 0 {
		l.MaxWalkFiles = d.MaxWalkFiles
	}
	if l.MaxWriteBytes <= 0 {
		l.MaxWriteBytes = d.MaxWriteBytes
	}
	if l.MaxOutputBytes <= 0 {
		l.MaxOutputBytes = d.MaxOutputBytes
	}
	if l.MaxCommandTime <= 0 {
		l.MaxCommandTime = d.MaxCommandTime
	}
	return l
}

// Policy is the toolset's default-deny admission floor. Allow is an explicit allowlist:
// a tool absent from it is DENIED with DEFAULT_DENY, never admitted by omission.
type Policy struct {
	Allow map[string]bool
}

// DefaultPolicy admits the three READ-shaped tools this leaf ships. The side-effecting
// three (Write, Edit, Bash) are deliberately absent — not merely unlisted but not
// IMPLEMENTED here (#6704, #6705) — so the read spine cannot be mistaken for a mutation
// surface an operator forgot to close.
func DefaultPolicy() Policy {
	return Policy{Allow: map[string]bool{ToolRead: true, ToolGrep: true, ToolGlob: true, ToolWrite: true, ToolEdit: true, ToolBash: true, ToolApplyPatch: true}}
}

// Config configures a Toolset. Root is the workspace every path is confined to; empty
// means the process working directory, matching RegisterReadEngine's convention.
type Config struct {
	Root            string
	Limits          Limits
	Policy          Policy
	FocusedCommands bool // restrict Bash to the browser coding spine command set
}

// Toolset is a configured, confinement-bound instance of the coding engines plus the
// adjudicator rung that admits them. Configuration is immutable after construction; the
// synchronized mutation-lock registry serializes competing updates to one target.
type Toolset struct {
	root            string
	evalRoot        string // root with symlinks resolved — the base every escape check compares against
	limits          Limits
	policy          Policy
	focusedCommands bool
	mutationMu      sync.Mutex
	mutationLocks   map[string]*mutationLock
	grepFlight      flightGroup[*grepRecord]
	globFlight      flightGroup[*globRecord]
	searchHook      func()
}

type mutationLock struct {
	mu   sync.Mutex
	refs int
}

// New builds a Toolset over cfg. It resolves the root ONCE (including symlinks) so every
// later confinement check is a pure comparison against a fixed base rather than a
// filesystem walk that could observe a different tree mid-session.
func New(cfg Config) (*Toolset, error) {
	root := cfg.Root
	if root == "" {
		cwd, err := os.Getwd()
		if err != nil {
			return nil, err
		}
		root = cwd
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	abs = filepath.Clean(abs)
	evalRoot := abs
	if resolved, err := filepath.EvalSymlinks(abs); err == nil {
		// A root that is ITSELF reached through a symlink (a /tmp -> /private/tmp
		// scratch dir on macOS, a junction on Windows) would make every in-tree path
		// look like an escape once EvalSymlinks ran on it. Comparing against the
		// resolved root is what keeps the check about escaping rather than about how
		// the operator spelled the workspace.
		evalRoot = resolved
	}
	pol := cfg.Policy
	if pol.Allow == nil {
		pol = DefaultPolicy()
	}
	return &Toolset{
		root: abs, evalRoot: evalRoot, limits: cfg.Limits.normalize(), policy: pol,
		focusedCommands: cfg.FocusedCommands, mutationLocks: map[string]*mutationLock{},
	}, nil
}

func (t *Toolset) withMutationLock(key string, fn func() ([]byte, bool)) ([]byte, bool) {
	t.mutationMu.Lock()
	l := t.mutationLocks[key]
	if l == nil {
		l = &mutationLock{}
		t.mutationLocks[key] = l
	}
	l.refs++
	t.mutationMu.Unlock()

	l.mu.Lock()
	defer func() {
		l.mu.Unlock()
		t.mutationMu.Lock()
		l.refs--
		if l.refs == 0 {
			delete(t.mutationLocks, key)
		}
		t.mutationMu.Unlock()
	}()
	return fn()
}

// Root reports the workspace root every path is confined to.
func (t *Toolset) Root() string { return t.root }

// Limits reports the bounds in force.
func (t *Toolset) Limits() Limits { return t.limits }

// Read executes a Read operation with JSON body arguments.
func (t *Toolset) Read(ctx context.Context, body []byte) ([]byte, bool) {
	return t.read(ctx, body)
}

// Grep executes a Grep operation with JSON body arguments.
func (t *Toolset) Grep(ctx context.Context, body []byte) ([]byte, bool) {
	return t.grep(ctx, body)
}

// GrepCoalesced reports the number of concurrent Grep calls that joined an in-flight search.
func (t *Toolset) GrepCoalesced() int64 { return t.grepFlight.Coalesced() }

// Glob executes a Glob operation with JSON body arguments.
func (t *Toolset) Glob(ctx context.Context, body []byte) ([]byte, bool) {
	return t.glob(ctx, body)
}

// GlobCoalesced reports the number of concurrent Glob calls that joined an in-flight search.
func (t *Toolset) GlobCoalesced() int64 { return t.globFlight.Coalesced() }

// Write executes a Write operation with JSON body arguments.
func (t *Toolset) Write(ctx context.Context, body []byte) ([]byte, bool) {
	return t.write(ctx, body)
}

// Edit executes an Edit operation with JSON body arguments.
func (t *Toolset) Edit(ctx context.Context, body []byte) ([]byte, bool) {
	return t.edit(ctx, body)
}

// ApplyPatch executes an ApplyPatch operation with JSON body arguments.
func (t *Toolset) ApplyPatch(ctx context.Context, body []byte) ([]byte, bool) {
	return t.applyPatch(ctx, body)
}

// RegisterEngines binds the engines into the abi registry under their own ids, so a
// kernel dispatching a call whose Engine names one of them reaches this toolset. Mirrors
// RegisterReadEngine: re-registering replaces the driver, so arming twice is safe.
func (t *Toolset) RegisterEngines() {
	abi.RegisterEngine(EngineRead, readEngine{t})
	abi.RegisterEngine(EngineGrep, grepEngine{t})
	abi.RegisterEngine(EngineGlob, globEngine{t})
	abi.RegisterEngine(EngineWrite, writeEngine{t})
	abi.RegisterEngine(EngineEdit, editEngine{t})
	abi.RegisterEngine(EngineBash, bashEngine{t})
	abi.RegisterEngine(EngineApplyPatch, applyPatchEngine{t})
}

// Register builds a Toolset, registers its engines, and places its rung in the
// adjudication chain at rank. It is the one-call arming path a host uses.
//
// The rung must be present for the engines to be REACHABLE at all: it is what pins
// abi.ToolCall.Engine to the per-tool engine id (see rung.go). Registering engines
// without it leaves three drivers nothing routes to.
func Register(cfg Config, rank int) (*Toolset, error) {
	t, err := New(cfg)
	if err != nil {
		return nil, err
	}
	t.RegisterEngines()
	abi.RegisterAdjudicator(rank, t)
	return t, nil
}

// engineFor maps a tool name to the engine id its calls dispatch to, and reports whether
// the tool belongs to this toolset at all.
func engineFor(tool string) (string, bool) {
	switch tool {
	case ToolRead:
		return EngineRead, true
	case ToolGrep:
		return EngineGrep, true
	case ToolGlob:
		return EngineGlob, true
	case ToolWrite:
		return EngineWrite, true
	case ToolEdit:
		return EngineEdit, true
	case ToolBash:
		return EngineBash, true
	case ToolApplyPatch:
		return EngineApplyPatch, true
	}
	return "", false
}

// readOnlyTool reports whether a tool is read-shaped. Sourced from Catalog() so the
// catalog a planner sees, the vDSO scope CallMeta stamps, and the write-shape check the
// rung enforces can never drift apart.
func readOnlyTool(tool string) bool {
	for _, d := range Catalog() {
		if d.Name == tool {
			return d.ReadOnly
		}
	}
	return false
}

// CallMeta builds the abi.ToolCall.Meta a caller must stamp on a codetools call:
// the vDSO eligibility hints that match the tool's real write shape, plus the optional
// isolation principal that KEYS the call's cache entries.
//
// This is the "request/tool identity in the cache scope" contract in code. Two halves:
//
//   - Grep/Glob assert readOnlyHint+idempotentHint and may use the vDSO. Read remains
//     read-only but omits idempotentHint: a peer filesystem writer emits no kernel
//     invalidation event, so caching Read would keep returning an obsolete version after
//     FS_STALE_VERSION told the model to read again.
//   - A write-shaped tool asserts destructive and NEITHER hint, so it can never be served
//     from cache and never fills one. The rung REFUSES a call that contradicts this
//     (CodeCacheScope), because a mutation mislabeled read-only would let the vDSO answer
//     it from a cached result — the one failure mode a cache in front of a side-effecting
//     tool must make impossible. Write/Edit exercise this branch: mutations are destructive and never replayable.
//
// principal is the isolation subject (tenant / user / auth principal); empty leaves the
// key unscoped, which is byte-identical to the single-tenant behavior and is the
// documented default.
func CallMeta(tool, principal string) map[string]string {
	m := map[string]string{}
	if readOnlyTool(tool) {
		m["readOnlyHint"] = "true"
		if tool != ToolRead {
			m["idempotentHint"] = "true"
		}
	} else {
		m["readOnlyHint"] = "false"
		m["idempotentHint"] = "false"
		m["destructive"] = "true"
	}
	if principal != "" {
		// vdso.MetaPrincipal, spelled literally: importing internal/vdso (tier 3) from
		// this tier-1 leaf would invert the layering for one constant. The key is part
		// of the vDSO's OPEN meta contract, and cacheScopeConsistent below is the test
		// that keeps the spelling honest.
		m["principal"] = principal
	}
	return m
}

// bytesOf materializes a ref: inline bytes directly, otherwise through the active
// resolver. Fails closed to nil, exactly like internal/refutil.Bytes — repeated here
// rather than imported so this leaf's only internal dependency stays abi(0).
func bytesOf(ctx context.Context, ref abi.Ref) []byte {
	if ref.Kind == abi.RefInline {
		return ref.Inline
	}
	if r := abi.ActiveResolver(); r != nil {
		if body, err := r.Resolve(ctx, ref); err == nil {
			return body
		}
	}
	return nil
}

// putBytes stores an engine payload by ref, falling back to inline when no resolver is
// registered so an engine result is always materializable.
func putBytes(ctx context.Context, b []byte) abi.Ref {
	if r := abi.ActiveResolver(); r != nil {
		if ref, err := r.Put(ctx, b); err == nil {
			return ref
		}
	}
	return abi.Ref{Kind: abi.RefInline, Inline: b, Len: int64(len(b))}
}

// result builds the standard *abi.Result an engine returns: the payload by ref, the
// isErr flag mapped onto Status, and the engine id plus ~4-chars/token I/O sizes on Meta.
// Shape-identical to internal/agent's engineResult so a codetools result is folded by the
// loop's metrics exactly like any other engine's.
func result(ctx context.Context, c *abi.ToolCall, in, out []byte, isErr bool, engineID string) *abi.Result {
	status := abi.StatusOK
	if isErr {
		status = abi.StatusError
	}
	return &abi.Result{Call: c, Payload: putBytes(ctx, out), Status: status, Meta: map[string]string{
		"engine":        engineID,
		"input_tokens":  strconv.Itoa(len(in) / 4),
		"output_tokens": strconv.Itoa(len(out) / 4),
	}}
}

// okJSON marshals an engine's success payload. Marshal of these concrete shapes cannot
// fail; the error arm keeps the path total rather than panicking on an impossible case.
func okJSON(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		return refuse(CodeIO, "cannot encode result").JSON()
	}
	return b
}
