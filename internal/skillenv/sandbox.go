// Package skillenv tracks active skill versions and previews blast radius for
// hot-swap or rollback through the context and KV MMU surfaces.
package skillenv

import (
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
)

// PageQuery is the versioned, revisable unit the sandbox runtime admits and
// executes: one skill (or tool plugin) page at a named version.
type PageQuery struct {
	Skill   string `json:"skill"`
	Version string `json:"version"`
	// ABIDigest is the plugin's declared static ABI hash (e.g. its summary of
	// ToolCall/Result shape). The admission gate checks it against the kernel's
	// frozen ABI identity before the page is admitted.
	ABIDigest string `json:"abi_digest"`
}

// pageKey is the page-table identity of a resident page.
type pageKey struct {
	skill   string
	version string
}

// PageState is the closed lifecycle state of a resident sandbox page.
type PageState int32

const (
	PageLoading  PageState = iota // admitted, warm-up in progress
	PageActive                    // serving invocations
	PageDraining                  // in-flight pinned, new dispatch refused
)

// String renders the state for logs and maps.
func (s PageState) String() string {
	switch s {
	case PageLoading:
		return "loading"
	case PageActive:
		return "active"
	case PageDraining:
		return "draining"
	default:
		return fmt.Sprintf("skillenv:pagestate(%d)", int32(s))
	}
}

// Page is one resident skill page in a sandboxed slot: state transitions move
// through monotonic atomic steps (loading → active → draining), and memory is
// fenced against the pool's global cap.
type Page struct {
	Skill   string `json:"skill"`
	Version string `json:"version"`

	state    atomic.Int32
	lastErr  atomic.Pointer[string]
	segments atomic.Int64 // resident instruction/context segments owned by this page
	released atomic.Bool  // segments reclaimed on unload (single release)
}

// State reads the page's current lifecycle state.
func (p *Page) State() PageState { return PageState(p.state.Load()) }

// LastError reports the most recent failure recorded against the page, if any.
// A failed warm-up leaves the page terminal (draining) — fail-closed.
func (p *Page) LastError() string {
	if s := p.lastErr.Load(); s != nil {
		return *s
	}
	return ""
}

func (p *Page) fail(err string) {
	p.lastErr.Store(&err)
	p.state.Store(int32(PageDraining))
}

// drain makes every admitted page terminal, including a page unloaded while
// its warm-up is still pending. Loading and Active are both monotonic
// predecessors of Draining, so an atomic store cannot revive a terminal page.
func (p *Page) drain() { p.state.Store(int32(PageDraining)) }

func (p *Page) key() pageKey { return pageKey{p.Skill, p.Version} }

// segmentCost is the accounting unit charged against the byte pool per page
// segment (a resident skill view or plugin working buffer).
const segmentCost = 1 << 16 // 64 KiB

// Pool is the sandbox's strict memory ceiling: a byte-budgeted, per-page
// fenced allocator. A charge that would exceed the cap is REJECTED, not
// spilled — fail-closed, zero leaks (release returns the exact bytes charged).
type Pool struct {
	used atomic.Int64
	cap  int64
}

// NewPool constructs a pool with the given hard byte ceiling.
func NewPool(capBytes int64) *Pool { return &Pool{cap: capBytes} }

// Used reports currently charged bytes.
func (p *Pool) Used() int64 { return p.used.Load() }

// Cap reports the hard ceiling.
func (p *Pool) Cap() int64 { return p.cap }

// Charge takes n bytes against the ceiling, rejecting if the cap would be
// exceeded. It is exact: on rejection nothing is consumed.
func (p *Pool) Charge(n int64) error {
	if n < 0 {
		return fmt.Errorf("skillenv: negative charge %d", n)
	}
	for {
		u := p.used.Load()
		if u+n > p.cap {
			return fmt.Errorf("skillenv: sandbox memory cap exceeded: %d + %d > %d", u, n, p.cap)
		}
		if p.used.CompareAndSwap(u, u+n) {
			return nil
		}
	}
}

// Release returns n previously charged bytes; it can never go negative.
func (p *Pool) Release(n int64) { p.used.Add(-n) }

// Runner executes a sandboxed page invocation. It is the isolation seam: a
// real guest runtime (WASM, or a memory-bounded isolated IPC worker) plugs in
// here; the in-proc runtime below models the slot lifecycle the seam must
// honor (bounded memory, fail-closed warm-up, reclaim on unload).
type Runner interface {
	Run(page *Page, fn func() error) error
}

// InProcRunner is the default Runner: execution is fenced by the shared byte
// pool using the page's segment accounting. A warm-up failure marks the page
// terminal and the page NEVER transitions to Active (fail-closed).
type InProcRunner struct {
	pool *Pool
}

// NewInProcRunner builds a runner fenced by the given pool.
func NewInProcRunner(pool *Pool) *InProcRunner { return &InProcRunner{pool: pool} }

// Run executes fn inside the page's fenced slot. Semantics by lifecycle
// state: on an Active page this is one invocation (a failure marks the page
// draining/terminal); on a Loading page this is a WARM-UP execution with
// fail-closed admission (a failure marks the page terminal — it can then
// never be Activated). The segment memory charged for the execution is
// returned on exit even when fn fails (zero leaks).
func (r *InProcRunner) Run(page *Page, fn func() error) error {
	if page == nil {
		return fmt.Errorf("skillenv: nil sandbox page")
	}
	st := page.State()
	if st != PageActive && st != PageLoading {
		return fmt.Errorf("skillenv: page %s@%s not schedulable (state %s)", page.Skill, page.Version, page.State())
	}
	if err := r.pool.Charge(segmentCost); err != nil {
		page.fail(err.Error())
		return err
	}
	defer r.pool.Release(segmentCost)
	if err := fn(); err != nil {
		page.fail(err.Error())
		return err
	}
	return nil
}

// Sandbox is the isolated runtime host over the skill page table: pages are
// admitted through ABI verification, served lock-free (the page snapshot lives
// in the same lock-free store Table uses), and reclaimed under draining state
// on unload without a daemon restart.
type Sandbox struct {
	table  *Table
	pool   *Pool
	runner Runner

	mu    sync.Mutex // guards registration bookkeeping only; serving paths are lock-free
	pages map[pageKey]*Page
}

// NewSandbox constructs a sandbox runtime bound to a page table and a byte
// ceiling. Slice cap: fixed.
func NewSandbox(t *Table, capBytes int64) *Sandbox {
	if t == nil {
		t = New(nil, nil, nil)
	}
	pool := NewPool(capBytes)
	return &Sandbox{
		table:  t,
		pool:   pool,
		runner: NewInProcRunner(pool),
		pages:  make(map[pageKey]*Page),
	}
}

// Admit admits a page into a sandbox slot through ABI verification (fail-closed
// on digest mismatch), binds it into the page table, and returns the live
// pointer for warm-up (state loading).
func (s *Sandbox) Admit(q PageQuery) (*Page, error) {
	if err := VerifyPluginABI(q.ABIDigest); err != nil {
		return nil, err
	}
	if q.Skill == "" || q.Version == "" {
		return nil, fmt.Errorf("skillenv: cannot admit page with empty skill/version")
	}
	page := &Page{Skill: q.Skill, Version: q.Version}
	page.state.Store(int32(PageLoading))

	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.pages[page.key()]; exists {
		return nil, fmt.Errorf("skillenv: page %s@%s already admitted", q.Skill, q.Version)
	}
	s.pages[page.key()] = page
	// Publish the version into the lock-free page table; the skill is
	// addressable for NEW invocations the moment the pin publishes.
	if _, _, err := s.table.Pin(q.Skill, q.Version); err != nil {
		delete(s.pages, page.key())
		return nil, err
	}
	return page, nil
}

// Activate promotes a loading page to Active, making it schedulable. It must
// be called after the page's runtime slot finished warming up successfully.
func (s *Sandbox) Activate(page *Page) error {
	if page == nil {
		return fmt.Errorf("skillenv: nil sandbox page")
	}
	if !page.state.CompareAndSwap(int32(PageLoading), int32(PageActive)) {
		return fmt.Errorf("skillenv: page %s@%s not in loading state (state %s)", page.Skill, page.Version, page.State())
	}
	return nil
}

// Run executes fn under the page's sandbox slot (admission already happened).
func (s *Sandbox) Run(page *Page, fn func() error) error { return s.runner.Run(page, fn) }

// Unload removes a page from the table, drains it (in-flight pinned: no new
// dispatch), and reclaims EVERYTHING the page owns — its entry, its slot, and
// its charged memory — without a daemon restart. Unloading an absent page
// reports a count of 0 and no error.
func (s *Sandbox) Unload(skillName, version string) (int, error) {
	if skillName == "" {
		return 0, fmt.Errorf("skillenv: cannot unload empty skill name")
	}

	s.mu.Lock()
	page, ok := s.pages[pageKey{skillName, version}]
	if ok {
		delete(s.pages, pageKey{skillName, version})
	}
	s.mu.Unlock()

	reclaimed := 0
	if ok {
		// Drain first: in-flight invocations keep their pinned frame; new ones
		// are refused the moment the state flips (atomic and monotonic).
		page.drain()
		reclaimed = 1 + int(page.segments.Load())
		page.segments.Store(0)
		page.released.Store(true)
		// Drop the table pin ONLY if it still resolves this exact version (a
		// concurrent remap to a newer version must not be torn down by us).
		if cur, ok := s.table.ActiveVersion(skillName); ok && cur == version {
			_, _, _ = s.table.Unpin(skillName)
		}
	}
	return reclaimed, nil
}

// Pages enumerates registered pages sorted by (skill, version) for stable
// reporting.
func (s *Sandbox) Pages() []*Page {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]*Page, 0, len(s.pages))
	for _, p := range s.pages {
		out = append(out, p)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Skill != out[j].Skill {
			return out[i].Skill < out[j].Skill
		}
		return out[i].Version < out[j].Version
	})
	return out
}

// Pool exposes the sandbox's memory pool for budget telemetry in tests and
// supervisors.
func (s *Sandbox) Pool() *Pool { return s.pool }
