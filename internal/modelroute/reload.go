// Hot-reload: follow an installed route Manifest file and atomically swap the live
// policy on a validated change, without restarting the server (#842).
//
// TWO PIECES, both pure-stdlib (the package keeps its zero-dependency contract — no
// fsnotify):
//
//   - Live is a concurrency-safe holder for the active Manifest. A classification
//     loads the WHOLE manifest pointer atomically, so a reload swaps the policy
//     without a torn read: every Route sees either the entire old manifest or the
//     entire new one, never a half-applied edit. This is the load-bearing property
//     the residency contract needs — a mis-route is a security boundary, so a
//     partially-swapped rule set must never be observable.
//
//   - Watcher polls the manifest file and reloads on a content change. A reload that
//     fails to parse or validate is REJECTED: the last-good manifest stays installed
//     (the fail-loud startup contract extended to reload — a malformed edit never
//     silently degrades the routing surface to a default) and the rejection is
//     surfaced via the OnEvent callback so an operator can confirm what happened.
//
// Polling, not inotify/fsnotify: a once-per-interval os.Stat over a single small file
// is negligible, keeps the package dependency-free, and works identically on every
// OS the kernel targets. The change gate is cheap (stat size+mtime) with a content
// compare behind it, so a touch with identical bytes is a no-op, not a spurious swap.
package modelroute

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

// Live is a concurrency-safe holder for the active routing Manifest. The whole
// manifest is published behind a single atomic pointer, so a hot-swap is seen by a
// concurrent Route as all-or-nothing — never a torn read. Build one with NewLive;
// the zero value is not usable (its pointer is nil and Route would panic).
type Live struct {
	cur     atomic.Pointer[Manifest]
	reloads atomic.Int64 // cumulative successful swaps (an operator-confirmable count)
	rejects atomic.Int64 // cumulative rejected (malformed) reload attempts
}

// NewLive installs m as the initial policy. m MUST be non-nil and already validated
// (gateway.New and LoadManifest both validate before this point); a nil manifest is
// a programming error the caller must avoid by not building a Live when routing is
// off.
func NewLive(m *Manifest) *Live {
	l := &Live{}
	l.cur.Store(m)
	return l
}

// Manifest returns the currently-installed policy (an atomic load). The returned
// pointer is immutable — a reload swaps to a fresh manifest, it never mutates an
// installed one — so a holder of this pointer keeps reading a consistent snapshot
// even across a concurrent reload.
func (l *Live) Manifest() *Manifest { return l.cur.Load() }

// Route classifies a Subject against the live policy. It loads the manifest pointer
// once and routes against that snapshot, so a concurrent Store cannot tear the
// decision.
func (l *Live) Route(s Subject) Decision { return l.cur.Load().Route(s) }

// Store atomically installs an already-validated manifest, returning the prior one.
// It increments the reload count. The caller is responsible for validation (Watcher
// only ever Stores a ParseManifest-checked manifest).
func (l *Live) Store(m *Manifest) *Manifest {
	prev := l.cur.Swap(m)
	l.reloads.Add(1)
	return prev
}

// Reloads reports the cumulative number of successful hot-swaps since construction.
func (l *Live) Reloads() int64 { return l.reloads.Load() }

// Rejects reports the cumulative number of rejected (malformed) reload attempts.
func (l *Live) Rejects() int64 { return l.rejects.Load() }

// markReject records a rejected reload and returns the new reject count.
func (l *Live) markReject() int64 { return l.rejects.Add(1) }

// ReloadEvent reports the outcome of one reload attempt, so the host can log a line
// or bump a metric (the observability the swap requires — an operator must be able
// to confirm a reload landed, or that a bad edit was rejected). Err != nil means the
// edit was REJECTED and the last-good manifest is still installed.
type ReloadEvent struct {
	Path     string // the watched manifest path
	Changed  bool   // the file content differed from the installed policy
	Reloaded bool   // a new manifest was atomically installed
	Reloads  int64  // cumulative successful reloads (after this event)
	Rejects  int64  // cumulative rejected reload attempts (after this event)
	Err      error  // non-nil => malformed edit rejected; last-good kept
}

// fileSig is the cheap change gate: a manifest whose size and mtime are unchanged is
// assumed unchanged, so the common (no-edit) poll never reads or re-parses the file.
type fileSig struct {
	size    int64
	modNano int64
}

// DefaultReloadInterval is the poll cadence when a non-positive interval is given.
// One second is imperceptible operator latency for a config edit and a negligible
// cost (one stat of one small file).
const DefaultReloadInterval = time.Second

// Watcher follows a manifest file and hot-swaps a Live on a validated content
// change. Construct it with NewWatcher; drive it with Run (the polling loop) or
// Reload (a single forced check, also the unit-test seam).
type Watcher struct {
	path     string
	live     *Live
	interval time.Duration
	onEvent  func(ReloadEvent)

	mu       sync.Mutex
	lastSig  fileSig
	lastGood []byte // bytes of the currently-installed manifest (content gate)
}

// NewWatcher builds a watcher for path that swaps live on change. interval <= 0 uses
// DefaultReloadInterval. onEvent (may be nil) is called for every reload and every
// rejection, so the host can log / meter it. The watcher seeds its baseline from the
// file at construction, so the first poll of an unedited file is a no-op (no
// spurious startup reload).
func NewWatcher(path string, live *Live, interval time.Duration, onEvent func(ReloadEvent)) *Watcher {
	if interval <= 0 {
		interval = DefaultReloadInterval
	}
	w := &Watcher{path: path, live: live, interval: interval, onEvent: onEvent}
	// Seed the baseline only when disk still matches the installed policy. The host
	// loads the manifest before constructing the watcher; if the file changes in that
	// small window, treating the new bytes as "last-good" would make the edit invisible.
	if b, err := os.ReadFile(path); err == nil {
		if disk, err := ParseManifest(b); err == nil && sameManifest(live.Manifest(), &disk) {
			w.lastGood = b
			if fi, err := os.Stat(path); err == nil {
				w.lastSig = sigOf(fi)
			}
		}
	}
	return w
}

// Run polls the manifest file every interval until ctx is cancelled (the serve
// lifetime). It returns ctx.Err() so a caller can distinguish a clean shutdown.
func (w *Watcher) Run(ctx context.Context) error {
	t := time.NewTicker(w.interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-t.C:
			w.poll()
		}
	}
}

// poll does one cheap-gated reload check: if the file's size+mtime are unchanged it
// returns without reading; otherwise it reads and applies. A transient stat/read
// failure (e.g. mid-rename of an atomically-replaced file) is treated as no-change —
// the last-good policy stays installed and the next tick retries — so a reload is
// never half-applied from a torn write.
func (w *Watcher) poll() ReloadEvent {
	fi, err := os.Stat(w.path)
	if err != nil {
		return ReloadEvent{Path: w.path} // transient / deleted: keep last-good, retry
	}
	sig := sigOf(fi)
	w.mu.Lock()
	unchanged := sig == w.lastSig && w.lastGood != nil
	w.lastSig = sig
	w.mu.Unlock()
	if unchanged {
		return ReloadEvent{Path: w.path}
	}
	return w.applyFromFile()
}

// Reload forces a reload check now, bypassing the size+mtime gate so even an edit
// that left the mtime unchanged (coarse filesystem timestamp granularity) is caught
// by the content compare. It is the SIGHUP-style manual trigger and the unit-test
// seam. It returns the resulting event.
func (w *Watcher) Reload() ReloadEvent {
	if fi, err := os.Stat(w.path); err == nil {
		w.mu.Lock()
		w.lastSig = sigOf(fi)
		w.mu.Unlock()
	}
	return w.applyFromFile()
}

// applyFromFile reads the file and, if its content differs from the installed
// policy, parses+validates it and atomically swaps it in. A parse/validate failure
// is rejected (last-good kept, rejects incremented). Identical content is a no-op.
func (w *Watcher) applyFromFile() ReloadEvent {
	b, err := os.ReadFile(w.path)
	if err != nil {
		return ReloadEvent{Path: w.path} // transient: keep last-good
	}
	w.mu.Lock()
	same := w.lastGood != nil && bytes.Equal(b, w.lastGood)
	w.mu.Unlock()
	if same {
		return ReloadEvent{Path: w.path}
	}
	m, perr := ParseManifest(b)
	if perr != nil {
		// Malformed edit: reject it and KEEP the last-good manifest installed. The
		// last-good bytes are deliberately not advanced, so once the operator fixes
		// the file the corrected content still reads as "changed" and reloads.
		ev := ReloadEvent{
			Path:    w.path,
			Changed: true,
			Reloads: w.live.Reloads(),
			Rejects: w.live.markReject(),
			Err:     fmt.Errorf("route-manifest reload rejected (last-good kept): %w", perr),
		}
		w.emit(ev)
		return ev
	}
	mp := &m
	w.live.Store(mp)
	w.mu.Lock()
	w.lastGood = b
	w.mu.Unlock()
	ev := ReloadEvent{
		Path:     w.path,
		Changed:  true,
		Reloaded: true,
		Reloads:  w.live.Reloads(),
		Rejects:  w.live.Rejects(),
	}
	w.emit(ev)
	return ev
}

func (w *Watcher) emit(ev ReloadEvent) {
	if w.onEvent != nil {
		w.onEvent(ev)
	}
}

func sigOf(fi os.FileInfo) fileSig {
	return fileSig{size: fi.Size(), modNano: fi.ModTime().UnixNano()}
}

func sameManifest(a, b *Manifest) bool {
	if a == nil || b == nil {
		return a == b
	}
	return bytes.Equal(a.JSON(), b.JSON())
}
