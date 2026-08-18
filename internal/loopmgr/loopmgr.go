package loopmgr

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/parentdir"
	"io"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dormancy"
	"github.com/anthony-chaudhary/fak/internal/flock"
	"github.com/anthony-chaudhary/fak/internal/lifecycle"
)

const (
	SchemaEvent  = "fak.loop-event.v1"
	SchemaStatus = "fak.loop-status.v1"
)

type EventKind string

const (
	EventArmed     EventKind = "armed"
	EventFire      EventKind = "fire"
	EventAdmit     EventKind = "admit"
	EventStart     EventKind = "start"
	EventHeartbeat EventKind = "heartbeat"
	EventEnd       EventKind = "end"
	EventWitness   EventKind = "witness"
	EventNotify    EventKind = "notify"
)

type LoopState string

const (
	// Armed and Disabled are supervisor-only schedule states — a live served
	// sequence has no peer for either, so they are spelled here, not in the shared
	// leaf. The four common states are SOURCED from internal/lifecycle (the single
	// definition the served session shares) so the two machines cannot drift apart.
	StateArmed    LoopState = "armed"
	StateRunning  LoopState = lifecycle.TokenRunning
	StatePaused   LoopState = lifecycle.TokenPaused
	StateDraining LoopState = lifecycle.TokenDraining
	StateStopped  LoopState = lifecycle.TokenStopped
	StateDisabled LoopState = "disabled"
)

// Phase projects a LoopState onto the shared lifecycle skeleton. The bool is false
// for the supervisor-only extras (Armed/Disabled) and any unknown string — the
// projection is explicit about the extras, never a silent default. This is the
// supervisor half of the #912 "one machine" converter; internal/lifebridge
// composes it with the served-session half.
func (s LoopState) Phase() (lifecycle.Phase, bool) {
	switch s {
	case StateRunning:
		return lifecycle.Running, true
	case StatePaused:
		return lifecycle.Paused, true
	case StateDraining:
		return lifecycle.Draining, true
	case StateStopped:
		return lifecycle.Stopped, true
	}
	return 0, false
}

// LoopStateFromPhase lifts a shared lifecycle Phase into a LoopState. It is total
// over the four Phases (every shared state has a LoopState peer); an out-of-range
// Phase yields ("", false).
func LoopStateFromPhase(p lifecycle.Phase) (LoopState, bool) {
	switch p {
	case lifecycle.Running:
		return StateRunning, true
	case lifecycle.Paused:
		return StatePaused, true
	case lifecycle.Draining:
		return StateDraining, true
	case lifecycle.Stopped:
		return StateStopped, true
	}
	return "", false
}

type RunStatus string

const (
	StatusAdmitted           RunStatus = "admitted"
	StatusRefused            RunStatus = "refused"
	StatusRunning            RunStatus = "running"
	StatusClaimedDone        RunStatus = "claimed_done"
	StatusWitnessedDone      RunStatus = "witnessed_done"
	StatusWitnessRefused     RunStatus = "witness_refused"
	StatusWitnessUnavailable RunStatus = "witness_unavailable"
	StatusFailed             RunStatus = "failed"
	StatusCanceled           RunStatus = "canceled"
)

type EvidenceRef struct {
	Kind    string `json:"kind"`
	Ref     string `json:"ref"`
	Summary string `json:"summary,omitempty"`
	SHA256  string `json:"sha256,omitempty"`
}

type Event struct {
	Schema       string           `json:"schema"`
	Seq          uint64           `json:"seq"`
	TSUnixNano   int64            `json:"ts_unix_nano"`
	LoopID       string           `json:"loop_id"`
	RunID        string           `json:"run_id,omitempty"`
	Kind         EventKind        `json:"kind"`
	Source       string           `json:"source,omitempty"`
	Principal    string           `json:"principal,omitempty"`
	State        LoopState        `json:"state,omitempty"`
	Status       RunStatus        `json:"status,omitempty"`
	Reason       string           `json:"reason,omitempty"`
	Summary      string           `json:"summary,omitempty"`
	EvidenceRefs []EvidenceRef    `json:"evidence_refs,omitempty"`
	Metrics      map[string]int64 `json:"metrics,omitempty"`
	PrevHash     string           `json:"prev_hash,omitempty"`
	Hash         string           `json:"hash"`
}

type Option func(*appendConfig)

type appendConfig struct {
	clock       func() time.Time
	rotateBytes int64
}

// WithRotateBytes overrides the active-ledger size bound. Zero disables automatic
// rotation (useful for tests and explicit maintenance callers).
func WithRotateBytes(n int64) Option {
	return func(cfg *appendConfig) { cfg.rotateBytes = n }
}

func WithClock(clock func() time.Time) Option {
	return func(cfg *appendConfig) {
		if clock != nil {
			cfg.clock = clock
		}
	}
}

// ErrLedgerBusy is returned by Append when the cross-process append lock could not
// be acquired within the bounded wait. It is deliberately fail-closed: a forked chain
// (two unserialized writers stamping the same seq + prev_hash) is worse than a
// retriable error, so a contended-out Append never falls back to an unlocked write.
// Callers (the one-shot `fak loop ...` producer) can retry.
var ErrLedgerBusy = errors.New("loopmgr: loop ledger append lock is busy")

// appendLockWait bounds how long Append polls for the append lock before failing with
// ErrLedgerBusy. A local single-line append holds the lock for microseconds, so this
// should essentially never elapse; it only bounds a pathological stuck holder.
const appendLockWait = 2 * time.Second

func Append(path string, ev Event, opts ...Option) (Event, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return Event{}, errors.New("loop ledger path is required")
	}
	cfg := appendConfig{clock: time.Now, rotateBytes: DefaultRotateBytes}
	for _, opt := range opts {
		opt(&cfg)
	}

	// Validate the caller's event before touching the lock (cheap, no I/O).
	if err := validateNewEvent(ev); err != nil {
		return Event{}, err
	}

	if err := parentdir.Ensure(path, 0o755); err != nil {
		return Event{}, fmt.Errorf("create loop ledger dir: %w", err)
	}

	// Bound the hot active ledger at its shared write site. RotateIfDue first pays
	// only an O(1) stat; it acquires the same ledger lock and verifies/seals the
	// chain only after the threshold is crossed. Every producer therefore inherits
	// the bound without a per-tick reaper process.
	if cfg.rotateBytes > 0 {
		if _, err := RotateIfDue(path, cfg.rotateBytes); err != nil {
			return Event{}, fmt.Errorf("rotate loop ledger before append: %w", err)
		}
	}

	// Cross-process critical section: hold an OS advisory lock on a sidecar
	// <path>.lock fd across the whole read-compute-write, so seq/prev_hash are
	// derived from the TRUE tail under exclusion and two processes cannot fork the
	// chain. Correctness rests on recomputing under the lock, NOT on the lock being
	// acquired within the budget — on timeout we fail (ErrLedgerBusy), never proceed.
	var out Event
	err := withLedgerLock(path, appendLockWait, func() error {
		// Fast path: derive the tail (next seq, prev_hash) from the O(1) sidecar instead
		// of re-parsing + hash-verifying the whole ledger on every append. fastTail
		// size-checks the sidecar AND re-hashes the final line, so it trusts the recorded
		// tip only when the tail is byte-intact; full-chain verification of the earlier
		// prefix lives on the read side (the strict Load reader), per issue #3462.
		nextSeq, prevHash, ok := fastTail(path)
		if !ok {
			// Slow path: the tail sidecar is absent, stale, or no longer matches the
			// file size (first append after upgrade, a crashed writer, a rotation, or
			// an external edit). Fall back to the full tolerant scan — which also
			// repairs a broken tail — then refresh the sidecar after the write.
			existing, integ, err := LoadPrefix(path)
			if err != nil {
				return err
			}
			if integ.Broken {
				if !repairableAppendBreak(integ) {
					return integrityError(integ)
				}
				if err := os.Truncate(path, integ.ValidBytes); err != nil {
					return fmt.Errorf("repair loop ledger tail: %w", err)
				}
			}
			nextSeq = uint64(len(existing) + 1)
			prevHash = ""
			if len(existing) > 0 {
				prevHash = existing[len(existing)-1].Hash
			}
		}

		ev.Schema = SchemaEvent
		ev.Seq = nextSeq
		ev.TSUnixNano = cfg.clock().UTC().UnixNano()
		ev.PrevHash = prevHash
		ev.Hash = hashEvent(ev)

		f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			return fmt.Errorf("open loop ledger: %w", err)
		}
		defer f.Close()

		line, err := json.Marshal(ev)
		if err != nil {
			return fmt.Errorf("marshal loop event: %w", err)
		}
		if _, err := f.Write(append(line, '\n')); err != nil {
			return fmt.Errorf("append loop event: %w", err)
		}
		// Refresh the O(1) tail sidecar from the true post-write size so the next
		// append can skip the full re-scan. Best-effort: the line is already durably
		// appended, so a sidecar write failure must not fail the append — on a stat
		// fault we drop the sidecar instead, forcing the safe slow path next time.
		if fi, serr := f.Stat(); serr == nil {
			writeTailCache(path, tailCache{Seq: ev.Seq, Hash: ev.Hash, Size: fi.Size()})
		} else {
			_ = os.Remove(path + tailSuffix)
		}
		out = ev
		return nil
	})
	if err != nil {
		return Event{}, err
	}
	return out, nil
}

// tailSuffix names the O(1) tail sidecar Append uses to derive the next seq /
// prev_hash without re-reading the whole ledger. It is deliberately NOT a
// *.jsonl/.log/.err name so growthgate's candidate filter skips it: unlike the
// append-only ledger it caches, the sidecar is a fixed tiny file overwritten in
// place, never a grower.
const tailSuffix = ".tail"

// tailCache is the sidecar payload: the ledger's last committed seq + hash and the
// exact file size at which they were true. Append trusts it ONLY when the current file
// size still equals Size AND the final line still re-hashes to Hash (see fastTail).
// Size-equality alone is NOT a proof of tail-integrity: a same-size in-place edit
// changes content without changing size, and a crash can persist the file's size
// metadata (and the sidecar) while the final line's data bytes are lost — the NTFS
// ValidDataLength window for an un-fsync'd extending write, or ext4 data=writeback.
// Trusting size alone there would chain the next event onto a zeroed/forged tip and
// silently fork the chain. Re-hashing the last line closes that gap: any divergence
// (crashed writer, torn/zeroed tail, rotation, external edit, first append after
// upgrade, an empty zero-value sidecar) fails a guard and routes the append through the
// full LoadPrefix scan, which repairs and rewrites the sidecar. So the sidecar can only
// ever speed Append up, never make it wrong.
type tailCache struct {
	Seq  uint64 `json:"seq"`
	Hash string `json:"hash"`
	Size int64  `json:"size"`
}

// fastTail returns the next seq and prev_hash for an append in O(1) via the tail
// sidecar, or ok=false when the sidecar is missing/unreadable, no longer matches the
// file size, or whose recorded tip no longer matches the ledger's final line — in which
// case the caller must fall back to the full LoadPrefix scan (which also repairs a
// broken tail). Any error is folded into ok=false: the slow path re-opens the file and
// surfaces a genuine fault properly.
//
// The size guard is a necessary cheap filter but not sufficient on its own: a same-size
// in-place edit, or a crash that persists the file size ahead of the tail data bytes
// (NTFS ValidDataLength / ext4 writeback), keeps the size while corrupting the tip. So
// after the size check passes we re-hash the actual last line and require it to equal
// the recorded hash before trusting it. Reading one trailing line stays O(1) in the
// ledger length, so this preserves the amortized-O(1) append while closing the
// silent-fork hole trusting size alone would leave.
func fastTail(path string) (nextSeq uint64, prevHash string, ok bool) {
	tc, have := readTailCache(path)
	if !have {
		return 0, "", false
	}
	fi, err := os.Stat(path)
	if err != nil || fi.Size() != tc.Size {
		return 0, "", false
	}
	if !tailLineMatches(path, fi.Size(), tc.Hash) {
		return 0, "", false
	}
	return tc.Seq + 1, tc.Hash, true
}

// tailWindow bounds how many trailing bytes tailLineMatches reads to isolate the final
// ledger line. A single loop event serializes to well under this; if a line ever
// exceeds it, tailLineMatches conservatively returns false (routing to the slow scan),
// so the bound only ever costs a re-scan, never correctness.
const tailWindow = 64 << 10

// tailLineMatches reports whether the ledger's final line decodes to an event whose
// RECOMPUTED hash equals want. It reads only a bounded window at the end of the file,
// so it is O(1) in the ledger length. It recomputes hashEvent rather than trusting the
// line's stored hash field, so a same-size content edit (which leaves that field
// untouched) is caught; and any read or decode failure — including the zeroed or torn
// tail a crash can leave behind — returns false, routing Append to the safe full-scan
// slow path instead of chaining onto an unverified tip.
func tailLineMatches(path string, size int64, want string) bool {
	if size <= 0 {
		return false
	}
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	start := size - tailWindow
	if start < 0 {
		start = 0
	}
	buf := make([]byte, size-start)
	if _, err := f.ReadAt(buf, start); err != nil && !errors.Is(err, io.EOF) {
		return false
	}
	// Isolate the last complete line: drop the trailing record terminator, then take
	// the bytes after the preceding newline.
	b := bytes.TrimRight(buf, "\r\n")
	if i := bytes.LastIndexByte(b, '\n'); i >= 0 {
		b = b[i+1:]
	} else if start > 0 {
		// The window held no line boundary, so the final line is longer than tailWindow
		// (or the tail is one run of non-newline bytes). We cannot isolate a whole line
		// from it — fail the guard and let the slow path read the file properly.
		return false
	}
	if len(b) == 0 {
		return false
	}
	var ev Event
	if err := json.Unmarshal(b, &ev); err != nil {
		return false
	}
	return hashEvent(ev) == want
}

// readTailCache loads and sanity-checks the sidecar. A missing, truncated, or
// zero-value sidecar (Seq 0 / empty Hash — impossible for a real event, whose seqs
// start at 1) is reported as absent so the caller takes the safe slow path.
func readTailCache(path string) (tailCache, bool) {
	b, err := os.ReadFile(path + tailSuffix)
	if err != nil {
		return tailCache{}, false
	}
	var tc tailCache
	if err := json.Unmarshal(b, &tc); err != nil {
		return tailCache{}, false
	}
	if tc.Seq == 0 || tc.Hash == "" {
		return tailCache{}, false
	}
	return tc, true
}

// writeTailCache overwrites the sidecar in place. The payload is a single tiny
// write, so a torn write just fails the next Unmarshal and degrades to the slow
// path — no atomic-rename dance is needed (and rename-over is not atomic on
// Windows anyway). Callers invoke this while holding the ledger lock.
func writeTailCache(path string, tc tailCache) {
	b, err := json.Marshal(tc)
	if err != nil {
		return
	}
	_ = os.WriteFile(path+tailSuffix, b, 0o644)
}

// withLedgerLock runs fn while holding an exclusive cross-process advisory lock on
// <path>.lock. flock.TryLock is non-blocking, so it polls until the lock
// is free or wait elapses (then ErrLedgerBusy). The lock fd is closed on return, which
// also releases the OS lock (and the OS releases it if this process dies mid-write).
func withLedgerLock(path string, wait time.Duration, fn func() error) error {
	lockPath := path + ".lock"
	f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return fmt.Errorf("open loop ledger lock: %w", err)
	}
	defer f.Close()

	deadline := time.Now().Add(wait)
	for {
		lerr := flock.TryLock(f)
		if lerr == nil {
			break
		}
		if !errors.Is(lerr, flock.ErrLockBusy) {
			return fmt.Errorf("lock loop ledger: %w", lerr)
		}
		if time.Now().After(deadline) {
			return ErrLedgerBusy
		}
		time.Sleep(25 * time.Millisecond)
	}
	defer func() { _ = flock.Unlock(f) }()
	return fn()
}

func Load(path string) ([]Event, error) {
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("open loop ledger: %w", err)
	}
	defer f.Close()
	return loadReader(f)
}

func SnapshotFile(path string, now time.Time) (Status, error) {
	return snapshotLedger(path, now, Load)
}

// snapshotLedger reads a ledger with `load`, folds it into a Status, and stamps the
// ledger path on the result. SnapshotFile and SnapshotFileAll differ ONLY in which
// loader they hand it (active segment vs. every sealed segment too), so the fold and
// the stamping live here once.
func snapshotLedger(path string, now time.Time, load func(string) ([]Event, error)) (Status, error) {
	events, err := load(path)
	if err != nil {
		return Status{}, err
	}
	st := Summarize(events, now)
	st.LedgerPath = path
	return st, nil
}

// SnapshotFileAll is the rotation-aware cumulative sibling of SnapshotFile: it folds the
// FULL event history across every sealed segment AND the active segment (via LoadAll), so
// its per-loop totals stay correct once rotation has sealed old events — whereas
// SnapshotFile folds only the active segment and would undercount sealed history (and can
// underflow Concurrent()=Started-Ended when an in-flight run's EventStart was sealed while
// its EventEnd is still in the active segment). Use it for any consumer whose correctness
// depends on cumulative lifetime totals, or on an in-flight count that can span a rotation
// boundary. With no sealed segments it is exactly SnapshotFile(path) — LoadAll == Load when
// unrotated. It is O(total-history) today; the carried-snapshot seal optimization reduces
// it to O(active) later without changing this result or its signature.
func SnapshotFileAll(path string, now time.Time) (Status, error) {
	return snapshotLedger(path, now, LoadAll)
}

// Summarize folds an ordered event slice into a per-loop Status snapshot, starting from
// empty. It is the from-scratch special case of SummarizeFrom:
// Summarize(events, now) == SummarizeFrom(nil, events, now).
func Summarize(events []Event, now time.Time) Status {
	return SummarizeFrom(nil, events, now)
}

// SummarizeFrom is the RESUMABLE form of Summarize: it seeds the per-loop fold with a
// prior snapshot — e.g. the cumulative per-loop state as of a sealed rotation boundary —
// then folds `events` on top. Because Summarize is a pure per-loop left-fold whose entire
// accumulator IS the LoopSnapshot struct, the fold is exactly resumable:
//
//	SummarizeFrom(Summarize(A, now).Loops, B, now) == Summarize(append(A, B...), now)
//
// field-for-field. The additive counters continue, the ConsecutiveRefusals streak resumes
// at its true position, the last-writer-wins scalars (State/LastSeq/LastKind/CurrentRunID)
// resolve against the real tail, LastRun's prior-run RunID inheritance is preserved across
// the seam, and Metrics per-key overwrite continues. This is the primitive a rotation-aware
// cumulative read is built on: fold only the active segment seeded from a carried baseline
// instead of an O(total-history) re-fold of every sealed segment.
//
// The seed is DEEP-copied (the Metrics map, the *LastRun, and its EvidenceRefs slice)
// before folding, so the caller's snapshot is never aliased into the result nor mutated by
// the fold. A nil or empty seed makes it identical to the from-empty Summarize.
func SummarizeFrom(seed []LoopSnapshot, events []Event, now time.Time) Status {
	byLoop := make(map[string]*LoopSnapshot, len(seed))
	for i := range seed {
		byLoop[seed[i].LoopID] = cloneLoopSnapshot(seed[i])
	}
	for _, ev := range events {
		loop := byLoop[ev.LoopID]
		if loop == nil {
			loop = &LoopSnapshot{LoopID: ev.LoopID}
			byLoop[ev.LoopID] = loop
		}
		loop.LastSeq = ev.Seq
		loop.LastEventUnixNano = ev.TSUnixNano
		loop.LastKind = ev.Kind
		if ev.RunID != "" {
			loop.CurrentRunID = ev.RunID
		}
		if ev.State != "" {
			loop.State = string(ev.State)
		}
		for k, v := range ev.Metrics {
			if loop.Metrics == nil {
				loop.Metrics = map[string]int64{}
			}
			loop.Metrics[k] = v
		}

		switch ev.Kind {
		case EventArmed:
			if loop.State == "" {
				loop.State = string(StateArmed)
			}
		case EventFire:
			loop.Fires++
			if loop.State == "" {
				loop.State = "fired"
			}
		case EventAdmit:
			if ev.Status == StatusRefused {
				loop.Refused++
				loop.ConsecutiveRefusals++
				loop.State = string(StatusRefused)
			} else {
				loop.Admitted++
				loop.ConsecutiveRefusals = 0
				if loop.State == "" {
					loop.State = string(StatusAdmitted)
				}
			}
			loop.setRun(ev, fallbackStatus(ev.Status, StatusAdmitted))
		case EventStart:
			loop.Started++
			loop.State = string(StateRunning)
			loop.setRun(ev, StatusRunning)
		case EventEnd:
			loop.Ended++
			loop.applyEndUtility(ev)
			status := fallbackStatus(ev.Status, StatusClaimedDone)
			loop.State = string(status)
			loop.setRun(ev, status)
		case EventWitness:
			status := fallbackStatus(ev.Status, StatusWitnessUnavailable)
			switch status {
			case StatusWitnessedDone:
				loop.Witnessed++
			case StatusWitnessRefused:
				loop.WitnessRefused++
			case StatusWitnessUnavailable:
				loop.WitnessUnavailable++
			}
			loop.State = string(status)
			loop.setRun(ev, status)
		case EventNotify:
			loop.Notifications++
		}
	}

	ids := make([]string, 0, len(byLoop))
	for id := range byLoop {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	loops := make([]LoopSnapshot, 0, len(ids))
	for _, id := range ids {
		loops = append(loops, *byLoop[id])
	}
	return Status{
		Schema:     SchemaStatus,
		TSUnixNano: now.UTC().UnixNano(),
		Loops:      loops,
	}
}

// cloneLoopSnapshot returns a deep copy of a seed snapshot so SummarizeFrom can fold new
// events onto it without aliasing or mutating the caller's baseline. The value fields copy
// by assignment; the Metrics map (mutated in place by the fold) and the *LastRun pointer
// with its EvidenceRefs slice (which the caller may still hold) are cloned explicitly.
func cloneLoopSnapshot(s LoopSnapshot) *LoopSnapshot {
	c := s
	if s.Metrics != nil {
		c.Metrics = make(map[string]int64, len(s.Metrics))
		for k, v := range s.Metrics {
			c.Metrics[k] = v
		}
	}
	if s.LastRun != nil {
		lr := *s.LastRun
		lr.EvidenceRefs = append([]EvidenceRef(nil), s.LastRun.EvidenceRefs...)
		c.LastRun = &lr
	}
	return &c
}

type Status struct {
	Schema     string         `json:"schema"`
	TSUnixNano int64          `json:"ts_unix_nano"`
	LedgerPath string         `json:"ledger_path,omitempty"`
	Loops      []LoopSnapshot `json:"loops"`
}

type LoopSnapshot struct {
	LoopID              string           `json:"loop_id"`
	State               string           `json:"state,omitempty"`
	LastSeq             uint64           `json:"last_seq"`
	LastEventUnixNano   int64            `json:"last_event_unix_nano,omitempty"`
	LastKind            EventKind        `json:"last_kind,omitempty"`
	CurrentRunID        string           `json:"current_run_id,omitempty"`
	Fires               uint64           `json:"fires"`
	Admitted            uint64           `json:"admitted"`
	Refused             uint64           `json:"refused"`
	ConsecutiveRefusals uint64           `json:"consecutive_refusals"`
	Started             uint64           `json:"started"`
	Ended               uint64           `json:"ended"`
	Witnessed           uint64           `json:"witnessed"`
	WitnessRefused      uint64           `json:"witness_refused"`
	WitnessUnavailable  uint64           `json:"witness_unavailable"`
	Notifications       uint64           `json:"notifications"`
	Metrics             map[string]int64 `json:"metrics,omitempty"`
	LastRun             *RunSnapshot     `json:"last_run,omitempty"`

	// The #6497 utility partition of Ended: Ended == Failed+Effects+NoFuel+Unattributed.
	// Ended alone counts an end regardless of its outcome, so a loop that has failed on
	// every recorded run still reads Ended>0 and looks like it is doing work. See
	// utility.go for the classification and why each bucket is kept distinct.

	// Failed is the count of ended runs that ended failed or canceled.
	Failed uint64 `json:"failed"`
	// ConsecutiveFailures is the length of the CURRENT trailing streak of failing
	// ends, reset to 0 by any end that completed. It is the alert lever
	// (FailureAlertThreshold), not a history.
	ConsecutiveFailures uint64 `json:"consecutive_failures"`
	// Effects is the count of completed runs that declared >= 1 useful effect.
	Effects uint64 `json:"effects"`
	// NoFuel is the count of completed runs that declared, in the typed vocabulary,
	// that there was no work available. A no-fuel tick is a success, not an effect.
	NoFuel uint64 `json:"no_fuel"`
	// Unattributed is the count of completed runs that declared neither an effect nor
	// no-fuel — a bare child exit 0, which proves only that the process ran.
	Unattributed uint64 `json:"unattributed"`
	// CostMilliUSD is the summed reported cost across ended runs, in thousandths of a
	// US dollar. Meaningful only alongside CostedRuns.
	CostMilliUSD int64 `json:"cost_milli_usd,omitempty"`
	// CostedRuns is how many ended runs actually reported a cost, so a zero
	// CostMilliUSD reads as "never measured" rather than "free".
	CostedRuns uint64 `json:"costed_runs,omitempty"`
}

// LastActive exposes the loop's dormancy clock (issue #1179, epic #1178): the durable
// LastActiveAt stamp derived from the last ledger-event timestamp the fold already
// carries. From it a loop's dormancy band (warm/cool/cold/frozen/ancient) is derivable
// without I/O via snap.LastActive().HorizonAt(now) — the input the #1180 dormant-vs-stuck
// split and the Phase-3 durable-wake rungs (#1188) key on. A loop with no events yet
// yields the zero (unknown) Stamp, which buckets to Ancient. Pure: it reads only the
// already-recorded LastEventUnixNano, adds no field, and changes no ledger byte.
func (s LoopSnapshot) LastActive() dormancy.Stamp {
	return dormancy.FromUnixNano(s.LastEventUnixNano)
}

// Concurrent reports the loop's in-flight run count: runs that have started and
// have no matching end yet, read straight off the fold as Started-Ended. This is
// the same unmatched-start signal the schedule's overlap-lock keys on, named once
// so the governor's concurrency budget (Policy.MaxConcurrent) and the binary
// overlap-lock derive from one definition. Pure: it reads only the already-folded
// Started/Ended counters and never lets the count go negative (a witnessed or
// claimed end both increment Ended, so Ended can never exceed Started in a
// well-formed ledger; the floor is defensive).
func (s LoopSnapshot) Concurrent() uint64 {
	if s.Ended >= s.Started {
		return 0
	}
	return s.Started - s.Ended
}

type RunSnapshot struct {
	RunID         string        `json:"run_id,omitempty"`
	Status        RunStatus     `json:"status,omitempty"`
	Reason        string        `json:"reason,omitempty"`
	Summary       string        `json:"summary,omitempty"`
	EvidenceRefs  []EvidenceRef `json:"evidence_refs,omitempty"`
	EndedUnixNano int64         `json:"ended_unix_nano,omitempty"`
}

func (s *LoopSnapshot) setRun(ev Event, status RunStatus) {
	runID := ev.RunID
	if runID == "" && s.LastRun != nil {
		runID = s.LastRun.RunID
	}
	s.LastRun = &RunSnapshot{
		RunID:         runID,
		Status:        status,
		Reason:        ev.Reason,
		Summary:       ev.Summary,
		EvidenceRefs:  append([]EvidenceRef(nil), ev.EvidenceRefs...),
		EndedUnixNano: ev.TSUnixNano,
	}
}

func loadReader(r io.Reader) ([]Event, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var out []Event
	var prev string
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var ev Event
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			return nil, fmt.Errorf("loop ledger line %d: decode: %w", lineNo, err)
		}
		if err := validateLoadedEvent(ev, uint64(len(out)+1), prev); err != nil {
			return nil, fmt.Errorf("loop ledger line %d: %w", lineNo, err)
		}
		prev = ev.Hash
		out = append(out, ev)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read loop ledger: %w", err)
	}
	return out, nil
}

// Integrity describes the first chain break a tolerant read encountered, if any. It
// is the structured form of "the strict reader would have aborted here": a console or
// other read-only consumer can render what was recovered and surface the break,
// instead of the whole pane going dark on a single forked/corrupt line.
type Integrity struct {
	Broken     bool   `json:"broken"`
	AtLine     int    `json:"at_line,omitempty"`
	AtSeq      uint64 `json:"at_seq,omitempty"`
	Reason     string `json:"reason,omitempty"`
	Recovered  int    `json:"recovered_events"`
	ValidBytes int64  `json:"valid_bytes,omitempty"`
}

// LoadPrefix is the tolerant sibling of Load: it reads the longest valid chained
// prefix and, instead of aborting on the first integrity break (forked seq, bad
// prev_hash, tampered hash), STOPS and returns the events recovered so far plus an
// Integrity describing the break. err is reserved for true I/O / scanner faults, not
// chain breaks — a forked ledger yields (prefix, Integrity{Broken:true}, nil). This
// mirrors internal/journal's Verify (strict) vs ReadRows (tolerant) split; Load stays
// strict so the tamper-evidence guarantee is never weakened.
func LoadPrefix(path string) ([]Event, Integrity, error) {
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, Integrity{}, nil
	}
	if err != nil {
		return nil, Integrity{}, fmt.Errorf("open loop ledger: %w", err)
	}
	defer f.Close()

	return loadPrefixReader(f)
}

func loadPrefixReader(r io.Reader) ([]Event, Integrity, error) {
	reader := bufio.NewReader(r)
	var out []Event
	var prev string
	var offset int64
	var validBytes int64
	lineNo := 0
	for {
		raw, err := reader.ReadBytes('\n')
		if len(raw) == 0 && errors.Is(err, io.EOF) {
			return out, Integrity{Recovered: len(out), ValidBytes: validBytes}, nil
		}
		if err != nil && !errors.Is(err, io.EOF) {
			return out, Integrity{}, fmt.Errorf("read loop ledger: %w", err)
		}

		lineNo++
		offset += int64(len(raw))
		line := strings.TrimSpace(string(raw))
		if line == "" {
			validBytes = offset
			if errors.Is(err, io.EOF) {
				return out, Integrity{Recovered: len(out), ValidBytes: validBytes}, nil
			}
			continue
		}
		var ev Event
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			return out, Integrity{Broken: true, AtLine: lineNo, Reason: "decode: " + err.Error(), Recovered: len(out), ValidBytes: validBytes}, nil
		}
		if verr := validateLoadedEvent(ev, uint64(len(out)+1), prev); verr != nil {
			return out, Integrity{Broken: true, AtLine: lineNo, AtSeq: ev.Seq, Reason: verr.Error(), Recovered: len(out), ValidBytes: validBytes}, nil
		}
		prev = ev.Hash
		out = append(out, ev)
		validBytes = offset
		if errors.Is(err, io.EOF) {
			return out, Integrity{Recovered: len(out), ValidBytes: validBytes}, nil
		}
	}
}

func repairableAppendBreak(integ Integrity) bool {
	return integ.Broken && integ.Recovered > 0 && strings.HasPrefix(integ.Reason, "seq = ")
}

func integrityError(integ Integrity) error {
	if integ.AtLine > 0 {
		return fmt.Errorf("loop ledger line %d: %s", integ.AtLine, integ.Reason)
	}
	if integ.Reason != "" {
		return errors.New(integ.Reason)
	}
	return errors.New("loop ledger integrity break")
}

// SnapshotFilePartial is the tolerant sibling of SnapshotFile: it Summarizes the
// recovered prefix from LoadPrefix and carries out the Integrity break (if any). Used
// by the loops console so a forked/corrupt ledger renders the loops it could recover
// plus a break banner, instead of exiting blank.
func SnapshotFilePartial(path string, now time.Time) (Status, Integrity, error) {
	events, integ, err := LoadPrefix(path)
	if err != nil {
		return Status{}, integ, err
	}
	st := Summarize(events, now)
	st.LedgerPath = path
	return st, integ, nil
}

// validateEventCore checks the fields every loop event must satisfy regardless of
// whether it is newly minted or loaded from the ledger: a non-empty loop id and a
// known kind. Shared by validateNewEvent and validateLoadedEvent.
func validateEventCore(ev Event) error {
	if strings.TrimSpace(ev.LoopID) == "" {
		return errors.New("loop_id is required")
	}
	if !validKind(ev.Kind) {
		return fmt.Errorf("unknown loop event kind %q", ev.Kind)
	}
	return nil
}

func validateNewEvent(ev Event) error {
	if ev.Schema != "" && ev.Schema != SchemaEvent {
		return fmt.Errorf("schema = %q, want %q", ev.Schema, SchemaEvent)
	}
	return validateEventCore(ev)
}

func validateLoadedEvent(ev Event, wantSeq uint64, wantPrev string) error {
	if ev.Schema != SchemaEvent {
		return fmt.Errorf("schema = %q, want %q", ev.Schema, SchemaEvent)
	}
	if ev.Seq != wantSeq {
		return fmt.Errorf("seq = %d, want %d", ev.Seq, wantSeq)
	}
	if ev.PrevHash != wantPrev {
		return fmt.Errorf("prev_hash = %q, want %q", ev.PrevHash, wantPrev)
	}
	if err := validateEventCore(ev); err != nil {
		return err
	}
	if ev.Hash == "" {
		return errors.New("hash is required")
	}
	if got := hashEvent(ev); got != ev.Hash {
		return fmt.Errorf("hash = %q, want %q", ev.Hash, got)
	}
	return nil
}

func validKind(kind EventKind) bool {
	switch kind {
	case EventArmed, EventFire, EventAdmit, EventStart, EventHeartbeat, EventEnd, EventWitness, EventNotify:
		return true
	default:
		return false
	}
}

func fallbackStatus(status, fallback RunStatus) RunStatus {
	if status != "" {
		return status
	}
	return fallback
}

func hashEvent(ev Event) string {
	ev.Hash = ""
	b, err := json.Marshal(ev)
	if err != nil {
		panic(err)
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
