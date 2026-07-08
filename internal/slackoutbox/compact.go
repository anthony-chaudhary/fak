package slackoutbox

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

// Compaction keeps the append-only outbox from growing without bound. Every drain appends
// transitions and a heartbeat, so a busy fleet's state.jsonl outgrows its live queue by
// orders of magnitude within a day — the fold stays correct but the read cost and disk
// footprint do not. Compaction folds the accumulated segments down to *only what a future
// reader still needs*: rows still owed a delivery, dead rows an operator may retry, and
// recently-settled rows kept briefly so a deferred producer's PostedTS probe still
// resolves. Everything older is dropped, the per-nonce transition history collapses to one
// line, and the heartbeat storm collapses to a single drain_pass.
//
// The mechanism is seal-quiesce-rewrite, not truncate-in-place: the live head is renamed
// to a transient seal segment so it can be read without racing a lock-free appender, the
// sealed inode is allowed to quiesce (the last in-flight O_APPEND lands), the archive is
// rewritten via temp-then-rename, and the seal is removed. A crash at any point leaves a
// segment Load already folds — never a torn or lost row. This is the retention exception
// to the leaf's "terminal states are terminal FOREVER" rule: a posted/refused/superseded
// row is forgotten once it is old enough that no reader will ask about it again.

// Retention windows: how long a terminal row survives compaction past its final
// transition. Settled (superseded/refused) rows are kept briefly for diagnostics; posted
// rows are kept long enough that a deferred producer resuming after a guard session can
// still resolve their PostedTS; dead rows are never dropped (an operator may retry them).
const (
	DefaultRetainSettled = time.Hour      // superseded / refused
	DefaultRetainPosted  = 48 * time.Hour // posted (past any guard session's PostedTS probes)
	DefaultRetainCards   = 48 * time.Hour // finalized <dir>/cards/*.json run-card state

	// CompactRowThreshold and CompactMinInterval gate the automatic post-drain
	// compaction: run when the fold carries more terminal rows than the threshold, or a
	// row is already past its window and the last compaction is older than the interval.
	CompactRowThreshold = 512
	CompactMinInterval  = 15 * time.Minute

	// cardsSubdir mirrors dispatchpost.OpenRunCard's <outboxDir>/cards layout — the run
	// cards whose durable identity files the outbox also garbage-collects.
	cardsSubdir = "cards"

	// quiesceSettleWait / quiesceMaxRounds bound the wait for a sealed segment to stop
	// growing. Convergence is near-immediate (an appender holds the old fd for one
	// syscall); the bound is only a safety cap against a pathological writer.
	quiesceSettleWait = 25 * time.Millisecond
	quiesceMaxRounds  = 200
)

// CompactOpts configures one compaction. Zero values take the documented defaults.
type CompactOpts struct {
	RetainSettled time.Duration       // superseded/refused window; <=0 => DefaultRetainSettled
	RetainPosted  time.Duration       // posted window; <=0 => DefaultRetainPosted
	RetainCards   time.Duration       // finalized run-card window; <=0 => DefaultRetainCards
	Now           time.Time           // reference clock; zero => o.now()
	Sleep         func(time.Duration) // seal-quiesce wait seam; nil => o's compactSleep, then time.Sleep
	DryRun        bool                // compute the report without mutating anything

	held bool // internal: the caller already holds drain.lock (the in-drain auto path)
}

func (c CompactOpts) norm(o *Outbox) CompactOpts {
	if c.RetainSettled <= 0 {
		c.RetainSettled = DefaultRetainSettled
	}
	if c.RetainPosted <= 0 {
		c.RetainPosted = DefaultRetainPosted
	}
	if c.RetainCards <= 0 {
		c.RetainCards = DefaultRetainCards
	}
	if c.Now.IsZero() {
		c.Now = o.now()
	}
	if c.Sleep == nil {
		if o.compactSleep != nil {
			c.Sleep = o.compactSleep
		} else {
			c.Sleep = time.Sleep
		}
	}
	return c
}

// CompactReport is what one pass did (or, for a dry run, would do), for the verb's
// human/JSON output.
type CompactReport struct {
	ScannedRows        int   `json:"scanned_rows"`         // spool rows folded from archive+seal
	KeptRows           int   `json:"kept_rows"`            // survivors written back to the archive
	DroppedPosted      int   `json:"dropped_posted"`       // posted rows past RetainPosted
	DroppedSuperseded  int   `json:"dropped_superseded"`   // superseded rows past RetainSettled
	DroppedRefused     int   `json:"dropped_refused"`      // refused rows past RetainSettled
	CollapsedDrainPass int   `json:"collapsed_drain_pass"` // drain_pass heartbeats folded to one
	DroppedCards       int   `json:"dropped_cards"`        // finalized run-card files removed
	SpoolBytesBefore   int64 `json:"spool_bytes_before"`   // archive+seal spool bytes folded
	SpoolBytesAfter    int64 `json:"spool_bytes_after"`    // rewritten archive spool bytes
	StateBytesBefore   int64 `json:"state_bytes_before"`
	StateBytesAfter    int64 `json:"state_bytes_after"`
	DryRun             bool  `json:"dry_run,omitempty"`
}

// Compact folds the outbox's accumulated segments down to what a future reader still
// needs, dropping terminal rows past their retention window and collapsing per-nonce
// history and the heartbeat storm. It serializes against the drainer via drain.lock (the
// in-drain auto path passes held=true because the drainer already holds it), and returns
// ErrDrainBusy when another holder owns the lock. A dry run computes the report without
// touching a single file.
func (o *Outbox) Compact(opts CompactOpts) (*CompactReport, error) {
	opts = opts.norm(o)

	if !opts.held {
		lock, err := os.OpenFile(filepath.Join(o.dir, lockFile), os.O_CREATE|os.O_RDWR, 0o644)
		if err != nil {
			return nil, err
		}
		defer lock.Close()
		if err := flock.TryLock(lock); err != nil {
			if errors.Is(err, flock.ErrLockBusy) {
				return nil, ErrDrainBusy
			}
			return nil, err
		}
		defer func() { _ = flock.Unlock(lock) }()
	}

	if opts.DryRun {
		return o.compactPreview(opts)
	}

	// Recover a seal left by a crashed prior compaction by folding it into the archive
	// first, so the fresh seal below never has to merge with an older one.
	if o.hasSeal() {
		if _, err := o.rewriteArchive(opts, nil); err != nil {
			return nil, err
		}
	}

	// Seal the live heads (atomic renames; the last in-flight append lands in the sealed
	// inode, which we then let quiesce). A missing head is a no-op.
	if err := o.sealHead(spoolFile, spoolSealFile); err != nil {
		return nil, err
	}
	if err := o.sealHead(stateFile, stateSealFile); err != nil {
		return nil, err
	}
	o.quiesce(spoolSealFile, opts)
	o.quiesce(stateSealFile, opts)

	rep := &CompactReport{}
	if _, err := o.rewriteArchive(opts, rep); err != nil {
		return nil, err
	}

	dropped, err := o.gcCards(opts, false)
	if err != nil {
		return nil, err
	}
	rep.DroppedCards = dropped

	// A compact_pass heartbeat records that a compaction ran, so the auto-trigger's
	// interval gate has a baseline on the next drain.
	if err := o.appendState(transition{State: stateCompactPass, At: opts.Now.UTC().Format(time.RFC3339)}); err != nil {
		return nil, err
	}
	return rep, nil
}

// hasSeal reports whether a transient seal segment is present (a crashed compaction).
func (o *Outbox) hasSeal() bool {
	for _, n := range []string{spoolSealFile, stateSealFile} {
		if _, err := os.Stat(filepath.Join(o.dir, n)); err == nil {
			return true
		}
	}
	return false
}

// sealHead renames the live head to its seal segment so it can be read without racing a
// lock-free appender. A missing head (no appends since the last seal) is a no-op.
func (o *Outbox) sealHead(head, seal string) error {
	src := filepath.Join(o.dir, head)
	if _, err := os.Stat(src); err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	return os.Rename(src, filepath.Join(o.dir, seal))
}

// quiesce waits until a sealed segment stops growing — the last O_APPEND opened before the
// rename lands, then the inode is stable to read. Missing segment: nothing to wait for.
func (o *Outbox) quiesce(seal string, opts CompactOpts) {
	path := filepath.Join(o.dir, seal)
	prev := int64(-1)
	for i := 0; i < quiesceMaxRounds; i++ {
		fi, err := os.Stat(path)
		if err != nil {
			return
		}
		if fi.Size() == prev {
			return
		}
		prev = fi.Size()
		opts.Sleep(quiesceSettleWait)
	}
}

// rewriteArchive folds archive+seal, writes the survivors back to the archive via
// temp-then-rename, and removes the seals. rep, when non-nil, gets the drop tallies and
// byte sizes (the recovery pass passes nil). It returns the survivor count.
func (o *Outbox) rewriteArchive(opts CompactOpts, rep *CompactReport) (int, error) {
	spoolBefore := fileSize(filepath.Join(o.dir, spoolArchFile)) + fileSize(filepath.Join(o.dir, spoolSealFile))
	stateBefore := fileSize(filepath.Join(o.dir, stateArchFile)) + fileSize(filepath.Join(o.dir, stateSealFile))

	snap, err := o.foldFiles(
		[]string{spoolArchFile, spoolSealFile},
		[]string{stateArchFile, stateSealFile},
	)
	if err != nil {
		return 0, err
	}

	var rows []any
	var states []any
	kept := 0
	for _, r := range snap.Rows {
		s := snap.States[r.Nonce]
		if rep != nil {
			rep.ScannedRows++
		}
		keep, class := keepRow(s, opts)
		if !keep {
			if rep != nil {
				switch class {
				case statePosted:
					rep.DroppedPosted++
				case stateSuperseded:
					rep.DroppedSuperseded++
				case stateRefused:
					rep.DroppedRefused++
				}
			}
			continue
		}
		kept++
		rows = append(rows, r)
		if s.State != statePending { // pending rows have no transition to carry forward
			states = append(states, collapse(r.Nonce, s))
		}
	}

	// Carry one drain_pass forward so LastDrainAgeS survives compaction, then the survivor
	// transitions. The compact_pass is appended to the fresh head by Compact, not here.
	if !snap.LastDrainAt.IsZero() {
		states = append([]any{transition{State: stateDrainPass, At: snap.LastDrainAt.UTC().Format(time.RFC3339)}}, states...)
	}

	spoolAfter, err := writeJSONLAtomic(filepath.Join(o.dir, spoolArchFile), rows)
	if err != nil {
		return 0, err
	}
	stateAfter, err := writeJSONLAtomic(filepath.Join(o.dir, stateArchFile), states)
	if err != nil {
		return 0, err
	}

	// The seals are folded into the archive now — remove them so they are not folded twice.
	if err := removeIfExists(filepath.Join(o.dir, spoolSealFile)); err != nil {
		return 0, err
	}
	if err := removeIfExists(filepath.Join(o.dir, stateSealFile)); err != nil {
		return 0, err
	}

	if rep != nil {
		rep.KeptRows = kept
		rep.CollapsedDrainPass = snap.drainPasses
		rep.SpoolBytesBefore, rep.SpoolBytesAfter = spoolBefore, spoolAfter
		rep.StateBytesBefore, rep.StateBytesAfter = stateBefore, stateAfter
	}
	return kept, nil
}

// keepRow decides whether a folded row survives compaction. Non-terminal rows (still owed
// a delivery) and dead rows (operator-actionable) always survive; a settled row survives
// until it is older than its retention window. class is the drop bucket when keep is false.
// An unparseable timestamp (zero At) keeps the row — compaction never drops on a guess.
func keepRow(s rowState, opts CompactOpts) (keep bool, class string) {
	switch s.State {
	case statePosted:
		if s.At.IsZero() || opts.Now.Sub(s.At) < opts.RetainPosted {
			return true, ""
		}
		return false, statePosted
	case stateSuperseded:
		if s.At.IsZero() || opts.Now.Sub(s.At) < opts.RetainSettled {
			return true, ""
		}
		return false, stateSuperseded
	case stateRefused:
		if s.At.IsZero() || opts.Now.Sub(s.At) < opts.RetainSettled {
			return true, ""
		}
		return false, stateRefused
	default: // pending / sending / failed / dead / unknown — all still needed
		return true, ""
	}
}

// collapse renders a folded rowState as the single transition that reproduces it on the
// next replay.
func collapse(nonce string, s rowState) transition {
	t := transition{
		Nonce:     nonce,
		State:     s.State,
		TS:        s.TS,
		Reason:    s.Reason,
		Attempts:  s.Attempts,
		Ambiguous: s.Ambiguous,
	}
	if !s.At.IsZero() {
		t.At = s.At.UTC().Format(time.RFC3339)
	}
	return t
}

// compactPreview computes the report a real compaction would produce, folding all three
// live layers, without mutating anything.
func (o *Outbox) compactPreview(opts CompactOpts) (*CompactReport, error) {
	snap, err := o.Load()
	if err != nil {
		return nil, err
	}
	rep := &CompactReport{DryRun: true, CollapsedDrainPass: snap.drainPasses}
	for _, r := range snap.Rows {
		rep.ScannedRows++
		keep, class := keepRow(snap.States[r.Nonce], opts)
		if keep {
			rep.KeptRows++
			continue
		}
		switch class {
		case statePosted:
			rep.DroppedPosted++
		case stateSuperseded:
			rep.DroppedSuperseded++
		case stateRefused:
			rep.DroppedRefused++
		}
	}
	dropped, err := o.gcCards(opts, true)
	if err != nil {
		return nil, err
	}
	rep.DroppedCards = dropped
	return rep, nil
}

// gcCards removes (or, in dryRun, counts) finalized run-card state files older than
// RetainCards. A non-final card is a live run's card and is never touched; an unreadable
// or unparseable file is left in place. Returns how many were (or would be) removed.
func (o *Outbox) gcCards(opts CompactOpts, dryRun bool) (int, error) {
	matches, err := filepath.Glob(filepath.Join(o.dir, cardsSubdir, "*.json"))
	if err != nil {
		return 0, err
	}
	dropped := 0
	for _, path := range matches {
		fi, err := os.Stat(path)
		if err != nil {
			continue
		}
		if opts.Now.Sub(fi.ModTime()) < opts.RetainCards {
			continue
		}
		b, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		var st CardState
		if json.Unmarshal(b, &st) != nil || !st.Final {
			continue // torn/live card — leave it
		}
		if !dryRun {
			if err := os.Remove(path); err != nil {
				return dropped, err
			}
		}
		dropped++
	}
	return dropped, nil
}

// maybeCompact runs a compaction after a drain when it is due, reusing the snapshot the
// drain already folded. It is called with drain.lock held.
func (o *Outbox) maybeCompact(snap *Snapshot, opts CompactOpts) (*CompactReport, error) {
	opts = opts.norm(o)
	if !o.compactionDue(snap, opts) {
		return nil, nil
	}
	opts.held = true
	return o.Compact(opts)
}

// compactionDue gates the automatic post-drain compaction. It skips when a compaction ran
// within CompactMinInterval (the archive has had no time to grow), then runs when any
// terminal row is already past its window or the terminal backlog is large. A never-yet
// compacted outbox (zero LastCompactAt) is allowed through the interval gate so the first
// droppable row triggers the first compaction. Rows all fresh and few in number: no pass.
func (o *Outbox) compactionDue(snap *Snapshot, opts CompactOpts) bool {
	if !snap.LastCompactAt.IsZero() && opts.Now.Sub(snap.LastCompactAt) < CompactMinInterval {
		return false
	}
	terminal := 0
	for _, r := range snap.Rows {
		s := snap.States[r.Nonce]
		if !s.terminal() {
			continue
		}
		terminal++
		if keep, _ := keepRow(s, opts); !keep {
			return true
		}
	}
	return terminal > CompactRowThreshold
}

// writeJSONLAtomic writes records as JSONL to path via temp-then-rename+fsync, returning
// the final file size. An empty record set writes an empty file (Load treats it as absent).
func writeJSONLAtomic(path string, records []any) (int64, error) {
	tmp := path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return 0, err
	}
	w := bufio.NewWriter(f)
	for _, rec := range records {
		b, err := json.Marshal(rec)
		if err != nil {
			f.Close()
			return 0, err
		}
		if _, err := w.Write(append(b, '\n')); err != nil {
			f.Close()
			return 0, err
		}
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return 0, err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return 0, err
	}
	if err := f.Close(); err != nil {
		return 0, err
	}
	if err := os.Rename(tmp, path); err != nil {
		return 0, err
	}
	return fileSize(path), nil
}

// fileSize returns a file's size, or 0 when it is missing/unstatable.
func fileSize(path string) int64 {
	fi, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return fi.Size()
}

// removeIfExists removes path, treating a missing file as success.
func removeIfExists(path string) error {
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// CompactReportLine renders the report as one human line (the CLI's non-JSON output).
func CompactReportLine(rep *CompactReport) string {
	verb := "compacted"
	if rep.DryRun {
		verb = "would compact"
	}
	return fmt.Sprintf(
		"%s: scanned %d  kept %d  dropped(posted %d superseded %d refused %d)  cards %d  drain_pass→1 (from %d)  spool %d→%dB  state %d→%dB",
		verb, rep.ScannedRows, rep.KeptRows, rep.DroppedPosted, rep.DroppedSuperseded, rep.DroppedRefused,
		rep.DroppedCards, rep.CollapsedDrainPass, rep.SpoolBytesBefore, rep.SpoolBytesAfter,
		rep.StateBytesBefore, rep.StateBytesAfter)
}
