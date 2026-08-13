package journal

// Chain-aware rotation (the "cut"). A hash-chained journal grows unbounded by
// design, but naive rotation FORKS OR BREAKS the chain: a fresh Open on a new
// file restarts at genesis (Seq 1, empty PrevHash), so a verifier that folds the
// old file and the new one together sees a sequence gap and a prev-hash
// discontinuity at the boundary. Cut rotates WITHOUT forking: it archives the
// current segment and opens a successor whose first row is a CUT anchor that
// CONTINUES the chain.
//
// No schema change is needed to make the anchor tamper-evident. The prior
// segment's chain head is already bound by the frozen Row fields: the anchor's
// PrevHash IS the prior segment's final row hash (the chained-in prefix), and the
// anchor's Seq is prevFinalSeq+1 (Seq is in the hash pre-image, so Seq-1 recovers
// and binds the prior final seq). VerifySegments follows this anchor across every
// boundary, so a multi-file rotated journal verifies end-to-end exactly like the
// single-file chain it was cut from.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// KindCut marks a segment-boundary ANCHOR row: the first row of a successor
// segment written by Cut. It is a genuine chained row — it consumes the next Seq
// and chains onto the prior segment's head via PrevHash — so the prior segment's
// final seq (this row's Seq-1) and final hash (this row's PrevHash) are both bound
// by the chain. It carries no decision; a decision-folding consumer skips it (its
// Kind is neither DECIDE/DENY nor any other verdict kind).
const KindCut = "CUT"

// cutSuffix is the archival-name discipline for a rotated segment: <path>.cut-<seq>
// where <seq> is the final seq of the archived segment. It mirrors loopmgr's
// .broken-<seq> naming so a rotated set sorts by seq and is greppable.
const cutSuffix = ".cut-"

// Cut rotates a file-backed journal without forking or breaking the chain. It
// flushes and closes the current segment, renames it aside to an archival sibling
// (<path>.cut-<finalSeq>), opens a fresh segment at the original path, and writes
// a KindCut anchor as that segment's first row — Seq = finalSeq+1, PrevHash = the
// prior segment's head hash — so the archived segment and its successor
// VerifySegments end-to-end through the anchor. The live seq/hash head is
// preserved (the successor CONTINUES the chain), so seq stays globally monotonic
// with no gap. Returns the archived segment's path.
//
// Cut errors (a no-op) on an in-memory journal (nothing durable to rotate) and on
// a journal with no committed rows (a cut before any row would anchor onto genesis
// and let a caller spam empty segments). On a rename/open fault it best-effort
// re-opens the original path so the journal is never left headless.
func (j *Journal) Cut() (archivedPath string, err error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.f == nil || j.path == "" {
		return "", fmt.Errorf("journal: cut: not a file-backed journal")
	}
	if j.seq == 0 {
		return "", fmt.Errorf("journal: cut: nothing to rotate (no committed rows)")
	}

	// Flush + fsync + close the current segment so the archive is complete.
	if err := j.bw.Flush(); err != nil {
		return "", fmt.Errorf("journal: cut: flush: %w", err)
	}
	_ = j.f.Sync()
	if err := j.f.Close(); err != nil {
		j.f, j.bw = nil, nil
		j.reopen()
		return "", fmt.Errorf("journal: cut: close: %w", err)
	}
	j.f, j.bw = nil, nil

	archivedPath = fmt.Sprintf("%s%s%d", j.path, cutSuffix, j.seq)
	if err := os.Rename(j.path, archivedPath); err != nil {
		j.reopen() // do not leave the journal headless on a failed rotate
		return "", fmt.Errorf("journal: cut: archive %s: %w", archivedPath, err)
	}

	// Open the fresh successor segment at the original path.
	f, oerr := os.OpenFile(j.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if oerr != nil {
		return "", fmt.Errorf("journal: cut: open successor %s: %w", j.path, oerr)
	}
	j.f, j.bw = f, bufio.NewWriter(f)

	// The anchor CONTINUES the chain: appendLocked stamps Seq=prevFinalSeq+1,
	// PrevHash=prevFinalHash, Hash over them (we already hold j.mu). Those two
	// fields ARE the recorded anchor — no extra schema.
	j.appendLocked(Row{Kind: KindCut})
	return archivedPath, nil
}

// reopen best-effort re-attaches the journal to its original path after a failed
// cut, so a rotate fault degrades to "kept writing the same file" rather than a
// headless journal that silently drops every subsequent row. Caller holds j.mu.
func (j *Journal) reopen() {
	if j.f != nil || j.path == "" {
		return
	}
	if f, err := os.OpenFile(j.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644); err == nil {
		j.f, j.bw = f, bufio.NewWriter(f)
	}
}

// Segments returns the ordered segment files for a rotated journal at path: every
// archived <path>.cut-<seq> sibling oldest-first (by seq), followed by the live
// path. It is the input to VerifySegments and to any read-fold that must not lose
// history across a cut. A journal that was never cut returns just [path].
func Segments(path string) ([]string, error) {
	segs, err := archivedSegments(path)
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, len(segs)+1)
	for _, s := range segs {
		out = append(out, s.path)
	}
	out = append(out, path)
	return out, nil
}

// archivedSeg is one sealed sibling of a live journal: the final seq the archive
// name records and the archive's path.
type archivedSeg struct {
	seq  uint64
	path string
}

// archivedSegments lists the SEALED siblings of the journal at path, oldest-first
// by their recorded final seq — Segments minus the live file. It is shared by
// Segments (which appends the live path) and by the tail read, which needs the
// sealed count and the last sealed seq without opening a single archive.
func archivedSegments(path string) ([]archivedSeg, error) {
	matches, err := filepath.Glob(path + cutSuffix + "*")
	if err != nil {
		return nil, fmt.Errorf("journal: segments %s: %w", path, err)
	}
	var segs []archivedSeg
	for _, m := range matches {
		tail := strings.TrimPrefix(m, path+cutSuffix)
		seq, perr := strconv.ParseUint(tail, 10, 64)
		if perr != nil {
			continue // not a <seq> archive (e.g. a further-suffixed sibling) — skip
		}
		segs = append(segs, archivedSeg{seq: seq, path: m})
	}
	sort.Slice(segs, func(a, b int) bool { return segs[a].seq < segs[b].seq })
	return segs, nil
}

// VerifySegments validates a rotated journal split across ordered segment files as
// ONE continuous hash chain, following the Cut anchor across each boundary. It
// verifies seg[0] from genesis, then continues the SAME running chain head +
// expected sequence into each successor (whose first row is the KindCut anchor
// binding the prior segment's final seq and head hash). It returns the total rows
// verified and the first broken link — a sequence gap, a prev-hash discontinuity,
// a tampered row, or a successor that does not begin with a continuing CUT anchor.
// Segments must be given oldest-first (the order Cut and Segments produce them).
func VerifySegments(paths ...string) (int, error) {
	var (
		prev    string
		wantSeq uint64
		total   int
	)
	for si, path := range paths {
		f, err := os.Open(path)
		if err != nil {
			return total, fmt.Errorf("journal: open segment %s: %w", path, err)
		}
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
		firstInSeg := true
		for sc.Scan() {
			line := sc.Bytes()
			if len(line) == 0 {
				continue
			}
			var row Row
			if err := json.Unmarshal(line, &row); err != nil {
				f.Close()
				return total, fmt.Errorf("journal: segment %s row %d: malformed JSON: %w", path, total+1, err)
			}
			if firstInSeg && si > 0 && row.Kind != KindCut {
				f.Close()
				return total, fmt.Errorf("journal: segment %s does not begin with a CUT anchor (kind=%q at seq %d)", path, row.Kind, row.Seq)
			}
			wantSeq++
			next, err := verifyStep(prev, wantSeq, row)
			if err != nil {
				f.Close()
				return total, err
			}
			prev = next
			firstInSeg = false
			total++
		}
		if err := sc.Err(); err != nil {
			f.Close()
			return total, fmt.Errorf("journal: scan segment %s: %w", path, err)
		}
		f.Close()
	}
	return total, nil
}

// ReadAllSegments reads every row of a rotated journal at path, in chain order,
// across all archived cut segments and the live path — the read-fold a consumer
// (audit-usage roll-up, loop-health, guard-RSI) uses so it loses NO history across
// a cut. It is robust like ReadRows (a torn final line stops that segment at its
// last well-formed row); use VerifySegments for the integrity check.
//
// The returned rows are the literal chain, so they include one KindCut anchor per
// boundary; a consumer whose TOTAL must match the same journal unrotated folds them
// through WithoutCutAnchors.
func ReadAllSegments(path string) ([]Row, error) {
	segs, err := Segments(path)
	if err != nil {
		return nil, err
	}
	var out []Row
	for _, seg := range segs {
		rows, err := ReadRows(seg)
		if err != nil {
			return out, err
		}
		out = append(out, rows...)
	}
	return out, nil
}

// WithoutCutAnchors drops the KindCut anchor rows from a segment-aware read so a
// fold over a ROTATED journal counts exactly what the same journal would have
// counted unrotated. The anchor is rotation bookkeeping, not history: it carries no
// decision, and an uncut journal has none of them. A roll-up that wants its totals
// to be cut-invariant reads ReadAllSegments and folds through this; a caller that
// wants the literal chain (verification, seq continuity) uses ReadAllSegments raw.
// It returns rows unchanged (no copy) when there is nothing to drop.
func WithoutCutAnchors(rows []Row) []Row {
	n := 0
	for _, r := range rows {
		if r.Kind == KindCut {
			n++
		}
	}
	if n == 0 {
		return rows
	}
	out := make([]Row, 0, len(rows)-n)
	for _, r := range rows {
		if r.Kind == KindCut {
			continue
		}
		out = append(out, r)
	}
	return out
}

// TailOmission is what a LIVE-FILE-ONLY read of a rotated journal did not see:
// how many sealed segments sit beside the live file, and how many rows were
// committed before the most recent cut. It exists so a truncated read cannot be
// mistaken for a complete one — before rotation shipped, ReadRows on a cut journal
// returned a short slice that looked exactly like a whole small journal, so a
// roll-up over it reported a tail as a total and said nothing (#6488).
type TailOmission struct {
	// SealedSegments is the number of archived <path>.cut-<seq> siblings the tail
	// read did NOT open.
	SealedSegments int `json:"sealed_segments,omitempty"`
	// RowsBeforeCut is the count of rows committed before the most recent cut, read
	// off the newest archive's recorded final seq (seq is globally monotonic across
	// the chain, so that seq IS the number of rows preceding the live segment).
	RowsBeforeCut uint64 `json:"rows_before_cut,omitempty"`
}

// Omitted reports whether the tail read left history unread. The zero
// TailOmission (an uncut journal) is false: the tail WAS the whole journal.
func (o TailOmission) Omitted() bool { return o.SealedSegments > 0 || o.RowsBeforeCut > 0 }

// String renders the omission as one operator-facing clause, or "" when nothing
// was omitted, so a complete read prints no disclaimer at all.
func (o TailOmission) String() string {
	if !o.Omitted() {
		return ""
	}
	return fmt.Sprintf("%d row(s) before the cut omitted (%d sealed segment(s) not read)", o.RowsBeforeCut, o.SealedSegments)
}

// ReadTail reads ONLY the live segment at path — the same rows ReadRows returns —
// and reports what that cost. It is the honest form of the tail read for a consumer
// that genuinely wants recent rows (a live pane) rather than a total: the returned
// TailOmission names the sealed segments and the pre-cut row count it skipped, so
// the truncation is on the record instead of being invisible. A consumer that
// produces a total, a rate, or a roll-up wants ReadAllSegments, not this.
//
// The omission costs no extra file reads: it comes from the archive NAMES (Cut
// records the archived segment's final seq in the suffix). A read error on the live
// file is returned as-is with a zero omission.
func ReadTail(path string) ([]Row, TailOmission, error) {
	rows, err := ReadRows(path)
	if err != nil {
		return rows, TailOmission{}, err
	}
	segs, serr := archivedSegments(path)
	if serr != nil {
		return rows, TailOmission{}, serr
	}
	om := TailOmission{SealedSegments: len(segs)}
	if len(segs) > 0 {
		om.RowsBeforeCut = segs[len(segs)-1].seq
	}
	return rows, om, nil
}

// CutIfOversized bounds a live file-backed journal without changing callers'
// append semantics. It is a no-op below maxBytes.
func CutIfOversized(j *Journal, maxBytes int64) (bool, error) {
	if j == nil || maxBytes <= 0 {
		return false, nil
	}
	path := j.Path()
	if path == "" {
		return false, nil
	}
	info, err := os.Stat(path)
	if err != nil {
		return false, err
	}
	if info.Size() <= maxBytes {
		return false, nil
	}
	_, err = j.Cut()
	return err == nil, err
}
