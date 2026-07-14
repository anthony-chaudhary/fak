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
	matches, err := filepath.Glob(path + cutSuffix + "*")
	if err != nil {
		return nil, fmt.Errorf("journal: segments %s: %w", path, err)
	}
	type seg struct {
		seq  uint64
		path string
	}
	var segs []seg
	for _, m := range matches {
		tail := strings.TrimPrefix(m, path+cutSuffix)
		seq, perr := strconv.ParseUint(tail, 10, 64)
		if perr != nil {
			continue // not a <seq> archive (e.g. a further-suffixed sibling) — skip
		}
		segs = append(segs, seg{seq: seq, path: m})
	}
	sort.Slice(segs, func(a, b int) bool { return segs[a].seq < segs[b].seq })
	out := make([]string, 0, len(segs)+1)
	for _, s := range segs {
		out = append(out, s.path)
	}
	out = append(out, path)
	return out, nil
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
