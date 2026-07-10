package toolproc

import (
	"bufio"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// JournalCompactThresholdBytes is the on-disk size past which CompactJournalFile
// rewrites the shared journal. It matches JournalTailWindowBytes on purpose: the
// window is exactly how much of the file a single hook firing parses
// (ParseTailFile), so once the journal grows beyond it two things become true at
// once — older history has begun falling outside every firing's view, and a
// still-live spawn can sit BEYOND the tail where a firing would miss its pairing
// exit (the "correlation target older than the window" fallback ParseTail
// documents). Compacting at precisely that point folds the whole journal back
// inside one tail read: it bounds disk AND restores correlation for old live
// spawns in a single pass.
const JournalCompactThresholdBytes = JournalTailWindowBytes

// JournalCompactTailKeep is the recent-event margin CompactJournalFile retains,
// on top of every un-exited spawn (which CompactJournal keeps regardless of
// age). At the journal's ~100–300 byte rows this is well under a MiB, so a
// compacted journal sits comfortably inside a single JournalTailWindowBytes
// read.
const JournalCompactTailKeep = 4096

// CompactJournalFile bounds the on-disk journal at path once it has grown past
// thresholdBytes: it reads the whole file, runs CompactJournal(tailKeep) to drop
// fully-terminal history while preserving every un-exited spawn and the last
// tailKeep events, and atomically replaces the file with the compacted set. It
// reports whether a rewrite actually happened.
//
// It is a cheap stat-only no-op below the threshold (the common case) and on a
// missing file, so a caller can invoke it unconditionally at a session boundary.
// The swap is a same-directory temp write + rename, so a concurrent reader
// (another guarded session's hook firing) never sees a torn file — it observes
// either the pre- or the post-compaction journal, both complete and fold-clean.
// The residual races are fail-open and platform-shaped, never corruption:
//   - POSIX: an append landing in the brief read→rename window (only from a
//     DIFFERENT session — this session's own append handle is already closed)
//     writes to the pre-rename inode and is lost: at worst a dropped row.
//   - Windows: the swap is a POSIX-semantics rename that supersedes the
//     destination even while another session's reader/appender holds it open,
//     because the journal's own openers (OpenShareDelete /
//     OpenAppendShareDelete) grant FILE_SHARE_DELETE (#3555). A FOREIGN handle
//     without that share mode (an antivirus scan, an ad-hoc `type`/editor, an
//     older fak binary) still makes the rename fail ERROR_ACCESS_DENIED;
//     replaceFileAtomic retries briefly, and if contention outlasts the
//     retries the rename is abandoned and this returns the error, so the
//     journal is simply left un-compacted this round (retried at the next
//     stop) rather than truncated.
//
// A row the reader cannot decode — a single record past the parser's token
// cap, torn/malformed JSON, or an invalid event (pathological: rows are bounded
// scalars ~100–300 bytes, so only a corrupt write or an out-of-band appender
// produces one) — is dropped by the compaction read rather than aborting the
// compaction (#3556). Aborting would leave such a journal un-boundable forever:
// the one pass that could shrink it is exactly the pass that errored. Dropping
// is compaction-only — the fold path's ParseEvents stays fail-closed — and a
// journal that had rows dropped is rewritten even when nothing terminal was
// reclaimable, so the bad row is expelled and the file comes back fold-clean.
func CompactJournalFile(path string, thresholdBytes int64, tailKeep int) (bool, error) {
	fi, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	if thresholdBytes > 0 && fi.Size() <= thresholdBytes {
		return false, nil // under the window: nothing has fallen out of view yet
	}
	f, err := OpenShareDelete(path)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	events, dropped, err := parseEventsLenient(f)
	f.Close()
	if err != nil {
		return false, err
	}
	compacted := CompactJournal(events, tailKeep)
	if dropped == 0 && len(compacted) >= len(events) {
		// Everything is either recent or a still-live spawn — nothing terminal to
		// reclaim. Leave the file untouched rather than churn an identical rewrite.
		// (With dropped rows the rewrite must happen regardless: expelling them is
		// the only way the file gets bounded.)
		return false, nil
	}
	var buf []byte
	for _, ev := range compacted {
		line, err := json.Marshal(ev)
		if err != nil {
			return false, err
		}
		buf = append(buf, append(line, '\n')...)
	}
	if err := replaceFileAtomic(path, buf); err != nil {
		return false, err
	}
	return true, nil
}

// parseEventsLenient reads a JSONL journal for compaction, dropping any row it
// cannot decode — a line past maxEventLineBytes, malformed JSON, or an invalid
// event — instead of refusing the whole file the way the fail-closed fold
// reader (ParseEvents) does (#3556). A bufio.Scanner cannot continue past
// bufio.ErrTooLong, so this reads bounded lines itself: an oversized line stops
// accumulating at the cap and drains to its newline, and reading resumes at the
// next row. Blank and comment rows are skipped as in ParseEvents and are not
// counted as dropped. err is non-nil only for a real read failure.
func parseEventsLenient(r io.Reader) (events []Event, dropped int, err error) {
	br := bufio.NewReaderSize(r, 64*1024)
	var line []byte
	overflow := false
	flush := func() {
		raw := strings.TrimSpace(string(line))
		switch {
		case overflow:
			dropped++
		case raw == "" || strings.HasPrefix(raw, "#"):
		default:
			var ev Event
			if json.Unmarshal([]byte(raw), &ev) != nil || ValidateEvent(ev) != nil {
				dropped++
			} else {
				events = append(events, ev)
			}
		}
		line = line[:0]
		overflow = false
	}
	for {
		frag, isPrefix, rerr := br.ReadLine()
		if len(frag) > 0 && !overflow {
			if len(line)+len(frag) > maxEventLineBytes {
				overflow = true
				line = line[:0]
			} else {
				line = append(line, frag...)
			}
		}
		if rerr != nil {
			if rerr == io.EOF {
				// ReadLine never returns data alongside an error, so anything
				// buffered was already flushed at its final !isPrefix return.
				return events, dropped, nil
			}
			return nil, 0, rerr
		}
		if !isPrefix {
			flush()
		}
	}
}

// replaceFileAtomic writes data to a temp file in path's directory and renames
// it over path. A reader therefore sees the whole old file or the whole new one,
// never a partial write: when the rename succeeds it replaces the destination
// atomically on both POSIX and Windows.
//
// On Windows the swap is renameOverOpenHandles — a POSIX-semantics rename that
// supersedes the destination even while the journal's own share-delete-opened
// readers/appenders (OpenShareDelete / OpenAppendShareDelete) hold it in
// another session (#3555). It can still fail against a FOREIGN handle opened
// without FILE_SHARE_DELETE (antivirus, an editor, an older fak binary), so a
// short bounded retry-with-backoff remains as a backstop; if contention
// outlasts it the error is returned and the caller leaves the file
// un-compacted (fail-open). POSIX renames over an open file fine, so there the
// first attempt succeeds.
func replaceFileAtomic(path string, data []byte) error {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".journal-compact-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName) // a no-op once the rename below has moved it away
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	// Retry the swap across a transient Windows sharing violation from a
	// FOREIGN no-share-delete handle: one initial attempt plus backoffs summing
	// to ~31ms, comfortably past a scanner's open→read→close. The journal's own
	// readers/appenders no longer contend at all (share-delete openers +
	// POSIX-semantics rename, #3555), so on both POSIX and Windows the first
	// attempt normally succeeds and the loop exits immediately.
	var rerr error
	for _, backoff := range []time.Duration{0, time.Millisecond, 2 * time.Millisecond, 4 * time.Millisecond, 8 * time.Millisecond, 16 * time.Millisecond} {
		if backoff > 0 {
			time.Sleep(backoff)
		}
		if rerr = renameOverOpenHandles(tmpName, path); rerr == nil {
			return nil
		}
	}
	return rerr
}
