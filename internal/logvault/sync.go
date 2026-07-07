package logvault

// sync.go — the off-box replication rung (#2454, part of epic #2447): replicate
// the vault's replayed CURRENT state to a second vault directory, gating every
// outbound byte through the redaction/scrub pass before it leaves the source
// vault. The local vault (rung 1) shares the box's fate; this rung is the
// second-disk step of the durability ladder (another disk first, remote later).
//
// The gate is MANDATORY and fail-closed, in three layers:
//
//  1. Nothing leaves a vault that cannot prove itself: the source manifest
//     chain must verify before any byte is read, and every mirror is re-hashed
//     against its manifest-attested SHA256 on the way out — a corrupt or
//     tampered mirror is refused (recorded as a skip-error row on the
//     receiving manifest), never shipped.
//  2. Every outbound byte passes the scrub: mirror content, rel paths, source
//     ids, and the note strings this rung itself authors all run through the
//     redaction substrate (the #880-epic reference floor,
//     wirescreen.PIIRedactor, plus the operator-selected FAK_WIRE_REDACT arm
//     when one is active — the seam the #1983 event-stream scrub plugs into
//     when it lands). Redaction here is deliberately IRREVERSIBLE on the
//     receiving side — no CAS pin of the original, unlike wirescreen.Apply —
//     because the unredacted original stays in the SOURCE vault; the replica
//     must hold only what may leave the box. A path that itself bears a
//     redactable span is refused outright rather than rewritten (a rewritten
//     path would break the mirror<->manifest binding).
//  3. The receiving side re-runs verify: SyncTo appends hash-chained rows to
//     the DESTINATION's own manifest (one sync-copy row per mirror, SHA256 =
//     the scrubbed content hash), anchors the head, then re-derives the chain
//     and re-hashes the mirrors on arrival. A sync that cannot prove chain
//     integrity on the receiving side reports the problems — it never
//     silently passes.
//
// Incrementality: each sync-copy row notes the SOURCE hash it was scrubbed
// from ("src=<hex>"), so the next pass skips versions the destination already
// holds. A final sync-mark row binds the destination chain to the source
// chain head (seq + hash), the same out-of-band anchoring posture as
// vault-head.json.
//
// Remote-transport design note (#2454 done condition): of the two scoped
// candidates, #2300's GH-comment backup channel is for SMALL state (KB-scale,
// publicly visible — vault content is exactly the material the scrub laws say
// must not be public), so the chosen remote transport is #2254's multi-node
// path: a plain file transport (rsync/scp over the existing node lease fabric)
// moving the scrubbed replica directory to a peer box, where the receiver
// re-runs `fak logvault verify` against the replica's own manifest + anchor.
// This rung is deliberately transport-agnostic to make that composition
// trivial: it produces a self-verifying, already-scrubbed vault directory;
// any file transport can move it; nothing unscrubbed ever exists off-box.

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
	"github.com/anthony-chaudhary/fak/internal/wirescreen"
)

// Op names for receiving-manifest rows written by SyncTo.
const (
	OpSyncCopy = "sync-copy" // one scrubbed copy of a source-vault mirror
	OpSyncMark = "sync-mark" // end-of-pass row binding the destination chain to the source chain head
)

// SyncStats folds one sync pass's outcome, including the receiving-side
// verify that SyncTo always runs (the fail-closed arrival check).
type SyncStats struct {
	Files         int   // source mirrors considered (the manifest-replayed current state)
	Copied        int   // scrubbed copies written this pass
	Unchanged     int   // source versions the destination already held
	Errors        int   // mirrors refused/unreadable (recorded as skip-error rows, never shipped)
	Redacted      int   // scrub spans redacted across the shipped bytes
	CopyBytes     int64 // scrubbed bytes written to the destination
	VerifyRows    int   // receiving-side manifest chain rows verified on arrival
	VerifyChecked int   // receiving-side mirrors re-hashed on arrival
}

// SyncTo replicates the vault's replayed current state into dstDir (a second
// vault directory — another disk first, a remote box via a file transport
// later), gating every outbound byte through the redaction scrub and proving
// chain integrity on arrival. sample bounds the receiving-side mirror
// re-hash exactly as Verify does (0 = all).
//
// The returned problems are the receiving-side verify findings: a green sync
// returns (stats, nil, nil). Fail-closed contract: a source vault whose chain
// does not verify, or a destination that overlaps the source, refuses before
// any byte is copied.
func (v *Vault) SyncTo(dstDir string, sample int) (SyncStats, []VerifyProblem, error) {
	var stats SyncStats
	if strings.TrimSpace(dstDir) == "" {
		return stats, nil, errors.New("logvault: sync: empty destination")
	}
	srcAbs, err := filepath.Abs(v.Dir)
	if err != nil {
		return stats, nil, err
	}
	dstAbs, err := filepath.Abs(dstDir)
	if err != nil {
		return stats, nil, err
	}
	if pathWithin(dstAbs, srcAbs) || pathWithin(srcAbs, dstAbs) {
		return stats, nil, fmt.Errorf("logvault: sync: destination %s overlaps the source vault %s", dstDir, v.Dir)
	}

	// Layer 1: nothing leaves a vault that cannot prove itself.
	manPath := filepath.Join(srcAbs, ManifestName)
	if _, err := VerifyManifest(manPath); err != nil {
		return stats, nil, fmt.Errorf("logvault: sync refused — source chain: %w", err)
	}
	rows, err := ReadManifestRows(manPath)
	if err != nil {
		return stats, nil, err
	}
	states := replayStates(rows)
	var srcHeadSeq uint64
	var srcHeadHash string
	if n := len(rows); n > 0 {
		srcHeadSeq, srcHeadHash = rows[n-1].Seq, rows[n-1].Hash
	}

	// The destination is a vault of its own: single-writer lock + chained manifest.
	if err := os.MkdirAll(dstAbs, 0o755); err != nil {
		return stats, nil, err
	}
	lock, err := os.OpenFile(filepath.Join(dstAbs, LockName), os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return stats, nil, err
	}
	defer lock.Close()
	if err := flock.TryLock(lock); err != nil {
		if errors.Is(err, flock.ErrLockBusy) {
			return stats, nil, fmt.Errorf("logvault: another sync holds %s", filepath.Join(dstAbs, LockName))
		}
		return stats, nil, err
	}
	defer flock.Unlock(lock)

	dman, err := OpenManifest(dstAbs)
	if err != nil {
		return stats, nil, err
	}
	dmanOpen := true
	defer func() {
		if dmanOpen {
			dman.Close()
		}
	}()
	drows, err := ReadManifestRows(filepath.Join(dstAbs, ManifestName))
	if err != nil {
		return stats, nil, err
	}
	// have: source version already on the receiving side, keyed like replayStates.
	have := make(map[string]string, len(drows))
	for _, r := range drows {
		if r.Op != OpSyncCopy {
			continue
		}
		if src, ok := syncNoteSrc(r.Note); ok {
			have[r.Source+"\x00"+r.RelPath] = src
		}
	}

	dst := &Vault{Dir: dstAbs}
	rs := syncRedactors()
	// appendRow scrubs the note (layer 2 covers even the strings this rung
	// authors — an error message can quote source content) before chaining it.
	appendRow := func(row ManifestRow) error {
		note, _ := scrubString(rs, row.Note)
		row.Note = note
		row.TSUnixNano = time.Now().UnixNano()
		_, err := dman.Append(row)
		return err
	}

	keys := make([]string, 0, len(states))
	for k := range states {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		srcID, rel, _ := strings.Cut(k, "\x00")
		st := states[k]
		stats.Files++
		if have[k] == st.SHA256 {
			stats.Unchanged++
			continue
		}
		cleanID, idSpans := scrubString(rs, srcID)
		cleanRel, relSpans := scrubString(rs, rel)
		if idSpans+relSpans > 0 {
			// A secret-shaped path must not travel, and a rewritten path would
			// break the mirror<->manifest binding: refuse the file outright.
			// The skip row carries the SCRUBBED path so the refusal itself
			// never leaks what it refused.
			stats.Errors++
			if err := appendRow(ManifestRow{Op: OpSkip, Source: cleanID, RelPath: cleanRel, Note: "sync: path bears a redactable span — refused"}); err != nil {
				return stats, nil, err
			}
			continue
		}
		body, readErr := os.ReadFile(v.mirrorPath(srcID, rel))
		if readErr != nil {
			stats.Errors++
			if err := appendRow(ManifestRow{Op: OpSkip, Source: srcID, RelPath: rel, Note: "sync: mirror unreadable: " + readErr.Error()}); err != nil {
				return stats, nil, err
			}
			continue
		}
		if got := hashBytes(body); got != st.SHA256 {
			// Only manifest-attested bytes travel: a mirror that disagrees with
			// its own chain is corrupt or tampered — refused, never shipped.
			stats.Errors++
			if err := appendRow(ManifestRow{Op: OpSkip, Source: srcID, RelPath: rel, Note: "sync: mirror hash disagrees with the source manifest — refused"}); err != nil {
				return stats, nil, err
			}
			continue
		}
		scrubbed, spans := scrubBytes(rs, body)
		stats.Redacted += spans
		written, sha, err := writeMirror(dst.mirrorPath(srcID, rel), scrubbed)
		if err != nil {
			return stats, nil, err
		}
		note := "src=" + st.SHA256
		if spans > 0 {
			note += fmt.Sprintf(" redacted=%d", spans)
		}
		if err := appendRow(ManifestRow{
			Op:        OpSyncCopy,
			Source:    srcID,
			RelPath:   rel,
			Bytes:     written,
			SizeAfter: written,
			MTimeNano: st.MTimeNano,
			SHA256:    sha,
			Note:      note,
		}); err != nil {
			return stats, nil, err
		}
		stats.Copied++
		stats.CopyBytes += written
	}
	if err := appendRow(ManifestRow{Op: OpSyncMark, Note: fmt.Sprintf("src_head_seq=%d src_head_hash=%s copied=%d unchanged=%d errors=%d", srcHeadSeq, srcHeadHash, stats.Copied, stats.Unchanged, stats.Errors)}); err != nil {
		return stats, nil, err
	}
	seq, hash := dman.Head()
	dmanOpen = false
	if err := dman.Close(); err != nil { // flush + fsync BEFORE anchoring/verifying
		return stats, nil, err
	}
	if err := WriteAnchor(dstAbs, seq, hash); err != nil {
		return stats, nil, err
	}

	// Layer 3: the receiving side re-runs verify. A sync that cannot prove
	// chain integrity on arrival fails closed (problems reported, never dropped).
	chainRows, checked, problems, err := dst.Verify(sample)
	if err != nil {
		return stats, nil, fmt.Errorf("logvault: sync: receiving-side chain broken: %w", err)
	}
	stats.VerifyRows, stats.VerifyChecked = chainRows, checked
	return stats, problems, nil
}

// syncRedactors is the outbound scrub gate: the deterministic reference floor
// is ALWAYS on (sync never runs unscrubbed — there is no opt-out), and the
// operator-selected FAK_WIRE_REDACT arm composes on top when one is active.
func syncRedactors() []wirescreen.Redactor {
	rs := []wirescreen.Redactor{wirescreen.PIIRedactor()}
	if a := wirescreen.ActiveRedactor(); a != nil && a.Name() != rs[0].Name() {
		rs = append(rs, a)
	}
	return rs
}

// scrubBytes runs every redactor over body in order, replacing each proposed
// span with a "[REDACTED:<kind>]" placeholder, and returns the scrubbed bytes
// plus the span count. Unlike wirescreen.Apply it pins NO original: the
// unredacted bytes stay in the source vault, so the replica-side redaction is
// deliberately irreversible and needs no CAS witness to be safe.
func scrubBytes(rs []wirescreen.Redactor, body []byte) ([]byte, int) {
	ctx := context.Background()
	total := 0
	for _, r := range rs {
		spans := disjointSpans(r.Propose(ctx, body, "logvault-sync"), len(body))
		if len(spans) == 0 {
			continue
		}
		var out bytes.Buffer
		out.Grow(len(body))
		prev := 0
		for _, s := range spans {
			out.Write(body[prev:s.Start])
			out.WriteString("[REDACTED:")
			out.WriteString(s.Kind)
			out.WriteByte(']')
			prev = s.End
			total++
		}
		out.Write(body[prev:])
		body = out.Bytes()
	}
	return body, total
}

// scrubString is scrubBytes over a string field (rel paths, source ids, notes).
func scrubString(rs []wirescreen.Redactor, s string) (string, int) {
	if s == "" {
		return s, 0
	}
	b, n := scrubBytes(rs, []byte(s))
	return string(b), n
}

// disjointSpans sorts proposed spans (start asc, longer first at a tie) and
// drops out-of-bounds or overlapping ones, mirroring wirescreen.Apply's
// coalesce so a third-party redactor's overlapping proposal cannot corrupt
// the rewrite. Keeping the earlier-starting span redacts the LARGER secret.
func disjointSpans(in []wirescreen.Span, bodyLen int) []wirescreen.Span {
	if len(in) == 0 {
		return nil
	}
	sort.Slice(in, func(i, j int) bool {
		if in[i].Start != in[j].Start {
			return in[i].Start < in[j].Start
		}
		return in[i].End > in[j].End
	})
	out := make([]wirescreen.Span, 0, len(in))
	lastEnd := -1
	for _, s := range in {
		if s.Start < 0 || s.End > bodyLen || s.Start >= s.End {
			continue
		}
		if s.Start >= lastEnd {
			out = append(out, s)
			lastEnd = s.End
		}
	}
	return out
}

// syncNoteSrc extracts the "src=<hex>" source-version binding from a
// sync-copy row's note (the incremental-skip key).
func syncNoteSrc(note string) (string, bool) {
	for _, f := range strings.Fields(note) {
		if h, ok := strings.CutPrefix(f, "src="); ok {
			return h, true
		}
	}
	return "", false
}

// writeMirror writes data to <mirror>.part while hashing, then renames it into
// place — the same crash posture as copyToMirror (a torn write never
// masquerades as a mirror).
func writeMirror(mirror string, data []byte) (written int64, sha string, err error) {
	if err := os.MkdirAll(filepath.Dir(mirror), 0o755); err != nil {
		return 0, "", err
	}
	part := mirror + ".part"
	out, err := os.Create(part)
	if err != nil {
		return 0, "", err
	}
	n, err := out.Write(data)
	if err == nil {
		err = out.Sync() // a manifest row must never outlive the bytes behind it
	}
	if closeErr := out.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		os.Remove(part)
		return 0, "", err
	}
	os.Remove(mirror) // Windows: rename does not replace
	if err := os.Rename(part, mirror); err != nil {
		os.Remove(part)
		return 0, "", err
	}
	h := sha256.Sum256(data)
	return int64(n), hex.EncodeToString(h[:]), nil
}

func hashBytes(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}
