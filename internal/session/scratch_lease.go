package session

// scratch_lease.go — the session scratchpad as a LEASED resource (issue #2420, part
// of the harness-native scratchpad-lifecycle epic #2392 / program #2387).
//
// THE GAP IT CLOSES.
// The harness scratchpad is a session-scoped temp dir that is permission-free by
// construction — the right idea with no lifecycle. Nothing owns it, nothing
// collects it (#2344 measures ~2 GB/day of orphans), and rewind/fork ignore it.
// This file binds that directory to a session trace as a leased resource with the
// whole lifecycle the affordance was missing: BIRTH (minted at session start, its
// path journaled), DEATH (garbage-collected at session end with a journaled GC
// event carrying bytes reclaimed + files dropped), FORK (each fork gets its own
// copy-on-write scratch dir so two forks cannot race one temp tree), and
// CHECKPOINT (a scratch digest is the third optional axis a checkpoint records, so
// a rewind can reproduce — or deliberately skip — scratch state). It generalizes
// #2345's carry-across-resume with the rest of the lifecycle.
//
// WHAT IT IS, WHAT IT IS NOT. A ScratchLease is a value: a TraceID + a directory
// path + a mint time. The lease does NOT own a session's drive (the Table does) and
// records nothing on the drive State — a scratchpad is a filesystem resource, not a
// drive field. The sessionledger ENTRY FORMAT is out of scope (#2416); this file
// journals through a small ScratchJournal seam (the same append-only, closed-
// vocabulary discipline rewind.go's RewindJournal takes) so the mint / GC / fork /
// checkpoint / restore events land wherever the host wires them. Every journal is
// nil-permissive: a caller with none wired gets the same lifecycle with no ledger.
//
// THE GC FENCE (#2420 confusion risk). GC is SESSION-END only — it reclaims the
// whole tree. A checkpoint is NOT a GC: Checkpoint archives a copy and never
// removes the live tree, so "record a digest each checkpoint" and "reclaim the tree
// at death" cannot be confused into reaping scratch mid-session.

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Closed-vocabulary event kinds recorded on the scratchpad journal. A scratchpad
// lifecycle never invents a free-text kind (the same discipline as EvRewind*).
const (
	EvScratchMinted         = "minted"          // a lease was minted at session start; Dir recorded
	EvScratchGC             = "gc"              // session-end GC reclaimed the tree; BytesReclaimed / FilesDropped set
	EvScratchForked         = "forked"          // a copy-on-write fork produced a child lease over its own dir
	EvScratchCheckpoint     = "checkpointed"    // a checkpoint archived the tree and recorded its Digest
	EvScratchRestored       = "restored"        // restore-with-scratch reproduced the checkpoint tree; Digest set
	EvScratchRestoreSkipped = "restore_skipped" // restore-without-scratch left the current tree untouched
)

// ScratchEvent is one journaled scratchpad record — the union of the fields the
// five lifecycle events carry. A minted/forked event fills TraceID + Dir; a GC
// event adds BytesReclaimed + FilesDropped; a checkpoint/restore event adds Digest.
// It is data-only and self-describing, so a ledger row needs no out-of-band context.
type ScratchEvent struct {
	Kind           string    `json:"kind"`                      // one of the EvScratch* tokens
	TraceID        string    `json:"trace_id,omitempty"`        // the session trace the lease is bound to
	Dir            string    `json:"dir,omitempty"`             // the scratch directory the event concerns
	BytesReclaimed int64     `json:"bytes_reclaimed,omitempty"` // GC only: total bytes freed
	FilesDropped   int       `json:"files_dropped,omitempty"`   // GC only: regular files removed
	Digest         string    `json:"digest,omitempty"`          // checkpoint/restore only: the content digest of the tree
	At             time.Time `json:"at"`                        // when the event was recorded
}

// ScratchJournal is the append-only ledger a scratchpad lifecycle records onto —
// the same seam shape rewind.go uses. It is nil-permissive at every call site, so a
// host with no ledger wired gets the identical lifecycle with the journaling elided.
type ScratchJournal interface {
	Record(e ScratchEvent) error
}

// ScratchLease binds a session-scoped scratch directory to a trace. It is the value
// the lifecycle passes around: minted by MintScratch, reclaimed by GC, branched by
// Fork, and snapshotted by Checkpoint. The zero lease is not usable — a lease is
// always the return of a mint/fork, so Dir is a real directory.
type ScratchLease struct {
	TraceID   string    `json:"trace_id"`
	Dir       string    `json:"dir"`
	CreatedAt time.Time `json:"created_at"`
}

// ScratchCheckpoint is the third optional axis a session checkpoint records for
// scratch state (alongside the drive/context axes the checkpoint epic already
// carries). It is a Digest — the checkable identity of the scratch tree at
// checkpoint time — plus an Archive path holding the copied bytes, so a rewind can
// REPRODUCE the tree (restore-with-scratch) rather than merely name it. The zero
// value means "no scratch axis recorded" (Digest == "").
type ScratchCheckpoint struct {
	TraceID string `json:"trace_id"`
	Digest  string `json:"digest"`
	Archive string `json:"archive"`
}

// IsZero reports whether the checkpoint carries no scratch axis — the safe default a
// rewind reads as "this checkpoint has no scratch to restore or skip".
func (c ScratchCheckpoint) IsZero() bool { return c.Digest == "" }

// MintScratch mints a fresh scratch directory for traceID under base (os.TempDir()
// when base is ""), records it on the journal (EvScratchMinted), and returns the
// lease — the BIRTH of the lifecycle. The directory is unique per mint (a session
// re-home mints a new one), so two live sessions never share a tree. now stamps the
// lease and the journal event (zero => time.Now), the injected-clock posture the
// rest of the package takes.
func MintScratch(base, traceID string, j ScratchJournal, now time.Time) (ScratchLease, error) {
	if now.IsZero() {
		now = time.Now()
	}
	if base == "" {
		base = os.TempDir()
	}
	if err := os.MkdirAll(base, 0o755); err != nil {
		return ScratchLease{}, err
	}
	dir, err := os.MkdirTemp(base, "fak-scratch-"+scratchSanitizeTrace(traceID)+"-*")
	if err != nil {
		return ScratchLease{}, err
	}
	lease := ScratchLease{TraceID: traceID, Dir: dir, CreatedAt: now}
	if err := recordScratch(j, ScratchEvent{Kind: EvScratchMinted, TraceID: traceID, Dir: dir, At: now}); err != nil {
		return lease, err
	}
	return lease, nil
}

// GC reclaims the whole scratch tree at session end (the DEATH of the lifecycle),
// journaling an EvScratchGC event with the bytes reclaimed and regular files
// dropped BEFORE the removal, so the ledger records what the reap freed even though
// the tree is then gone. GC is idempotent — reclaiming an already-removed lease
// reports zero bytes/files and does not error. It is SESSION-END only; a checkpoint
// never calls it.
func (l ScratchLease) GC(j ScratchJournal, now time.Time) (ScratchEvent, error) {
	if now.IsZero() {
		now = time.Now()
	}
	bytes, files, err := scratchTreeUsage(l.Dir)
	if err != nil {
		return ScratchEvent{}, err
	}
	if err := os.RemoveAll(l.Dir); err != nil {
		return ScratchEvent{}, err
	}
	ev := ScratchEvent{Kind: EvScratchGC, TraceID: l.TraceID, Dir: l.Dir, BytesReclaimed: bytes, FilesDropped: files, At: now}
	if err := recordScratch(j, ev); err != nil {
		return ev, err
	}
	return ev, nil
}

// Fork gives a child trace its OWN copy-on-write scratch dir under base, seeded with
// a copy of this lease's current contents, and journals EvScratchForked. Because
// each fork owns a distinct directory, two forks that write the SAME filename cannot
// collide — the isolation #2420 requires. The copy is eager (a real byte copy at
// fork time); "copy-on-write" is the SEMANTIC guarantee (neither fork sees the
// other's later writes), not a filesystem reflink dependency.
func (l ScratchLease) Fork(base, childTrace string, j ScratchJournal, now time.Time) (ScratchLease, error) {
	if now.IsZero() {
		now = time.Now()
	}
	if base == "" {
		base = os.TempDir()
	}
	if err := os.MkdirAll(base, 0o755); err != nil {
		return ScratchLease{}, err
	}
	dir, err := os.MkdirTemp(base, "fak-scratch-"+scratchSanitizeTrace(childTrace)+"-*")
	if err != nil {
		return ScratchLease{}, err
	}
	if err := scratchCopyTree(l.Dir, dir); err != nil {
		return ScratchLease{}, err
	}
	child := ScratchLease{TraceID: childTrace, Dir: dir, CreatedAt: now}
	if err := recordScratch(j, ScratchEvent{Kind: EvScratchForked, TraceID: childTrace, Dir: dir, At: now}); err != nil {
		return child, err
	}
	return child, nil
}

// Digest is the content identity of the scratch tree: a sha256 over every regular
// file's repo-relative path and bytes, in sorted path order (so the digest is a pure
// function of content, not of walk order or wall-clock). An empty or missing tree
// digests to a stable sentinel. It is the checkable equality "the scratch was
// preserved" the checkpoint axis records.
func (l ScratchLease) Digest() (string, error) { return scratchTreeDigest(l.Dir) }

// Checkpoint records the scratch axis for a session checkpoint: it archives a copy
// of the live tree under archiveBase (os.TempDir() when ""), computes its Digest,
// journals EvScratchCheckpoint, and returns the ScratchCheckpoint. It NEVER removes
// the live tree — a checkpoint is not a GC (the #2420 fence) — so the session keeps
// writing scratch after a checkpoint is taken.
func (l ScratchLease) Checkpoint(archiveBase string, j ScratchJournal, now time.Time) (ScratchCheckpoint, error) {
	if now.IsZero() {
		now = time.Now()
	}
	if archiveBase == "" {
		archiveBase = os.TempDir()
	}
	if err := os.MkdirAll(archiveBase, 0o755); err != nil {
		return ScratchCheckpoint{}, err
	}
	archive, err := os.MkdirTemp(archiveBase, "fak-scratch-ckpt-"+scratchSanitizeTrace(l.TraceID)+"-*")
	if err != nil {
		return ScratchCheckpoint{}, err
	}
	if err := scratchCopyTree(l.Dir, archive); err != nil {
		return ScratchCheckpoint{}, err
	}
	digest, err := scratchTreeDigest(archive)
	if err != nil {
		return ScratchCheckpoint{}, err
	}
	cp := ScratchCheckpoint{TraceID: l.TraceID, Digest: digest, Archive: archive}
	if err := recordScratch(j, ScratchEvent{Kind: EvScratchCheckpoint, TraceID: l.TraceID, Dir: l.Dir, Digest: digest, At: now}); err != nil {
		return cp, err
	}
	return cp, nil
}

// Restore applies (or deliberately skips) a checkpoint's scratch axis onto target.
//
//   - includeScratch=true  — RESTORE-WITH-SCRATCH: the target tree is cleared and the
//     checkpoint's archived tree is copied back in, so target.Digest() reproduces the
//     checkpoint Digest. Journals EvScratchRestored.
//   - includeScratch=false — RESTORE-WITHOUT-SCRATCH: the current scratch is left
//     UNTOUCHED (a rewind that deliberately keeps live scratch state). Journals
//     EvScratchRestoreSkipped and makes zero tree mutations.
//
// A zero checkpoint (no scratch axis) is a no-op in either mode. This is the rewind
// consumer of the checkpoint axis — the checkpoint/rewind VERBS themselves (#2425 /
// #2426) stay out of scope; this only supplies the axis they read.
func (c ScratchCheckpoint) Restore(target ScratchLease, includeScratch bool, j ScratchJournal, now time.Time) error {
	if now.IsZero() {
		now = time.Now()
	}
	if c.IsZero() {
		return nil
	}
	if !includeScratch {
		return recordScratch(j, ScratchEvent{Kind: EvScratchRestoreSkipped, TraceID: target.TraceID, Dir: target.Dir, Digest: c.Digest, At: now})
	}
	if err := os.RemoveAll(target.Dir); err != nil {
		return err
	}
	if err := os.MkdirAll(target.Dir, 0o755); err != nil {
		return err
	}
	if err := scratchCopyTree(c.Archive, target.Dir); err != nil {
		return err
	}
	return recordScratch(j, ScratchEvent{Kind: EvScratchRestored, TraceID: target.TraceID, Dir: target.Dir, Digest: c.Digest, At: now})
}

// recordScratch journals e when j is non-nil — the single nil-permissive seam every
// lifecycle method records through, so a host with no ledger wired never nil-panics.
func recordScratch(j ScratchJournal, e ScratchEvent) error {
	if j == nil {
		return nil
	}
	return j.Record(e)
}

// scratchSanitizeTrace reduces a trace id to a filesystem-safe token for the temp-dir
// prefix (letters/digits/dash/underscore kept, everything else dropped), so a trace
// with a slash or colon cannot escape or malform the scratch path. An empty result
// falls back to "session".
func scratchSanitizeTrace(trace string) string {
	var b strings.Builder
	for _, r := range trace {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			b.WriteRune(r)
		}
	}
	if b.Len() == 0 {
		return "session"
	}
	return b.String()
}

// scratchTreeUsage sums the bytes and counts the regular files under dir — the pre-removal
// measurement a GC event reports. A missing dir reports zero/zero with no error (an
// already-reclaimed lease GCs idempotently). Only regular files count toward
// FilesDropped; directories are structure, not dropped payload.
func scratchTreeUsage(dir string) (int64, int, error) {
	var bytes int64
	var files int
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			if os.IsNotExist(err) && path == dir {
				return filepath.SkipAll
			}
			return err
		}
		if d.IsDir() {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		if info.Mode().IsRegular() {
			bytes += info.Size()
			files++
		}
		return nil
	})
	if err != nil {
		return 0, 0, err
	}
	return bytes, files, nil
}

// scratchTreeDigest hashes every regular file's repo-relative path and content under dir in
// sorted path order, so the digest is a pure function of the tree's content. A
// missing/empty tree digests to a stable sentinel (the sha256 of a fixed marker), so
// "no scratch" is a checkable value, not an error.
func scratchTreeDigest(dir string) (string, error) {
	type entry struct {
		rel  string
		path string
	}
	var entries []entry
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			if os.IsNotExist(err) && path == dir {
				return filepath.SkipAll
			}
			return err
		}
		if d.IsDir() {
			return nil
		}
		if !d.Type().IsRegular() {
			return nil
		}
		rel, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}
		entries = append(entries, entry{rel: filepath.ToSlash(rel), path: path})
		return nil
	})
	if err != nil {
		return "", err
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].rel < entries[j].rel })
	h := sha256.New()
	io.WriteString(h, "fak-scratch-digest/1\x00")
	for _, e := range entries {
		io.WriteString(h, e.rel)
		h.Write([]byte{0})
		f, err := os.Open(e.path)
		if err != nil {
			return "", err
		}
		if _, err := io.Copy(h, f); err != nil {
			f.Close()
			return "", err
		}
		f.Close()
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// scratchCopyTree copies every regular file (preserving the directory structure and file
// mode) from src into dst, which is created if absent. It is the eager byte copy a
// fork and a checkpoint archive share. A missing src is treated as an empty tree
// (dst is created and left empty), so forking/checkpointing a never-written lease is
// well-defined.
func scratchCopyTree(src, dst string) error {
	if err := os.MkdirAll(dst, 0o755); err != nil {
		return err
	}
	return filepath.WalkDir(src, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			if os.IsNotExist(err) && path == src {
				return filepath.SkipAll
			}
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		if !d.Type().IsRegular() {
			return nil // skip symlinks/devices — scratch payload is regular files
		}
		return scratchCopyFile(path, target)
	})
}

// scratchCopyFile copies one regular file from src to dst, preserving its mode. dst's parent
// is assumed to exist (scratchCopyTree creates it via the directory walk).
func scratchCopyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	info, err := in.Stat()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, info.Mode().Perm())
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	if err := out.Close(); err != nil {
		return err
	}
	return nil
}
