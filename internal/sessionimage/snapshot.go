package sessionimage

// snapshot.go — CAPTURE a session bundle into a fresh, addressable point-in-time SNAPSHOT
// (issue #2760, the operator-control epic's checkpoint op). It is the on-disk twin of the
// live capture DumpDir performs from an in-memory session: given a session's durable bundle
// dir, SnapshotDir mints a second, integrity-verified copy an operator can restore from
// later — while the source bundle (owned by the still-running session) is only read.
//
// # Snapshot vs branch
//
// A SNAPSHOT preserves the session id: it is the SAME session captured at a point in time,
// the substrate for crash-durable resume of THIS session. A BRANCH (branch.go) re-keys to a
// NEW id with a parent_id link: it is a SECOND session forked to diverge. So SnapshotDir
// shares session.json copy-on-write verbatim (nothing re-keyed), where BranchDir writes a
// fresh drive re-keyed to the branch id. Everything else — the content-addressed sharing
// that makes the capture cheap (hardlink, byte copy as the fallback) and the read-only
// treatment of the source — is identical, and both reuse the shared part-copy helper.
//
// # What is captured, what is shared, what is untouched
//
//   - SHARED : every part the source carries — session.json (the drive, verbatim: same id),
//              manifest.json, cas.json, index.json, trajectory.jsonl, witness.json,
//              quality.json, drive.json — linked copy-on-write, no page bytes re-serialized.
//   - FRESH  : image.json — the same identity/provenance as the source, re-indexed over the
//              snapshot's parts, with UpdatedUnix restamped and a migration-log entry
//              recording "checkpoint of <id> at <sha>" so the capture is an audited fact.
//   - SOURCE : never written. SnapshotDir LoadDir's the source first (so a capture always
//              starts from an integrity-verified bundle — a torn mid-write bundle fails
//              closed) and thereafter only READS it, so the running session is unaffected.

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/recall"
)

// SnapshotOptions configures an on-demand checkpoint. Reason is an optional operator note
// folded into the snapshot's migration-log entry. Now is an injected unix clock for
// deterministic stamps (0 => wall time).
type SnapshotOptions struct {
	Reason string
	Now    int64
}

// SnapshotDir captures the session bundle in srcDir into a fresh point-in-time snapshot at
// destDir, PRESERVING the session id, and returns the snapshot's Meta. It first LoadDir's the
// source (failing closed on a truncated or tampered bundle), shares every part copy-on-write,
// re-indexes the snapshot's integrity table, and writes an image.json that restamps
// UpdatedUnix and appends the checkpoint migration entry. The source bundle is only read,
// never written.
func SnapshotDir(srcDir, destDir string, opts SnapshotOptions) (Meta, error) {
	if strings.TrimSpace(srcDir) == "" || strings.TrimSpace(destDir) == "" {
		return Meta{}, fmt.Errorf("sessionimage: SnapshotDir requires a source and destination dir")
	}
	if srcDir == destDir {
		return Meta{}, fmt.Errorf("sessionimage: snapshot dir must differ from the source dir")
	}

	src, err := LoadDir(srcDir)
	if err != nil {
		return Meta{}, fmt.Errorf("sessionimage: snapshot: load source: %w", err)
	}

	// The checkpoint sha that pins WHICH source state this snapshot captured: the digest over
	// the source's image.json (its integrity root), read back from disk so the lineage names
	// the exact bytes on the record, not a reconstruction — the same discipline BranchDir uses.
	srcImageBytes, err := os.ReadFile(filepath.Join(srcDir, ImageFile))
	if err != nil {
		return Meta{}, fmt.Errorf("sessionimage: snapshot: read source %s: %w", ImageFile, err)
	}
	srcSHA := recall.Digest(srcImageBytes)

	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return Meta{}, err
	}

	// Share every part copy-on-write, INCLUDING session.json: a checkpoint preserves the drive
	// verbatim (same id), so — unlike a branch — nothing is re-keyed or written fresh here.
	if err := shareKnownParts(srcDir, destDir, "snapshot", ""); err != nil {
		return Meta{}, err
	}

	// The integrity index over the snapshot's parts (shared parts keep the source's digests —
	// content-addressed sharing is why the capture is cheap), then the fresh image.json:
	// same identity/provenance, UpdatedUnix restamped, and the migration entry that makes the
	// checkpoint an audited fact.
	parts, now, err := indexPartsAndStamp(destDir, opts.Now)
	if err != nil {
		return Meta{}, err
	}
	meta := src.Meta
	meta.UpdatedUnix = now
	meta.Parts = parts
	meta.Migrations = append(append([]Migration(nil), src.Meta.Migrations...), snapshotMigration(src.Meta, opts, srcSHA, now))
	if err := writeImageJSON(destDir, meta); err != nil {
		return Meta{}, err
	}
	return meta, nil
}

// snapshotMigration builds the migration-log entry that records the capture: the Reason
// carries the "checkpoint of <id> at <sha>" lineage, plus any operator note.
func snapshotMigration(srcMeta Meta, opts SnapshotOptions, srcSHA string, now int64) Migration {
	reason := fmt.Sprintf("checkpoint of %s at %s", srcMeta.SessionID, short(srcSHA))
	if note := strings.TrimSpace(opts.Reason); note != "" {
		reason += " (" + note + ")"
	}
	return Migration{WhenUnix: now, Reason: reason}
}
