package sessionimage

// branch.go — FORK a checkpoint into a NEW durable session that diverges from its parent
// while the parent keeps running (issue #1200, part of Pillar 2 / #1193). This is the
// net-new lifecycle move beyond restore-in-place (#748 / Rehydrate): a restore brings a
// session back AS ITSELF; a branch mints a SECOND session from the same checkpoint so an
// operator can "try a risky path without losing my place".
//
// # Why a branch is cheap (copy-on-write, not a deep copy)
//
// The recall core image is a content-addressed page table: manifest.json lists pages by
// their sha256 digest, cas.json is the swap device keyed by that same digest, and the
// image is KVIncluded=false (the KV cache is rebuilt on the first turn — see the package
// doc). So forking does NOT re-serialize the page bytes: the branch SHARES the parent's
// page table (manifest.json) and swap device (cas.json) by REFERENCE — a hardlink when the
// filesystem supports it (one inode, one copy of the bytes), a byte copy only as the
// fallback. Because pages are content-addressed, a page the branch later CHANGES gets a
// NEW digest and is written fresh under that new address; the unchanged pages keep their
// old digests and stay shared. At branch-creation time NOTHING has diverged, so zero page
// bytes are written fresh — the copy-on-write invariant the acceptance names.
//
// # What is fresh, what is shared, what is untouched
//
//   - FRESH  : session.json (the branch's own drive, re-keyed to the branch id) and
//              image.json (new id, parent_id link, and a migration-log entry recording
//              "branched from <parent> at <sha>" so lineage is an audited fact).
//   - SHARED : manifest.json, cas.json, index.json, trajectory.jsonl, witness.json,
//              quality.json — the content-addressed parts, shared copy-on-write.
//   - PARENT : never written. BranchDir only READS the parent bundle (LoadDir verifies its
//              integrity first), so the parent session's state / lease / descriptor are
//              unaffected by the fork.

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/recall"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// BranchOptions configures a fork. BranchID is the new session's durable id (required —
// it must differ from the parent so the two are distinct in the C1 registry). ToModel /
// ToHost / Account / Residency, when set, override the branch's identity metadata (a fork
// may re-home to a different model or host); unset fields inherit the parent's. Reason is
// an optional operator note folded into the migration-log entry. Now is an injected unix
// clock for deterministic stamps (0 => wall time).
type BranchOptions struct {
	BranchID  string
	ToModel   string
	ToHost    string
	Account   string
	Residency string
	Reason    string
	Now       int64
}

// BranchDir forks the checkpoint in parentDir into a fresh bundle at branchDir, returning
// the branch's Meta. It first LoadDir's the parent (so a fork always starts from an
// integrity-verified checkpoint — a truncated or tampered parent fails closed), then
// writes the branch: a fresh drive re-keyed to the branch id, the content-addressed parts
// shared copy-on-write, and an image.json carrying the parent_id link and a migration-log
// entry recording the lineage. The parent bundle is only read, never written.
func BranchDir(parentDir, branchDir string, opts BranchOptions) (Meta, error) {
	branchID := strings.TrimSpace(opts.BranchID)
	if branchID == "" {
		return Meta{}, fmt.Errorf("sessionimage: BranchDir requires a BranchID")
	}

	parent, err := LoadDir(parentDir)
	if err != nil {
		return Meta{}, fmt.Errorf("sessionimage: branch: load parent: %w", err)
	}
	if branchID == parent.Meta.SessionID {
		return Meta{}, fmt.Errorf("sessionimage: branch id %q must differ from the parent id", branchID)
	}
	if parentDir == branchDir {
		return Meta{}, fmt.Errorf("sessionimage: branch dir must differ from the parent dir")
	}

	// The checkpoint sha that pins WHICH parent state this branch forked from: the digest
	// over the parent's image.json (its integrity root — it indexes every other part), read
	// back from disk so lineage names the exact bytes on the record, not a reconstruction.
	parentImageBytes, err := os.ReadFile(filepath.Join(parentDir, ImageFile))
	if err != nil {
		return Meta{}, fmt.Errorf("sessionimage: branch: read parent %s: %w", ImageFile, err)
	}
	parentSHA := recall.Digest(parentImageBytes)

	if err := os.MkdirAll(branchDir, 0o755); err != nil {
		return Meta{}, err
	}

	// (1) The branch's own drive — the parent's State re-keyed to the branch id, so the two
	// sessions are distinct rows in the live table and the C1 registry. Written fresh (the
	// TraceID differs), never shared.
	drive := parent.Drive
	drive.TraceID = branchID
	if err := writeSessionJSON(branchDir, drive); err != nil {
		return Meta{}, err
	}

	// (2) The content-addressed parts, shared copy-on-write (hardlink, byte copy only as the
	// fallback). image.json and session.json are excluded: image.json is re-minted below with
	// the branch identity, and session.json was just written fresh for the branch's drive.
	if err := shareKnownParts(parentDir, branchDir, "branch", SessionFile); err != nil {
		return Meta{}, err
	}

	// (3) The integrity index over the branch's parts (shared parts keep the parent's
	// digests — content-addressed sharing is why the fork is cheap), then the fresh
	// image.json: new id, parent_id link, inherited identity (with any override), and the
	// migration-log entry that makes the fork an audited fact.
	parts, now, err := indexPartsAndStamp(branchDir, opts.Now)
	if err != nil {
		return Meta{}, err
	}
	meta := Meta{
		Version:     Version,
		SessionID:   branchID,
		ParentID:    parent.Meta.SessionID,
		CreatedUnix: now,
		UpdatedUnix: now,
		AppVersion:  appversion.Current(),
		Model:       coalesce(opts.ToModel, parent.Meta.Model),
		Engine:      parent.Meta.Engine,
		Account:     coalesce(opts.Account, parent.Meta.Account),
		Residency:   coalesce(opts.Residency, parent.Meta.Residency),
		Host:        coalesce(opts.ToHost, parent.Meta.Host),
		Labels:      parent.Meta.Labels,
		Portability: parent.Meta.Portability,
		Parts:       parts,
		Migrations:  append(append([]Migration(nil), parent.Meta.Migrations...), branchMigration(parent.Meta, opts, parentSHA, now)),
	}
	if err := writeImageJSON(branchDir, meta); err != nil {
		return Meta{}, err
	}
	return meta, nil
}

// branchMigration builds the migration-log entry that records the fork: the Reason carries
// the human-readable "branched from <parent_id> at <sha>" lineage the acceptance names, and
// the model/host fields capture a re-home when the branch targets a different one.
func branchMigration(parentMeta Meta, opts BranchOptions, parentSHA string, now int64) Migration {
	reason := fmt.Sprintf("branched from %s at %s", parentMeta.SessionID, short(parentSHA))
	if note := strings.TrimSpace(opts.Reason); note != "" {
		reason += " (" + note + ")"
	}
	m := Migration{WhenUnix: now, Reason: reason}
	if opts.ToModel != "" && opts.ToModel != parentMeta.Model {
		m.FromModel, m.ToModel = parentMeta.Model, opts.ToModel
	}
	if opts.ToHost != "" && opts.ToHost != parentMeta.Host {
		m.FromHost, m.ToHost = parentMeta.Host, opts.ToHost
	}
	return m
}

// shareKnownParts shares each present part from srcDir into dstDir, except excludedPart.
// Missing optional parts are skipped, matching the bundle's sparse-part semantics.
func shareKnownParts(srcDir, dstDir, operation, excludedPart string) error {
	for _, name := range knownParts {
		if name == excludedPart {
			continue
		}
		src := filepath.Join(srcDir, name)
		if _, statErr := os.Stat(src); statErr != nil {
			continue
		}
		if err := shareOrCopy(src, filepath.Join(dstDir, name)); err != nil {
			return fmt.Errorf("sessionimage: %s: share %s: %w", operation, name, err)
		}
	}
	return nil
}

// shareOrCopy links src to dst so the two share one on-disk copy of the bytes (the
// copy-on-write share). It hardlinks when the filesystem supports it — the cheap path that
// makes a branch not a deep copy — and falls back to a byte copy when the link fails (a
// cross-volume dst, or a filesystem without hardlinks). Either way the bytes are identical,
// so the content-address integrity check over the branch is satisfied.
func shareOrCopy(src, dst string) error {
	_ = os.Remove(dst) // a stale dst would make os.Link fail with EEXIST
	if err := os.Link(src, dst); err == nil {
		return nil
	}
	return copyFileBytes(src, dst)
}

// copyFileBytes is the hardlink fallback: a plain byte copy of src to dst at mode 0644.
func copyFileBytes(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}

// writeSessionJSON marshals a drive State to session.json — the one place BranchDir writes
// a fresh drive, matching DumpDir's exact serialization (indented, mode 0644).
func writeSessionJSON(dir string, drive session.State) error {
	b, err := json.MarshalIndent(drive, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, SessionFile), b, 0o644)
}

// coalesce returns a when non-empty, else b — the "override, else inherit" rule for a
// branch's identity metadata.
func coalesce(a, b string) string {
	if strings.TrimSpace(a) != "" {
		return a
	}
	return b
}
