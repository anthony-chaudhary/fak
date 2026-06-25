package sessionimage

// witness.go — the WITNESS sibling: the non-forgeable keep-bit, persisted into the
// portable image so a RESTORE can ask "did this effect already complete?" from
// integrity-checked bytes, not from the agent's own re-narration.
//
// # Why this part exists (the ACRFence distinction)
//
// The deepest hole in agent durable-execution is the semantic-rollback attack
// (ACRFence, arXiv:2603.20625): on restore, a framework re-executes side effects
// (a payment, a DB write) it already performed, because its checkpoint cannot
// distinguish ALREADY-COMPLETED from NEEDS-REPLAYING. fak already mints that
// distinction — the taskmgr keep-bit (VerifiedDone), raised only by a Witness that
// read the effect back from a source the process did not author — but until now the
// keep-bit lived ONLY in a per-process taskmgr snapshot. It did not ride the offload
// boundary, so a restore in a fresh process had nothing to consult.
//
// This sibling closes that: witness.json carries the VerifiedDone rung as a
// first-class, sha256-indexed part of the image. It is verified on Load exactly like
// every other part (verifyParts), so a tampered or truncated keep-bit fails the image
// closed rather than waving a forged "already done" through. A restore reads it back
// (Image.Witness / Image.VerifiedDone) and gates re-execution on real evidence.
//
// The keep-bit is reused verbatim from taskmgr.WitnessRecord — sessionimage does not
// mint a parallel verdict type (taskmgr imports no sessionimage/recall/session, so the
// import is one-way and cycle-free). An effect is keyed by a free-form EffectID (a task
// id, or "taskID/stepID") so a restore can ask about a specific effect.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

// WitnessEntry binds one effect's keep-bit to a stable id. EffectID is the join key a
// restore asks about ("task-refund-500", or "taskID/stepID" for a step-level effect);
// Record is the taskmgr verdict, reused unchanged so the rung's meaning is identical on
// both sides of the offload boundary.
type WitnessEntry struct {
	EffectID string                `json:"effect_id"`
	Record   taskmgr.WitnessRecord `json:"record"`
}

// WitnessSet is the on-disk shape of witness.json: a versioned, deterministically
// ordered list of per-effect keep-bits. Version mirrors the image Version so a reader
// fails closed on a format it does not recognize (a wrong-version keep-bit is worse
// than none). Entries are sorted by EffectID at write time, so a fixed set of effects
// serializes to byte-identical bytes — the same determinism the .faksession archive
// and the sha256 integrity index rely on.
type WitnessSet struct {
	Version string         `json:"version"`
	Entries []WitnessEntry `json:"entries"`
}

// writeWitness writes the witness sibling as a deterministically-ordered WitnessSet.
// It sorts a COPY by EffectID (never mutating the caller's slice) so the on-disk bytes
// — and therefore the part's digest and the packed archive — are stable across runs.
func writeWitness(path string, entries []WitnessEntry) error {
	sorted := make([]WitnessEntry, len(entries))
	copy(sorted, entries)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].EffectID < sorted[j].EffectID })
	set := WitnessSet{Version: Version, Entries: sorted}
	b, err := json.MarshalIndent(set, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

// Witness reads the persisted keep-bits back, or nil when the image carries none (a
// session that completed no witnessed effect). The bytes were already integrity-checked
// by LoadDir/verifyParts before this is reachable, so a returned entry is proven whole;
// this re-reads them only to decode. A version mismatch fails closed.
func (img *Image) Witness() ([]WitnessEntry, error) {
	b, err := os.ReadFile(filepath.Join(img.Dir, WitnessFile))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var set WitnessSet
	if err := json.Unmarshal(b, &set); err != nil {
		return nil, fmt.Errorf("sessionimage: bad %s: %w", WitnessFile, err)
	}
	if set.Version != Version {
		return nil, fmt.Errorf("sessionimage: witness version %q != %q", set.Version, Version)
	}
	return set.Entries, nil
}

// VerifiedDone is the one-call "already-completed?" a restore asks before re-firing an
// effect: it reports whether the image's keep-bit marks effectID as VerifiedDone. It is
// fail-closed on every uncertainty — an unreadable/bad witness sibling, a missing entry,
// or any verified state other than VerifiedDone (refused/unavailable/unknown) returns
// false, so a restore re-executes only when the keep-bit is ABSENT, never when it is
// present-but-not-confirmed. effectID is matched exactly.
func (img *Image) VerifiedDone(effectID string) bool {
	effectID = strings.TrimSpace(effectID)
	if effectID == "" {
		return false
	}
	entries, err := img.Witness()
	if err != nil {
		return false // fail closed: an unreadable keep-bit is not a confirmation
	}
	for _, e := range entries {
		if e.EffectID == effectID {
			return e.Record.VerifiedState == taskmgr.VerifiedDone
		}
	}
	return false
}
