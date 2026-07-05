package sessionimage

// quality_delta.go — the QUALITY-DELTA sibling (issue #1979, QA-dogfood spine
// #1961/QD-019): durable session checkpoints preserve runtime drive state, but not the
// quality-score movement observed during the session. A session that ran a scorecard
// (code-quality, milestone, dogfood, ...) mid-run had that before/after delta live only
// in whatever transient report the scorecard wrote — nothing rode the offload boundary,
// so a restored/inspected session could not answer "did quality move, and against what
// evidence?" without re-deriving it.
//
// This sibling closes that the same way witness.go closed the keep-bit gap: quality.json
// carries the latest known delta per scorecard as a first-class, sha256-indexed part of
// the image, verified on Load exactly like every other part (verifyParts). A restore or
// `fak snapshot info` reads it back (Image.QualityDeltas) instead of re-running a
// scorecard or trusting a narrated claim.
//
// One entry per CardKey (the scorecard's stable name, e.g. "code_quality",
// "milestone_scorecard") — a caller passing a fresh delta for a key already recorded
// replaces it, so the part always carries the LATEST delta per card, not a growing
// history (the done condition names "the latest session quality deltas").

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// QualityDelta is one scorecard's before/after movement observed during the session.
// CardKey names the scorecard (its stable identifier, e.g. "code_quality"); Before/After
// are that scorecard's own score unit (whatever the card reports — a count, a ratio, a
// debt total); Evidence points at the source that produced the numbers (a report path, a
// witness command, a commit SHA) so the delta is checkable, not just asserted.
type QualityDelta struct {
	CardKey  string  `json:"card_key"`
	Before   float64 `json:"before"`
	After    float64 `json:"after"`
	Evidence string  `json:"evidence,omitempty"`
}

// QualityDeltaSet is the on-disk shape of quality.json: a versioned, deterministically
// ordered list of per-card deltas. Version mirrors the image Version so a reader fails
// closed on a format it does not recognize.
type QualityDeltaSet struct {
	Version string         `json:"version"`
	Entries []QualityDelta `json:"entries"`
}

// writeQualityDelta writes the quality-delta sibling, deduplicated to the latest entry
// per CardKey and sorted by CardKey — so a fixed set of cards serializes to
// byte-identical bytes regardless of caller order, the same determinism the integrity
// index and the .faksession archive rely on. A later entry for a CardKey already seen
// replaces the earlier one (last-write-wins), keeping the part latest-only, never a
// growing history.
func writeQualityDelta(path string, entries []QualityDelta) error {
	latest := make(map[string]QualityDelta, len(entries))
	for _, e := range entries {
		latest[e.CardKey] = e
	}
	deduped := make([]QualityDelta, 0, len(latest))
	for _, e := range latest {
		deduped = append(deduped, e)
	}
	sort.Slice(deduped, func(i, j int) bool { return deduped[i].CardKey < deduped[j].CardKey })
	set := QualityDeltaSet{Version: Version, Entries: deduped}
	b, err := json.MarshalIndent(set, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

// QualityDeltas reads the persisted quality deltas back, or nil when the image carries
// none (a session no scorecard ran against). The bytes were already integrity-checked by
// LoadDir/verifyParts before this is reachable; this re-reads them only to decode. A
// version mismatch fails closed.
func (img *Image) QualityDeltas() ([]QualityDelta, error) {
	b, err := os.ReadFile(filepath.Join(img.Dir, QualityFile))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var set QualityDeltaSet
	if err := json.Unmarshal(b, &set); err != nil {
		return nil, fmt.Errorf("sessionimage: bad %s: %w", QualityFile, err)
	}
	if set.Version != Version {
		return nil, fmt.Errorf("sessionimage: quality delta version %q != %q", set.Version, Version)
	}
	return set.Entries, nil
}
