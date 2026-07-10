package loopmgr

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// SchemaSegmentSeal is the schema tag of a manifest seal row (see segmentSeal).
const SchemaSegmentSeal = "fak.loop-segment-seal.v1"

// manifestSuffix names the sealed-segment manifest: a tiny, hash-chained sidecar with
// one row per rotation. It is NOT a *.jsonl/.log name so growthgate's grower filter
// skips it, and it grows one line per rotation (not per append), so it never becomes a
// hot file. The manifest is what carries hash-chain continuity ACROSS segments: each
// row binds one sealed segment's content hashes and links to the prior row.
const manifestSuffix = ".segments"

// segmentIndexWidth is the zero-pad width Rotate uses when naming a sealed segment
// "<path>.NNN". Padding is cosmetic: segments are ordered by their PARSED numeric
// index, not lexically, so indices past 999 still order correctly.
const segmentIndexWidth = 3

// RotateResult reports the outcome of a Rotate call.
type RotateResult struct {
	Rotated      bool   `json:"rotated"`
	SealedPath   string `json:"sealed_path,omitempty"`
	SealedIndex  uint64 `json:"sealed_index,omitempty"`
	SealedEvents int    `json:"sealed_events,omitempty"`
	SealedBytes  int64  `json:"sealed_bytes,omitempty"`
	Reason       string `json:"reason,omitempty"`
}

// segmentSeal is one hash-chained manifest row, written by Rotate when it seals a
// segment. It binds the sealed segment's content (FirstHash/FinalHash/Events) and
// links to the previous seal (PrevSealHash), so the manifest is itself a tamper-
// evident chain that certifies the order and integrity of every sealed segment. Hash
// is sha256 over the row with Hash="" — the same self-hash discipline the event chain
// uses.
type segmentSeal struct {
	Schema       string `json:"schema"`
	Index        uint64 `json:"index"`
	Events       int    `json:"events"`
	FirstHash    string `json:"first_hash"`
	FinalHash    string `json:"final_hash"`
	PrevSealHash string `json:"prev_seal_hash,omitempty"`
	Hash         string `json:"hash"`
}

// Rotate seals the active ledger segment at path into an immutable "<path>.NNN" file
// and records a hash-chained seal row in the "<path>.segments" manifest binding that
// segment's first/final event hashes. Appends then continue into a fresh active
// segment. Rotation bounds the hot active file — the one Append flock-serializes and
// growthgate size-caps — without dropping or rewriting any durable event; every sealed
// segment stays on disk and the whole history is re-verifiable across the seams via
// LoadAll, which walks the manifest chain and cross-checks each sealed segment against
// its seal.
//
// It is a no-op (Rotated=false) when the active file is absent, empty, or — with
// minBytes>0 — still under minBytes, so an operator or cron can call it cheaply on a
// schedule and it only seals once the active file has actually grown. The whole
// operation runs under the same cross-process append lock as Append, so it never races
// a concurrent writer, and a ledger whose active chain does not verify is refused
// rather than sealed, so a broken tail is never frozen into an immutable segment.
func Rotate(path string, minBytes int64) (RotateResult, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return RotateResult{}, errors.New("loop ledger path is required")
	}
	var res RotateResult
	err := withLedgerLock(path, appendLockWait, func() error {
		fi, err := os.Stat(path)
		if errors.Is(err, os.ErrNotExist) {
			res = RotateResult{Reason: "no active ledger"}
			return nil
		}
		if err != nil {
			return fmt.Errorf("stat loop ledger: %w", err)
		}
		if fi.Size() == 0 {
			res = RotateResult{Reason: "active ledger empty"}
			return nil
		}
		if minBytes > 0 && fi.Size() < minBytes {
			res = RotateResult{Reason: "active ledger under threshold"}
			return nil
		}
		// Verify the active chain before sealing — never freeze a broken tail into an
		// immutable segment. Load reads the active segment only, which is exactly the
		// chain being sealed.
		events, err := Load(path)
		if err != nil {
			return fmt.Errorf("verify active ledger before rotate: %w", err)
		}
		if len(events) == 0 {
			res = RotateResult{Reason: "active ledger has no events"}
			return nil
		}
		// Read + verify the existing manifest chain so the new seal links onto a sound
		// tip (and so we never append onto a tampered manifest).
		seals, err := loadSeals(path)
		if err != nil {
			return fmt.Errorf("read segment manifest before rotate: %w", err)
		}
		prevSealHash := ""
		if len(seals) > 0 {
			prevSealHash = seals[len(seals)-1].Hash
		}
		idx, err := nextSegmentIndex(path)
		if err != nil {
			return err
		}
		sealed := segmentPath(path, idx)
		// Rename first: once the active file is moved aside it is immutable, so the seal
		// we write next describes a file that can no longer change under us. A crash in
		// the one-syscall gap leaves a sealed file with no seal row, which LoadAll
		// surfaces as a torn rotation (fail-visible) rather than silent corruption.
		if err := os.Rename(path, sealed); err != nil {
			return fmt.Errorf("seal loop ledger segment: %w", err)
		}
		_ = os.Remove(path + tailSuffix)
		seal := segmentSeal{
			Schema:       SchemaSegmentSeal,
			Index:        idx,
			Events:       len(events),
			FirstHash:    events[0].Hash,
			FinalHash:    events[len(events)-1].Hash,
			PrevSealHash: prevSealHash,
		}
		seal.Hash = hashSeal(seal)
		if err := appendSeal(path, seal); err != nil {
			return fmt.Errorf("record segment seal: %w", err)
		}
		res = RotateResult{
			Rotated:      true,
			SealedPath:   sealed,
			SealedIndex:  idx,
			SealedEvents: len(events),
			SealedBytes:  fi.Size(),
		}
		return nil
	})
	if err != nil {
		return RotateResult{}, err
	}
	return res, nil
}

// ShouldRotate reports whether the active ledger at path has grown to at least minBytes
// (and is therefore due for a Rotate), together with its current size. It is a cheap
// O(1) stat with NO lock and NO chain read, so a hot producer or a scheduled sweep can
// gate the (locking, chain-verifying) Rotate call on it and pay the rotation cost only
// once the active file has actually crossed the bound — keeping the size check off the
// append hot path that issue #3465 flagged as growing with the ledger. A missing or
// empty file, or minBytes<=0 ("no bound configured"), reports not-due.
//
// This is the cheap trigger a production auto-rotation wiring gates on; Rotate is the
// seam that bounds the hot file, but nothing invokes it on a schedule yet. Wiring that
// in front of the cumulative Load consumers — which must move to LoadAll first so they
// keep counting sealed history rather than silently dropping it at a rotation boundary —
// is the remaining activation work. Rotate stays the source of truth: it re-checks the
// size under the append lock, so a false positive here only ever costs one no-op Rotate,
// never a mis-seal.
func ShouldRotate(path string, minBytes int64) (bool, int64, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return false, 0, errors.New("loop ledger path is required")
	}
	if minBytes <= 0 {
		return false, 0, nil
	}
	fi, err := os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		return false, 0, nil
	}
	if err != nil {
		return false, 0, fmt.Errorf("stat loop ledger: %w", err)
	}
	return fi.Size() >= minBytes, fi.Size(), nil
}

// LoadAll walks the ledger's sealed segments (oldest first) and then the active
// segment, returning the full event history across all segments. Unlike Load — which
// reads only the active segment for the hot per-tick fold — LoadAll is the audit-grade
// reader that proves nothing was dropped, forged, or spliced at a rotation boundary:
// it verifies the hash-chained manifest, then requires every sealed segment to match
// its manifest seal (event count + first/final hash) exactly. Any swapped, truncated,
// or missing segment, or any tampered manifest row, is rejected. With no sealed
// segments LoadAll is exactly Load(path).
func LoadAll(path string) ([]Event, error) {
	seals, err := loadSeals(path)
	if err != nil {
		return nil, err
	}
	segFiles, err := sealedSegmentFiles(path)
	if err != nil {
		return nil, fmt.Errorf("list loop ledger segments: %w", err)
	}
	if len(segFiles) != len(seals) {
		return nil, fmt.Errorf("segment manifest lists %d seal(s) but %d sealed segment file(s) exist", len(seals), len(segFiles))
	}
	sealByIndex := make(map[uint64]segmentSeal, len(seals))
	for _, s := range seals {
		sealByIndex[s.Index] = s
	}

	var all []Event
	for _, sf := range segFiles {
		seal, ok := sealByIndex[sf.idx]
		if !ok {
			return nil, fmt.Errorf("%s: no manifest seal for segment index %d", filepath.Base(sf.path), sf.idx)
		}
		evs, err := Load(sf.path)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", filepath.Base(sf.path), err)
		}
		if err := checkSegmentAgainstSeal(evs, seal, filepath.Base(sf.path)); err != nil {
			return nil, err
		}
		all = append(all, evs...)
	}

	active, err := Load(path)
	if err != nil {
		return nil, err
	}
	return append(all, active...), nil
}

// checkSegmentAgainstSeal verifies a loaded sealed segment matches its manifest seal.
func checkSegmentAgainstSeal(evs []Event, seal segmentSeal, name string) error {
	if len(evs) != seal.Events {
		return fmt.Errorf("%s: %d event(s), manifest seal records %d", name, len(evs), seal.Events)
	}
	if len(evs) == 0 {
		return nil
	}
	if evs[0].Hash != seal.FirstHash {
		return fmt.Errorf("%s: first-event hash %q does not match manifest seal %q", name, evs[0].Hash, seal.FirstHash)
	}
	if evs[len(evs)-1].Hash != seal.FinalHash {
		return fmt.Errorf("%s: final-event hash %q does not match manifest seal %q", name, evs[len(evs)-1].Hash, seal.FinalHash)
	}
	return nil
}

// loadSeals reads and verifies the manifest chain: every row must carry the seal
// schema, re-hash to its stored Hash, and link to the prior row via PrevSealHash. A
// missing manifest yields (nil, nil). Any break is a hard error — the manifest is the
// cross-segment tamper-evidence, so a forked/edited manifest must never be trusted.
func loadSeals(path string) ([]segmentSeal, error) {
	b, err := os.ReadFile(path + manifestSuffix)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read segment manifest: %w", err)
	}
	var out []segmentSeal
	prev := ""
	for lineNo, raw := range strings.Split(string(b), "\n") {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		var s segmentSeal
		if err := json.Unmarshal([]byte(line), &s); err != nil {
			return nil, fmt.Errorf("segment manifest line %d: decode: %w", lineNo+1, err)
		}
		if s.Schema != SchemaSegmentSeal {
			return nil, fmt.Errorf("segment manifest line %d: schema = %q, want %q", lineNo+1, s.Schema, SchemaSegmentSeal)
		}
		if s.PrevSealHash != prev {
			return nil, fmt.Errorf("segment manifest line %d: prev_seal_hash = %q, want %q", lineNo+1, s.PrevSealHash, prev)
		}
		if got := hashSeal(s); got != s.Hash {
			return nil, fmt.Errorf("segment manifest line %d: hash = %q, want %q", lineNo+1, s.Hash, got)
		}
		prev = s.Hash
		out = append(out, s)
	}
	return out, nil
}

// appendSeal appends one seal row to the manifest. Callers hold the ledger lock.
func appendSeal(path string, seal segmentSeal) error {
	line, err := json.Marshal(seal)
	if err != nil {
		return fmt.Errorf("marshal segment seal: %w", err)
	}
	f, err := os.OpenFile(path+manifestSuffix, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("open segment manifest: %w", err)
	}
	defer f.Close()
	if _, err := f.Write(append(line, '\n')); err != nil {
		return fmt.Errorf("write segment seal: %w", err)
	}
	return nil
}

// hashSeal computes a seal row's self-hash over its content with Hash="" — the same
// tamper-evidence discipline the event chain uses.
func hashSeal(seal segmentSeal) string {
	seal.Hash = ""
	b, err := json.Marshal(seal)
	if err != nil {
		panic(err)
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// segmentFile pairs a sealed segment's parsed numeric index with its path.
type segmentFile struct {
	idx  uint64
	path string
}

// segmentPath is the sealed-segment name for a given index: "<path>.NNN".
func segmentPath(path string, idx uint64) string {
	return fmt.Sprintf("%s.%0*d", path, segmentIndexWidth, idx)
}

// sealedSegmentFiles lists the sealed segments for the ledger at path in chain order
// (oldest first, by parsed numeric index). A sealed segment is "<path>.NNN" with an
// all-digits suffix; the active file, the .tail/.lock/.segments sidecars, and any
// .broken-* repair archive are all excluded because their suffix is not all digits.
func sealedSegmentFiles(path string) ([]segmentFile, error) {
	dir := filepath.Dir(path)
	prefix := filepath.Base(path) + "."
	entries, err := os.ReadDir(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	var segs []segmentFile
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasPrefix(name, prefix) {
			continue
		}
		idx, err := parseSegmentIndex(name[len(prefix):])
		if err != nil {
			continue
		}
		segs = append(segs, segmentFile{idx: idx, path: filepath.Join(dir, name)})
	}
	sort.Slice(segs, func(i, j int) bool { return segs[i].idx < segs[j].idx })
	return segs, nil
}

// sealedSegments returns just the ordered sealed-segment paths.
func sealedSegments(path string) ([]string, error) {
	segs, err := sealedSegmentFiles(path)
	if err != nil {
		return nil, err
	}
	out := make([]string, len(segs))
	for i, s := range segs {
		out[i] = s.path
	}
	return out, nil
}

// nextSegmentIndex returns one past the highest existing sealed index, so a new seal
// never reuses or renumbers an index (renumbering would break the manifest chain).
// With no sealed segments yet it returns 1.
func nextSegmentIndex(path string) (uint64, error) {
	segs, err := sealedSegmentFiles(path)
	if err != nil {
		return 0, fmt.Errorf("list loop ledger segments: %w", err)
	}
	var max uint64
	for _, s := range segs {
		if s.idx > max {
			max = s.idx
		}
	}
	return max + 1, nil
}

// parseSegmentIndex accepts an all-digits suffix and returns its numeric value; any
// non-digit (e.g. "tail", "lock", "segments", "broken-7") is rejected so those
// siblings are never mistaken for a sealed segment.
func parseSegmentIndex(suffix string) (uint64, error) {
	if suffix == "" {
		return 0, errors.New("empty segment suffix")
	}
	for _, r := range suffix {
		if r < '0' || r > '9' {
			return 0, fmt.Errorf("non-numeric segment suffix %q", suffix)
		}
	}
	return strconv.ParseUint(suffix, 10, 64)
}
