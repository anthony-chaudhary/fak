// Package rsl is the git Reference State Log — a forge-independent, append-only,
// hash-chained record of every observed trunk ref transition. It is the offline
// no-force-push proof (#3190, borrowed from gittuf's RSL, OpenSSF/NDSS 2025):
// GitHub's server-side ruleset (non_fast_forward + required_signatures) and the
// local push/hook refusals PREVENT a rewrite but leave no PORTABLE artifact an
// auditor who never trusted the forge can re-check. This package is that artifact.
//
// WHAT IT RECORDS. One Row per observed transition of a ref: {ref, old_sha,
// new_sha, signer} plus the chain link {prev_hash, hash} and a monotonic Seq.
// The writer records what it OBSERVED — old_sha is the value the ref held before
// the move, new_sha the value after. A fast-forward-only history is one where,
// per ref, every transition CONTINUES the recorded head (old_sha == the prior
// new_sha) and never revisits a target already seen; a rewind/force-push shows
// up as a transition whose old_sha does NOT match the recorded head, or whose
// new_sha returns to a prior target. Verify detects both — PURELY, over the
// recorded log, with no git remote and no object-graph walk.
//
// WHAT IT GUARANTEES (and does not).
//
//   - TAMPER-EVIDENT: every row carries the hash of the previous row's hash
//     chained with its own content — the SAME construction internal/journal uses
//     (see chainHash). A single edited byte breaks the link at that row and
//     Verify fails there. Like the decision journal, the RSL does not PREVENT a
//     privileged edit; it makes one DETECTABLE.
//   - FAST-FORWARD-WITNESSING: Verify FAILS, naming the offending ref, on a
//     non-fast-forward gap — the offline proof the trunk was never rewritten.
//   - DETECT-ONLY, ADVISORY: an RSL detects a rewritten trunk AFTER the fact, it
//     does not block one. The server-side ruleset + hooks stay the prevention
//     layer (issue #3190 Fences).
//
// SIGNING. Attribution is operator-supplied Ed25519 (per the witness-attestation
// signing decision). It is OPTIONAL: unsigned rows are DETECT-ONLY — still
// tamper-evident via the hash chain, but not cryptographically attributable. The
// Signer interface is the seam; AppendSigned wires it. This package ships the
// unsigned path fully and the seam, never a key.
package rsl

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// Row is one observed ref transition — the on-disk JSONL schema AND the unit the
// pure Verify folds over. Field order in the hash pre-image (see chainHash) is
// fixed; do not reorder without invalidating every existing log.
type Row struct {
	Seq      uint64 `json:"seq"`              // monotonic 1-based order anchor (global across refs)
	Ref      string `json:"ref"`              // the ref that moved, e.g. refs/heads/main
	OldSHA   string `json:"old_sha"`          // the ref's value BEFORE the transition (the observed parent)
	NewSHA   string `json:"new_sha"`          // the ref's value AFTER the transition
	Signer   string `json:"signer,omitempty"` // operator identity attributing the row ("" = unsigned)
	Sig      string `json:"sig,omitempty"`    // detached signature over Hash ("" = unsigned/detect-only)
	PrevHash string `json:"prev_hash"`        // hash of the previous row ("" at genesis)
	Hash     string `json:"hash"`             // chainHash(PrevHash, this row)
}

// Signer is the optional operator-supplied signing seam. When a non-nil Signer
// is passed to AppendSigned the row is attributed (Row.Signer = Identity()) and
// its chain hash is signed (Row.Sig = Sign(hash)); when nil (the plain Append
// path) rows are unsigned and DETECT-ONLY. Ed25519 is the intended
// implementation; this package defines the seam, not a key.
type Signer interface {
	// Identity names the signer recorded in Row.Signer (e.g. a key id/fingerprint).
	Identity() string
	// Sign returns a detached signature over the row's chain hash pre-image (the
	// hex Hash string). An error aborts the append before anything is written.
	Sign(hash string) (string, error)
}

// chainHash is the tamper-evident link: sha256 over the previous row's hash
// chained with this row's content fields (Seq, Ref, OldSHA, NewSHA, Signer), in
// declaration order, unit-separated by 0x1f so no concatenation collision is
// possible. This is the IDENTICAL construction internal/journal.chainHash uses;
// that function is unexported and typed to journal.Row, so the RSL applies the
// same primitive to its own ref-transition fields rather than importing it (the
// established in-repo pattern — cf. chainHashForTest in cmd/fak). PrevHash and
// Hash are excluded from the pre-image (PrevHash is the chained-in prefix, Hash
// the output); Signer IS chained so a swapped attribution breaks the link, but
// Sig is NOT — it is computed OVER Hash, so it cannot also be part of Hash's
// pre-image.
func chainHash(prev string, r Row) string {
	h := sha256.New()
	io.WriteString(h, prev)
	fmt.Fprintf(h, "\x1f%d\x1f%s\x1f%s\x1f%s\x1f%s", r.Seq, r.Ref, r.OldSHA, r.NewSHA, r.Signer)
	return hex.EncodeToString(h.Sum(nil))
}

// Append records one observed ref transition to the RSL at path, CONTINUING the
// existing hash chain (it recovers the chain head from any existing content, so
// a restart extends the same log instead of forking it). The caller supplies
// Ref/OldSHA/NewSHA (and may set Signer to an identity string for an unsigned
// attribution); Append stamps Seq, PrevHash, and Hash and returns the committed
// row. It is DETECT-ONLY: it does NOT enforce the fast-forward invariant — an
// RSL records what was OBSERVED, INCLUDING a rewrite, so the evidence survives;
// Verify is the surface that FAILS on a non-ff gap. Unsigned path (Sig stays "").
func Append(path string, row Row) (Row, error) {
	return appendRow(path, row, nil)
}

// AppendSigned is the signing seam wired: it attributes the row (Row.Signer =
// s.Identity()) BEFORE the hash is computed (so the attribution is chained), then
// signs the committed hash (Row.Sig = s.Sign(Hash)). A signing error aborts
// before the row is written. A nil Signer degrades to the plain unsigned Append.
func AppendSigned(path string, row Row, s Signer) (Row, error) {
	return appendRow(path, row, s)
}

// appendRow is the shared commit core for Append/AppendSigned: recover the chain
// head, stamp Seq/PrevHash/Hash, optionally sign, then append one JSONL line and
// flush it (per-row durability, mirroring journal.writeRow).
func appendRow(path string, row Row, s Signer) (Row, error) {
	seq, last, err := recoverHead(path)
	if err != nil {
		return Row{}, err
	}
	if s != nil {
		row.Signer = s.Identity()
	}
	row.Seq = seq + 1
	row.PrevHash = last
	row.Hash = chainHash(row.PrevHash, row)
	if s != nil {
		sig, serr := s.Sign(row.Hash)
		if serr != nil {
			return Row{}, fmt.Errorf("rsl: sign row seq %d: %w", row.Seq, serr)
		}
		row.Sig = sig
	}
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return Row{}, fmt.Errorf("rsl: create dir %s: %w", dir, err)
		}
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return Row{}, fmt.Errorf("rsl: open %s: %w", path, err)
	}
	defer f.Close()
	b, err := json.Marshal(row)
	if err != nil {
		return Row{}, err
	}
	if _, err := f.Write(append(b, '\n')); err != nil {
		return Row{}, fmt.Errorf("rsl: append %s: %w", path, err)
	}
	return row, nil
}

// recoverHead scans an existing RSL to recover the chain head (last seq + last
// hash) so an append continues the same chain. A missing file is genesis (seq 0,
// empty hash). It does NOT validate the chain (Verify's job) so a damaged log
// never blocks an append; a torn final line (crash mid-write) is tolerated by
// stopping at the last well-formed row. Mirrors journal.recoverHead.
func recoverHead(path string) (seq uint64, lastHash string, err error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, "", nil
		}
		return 0, "", fmt.Errorf("rsl: stat %s: %w", path, err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var r Row
		if err := json.Unmarshal(line, &r); err != nil {
			break // torn final line: stop at the last well-formed row (Verify catches real corruption)
		}
		seq = r.Seq
		lastHash = r.Hash
	}
	if err := sc.Err(); err != nil {
		return 0, "", fmt.Errorf("rsl: scan %s: %w", path, err)
	}
	return seq, lastHash, nil
}

// ReadRows reads all committed rows from an RSL file, in order — the READ side
// for a consumer (Verify, an exporter). A missing file is the empty log (nil,
// nil); a torn final line is tolerated by stopping at the last well-formed row.
// Genuine I/O errors are returned. Verify, not ReadRows, detects tampering.
func ReadRows(path string) ([]Row, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("rsl: read %s: %w", path, err)
	}
	defer f.Close()
	var out []Row
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var r Row
		if err := json.Unmarshal(line, &r); err != nil {
			break // torn final line: stop at the last well-formed row
		}
		out = append(out, r)
	}
	if err := sc.Err(); err != nil {
		return out, fmt.Errorf("rsl: scan %s: %w", path, err)
	}
	return out, nil
}

// VerifyFile reads an RSL file and runs Verify over it — the file-backed entry
// the CLI drives. A missing file is a trivially sound empty log (0, nil).
func VerifyFile(path string) (int, error) {
	rows, err := ReadRows(path)
	if err != nil {
		return 0, err
	}
	return Verify(rows)
}

// Verify is the PURE core (the DecidePush-style seam): it validates a recorded
// transition list end to end with no git remote and no object-graph walk, and
// returns the number of rows checked plus a non-nil error naming the FIRST
// violation. Two invariants are enforced, in row order:
//
//   - CHAIN (tamper-evidence): monotonic Seq, PrevHash continuity, and a
//     recomputed Hash — a single edited byte breaks the link and fails here.
//   - FAST-FORWARD (no-force-push): per ref, every transition must CONTINUE the
//     recorded head — its OldSHA must equal the ref's prior NewSHA — and must
//     never move to a target already seen for that ref. A rewind/force-push
//     violates one of these and fails, NAMING the ref. This is the offline proof
//     the trunk was fast-forward-only: the log itself, read purely, witnesses an
//     unbroken forward chain, or exhibits the exact row where it was rewritten.
//
// The first observation of a ref establishes its baseline (its OldSHA and NewSHA
// seed the seen-set and head); there is no prior state to contradict.
func Verify(rows []Row) (int, error) {
	var (
		prev    string
		wantSeq uint64
	)
	heads := map[string]string{}         // ref -> last recorded NewSHA (the chain head)
	seen := map[string]map[string]bool{} // ref -> set of SHAs already recorded (rewind guard)
	for i, row := range rows {
		wantSeq++
		if row.Seq != wantSeq {
			return i, fmt.Errorf("rsl: sequence gap: seq=%d want %d", row.Seq, wantSeq)
		}
		if row.PrevHash != prev {
			return i, fmt.Errorf("rsl: broken chain at seq %d: prev_hash=%q want %q", row.Seq, row.PrevHash, prev)
		}
		if got := chainHash(row.PrevHash, row); got != row.Hash {
			return i, fmt.Errorf("rsl: tampered row at seq %d: hash=%q recomputed %q", row.Seq, row.Hash, got)
		}

		refSeen := seen[row.Ref]
		if refSeen == nil {
			// First observation of this ref: seed the baseline, no prior state to check.
			refSeen = map[string]bool{}
			if row.OldSHA != "" {
				refSeen[row.OldSHA] = true
			}
			seen[row.Ref] = refSeen
		} else {
			// The transition must CONTINUE the recorded head (fast-forward), i.e. move
			// from the ref's current head — anything else is a non-ff gap (a rewrite the
			// log did not observe contiguously, the force-push signature).
			if head := heads[row.Ref]; row.OldSHA != head {
				return i, fmt.Errorf("rsl: non-fast-forward gap on ref %s at seq %d: old_sha=%q does not continue recorded head %q (trunk rewritten/force-pushed)", row.Ref, row.Seq, row.OldSHA, head)
			}
			// A fast-forward never returns to a target it already held.
			if refSeen[row.NewSHA] {
				return i, fmt.Errorf("rsl: non-fast-forward rewind on ref %s at seq %d: new_sha=%q revisits a prior target (trunk rewound/force-pushed)", row.Ref, row.Seq, row.NewSHA)
			}
		}
		refSeen[row.NewSHA] = true
		heads[row.Ref] = row.NewSHA
		prev = row.Hash
	}
	return len(rows), nil
}
