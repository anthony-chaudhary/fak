package logvault

// restore.go — the restore verb + restore drill (#2453, part of epic #2447). A
// backup nobody has restored from is a hypothesis; this file turns the vault's
// recoverability promise into a witnessed capability, in two layers:
//
//   - Restore copies one source's manifest-replayed state back OUT of the vault
//     into a target directory that is a fresh tree by default (never in-place
//     over a live store without an explicit Force grant). Every restored byte is
//     re-hashed against the manifest chain WHILE it is written — a restore that
//     copies bytes but cannot prove them is reported as a problem, never as
//     done — and restored chained journals (guard decision journals, usage
//     logs) are re-verified with their own verifiers so the restored chain is
//     proven sound end-to-end, not just byte-copied.
//   - Drill is the cadence hook: restore one source into a temp dir, verify,
//     append one durable DrillRow to the vault's drill-log (and optionally a
//     repo ledger), clean up. Run it on a schedule and the restore path cannot
//     rot unnoticed.
//
// `At` reconstructs an OLDER state: the manifest records, per row, the byte
// position its hash covers (SizeAfter), so any historical state is a verified
// prefix of either the current mirror (append-grown journals) or a retired
// .history/ version (rewritten files). Restore tries those candidates in order
// and admits only a prefix whose re-hash matches the chain.
//
// Untrusted-path discipline: rel paths come from the manifest, and although the
// chain is verified before any byte moves, a malicious writer could have chained
// a traversal path honestly. Every manifest-supplied rel path is therefore
// forced through the same refusal posture as internal/sessionimage's Unpack
// (no absolute paths, no "..", no drive-letter smuggling) before it may name a
// target file. Cold-park archives are restored as the banked archive FILE
// (hash-verified like any mirror) — restore never unpacks them, so the
// hardened-unpack size budgets do not come into play here.

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/usagelog"
)

// RestoreOptions selects what to restore and where.
type RestoreOptions struct {
	Source string // source id (a by-source/<id> subtree) — required
	To     string // target directory — required (callers default it to a FRESH dir)
	At     uint64 // reconstruct the state as of this manifest seq (0 = current head)
	Force  bool   // allow restoring into a non-empty existing directory (never the default)
}

// RestoreProblem is one file the restore could not prove: a hash mismatch, an
// unrestorable historical state, or a refused path. The acceptance bar for a
// sound restore is an EMPTY problem list.
type RestoreProblem struct {
	RelPath string
	Reason  string
}

// JournalCheck is the end-to-end verifier verdict for one restored chained
// journal: its own verifier re-run against the restored copy.
type JournalCheck struct {
	RelPath string
	Kind    string // "decision-journal" (internal/journal) | "usage-log" (internal/usagelog)
	Rows    int    // sound rows the verifier counted
	Err     string // "" = chain intact end-to-end
}

// RestoreReport folds one restore's outcome.
type RestoreReport struct {
	Source      string
	To          string
	HeadSeq     uint64 // the manifest seq the restore replayed to (At, clamped to the head)
	Files       int    // files restored AND re-hash-verified against the chain
	Bytes       int64  // bytes written to the target
	FromHistory int    // files reconstructed from .history/ retires rather than the current mirror
	Problems    []RestoreProblem
	Journals    []JournalCheck
}

// JournalFailures counts restored chained journals whose own verifier refused.
func (r RestoreReport) JournalFailures() int {
	n := 0
	for _, j := range r.Journals {
		if j.Err != "" {
			n++
		}
	}
	return n
}

// OK reports the acceptance condition: zero hash mismatches/problems and every
// restored chained journal verifying clean.
func (r RestoreReport) OK() bool { return len(r.Problems) == 0 && r.JournalFailures() == 0 }

// Restore copies one source's replayed state out of the vault into opts.To,
// re-hashing every restored byte against the manifest chain and re-running the
// chained-journal verifiers over restored journals. Fail-closed: a vault whose
// chain does not verify refuses before any byte is copied, a target that
// overlaps the vault is always refused, and a non-empty target needs an
// explicit Force grant. The vault is read-only throughout.
func (v *Vault) Restore(opts RestoreOptions) (RestoreReport, error) {
	rep := RestoreReport{Source: opts.Source, To: opts.To}
	if strings.TrimSpace(opts.Source) == "" {
		return rep, errors.New("logvault: restore: a source id is required")
	}
	if strings.TrimSpace(opts.To) == "" {
		return rep, errors.New("logvault: restore: a target directory is required")
	}

	// Nothing leaves a vault that cannot prove itself (the sync posture in reverse).
	manPath := filepath.Join(v.Dir, ManifestName)
	if _, err := VerifyManifest(manPath); err != nil {
		return rep, fmt.Errorf("logvault: restore refused — vault chain: %w", err)
	}
	rows, err := ReadManifestRows(manPath)
	if err != nil {
		return rep, err
	}
	if opts.At > 0 {
		cut := rows[:0:0]
		for _, r := range rows {
			if r.Seq <= opts.At {
				cut = append(cut, r)
			}
		}
		rows = cut
	}
	if n := len(rows); n > 0 {
		rep.HeadSeq = rows[n-1].Seq
	}
	states := replayStates(rows)
	var rels []string
	known := map[string]bool{}
	for k := range states {
		src, rel, _ := strings.Cut(k, "\x00")
		known[src] = true
		if src == opts.Source {
			rels = append(rels, rel)
		}
	}
	if len(rels) == 0 {
		ids := make([]string, 0, len(known))
		for id := range known {
			ids = append(ids, id)
		}
		sort.Strings(ids)
		return rep, fmt.Errorf("logvault: restore: no captured state for source %q at seq %d (captured sources: %s)",
			opts.Source, rep.HeadSeq, strings.Join(ids, ", "))
	}
	sort.Strings(rels)

	// Target discipline: never inside the vault (or the vault inside it) — Force
	// does not override that — and never over an existing non-empty tree without
	// the explicit grant (the "fresh directory by default" contract).
	vaultAbs, err := filepath.Abs(v.Dir)
	if err != nil {
		return rep, err
	}
	dstAbs, err := filepath.Abs(opts.To)
	if err != nil {
		return rep, err
	}
	if pathWithin(dstAbs, vaultAbs) || pathWithin(vaultAbs, dstAbs) {
		return rep, fmt.Errorf("logvault: restore: target %s overlaps the vault %s — refused", opts.To, v.Dir)
	}
	if ents, rdErr := os.ReadDir(dstAbs); rdErr == nil && len(ents) > 0 && !opts.Force {
		return rep, fmt.Errorf("logvault: restore: target %s is not empty — restoring over an existing tree needs an explicit Force grant (the default target is a fresh directory, never in-place over a live store)", opts.To)
	} else if rdErr != nil && !os.IsNotExist(rdErr) {
		return rep, rdErr
	}
	if err := os.MkdirAll(dstAbs, 0o755); err != nil {
		return rep, err
	}

	for _, rel := range rels {
		st := states[opts.Source+"\x00"+rel]
		if !safeRestoreRel(rel) {
			rep.Problems = append(rep.Problems, RestoreProblem{rel, "manifest rel path is not a safe in-tree name (traversal?) — refused"})
			continue
		}
		target := filepath.Join(dstAbs, filepath.FromSlash(rel))
		restored := false
		fromHistory := false
		for _, cand := range v.restoreCandidates(opts.Source, rel) {
			ok, written, cpErr := copyPrefixVerified(cand.path, target, st.SizeAfter, st.SHA256)
			if cpErr != nil {
				return rep, cpErr
			}
			if ok {
				rep.Files++
				rep.Bytes += written
				restored, fromHistory = true, cand.history
				break
			}
		}
		if !restored {
			rep.Problems = append(rep.Problems, RestoreProblem{rel, fmt.Sprintf("no vault copy re-hashes to the manifest-attested sha256 %s… over %d bytes (mirror and .history/ checked)", shortSHA(st.SHA256), st.SizeAfter)})
			continue
		}
		if fromHistory {
			rep.FromHistory++
		}
		if kind := chainedJournalKind(target); kind != "" {
			jc := JournalCheck{RelPath: rel, Kind: kind}
			var vErr error
			switch kind {
			case "usage-log":
				jc.Rows, vErr = usagelog.Verify(target)
			default:
				jc.Rows, vErr = journal.Verify(target)
			}
			if vErr != nil {
				jc.Err = vErr.Error()
			}
			rep.Journals = append(rep.Journals, jc)
		}
	}
	return rep, nil
}

// restoreCandidate is one vault-side file that may hold (a prefix of) the bytes
// a restore needs.
type restoreCandidate struct {
	path    string
	history bool
}

// restoreCandidates lists the vault files that can witness rel's state: the
// current mirror first, then every .history/ retire of the same original (in
// deterministic slot order). An `At`-state of an append-grown journal is a
// verified PREFIX of the current mirror; a rewritten file's older state lives
// whole (or as a prefix) in a retired slot — so candidates are tried by prefix
// re-hash, not by slot name alone (the slot is keyed by the retire-time content
// hash, which an earlier append state need not equal).
func (v *Vault) restoreCandidates(srcID, rel string) []restoreCandidate {
	cands := []restoreCandidate{{path: v.mirrorPath(srcID, rel), history: false}}
	histDir := filepath.Join(v.Dir, "by-source", srcID, ".history", filepath.Dir(filepath.FromSlash(rel)))
	base := filepath.Base(filepath.FromSlash(rel))
	ents, err := os.ReadDir(histDir)
	if err != nil {
		return cands
	}
	var slots []string
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		if orig, _, ok := parseHistorySlot(e.Name()); ok && orig == base {
			slots = append(slots, e.Name())
		}
	}
	sort.Strings(slots)
	for _, s := range slots {
		cands = append(cands, restoreCandidate{path: filepath.Join(histDir, s), history: true})
	}
	return cands
}

// copyPrefixVerified streams the first size bytes of from into target while
// hashing, and commits the copy (via a .part rename, the copyToMirror crash
// posture) ONLY when the streamed prefix re-hashes to wantSHA — the manifest
// chain's attestation. ok=false with a nil error means this candidate cannot
// witness the wanted state (missing, too short, or hash divergence); the caller
// tries the next candidate. Restored bytes are therefore chain-verified by
// construction, never trusted.
func copyPrefixVerified(from, target string, size int64, wantSHA string) (ok bool, written int64, err error) {
	f, err := os.Open(from)
	if err != nil {
		if os.IsNotExist(err) {
			return false, 0, nil
		}
		return false, 0, err
	}
	defer f.Close()
	if fi, sErr := f.Stat(); sErr != nil || fi.Size() < size {
		return false, 0, nil // shorter than the attested state: cannot hold it
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return false, 0, err
	}
	part := target + ".part"
	out, err := os.Create(part)
	if err != nil {
		return false, 0, err
	}
	h := sha256.New()
	written, err = io.CopyN(io.MultiWriter(out, h), f, size)
	if err == io.EOF && size == 0 {
		err = nil
	}
	if err == nil {
		err = out.Sync()
	}
	if closeErr := out.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		os.Remove(part)
		return false, 0, err
	}
	if hex.EncodeToString(h.Sum(nil)) != wantSHA {
		os.Remove(part)
		return false, 0, nil
	}
	os.Remove(target) // Windows: rename does not replace
	if err := os.Rename(part, target); err != nil {
		os.Remove(part)
		return false, 0, err
	}
	return true, written, nil
}

// safeRestoreRel refuses any manifest-supplied rel path that could name a file
// outside the restore target — the sessionimage Unpack safeName discipline
// applied to nested paths: forward-slash segments only, no absolute paths, no
// "."/".." hops, no backslashes or drive-letter colons a Windows join could
// reinterpret. The chain being intact does not make a path safe; a malicious
// writer can chain a traversal honestly.
func safeRestoreRel(rel string) bool {
	if rel == "" || strings.HasPrefix(rel, "/") || strings.Contains(rel, "\\") || filepath.IsAbs(rel) {
		return false
	}
	for _, seg := range strings.Split(rel, "/") {
		if seg == "" || seg == "." || seg == ".." || strings.ContainsRune(seg, ':') {
			return false
		}
	}
	return true
}

// chainedJournalKind peeks the first well-formed JSON line of a restored file
// and names the chained-journal verifier that owns it: a usagelog schema stamp
// wins, else the decision-journal row shape (always-present kind + seq +
// prev_hash/hash). Anything else — including logvault's own manifest schema
// ("op", not "kind") — is not a journal this restore can re-verify.
func chainedJournalKind(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var probe struct {
			Schema   string  `json:"schema"`
			Kind     string  `json:"kind"`
			Seq      *uint64 `json:"seq"`
			Hash     *string `json:"hash"`
			PrevHash *string `json:"prev_hash"`
		}
		if json.Unmarshal(line, &probe) != nil {
			return ""
		}
		if probe.Schema == usagelog.SchemaV1 {
			return "usage-log"
		}
		if probe.Kind != "" && probe.Seq != nil && probe.Hash != nil && probe.PrevHash != nil {
			return "decision-journal"
		}
		return ""
	}
	return ""
}

func shortSHA(sha string) string {
	if len(sha) > 12 {
		return sha[:12]
	}
	return sha
}

// DrillSchema stamps every drill-log row.
const DrillSchema = "fak-logvault-drill/1"

// DrillLogName is the vault-root ledger every drill run appends its row to —
// the durable "the restore path was exercised on <date> and passed/failed"
// record a cadence runner leaves behind.
const DrillLogName = "drill-log.jsonl"

// DrillRow is one drill run's durable record.
type DrillRow struct {
	Schema          string `json:"schema"`
	TSUnixNano      int64  `json:"ts_unix_nano"`
	Vault           string `json:"vault"`
	Source          string `json:"source"`
	HeadSeq         uint64 `json:"head_seq"`
	Files           int    `json:"files"`
	Bytes           int64  `json:"bytes"`
	FromHistory     int    `json:"from_history"`
	Mismatches      int    `json:"mismatches"`
	JournalsChecked int    `json:"journals_checked"`
	JournalsFailed  int    `json:"journals_failed"`
	Pass            bool   `json:"pass"`
	Err             string `json:"err,omitempty"` // a restore that refused outright (vs. one that ran and found problems)
}

// Drill restores one source into a fresh temp directory, verifies it (re-hash
// against the chain + chained-journal verifiers), appends one DrillRow to the
// vault's drill-log (and to ledgerPath when non-empty — e.g. a committed repo
// ledger), and removes the temp tree. source == "" picks the smallest captured
// source, so a cadence run stays cheap by default. A failed restore still
// journals its row — the drill's whole point is that a rotten restore path
// becomes a recorded red, not a silent skip.
func (v *Vault) Drill(source, ledgerPath string) (DrillRow, RestoreReport, error) {
	row := DrillRow{Schema: DrillSchema, TSUnixNano: time.Now().UnixNano(), Vault: v.Dir, Source: source}
	if source == "" {
		picked, err := v.smallestSource()
		if err != nil {
			return row, RestoreReport{}, err
		}
		source, row.Source = picked, picked
	}
	to, err := os.MkdirTemp("", "fak-logvault-drill-")
	if err != nil {
		return row, RestoreReport{}, err
	}
	defer os.RemoveAll(to)
	rep, restoreErr := v.Restore(RestoreOptions{Source: source, To: to})
	row.HeadSeq = rep.HeadSeq
	row.Files, row.Bytes, row.FromHistory = rep.Files, rep.Bytes, rep.FromHistory
	row.Mismatches = len(rep.Problems)
	row.JournalsChecked, row.JournalsFailed = len(rep.Journals), rep.JournalFailures()
	row.Pass = restoreErr == nil && rep.OK()
	if restoreErr != nil {
		row.Err = restoreErr.Error()
	}
	if err := appendDrillRow(filepath.Join(v.Dir, DrillLogName), row); err != nil {
		return row, rep, err
	}
	if ledgerPath != "" {
		if err := appendDrillRow(ledgerPath, row); err != nil {
			return row, rep, err
		}
	}
	return row, rep, restoreErr
}

// smallestSource picks the captured source with the fewest replayed bytes (id
// tiebreak) — the cheapest real restore a cadence drill can exercise.
func (v *Vault) smallestSource() (string, error) {
	rows, err := ReadManifestRows(filepath.Join(v.Dir, ManifestName))
	if err != nil {
		return "", err
	}
	totals := map[string]int64{}
	for k, st := range replayStates(rows) {
		src, _, _ := strings.Cut(k, "\x00")
		totals[src] += st.SizeAfter
	}
	if len(totals) == 0 {
		return "", errors.New("logvault: drill: the vault holds no captured state to restore")
	}
	ids := make([]string, 0, len(totals))
	for id := range totals {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	pick := ids[0]
	for _, id := range ids[1:] {
		if totals[id] < totals[pick] {
			pick = id
		}
	}
	return pick, nil
}

// appendDrillRow appends one JSONL row to path, creating parents as needed.
func appendDrillRow(path string, row DrillRow) error {
	b, err := json.Marshal(row)
	if err != nil {
		return err
	}
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	_, wErr := f.Write(append(b, '\n'))
	if sErr := f.Sync(); wErr == nil {
		wErr = sErr
	}
	if cErr := f.Close(); wErr == nil {
		wErr = cErr
	}
	return wErr
}
