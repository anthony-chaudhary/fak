package logvault

// Vault retention — the propose-by-default half of the memory-operation algebra
// applied to the log vault. The only forgetting the vault offers is bounded
// .history/ depth: superseded versions of a rewritten file pile up under
// by-source/<src>/.history/, and a retention pass keeps at most N of them per
// file. Everything else the vault holds is either a current mirror (never a GC
// target) or an append-only manifest row (never deleted — the hash chain is the
// audit trail). This mirrors internal/memq's stance: there is NO hard-delete of
// captured data, effects default to PROPOSED, and only an explicit --live grant
// mutates the vault, witnessing each prune with its own manifest row.

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

// OpGCPrune is the manifest op stamped when a retention pass reclaims one
// superseded .history/ version. Like skip-error it carries no capture SHA, so
// replayStates never folds it into mirror state — the chain simply witnesses
// its own pruning, and Verify keeps re-deriving cleanly across it.
const OpGCPrune = "gc-prune"

// GCPolicy is the retention policy a pass enforces. It is intentionally small:
// bounded .history/ depth is the whole lever.
type GCPolicy struct {
	// HistoryDepth is the maximum number of superseded versions to KEEP per
	// (source, rel-path) in .history/. Zero (or negative) means unlimited —
	// nothing is ever proposed for prune (the fail-safe default: keep everything).
	HistoryDepth int
}

// GCCandidate is one .history/ version a retention pass would prune (propose)
// or did prune (live).
type GCCandidate struct {
	Source   string // source id (the by-source/<id> directory name)
	RelPath  string // the ORIGINAL file's forward-slash rel path (not the .history slot name)
	HistFile string // vault-relative path of the .history/ slot file, forward slash
	SHA16    string // the 16-hex content slot in the slot file name
	Bytes    int64  // bytes the prune reclaims
}

// GCReport is the outcome of a retention pass.
type GCReport struct {
	Policy        GCPolicy      // the policy this pass applied
	Candidates    []GCCandidate // proposed (or, when Applied, the pruned set), deterministic order
	ReclaimBytes  int64         // total bytes across Candidates
	SkipErrorRows int           // advisory: skip-error rows in the manifest (noise — never deleted; the chain is append-only)
	Applied       bool          // true only under an explicit --live grant
}

// GC runs a retention pass. With live=false (the default) it PROPOSES: it walks
// the vault, computes what would be pruned, and returns the report without
// touching a single byte. With live=true it deletes the proposed slot files and
// appends one OpGCPrune manifest row per deletion, all under the vault's
// single-writer lock. Fail-closed: an empty candidate set never takes the lock.
func (v *Vault) GC(pol GCPolicy, live bool) (GCReport, error) {
	rep := GCReport{Policy: pol}
	rows, err := ReadManifestRows(filepath.Join(v.Dir, ManifestName))
	if err != nil {
		return rep, err
	}
	for _, r := range rows {
		if r.Op == OpSkip {
			rep.SkipErrorRows++
		}
	}
	order := manifestShaOrder(rows)

	bySrc := filepath.Join(v.Dir, "by-source")
	ents, err := os.ReadDir(bySrc)
	if err != nil && !os.IsNotExist(err) {
		return rep, err
	}
	for _, e := range ents {
		if !e.IsDir() {
			continue
		}
		srcID := e.Name()
		cands, err := v.historyPruneCandidates(srcID, filepath.Join(bySrc, srcID, ".history"), order, pol.HistoryDepth)
		if err != nil {
			return rep, err
		}
		rep.Candidates = append(rep.Candidates, cands...)
	}
	sort.Slice(rep.Candidates, func(i, j int) bool {
		if rep.Candidates[i].Source != rep.Candidates[j].Source {
			return rep.Candidates[i].Source < rep.Candidates[j].Source
		}
		return rep.Candidates[i].HistFile < rep.Candidates[j].HistFile
	})
	for _, c := range rep.Candidates {
		rep.ReclaimBytes += c.Bytes
	}

	if !live || len(rep.Candidates) == 0 {
		return rep, nil // propose-by-default: no deletion, no manifest row, no lock
	}
	err = v.withManifest(func(man *Manifest) error {
		for _, c := range rep.Candidates {
			abs := filepath.Join(v.Dir, filepath.FromSlash(c.HistFile))
			if err := os.Remove(abs); err != nil && !os.IsNotExist(err) {
				return err
			}
			if _, err := man.Append(ManifestRow{
				TSUnixNano: time.Now().UnixNano(),
				Op:         OpGCPrune,
				Source:     c.Source,
				RelPath:    c.RelPath,
				Note:       fmt.Sprintf("gc-prune history depth>%d reclaimed=%d sha=%s", pol.HistoryDepth, c.Bytes, c.SHA16),
			}); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return rep, err
	}
	rep.Applied = true
	return rep, nil
}

// historyPruneCandidates gathers the prune candidates under one source's
// .history/ tree: it groups slot files by their original rel-path and, for any
// group deeper than the policy, marks the oldest versions (retirement order
// derived from the manifest) as candidates while keeping the newest depth.
func (v *Vault) historyPruneCandidates(srcID, histRoot string, order map[string]map[string]uint64, depth int) ([]GCCandidate, error) {
	if depth < 0 {
		depth = 0
	}
	type slot struct {
		sha16, histRel string
		bytes          int64
		seq            uint64
		known          bool
	}
	groups := map[string][]slot{}
	err := filepath.WalkDir(histRoot, func(path string, d fs.DirEntry, we error) error {
		if we != nil {
			if os.IsNotExist(we) {
				return nil // no .history/ for this source yet — valid-empty
			}
			return we
		}
		if d.IsDir() || !d.Type().IsRegular() {
			return nil
		}
		rel, rErr := filepath.Rel(histRoot, path)
		if rErr != nil {
			return rErr
		}
		histRel := filepath.ToSlash(rel)
		orig, sha16, ok := parseHistorySlot(histRel)
		if !ok {
			return nil // not a recognizable content slot — never delete what we can't classify
		}
		info, iErr := d.Info()
		if iErr != nil {
			return iErr
		}
		seq, known := uint64(0), false
		if m := order[srcID+"\x00"+orig]; m != nil {
			if s, hit := m[sha16]; hit {
				seq, known = s, true
			}
		}
		groups[orig] = append(groups[orig], slot{sha16, histRel, info.Size(), seq, known})
		return nil
	})
	if err != nil {
		return nil, err
	}
	if depth == 0 {
		return nil, nil // unlimited retention: nothing proposed
	}
	var cands []GCCandidate
	for orig, slots := range groups {
		if len(slots) <= depth {
			continue
		}
		sort.Slice(slots, func(i, j int) bool {
			// newest last: manifest-ordered versions by seq ascending; versions
			// with no manifest row sort first (treated oldest); sha16 tiebreak.
			if slots[i].known != slots[j].known {
				return !slots[i].known
			}
			if slots[i].seq != slots[j].seq {
				return slots[i].seq < slots[j].seq
			}
			return slots[i].sha16 < slots[j].sha16
		})
		for _, s := range slots[:len(slots)-depth] {
			cands = append(cands, GCCandidate{
				Source:   srcID,
				RelPath:  orig,
				HistFile: "by-source/" + srcID + "/.history/" + s.histRel,
				SHA16:    s.sha16,
				Bytes:    s.bytes,
			})
		}
	}
	return cands, nil
}

// manifestShaOrder folds the manifest into a per-(source, rel-path) map from a
// content sha's 16-hex prefix to the seq at which that content first became the
// mirror. That first-capture seq is the retirement-order key: a version retired
// to .history/ was the mirror from its own capture until the next rewrite, so
// ordering by it recovers oldest→newest without trusting file mtimes.
func manifestShaOrder(rows []ManifestRow) map[string]map[string]uint64 {
	order := map[string]map[string]uint64{}
	for _, r := range rows {
		if len(r.SHA256) < 16 {
			continue
		}
		key := r.Source + "\x00" + r.RelPath
		sha16 := r.SHA256[:16]
		m := order[key]
		if m == nil {
			m = map[string]uint64{}
			order[key] = m
		}
		if _, seen := m[sha16]; !seen {
			m[sha16] = r.Seq
		}
	}
	return order
}

// parseHistorySlot splits a .history/ slot rel path (e.g. "sub/b.log.deadbeefdeadbeef")
// into the original rel path and the 16-hex content slot. A name whose final
// dot-segment is not exactly 16 hex chars is not a recognizable slot.
func parseHistorySlot(histRel string) (orig, sha16 string, ok bool) {
	i := strings.LastIndexByte(histRel, '.')
	if i < 0 || i == len(histRel)-1 {
		return "", "", false
	}
	suf := histRel[i+1:]
	if len(suf) != 16 || !isLowerHex(suf) {
		return "", "", false
	}
	return histRel[:i], suf, true
}

func isLowerHex(s string) bool {
	for _, c := range s {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			return false
		}
	}
	return true
}

// withManifest runs fn holding the vault's single-writer lock with an open
// manifest, then refreshes the head anchor — the shared mutating preamble that
// GC --live and AdoptCold use so their appends can never interleave with a
// concurrent capture (or each other).
func (v *Vault) withManifest(fn func(man *Manifest) error) error {
	if err := os.MkdirAll(v.Dir, 0o755); err != nil {
		return err
	}
	lock, err := os.OpenFile(filepath.Join(v.Dir, LockName), os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return err
	}
	defer lock.Close()
	if err := flock.TryLock(lock); err != nil {
		if errors.Is(err, flock.ErrLockBusy) {
			return fmt.Errorf("logvault: another writer holds %s", LockName)
		}
		return err
	}
	defer flock.Unlock(lock)
	man, err := OpenManifest(v.Dir)
	if err != nil {
		return err
	}
	if err := fn(man); err != nil {
		man.Close()
		return err
	}
	if err := man.Close(); err != nil {
		return err
	}
	if seq, hash := man.Head(); seq > 0 {
		return WriteAnchor(v.Dir, seq, hash)
	}
	return nil
}
