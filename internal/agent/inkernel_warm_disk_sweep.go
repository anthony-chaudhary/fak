package agent

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// sweepCandidate is one eviction candidate in a cross-identity sweep: an identity
// dir (or the legacy flat pseudo-identity) with its records ordered LRU-first.
type sweepCandidate struct {
	name    string
	legacy  bool
	used    int64
	bytes   int64
	records []sweepRecord
}

type sweepRecord struct {
	key  string
	size int64
}

// loadSweepCandidate computes one candidate's total bytes and LRU-ordered record
// list. Identities use their index when readable (falling back to a stat scan);
// the legacy pseudo-identity is always a stat scan with used=0 so pre-upgrade
// records are reclaimed first.
func loadSweepCandidate(root, name string, legacy bool) (*sweepCandidate, error) {
	cand := &sweepCandidate{name: name, legacy: legacy}
	if legacy {
		info, err := os.Stat(filepath.Join(root, name))
		if err != nil || !info.Mode().IsRegular() {
			return nil, nil
		}
		cand.bytes = info.Size()
		cand.records = []sweepRecord{{key: name, size: info.Size()}}
		return cand, nil
	}
	idDir := filepath.Join(root, name)
	if used, ok := readWordFile(filepath.Join(idDir, learnedUsedFile)); ok {
		cand.used = int64(used)
	}
	if idx, _, valid := loadLearnedIndex(filepath.Join(idDir, learnedIndexFile)); valid {
		order := make([]int, len(idx.States))
		for i := range order {
			order[i] = i
		}
		sort.SliceStable(order, func(a, b int) bool {
			ia, ib := idx.States[order[a]], idx.States[order[b]]
			if ia.Used != ib.Used {
				return ia.Used < ib.Used
			}
			return ia.Key < ib.Key
		})
		for _, i := range order {
			st := idx.States[i]
			info, err := os.Stat(filepath.Join(idDir, st.Key))
			if err != nil || !info.Mode().IsRegular() {
				continue
			}
			cand.bytes += info.Size()
			cand.records = append(cand.records, sweepRecord{key: st.Key, size: info.Size()})
		}
		return cand, nil
	}
	entries, err := os.ReadDir(idDir)
	if err != nil {
		return nil, fmt.Errorf("warm disk: sweep scan %s: %w", idDir, err)
	}
	var scanned []struct {
		key  string
		size int64
		mod  int64
	}
	for _, entry := range entries {
		if entry.IsDir() || !isLearnedRecordKey(entry.Name()) {
			continue
		}
		info, infoErr := entry.Info()
		if infoErr != nil || !info.Mode().IsRegular() {
			continue
		}
		scanned = append(scanned, struct {
			key  string
			size int64
			mod  int64
		}{entry.Name(), info.Size(), info.ModTime().UnixNano()})
	}
	sort.Slice(scanned, func(i, j int) bool {
		if scanned[i].mod == scanned[j].mod {
			return scanned[i].key < scanned[j].key
		}
		return scanned[i].mod < scanned[j].mod
	})
	for _, rec := range scanned {
		cand.bytes += rec.size
		cand.records = append(cand.records, sweepRecord{key: rec.key, size: rec.size})
	}
	return cand, nil
}

// learnedRootSweep enforces the byte cap SHARED by every identity under root:
// while total bytes plus need exceed capBytes it evicts the LRU record of the
// least-recently-opened candidate (oldest used word; ties go to the legacy
// pseudo-identity, then lexicographic name). keepKey inside keepIdentity is
// never evicted. slots/, clock, word files and any non-matching name are never
// touched. It returns nil once the cap fits and errLearnedSweepExhausted when
// nothing evictable remains.
func learnedRootSweep(ctx context.Context, root, keepIdentity, keepKey string, need, capBytes int64) error {
	if ctx == nil {
		ctx = context.Background()
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		return fmt.Errorf("warm disk: sweep root %s: %w", root, err)
	}
	var candidates []*sweepCandidate
	total := int64(0)
	for _, entry := range entries {
		if err := ctx.Err(); err != nil {
			return err
		}
		var cand *sweepCandidate
		switch {
		case entry.IsDir():
			if !isLearnedIdentityName(entry.Name()) {
				continue
			}
			cand, err = loadSweepCandidate(root, entry.Name(), false)
		default:
			if !isLearnedRecordKey(entry.Name()) {
				continue
			}
			cand, err = loadSweepCandidate(root, entry.Name(), true)
		}
		if err != nil {
			return err
		}
		if cand == nil || cand.bytes == 0 {
			continue
		}
		if cand.name == keepIdentity && keepKey != "" {
			// keepKey's bytes count toward the cap but are never evictable.
			kept := cand.records[:0]
			for _, rec := range cand.records {
				if rec.key != keepKey {
					kept = append(kept, rec)
				}
			}
			cand.records = kept
		}
		candidates = append(candidates, cand)
		total += cand.bytes
	}
	for total+need > capBytes {
		if err := ctx.Err(); err != nil {
			return err
		}
		best := -1
		for i, cand := range candidates {
			if len(cand.records) == 0 {
				continue
			}
			if best == -1 || sweepCandidateLess(*cand, *candidates[best]) {
				best = i
			}
		}
		if best == -1 {
			return errLearnedSweepExhausted
		}
		rec := candidates[best].records[0]
		path := filepath.Join(root, candidates[best].name, rec.key)
		if candidates[best].legacy {
			path = filepath.Join(root, rec.key)
		}
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("warm disk: sweep evict %s: %w", path, err)
		}
		candidates[best].records = candidates[best].records[1:]
		candidates[best].bytes -= rec.size
		total -= rec.size
	}
	return nil
}

// sweepCandidateLess orders eviction: smallest used word first; on a tie the
// legacy pseudo-identity goes first, then lexicographic name.
func sweepCandidateLess(a, b sweepCandidate) bool {
	if a.used != b.used {
		return a.used < b.used
	}
	if a.legacy != b.legacy {
		return a.legacy
	}
	return a.name < b.name
}
