package journal

import "fmt"

// Concurrent-writer integrity: the hash FOREST.
//
// VerifyRows reads a journal as ONE LINEAR chain: row i must carry Seq==i+1 and
// PrevHash==row[i-1].Hash. That is the correct and complete check for a journal
// file with a SINGLE writer process, which is the shipped default — `fak guard`
// gives every session its own file (cmd/fak/guard_support.go's
// guardDefaultAuditPath: .dispatch-runs/guard-audit/interactive-<pid>-<hash>.jsonl),
// and every such file verifies linearly.
//
// It is NOT the right check for a file that MORE THAN ONE process appended to.
// Open recovers the chain head (seq + last hash) into PROCESS-LOCAL state and
// appendLocked allocates Seq from that process-local counter, so two processes
// holding the same path each append from their own snapshot of the head. Their
// writes are still serialized by the OS (O_APPEND, one flushed row at a time) and
// still individually authentic, but the file stops being a chain and becomes a
// TREE: rows fork off a shared prefix, and the same Seq value is issued down more
// than one branch. A linear read then reports a "sequence gap" or a "broken
// chain" on a journal that no one edited.
//
// WHAT STILL HOLDS ON A FORKED FILE, and what VerifyForest checks:
//
//   - Every row is individually authentic: chainHash(PrevHash, row) == Hash. A
//     flipped byte in any field of the hash pre-image still breaks that row.
//   - Every non-genesis row's parent is PRESENT in the file. Deleting or editing
//     an interior row orphans its children, which no concurrency explains.
//   - Every root-to-leaf path is itself a sound linear chain, checked by the
//     SAME unmodified VerifyRows the single-writer case is held to — including
//     its Seq check, because a forking writer keeps counting from the head it
//     read, so each branch stays Seq-contiguous from genesis.
//
// So this is a re-aim of the check, not a relaxation of it: VerifyRows is applied
// once per branch instead of once per file, and the parent-resolution requirement
// is ADDED on top (a linear read never checks it at all).
//
// WHAT DOES NOT HOLD, stated plainly rather than glossed. A forked file has no
// total order — Seq is no longer a global sequence, only a per-branch depth — and
// a TIP row (one no other row names as its parent) can be truncated without
// leaving a hole, because nothing points at it. A linear chain has exactly one
// such point; a forest has one per branch. Concurrent writers therefore buy a
// weaker, though still non-trivial, guarantee than a single writer does, and the
// cure is to keep one writer per file, not to widen the verifier further.

// Forest is the shape of a journal read as a hash tree, and the report
// VerifyForest returns alongside its verdict. Linear is true when the rows
// verified as one plain chain and no reconstruction was needed.
type Forest struct {
	Rows         int  // chained rows examined
	Linear       bool // the whole file verified as ONE linear chain (VerifyRows passed)
	Genesis      int  // rows with PrevHash=="" — chain roots
	BranchPoints int  // rows named as parent by more than one row — concurrent-writer forks
	Tips         int  // rows no other row names as parent — one per distinct writer branch
	Orphans      int  // non-genesis rows whose parent hash is ABSENT — a dropped or edited row
	// Duplicates counts rows repeating a hash already seen — a replayed or
	// double-appended row. Two forked writers COULD in principle collide here
	// without malice, by forking at the same parent, reissuing the same Seq, and
	// emitting an identical decision inside one clock tick, since TSUnixNano is
	// then their only differing pre-image field. That is refused rather than
	// excused: real sessions carry distinct TraceIDs, and the reference host's
	// 60,514-row shared capture contains zero duplicate hashes. A benign
	// collision would be indistinguishable from a replay, which is precisely why
	// one writer per file is the cure and this verifier is only the reader-side
	// diagnosis.
	Duplicates   int
	IntactChains int // reconstructed root-to-tip chains that passed VerifyRows
	BrokenChains int // reconstructed root-to-tip chains that did not
}

// VerifyForest validates a journal that may have been appended to by more than
// one process. It returns the forest shape and a nil error when the rows are
// cryptographically intact — either as one linear chain (Linear true, the
// single-writer case) or as a tree in which every row's parent resolves and
// every root-to-tip branch passes VerifyRows.
//
// A non-nil error is a genuine integrity failure that concurrency does NOT
// explain: an absent parent, a repeated hash, or a branch whose recomputed hash
// or per-branch sequence does not check out. Callers that must distinguish
// "interleaved but trustworthy" from "tampered" should use this rather than
// VerifyRows; callers that require a single writer should keep using VerifyRows,
// which is strictly stronger and stays the default.
func VerifyForest(rows []Row) (Forest, error) {
	f := Forest{Rows: len(rows)}
	if len(rows) == 0 {
		f.Linear = true
		return f, nil
	}
	if _, err := VerifyRows(rows); err == nil {
		f.Linear, f.Genesis, f.Tips, f.IntactChains = true, 1, 1, 1
		return f, nil
	}

	// Index every row by its own hash. A repeated hash is refused rather than
	// silently collapsed: the index is what resolves parents, so letting a
	// duplicate overwrite its twin would hide an inserted or replayed row behind
	// an apparently intact tree.
	byHash := make(map[string]Row, len(rows))
	for _, r := range rows {
		if _, dup := byHash[r.Hash]; dup {
			f.Duplicates++
			continue
		}
		byHash[r.Hash] = r
	}

	childCount := make(map[string]int, len(rows))
	for _, r := range rows {
		if r.PrevHash == "" {
			f.Genesis++
			continue
		}
		if _, ok := byHash[r.PrevHash]; !ok {
			f.Orphans++
			continue
		}
		childCount[r.PrevHash]++
	}
	for _, c := range childCount {
		if c > 1 {
			f.BranchPoints++
		}
	}

	var firstErr error
	fail := func(err error) {
		f.BrokenChains++
		if firstErr == nil {
			firstErr = err
		}
	}
	for _, r := range rows {
		if childCount[r.Hash] != 0 {
			continue // not a tip
		}
		f.Tips++
		chain, ok := chainToRoot(byHash, r)
		if !ok {
			fail(fmt.Errorf("journal: forest: branch ending seq=%d has a missing ancestor (a dropped or edited row, not a concurrent write)", r.Seq))
			continue
		}
		if _, err := VerifyRows(chain); err != nil {
			fail(fmt.Errorf("journal: forest: branch ending seq=%d: %w", r.Seq, err))
			continue
		}
		f.IntactChains++
	}

	switch {
	case firstErr != nil:
		return f, firstErr
	case f.Duplicates > 0:
		return f, fmt.Errorf("journal: forest: %d row(s) repeat a hash already present (a replayed or double-appended row)", f.Duplicates)
	case f.Orphans > 0:
		return f, fmt.Errorf("journal: forest: %d row(s) name an absent parent hash (a dropped or edited row, not a concurrent write)", f.Orphans)
	case f.Genesis == 0:
		return f, fmt.Errorf("journal: forest: no genesis row (every row names a parent) — the head of the chain was removed")
	}
	return f, nil
}

// chainToRoot walks a tip up to its genesis via the unique parent pointer and
// returns the branch in genesis-first order, so VerifyRows can read it exactly
// as it reads a single-writer file. ok is false when an ancestor is missing or
// the walk runs longer than the corpus (a cycle in a malformed file).
func chainToRoot(byHash map[string]Row, tip Row) (chain []Row, ok bool) {
	cur := tip
	limit := len(byHash) + 1
	for steps := 0; steps <= limit; steps++ {
		chain = append(chain, cur)
		if cur.PrevHash == "" {
			for i, j := 0, len(chain)-1; i < j; i, j = i+1, j-1 {
				chain[i], chain[j] = chain[j], chain[i]
			}
			return chain, true
		}
		parent, found := byHash[cur.PrevHash]
		if !found {
			return nil, false
		}
		cur = parent
	}
	return nil, false
}
