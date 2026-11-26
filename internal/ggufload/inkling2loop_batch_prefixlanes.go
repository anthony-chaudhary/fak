package ggufload

import "errors"

// reuseOn reports whether this cohort restores and records prefixes: the store is on
// and the cohort prefills prompts in chunks, which is what lets a row start mid-prompt.
func (b *inkling2Batch) reuseOn() bool { return b.store != nil && b.prefillT > 0 }

// restoreLane starts lane's row from the longest stored prefix of its prompt. A
// checkpoint this cohort cannot hold (another cache dtype, a shorter context) is a
// miss, not an error; the row then prefills from position 0 as it would without the
// store.
func (b *inkling2Batch) restoreLane(i int, lane *inkling2BatchLane) error {
	if !b.reuseOn() {
		return nil
	}
	e := b.store.lookup(lane.req.Prompt)
	if e == nil {
		return nil
	}
	if err := b.restoreRow(i, e.ck); err != nil {
		if errors.Is(err, errInkling2RowCkptShape) {
			return nil
		}
		return err
	}
	lane.cached = e.ck.pos
	return nil
}

// prefillBranches prefills each live lane only up to the point where its prompt
// leaves a stored entry or another lane's prompt, and checkpoints it there, so the
// next request that shares that prefix (a shared system prompt, a sibling of the
// same conversation) restores it. Lanes branching at the same tokens share one
// prefill: the first computes them and the rest restore its checkpoint. Every lane
// is left at its cursor for the prefill that follows.
func (b *inkling2Batch) prefillBranches(lanes []inkling2BatchLane, keep func(int) bool) error {
	if !b.reuseOn() {
		return nil
	}
	prompts := make([][]int, len(lanes))
	for i := range lanes {
		prompts[i] = lanes[i].req.Prompt
	}
	cursor := append([]int(nil), b.pos...)
	ends := append([]int(nil), cursor...)
	var leads []int
	follow := map[int][]int{}
	for i := range lanes {
		if lanes[i].done {
			continue
		}
		k := b.store.branch(prompts[i], cursor[i], prompts, i)
		if k == 0 {
			continue
		}
		lead := -1
		for _, j := range leads {
			if ends[j] == k && inkling2HasPrefix(prompts[i], prompts[j][:k]) && cursor[i] == cursor[j] {
				lead = j
				break
			}
		}
		if lead < 0 {
			leads = append(leads, i)
			ends[i] = k
		} else {
			follow[lead] = append(follow[lead], i)
		}
	}
	if len(leads) == 0 {
		return nil
	}
	for {
		more := false
		for _, i := range leads {
			if cursor[i] < ends[i] && keep != nil && !keep(i) {
				ends[i] = cursor[i]
			}
			more = more || cursor[i] < ends[i]
		}
		if !more {
			break
		}
		segs := inkling2PrefillNextChunk(ends, cursor, b.prefillT)
		toks := make([][]int, len(segs))
		for k, sg := range segs {
			toks[k] = prompts[sg.row][sg.pos0 : sg.pos0+sg.n]
		}
		if err := b.prefillChunk(segs, toks, nil); err != nil {
			return err
		}
	}
	b.endPrefill(cursor)
	for _, i := range leads {
		k := ends[i]
		if lanes[i].done || cursor[i] != k {
			continue
		}
		ck, err := b.saveRow(i)
		if errors.Is(err, errInkling2RowCkptShape) {
			continue
		}
		if err != nil {
			return err
		}
		b.store.put(prompts[i][:k], ck)
		for _, j := range follow[i] {
			if lanes[j].done {
				continue
			}
			if err := b.restoreRow(j, ck); err != nil {
				return err
			}
			lanes[j].cached = k
		}
	}
	return nil
}

// saveFinished checkpoints every lane that finished cleanly since the last call,
// before its row is parked or handed to an admitted request: the row then holds the
// state after its prompt and every generated token but the last, which is exactly the
// prefix the next turn of the same conversation starts with.
func (b *inkling2Batch) saveFinished(lanes []inkling2BatchLane) error {
	if !b.reuseOn() {
		return nil
	}
	for i := range lanes {
		lane := &lanes[i]
		if !lane.done || lane.saved {
			continue
		}
		lane.saved = true
		if lane.result.Err != nil || len(lane.gen) == 0 {
			continue
		}
		fed := make([]int, 0, len(lane.req.Prompt)+len(lane.gen)-1)
		fed = append(append(fed, lane.req.Prompt...), lane.gen[:len(lane.gen)-1]...)
		if len(fed) != b.pos[i] || len(fed) < b.store.minTokens {
			continue
		}
		ck, err := b.saveRow(i)
		if errors.Is(err, errInkling2RowCkptShape) {
			continue
		}
		if err != nil {
			return err
		}
		b.store.put(fed, ck)
	}
	return nil
}
