package ggufload

// inkling2SharedPrefixLen is the longest token prefix every listed row's prompt
// shares, held one token short of the shortest prompt so every row still prefills its
// own last token and gets its first greedy token from its own head.
func inkling2SharedPrefixLen(prompts [][]int, rows []int) int {
	if len(rows) < 2 {
		return 0
	}
	lead := prompts[rows[0]]
	n := len(lead) - 1
	for _, i := range rows[1:] {
		p := prompts[i]
		n = min(n, len(p)-1)
		for k := 0; k < n; k++ {
			if p[k] != lead[k] {
				n = k
				break
			}
		}
	}
	return max(n, 0)
}

// prefillShared prefills the prefix every row with prompt left shares once, in the
// first such row, and copies that row's caches and histories into the others, leaving
// each sharing row's cursor past the prefix; a prefix shorter than sharedMin is not
// worth the copy and leaves every row untouched. keep is polled before each prefix
// chunk for every sharing row, and a refused row ends at its cursor as prefill drops
// it, and is not copied into. A row's prefill writes only that row's state, from that
// row's tokens at its own positions, so the copy is the state the row would have
// computed wherever a row's results do not depend on the chunk it is packed in.
func (b *inkling2Batch) prefillShared(prompts [][]int, end, cursor []int, keep func(row int) bool) error {
	if b.sharedMin < 1 {
		return nil
	}
	var rows []int
	for i := range end {
		if cursor[i] == 0 && end[i] == len(prompts[i]) {
			rows = append(rows, i)
		}
	}
	P := inkling2SharedPrefixLen(prompts, rows)
	if P < b.sharedMin {
		return nil
	}
	lead := rows[0]
	// only the lead row is packed, up to the shared prefix
	ends := make([]int, b.C)
	ends[lead] = P
	done := make([]int, b.C)
	for done[lead] < P {
		live := rows[:0]
		for _, i := range rows {
			if keep == nil || keep(i) {
				live = append(live, i)
			} else {
				end[i] = cursor[i]
			}
		}
		rows = live
		if len(rows) == 0 {
			return nil
		}
		segs := inkling2PrefillNextChunk(ends, done, b.prefillT)
		toks := make([][]int, len(segs))
		for k, sg := range segs {
			toks[k] = prompts[sg.row][sg.pos0 : sg.pos0+sg.n]
		}
		if err := b.prefillChunk(segs, toks, nil); err != nil {
			return err
		}
	}
	for _, i := range rows {
		if i != lead {
			if err := b.copyRowState(lead, i); err != nil {
				return err
			}
		}
		cursor[i] = P
	}
	return nil
}

// copyRowState makes row dst's caches and conv histories in every slot a bitwise copy
// of row src's, on the device when the backend can copy rows there.
func (b *inkling2Batch) copyRowState(src, dst int) error {
	hp := b.s.hp
	kvDim := hp.KVHeads * hp.HeadDim
	for slot, cap := range b.caps {
		n := inkling2CohortKVRow(1, cap, kvDim, b.kvHalf)
		for _, arena := range []ink2Ref{b.kc[slot], b.vc[slot]} {
			if err := b.copyFloats(at(arena, src*n), at(arena, dst*n), n); err != nil {
				return err
			}
		}
		if err := b.copyFloats(at(b.hist[slot], src*b.histStride), at(b.hist[slot], dst*b.histStride), b.histStride); err != nil {
			return err
		}
	}
	return nil
}

// copyFloats copies n floats from src to dst; the host round trip is the fallback
// for a backend without a device row copy.
func (b *inkling2Batch) copyFloats(src, dst ink2Ref, n int) error {
	if n == 0 {
		return nil
	}
	if r, ok := b.s.be.(interface {
		Ink2VRepeatRows(x, out ink2Ref, n, rows, k int) error
	}); ok {
		return r.Ink2VRepeatRows(src, dst, n, 1, 1)
	}
	data, err := b.s.dev.Ink2Read(src, n)
	if err != nil {
		return err
	}
	return b.s.dev.Ink2Write(dst, data)
}
