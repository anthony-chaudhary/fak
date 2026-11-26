package ggufload

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// inkling2DeviceEmbedding is a host token table the device can also read in
// place, so a forward can embed the token the previous forward's argmax chose
// without the host reading it first.
type inkling2DeviceEmbedding struct {
	inkling2QuantEmbedding
	table compute.Tensor
}

// ink2Pipeliner is a device that embeds a token from a device index, copies
// spans, and submits a forward without waiting for it.
type ink2Pipeliner interface {
	Ink2CanEmbed(table compute.Tensor) bool
	Ink2EmbedRow(table compute.Tensor, idx ink2Ref, pos int, out ink2Ref) error
	Ink2CopySpans(src, dst []ink2Ref, n []int) error
	Ink2Commit() (uint64, error)
	Ink2ReadIntsAfter(ticket uint64, src ink2Ref, n int) ([]int32, error)
}

// ink2Pipe overlaps greedy decode steps: once forward t is submitted, forward t+1
// is encoded and submitted fed from t's argmax on the device, before the host
// reads t's token. Forward t+1 is speculative until the next step names the token
// it consumed; any other call discards it and restores the state it changed.
type ink2Pipe struct {
	dev   ink2Pipeliner
	table compute.Tensor
	rows  int
	// a forward writes its argmax and logits to one slot; the forward after it,
	// already running while that argmax is read, writes the other
	arg, logits [2]ink2Ref
	cur         int

	// the pre-forward conv histories of every slot and the local ring rows the
	// speculative forward overwrites, as spans of save
	save                ink2Ref
	histSave, ringSaveK []ink2Ref
	ringSaveV           []ink2Ref

	// tok is the token the last step returned. Only a step that consumes it
	// speculates on the next, so a caller forcing tokens wastes no forward.
	tok     int
	haveTok bool

	// the speculative forward in flight: the position it consumed tok at, the slot
	// it writes and the ticket its submission completes
	live   bool
	pos    int
	slot   int
	ticket uint64
}

// enablePipe sets up decode pipelining when the device and the token table allow
// it; any session state it cannot cover leaves the plain step in place.
func (s *inkling2Resident) enablePipe(ctx int) {
	emb, ok := s.w.TokEmbd.(inkling2DeviceEmbedding)
	if !ok || ctx == 0 || s.hostRouting || s.err != nil || inkling2SpecDepth() > 0 {
		return
	}
	dev, ok := s.dev.(ink2Pipeliner)
	if !ok || !dev.Ink2CanEmbed(emb.table) || emb.m.cols != s.hp.Hidden {
		return
	}
	p := &ink2Pipe{dev: dev, table: emb.table, rows: emb.m.rows}
	p.arg[0], p.arg[1] = s.argIdx, s.alloc(4)
	rows := s.output.rows
	p.logits[0], p.logits[1] = s.alloc(rows), s.alloc(rows)
	kvDim := s.hp.KVHeads * s.hp.HeadDim
	hf := s.histFloats()
	n := len(s.hist) * hf
	for slot := range s.hist {
		if s.localAt(slot) {
			n += 2 * kvDim
		}
	}
	p.save = s.alloc(n)
	if s.err != nil {
		return
	}
	off := 0
	p.histSave = make([]ink2Ref, len(s.hist))
	p.ringSaveK, p.ringSaveV = make([]ink2Ref, len(s.hist)), make([]ink2Ref, len(s.hist))
	for slot := range s.hist {
		p.histSave[slot] = at(p.save, off)
		off += hf
		if s.localAt(slot) {
			p.ringSaveK[slot], p.ringSaveV[slot] = at(p.save, off), at(p.save, off+kvDim)
			off += 2 * kvDim
		}
	}
	s.pipe = p
}

// pipeOn reports whether this step may leave a speculative forward in flight.
func (s *inkling2Resident) pipeOn() bool {
	return s.pipe != nil && s.specMax == 0 && s.draft == nil && !s.graphReady
}

// pipeSpans lists, for a forward at pos, each piece of state it overwrites and
// where its save lives.
func (s *inkling2Resident) pipeSpans(pos int) (state, save []ink2Ref, n []int) {
	p := s.pipe
	kvDim := s.hp.KVHeads * s.hp.HeadDim
	hf := s.histFloats()
	for slot := range s.hist {
		state, save, n = append(state, s.hist[slot]), append(save, p.histSave[slot]), append(n, hf)
		if !s.localAt(slot) {
			// a global row at pos is rewritten by whichever forward next runs there,
			// before any query reads it
			continue
		}
		row := (pos % s.caps[slot]) * kvDim
		state = append(state, at(s.kc[slot], row), at(s.vc[slot], row))
		save = append(save, p.ringSaveK[slot], p.ringSaveV[slot])
		n = append(n, kvDim, kvDim)
	}
	return state, save, n
}

// pipeLaunch encodes and submits the forward at pos of the token the argmax in
// slot src names, writing slot src^1.
func (s *inkling2Resident) pipeLaunch(src, pos int) {
	p := s.pipe
	state, save, n := s.pipeSpans(pos)
	s.fail(p.dev.Ink2CopySpans(state, save, n))
	if s.err == nil {
		s.fail(p.dev.Ink2EmbedRow(p.table, p.arg[src], pos, s.in))
	}
	dst := src ^ 1
	s.forwardTo(p.arg[dst], p.logits[dst])
	if s.err != nil {
		return
	}
	ticket, err := p.dev.Ink2Commit()
	if err != nil {
		s.fail(err)
		return
	}
	p.live, p.pos, p.slot, p.ticket = true, pos, dst, ticket
}

// pipeDiscard drops the speculative forward: the restores it encodes run after it
// and leave every piece of state as it was before it.
func (s *inkling2Resident) pipeDiscard() {
	p := s.pipe
	if p == nil || !p.live {
		return
	}
	p.live = false
	state, save, n := s.pipeSpans(p.pos)
	s.fail(p.dev.Ink2CopySpans(save, state, n))
}

// settle discards any speculative forward, before a call that reads or replaces
// session state other than through the next step.
func (s *inkling2Resident) settle() { s.pipeDiscard() }

// pipeExpect records next as the token the next step is expected to consume.
func (s *inkling2Resident) pipeExpect(next int) {
	if s.pipe != nil {
		s.pipe.tok, s.pipe.haveTok = next, true
	}
}

// pipeStep runs step(token) on the pipelined path, or reports false when the plain
// step should run it.
func (s *inkling2Resident) pipeStep(token int) (int, bool, error) {
	p := s.pipe
	if p == nil {
		return 0, false, nil
	}
	follows := p.haveTok && token == p.tok
	p.haveTok = false
	if p.live && (!follows || !s.pipeOn() || p.pos != s.pos) {
		s.pipeDiscard()
	}
	if !s.pipeOn() || s.err != nil {
		return 0, s.err != nil, s.err
	}
	var slot int
	var ticket uint64
	if p.live {
		p.live = false
		slot, ticket = p.slot, p.ticket
	} else {
		if !follows {
			return 0, false, nil
		}
		s.embedHost(token)
		slot = p.cur
		s.forwardTo(p.arg[slot], p.logits[slot])
		s.warm = true
		if s.err != nil {
			return 0, true, s.err
		}
		var err error
		if ticket, err = p.dev.Ink2Commit(); err != nil {
			// a device that cannot submit early runs every step plainly from here
			s.pipe = nil
			return s.readStep(p.arg[slot], p.logits[slot])
		}
	}
	p.cur = slot ^ 1
	if s.pos+1 < s.ctx {
		s.pipeLaunch(slot, s.pos+1)
	}
	if s.err != nil {
		return 0, true, s.err
	}
	got, err := p.dev.Ink2ReadIntsAfter(ticket, p.arg[slot], 1)
	if err != nil {
		return 0, true, err
	}
	next := int(got[0])
	if next < 0 || next >= p.rows {
		// the device clamps an index past the table; the host would not embed it
		s.pipeDiscard()
	}
	s.logits = p.logits[slot]
	return s.finishStep(next), true, nil
}

// readStep fences the forward whose argmax is at arg and finishes the step.
func (s *inkling2Resident) readStep(arg, logits ink2Ref) (int, bool, error) {
	got, err := s.dev.Ink2ReadInts(arg, 1)
	if err != nil {
		return 0, true, err
	}
	s.logits = logits
	return s.finishStep(int(got[0])), true, nil
}

// pipeDrain waits for any submitted forward, so its buffers can be freed.
func (s *inkling2Resident) pipeDrain() {
	if s.pipe == nil || !s.pipe.live {
		return
	}
	s.pipe.live = false
	if _, err := s.dev.Ink2ReadInts(s.pipe.arg[s.pipe.slot], 1); err != nil {
		s.fail(fmt.Errorf("inkling2 resident: drain the pipelined forward: %w", err))
	}
}
