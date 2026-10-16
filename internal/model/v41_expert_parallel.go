package model

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
)

func (m *Model) v41EPRequested() bool {
	return m != nil && m.Cfg.IsDeepSeekV41() && m.epRanks > 1
}

// Only an explicitly bound, rank-consistent exact transport permits omission of
// nonowned expert weights. Generic Local/GLM/device reductions cannot admit it.
func (m *Model) v41EPPlan() (*v41ExpertParallelCollective, TPPlan, error) {
	if !m.v41EPRequested() {
		return nil, TPPlan{}, nil
	}
	c, ok := m.epColl.(*v41ExpertParallelCollective)
	if !ok || c == nil || c.g == nil || !m.epRankSet || m.epRanks != c.g.Size() || m.epRank != c.g.Rank() {
		return nil, TPPlan{}, fmt.Errorf("%w: requires an explicitly bound exact transport and matching rank/world", ErrV41ExpertParallel)
	}
	full, err := v41ForwardGeometry(m.Cfg)
	if err != nil || !full {
		return nil, TPPlan{}, fmt.Errorf("%w: requires full V4.1 geometry: %v", ErrV41ExpertParallel, err)
	}
	owners, err := ExpertParallelPlan(m.Cfg.NumExperts, m.epRanks)
	if err != nil {
		return nil, TPPlan{}, fmt.Errorf("%w: ownership: %v", ErrV41ExpertParallel, err)
	}
	return c, owners, nil
}

// v41EPGuard is deferred before announcement/admission. A local failure closes
// the selected transport, unblocking peer reads without authorizing any replay.
func (s *Session) v41EPGuard(completed *bool) {
	if *completed {
		return
	}
	// Preserve the caller's active panic or Goexit. Recover only a secondary
	// abort panic; the existing first-fault transport sequence stays unchanged.
	defer func() { _ = recover() }()
	if c, ok := s.M.epColl.(*v41ExpertParallelCollective); ok && c != nil {
		c.g.abortV41EP()
	}
	if s.M.epCoord != nil && s.M.Cfg.IsDeepSeekV41() {
		s.M.epCoord.g.abortV41EP()
	}
}

func (s *Session) bindV41ExpertParallel(st *v41ForwardState) {
	if !s.M.v41EPRequested() {
		return
	}
	c, owners, err := s.M.v41EPPlan()
	if err != nil {
		panic(err)
	}
	routed, err := s.v41EPOwnerDeviceBinding(owners, st.routedExpert)
	if err != nil {
		// v41State publishes callbackOwner before binding EP. A refused
		// binding must not leave its local-only callbacks reusable on retry.
		st.callbackOwner = nil
		c.g.abortV41EP()
		panic(err)
	}
	st.routedExpert = routed
	config, err := json.Marshal(s.M.Cfg)
	if err != nil {
		panic(fmt.Errorf("%w: config identity: %v", ErrV41ExpertParallel, err))
	}
	configDigest := sha256.Sum256(config)
	// The session-local ordinal binds repeated layer/route calls; the existing
	// coordinator serializes frames and resets follower sessions on pos=0.
	var ordinal uint64
	nextIdentity := func(layer, position int, prefill bool, picks []routePick, input []float32) [32]byte {
		ordinal++
		digest := v41EPIdentity(c.binding, configDigest, ordinal, layer, s.v41Forward.history, picks, input)
		return v41EPPositionIdentity(digest, position, prefill)
	}
	st.routedPrefill = func(layer int, picks [][]routePick, inputs [][]float32, scratch *v41ProjScratch) ([][]float32, error) {
		return s.M.v41EPPrefillSum(c, owners, layer, picks, inputs, scratch, st, nextIdentity)
	}
	st.routedLayer = func(layer int, picks []routePick, input []float32) ([]float32, v41RoutedExpertOutcome, error) {
		// Incremental Step appends its current token only after the head commits.
		digest := nextIdentity(layer, len(s.v41Forward.history), false, picks, input)
		scratch := &v41ProjScratch{}
		sum, cause := s.M.v41EPOwnerSum(c, owners, digest, layer, picks, input, scratch, st)
		if cause != nil {
			return nil, v41RoutedExpertError, cause
		}
		return sum, v41RoutedExpertHandled, nil
	}
}

func v41EPIdentity(binding [64]byte, config [32]byte, ordinal uint64, layer int, history []int, picks []routePick, input []float32) [32]byte {
	h := sha256.New()
	_, _ = h.Write(binding[:])
	_, _ = h.Write(config[:])
	var word [8]byte
	put := func(v uint64) { binary.LittleEndian.PutUint64(word[:], v); _, _ = h.Write(word[:]) }
	put(ordinal)
	put(uint64(layer))
	put(uint64(len(history)))
	for _, id := range history {
		put(uint64(id))
	}
	put(uint64(len(picks)))
	for _, pick := range picks {
		put(uint64(pick.expert))
		put(uint64(math.Float32bits(pick.weight)))
	}
	put(uint64(len(input)))
	for _, value := range input {
		put(uint64(math.Float32bits(value)))
	}
	var out [32]byte
	copy(out[:], h.Sum(nil))
	return out
}

func (m *Model) v41EPOwnerSum(c *v41ExpertParallelCollective, owners TPPlan, digest [32]byte, layer int, picks []routePick, input []float32, scratch *v41ProjScratch, st *v41ForwardState) (sum []float32, err error) {
	defer func() {
		if err != nil {
			if c.failure == nil {
				c.failure = err
			}
			c.g.abortV41EP()
		}
	}()
	if c.failure != nil {
		return nil, c.failure
	}
	H := m.Cfg.HiddenSize
	plan, err := v41EPGatherPlan(picks, owners, H)
	if err != nil {
		return nil, err
	}
	rank := c.g.Rank()
	part := make([]float32, plan.Shards[rank].Width())
	band := owners.Shards[rank]
	localErr := func() (cause error) {
		complete := false
		defer func() {
			if r := recover(); !complete {
				if e, ok := r.(error); ok {
					cause = e
				} else {
					cause = fmt.Errorf("expert panic: %v", r)
				}
			}
		}()
		cause = func() error {
			if len(input) != H {
				return fmt.Errorf("invalid hidden width")
			}
			offset := v41EPHeaderWords
			for _, pick := range picks {
				if pick.expert < band.Lo || pick.expert >= band.Hi {
					continue
				}
				stem := "ffn.experts." + itoa(pick.expert)
				row, cause := m.v41FullRoutedExpertPublished(layer, stem, input, pick.weight, m.Cfg, st.routedExpert, st.expertGateUp, st.expertDown,
					func() ([]float32, []float32, []float32, error) {
						return m.v41ExpertTripleInto(layer, stem, scratch, false)
					},
					func() ([]float32, error) { return m.hostExpertDown(layer, stem, scratch) }, false, st.gatherOps.bf16PublishFunc())
				if cause != nil {
					return cause
				}
				if len(row) != H {
					return errV41ExpertResult
				}
				copy(part[offset:offset+H], row)
				offset += H
			}
			return nil
		}()
		complete = true
		return cause
	}()
	v41EPHeader(part, digest, localErr != nil)
	sum, err = c.reduce(part, v41EPReduction{owners: owners, digest: digest, layer: layer, picks: picks, width: H, sum: st.routedSumFunc()})
	// A failed owner receives only the generic root refusal. Preserve its own
	// selected-operation cause locally, after participating in the failed round.
	if localErr != nil {
		c.failure = fmt.Errorf("%w: owner %d layer %d: %w", ErrV41ExpertParallel, rank, layer, localErr)
		return nil, c.failure
	}
	return sum, err
}

func v41EPUnpack(gathered []float32, plan, owners TPPlan, digest [32]byte, picks []routePick, width int) ([][]float32, error) {
	expected, err := v41EPGatherPlan(picks, owners, width)
	if err != nil || len(gathered) != expected.Dim || plan.Dim != expected.Dim || len(plan.Shards) != len(expected.Shards) {
		return nil, fmt.Errorf("%w: invalid gathered dimensions", ErrV41ExpertParallel)
	}
	outputs := make([][]float32, len(picks))
	wantHeader := make([]float32, v41EPHeaderWords)
	v41EPHeader(wantHeader, digest, false)
	for rank, shard := range expected.Shards {
		if plan.Shards[rank] != shard {
			return nil, fmt.Errorf("%w: gather ownership mismatch", ErrV41ExpertParallel)
		}
		part := gathered[shard.Lo:shard.Hi]
		for i, value := range wantHeader {
			if math.Float32bits(part[i]) != math.Float32bits(value) {
				return nil, fmt.Errorf("%w: owner %d refused or identity differs", ErrV41ExpertParallel, rank)
			}
		}
		band, offset := owners.Shards[rank], v41EPHeaderWords
		for slot, pick := range picks {
			if pick.expert < band.Lo || pick.expert >= band.Hi {
				continue
			}
			if outputs[slot] != nil {
				return nil, fmt.Errorf("%w: duplicate owner of slot %d", ErrV41ExpertParallel, slot)
			}
			row := part[offset : offset+width]
			for _, value := range row {
				if !finite32(value) {
					return nil, fmt.Errorf("%w: nonfinite owner contribution", ErrV41ExpertParallel)
				}
			}
			outputs[slot] = append([]float32(nil), row...)
			offset += width
		}
	}
	for slot, row := range outputs {
		if row == nil {
			return nil, fmt.Errorf("%w: missing owner of slot %d", ErrV41ExpertParallel, slot)
		}
	}
	return outputs, nil
}

// Public Close calls this after releasing backend execution ownership. Internal
// selected-operation cleanup must not take the coordinator mutex while its
// forward holds it. A cleanup failure also invalidates the distributed group.
func (s *Session) finishV41EPOwner(complete bool) {
	if !complete {
		// The original panic or Goexit is still unwinding in Close. Discard
		// only a secondary transport-cleanup panic, including panic(nil).
		defer func() { _ = recover() }()
	}
	if s != nil && s.M != nil && s.M.Cfg.IsDeepSeekV41() && s.M.epCoord != nil {
		c := s.M.epCoord
		c.mu.Lock()
		defer c.mu.Unlock()
		if c.v41Owner == s {
			c.v41Owner = nil
			if !complete {
				c.g.abortV41EP()
			}
		}
	}
}
