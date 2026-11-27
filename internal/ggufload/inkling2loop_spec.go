package ggufload

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"slices"
	"strconv"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// inkling2SpecMaxDraft bounds the drafted tokens of one speculative step, so a
// verify chunk is at most inkling2SpecMaxDraft+1 tokens.
const inkling2SpecMaxDraft = 8

// Device graph slots of a speculative session beside the decode graph: the full-width
// verify and its commit, the draft (inkling2DraftSlot), then a verify and a commit
// slot for each shorter chunk width.
const (
	inkling2VerifySlot = 1
	inkling2CommitSlot = 2
	inkling2TailSlot0  = 4
)

// inkling2TailSlots returns the verify and commit graph slots of a T-token chunk
// shorter than the full width.
func inkling2TailSlots(T int) (verify, commit int) {
	return inkling2TailSlot0 + 2*(T-1), inkling2TailSlot0 + 2*(T-1) + 1
}

// intBits carries an int32 through a float32 device word.
func intBits(v int) float32 { return math.Float32frombits(uint32(int32(v))) }

// inkling2SpecDepth is the drafted-token count of a speculative step from
// FAK_INKLING2_SPEC (unset = the default depth 4; 0 = plain decode; 1 = depth 4).
func inkling2SpecDepth() int {
	v := os.Getenv("FAK_INKLING2_SPEC")
	if v == "" {
		return 4
	}
	if v == "0" {
		return 0
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 1 {
		return 0
	}
	if n == 1 {
		return 4
	}
	return min(n, inkling2SpecMaxDraft)
}

// inkling2Drafter proposes up to k tokens to follow history, whose last token is
// the one the next forward consumes. A nil proposal runs the plain decode step; an
// empty one still runs that token through the batched verify forward, which a
// drafter whose context only that forward commits needs. Observe reports each verify
// outcome: the tokens fed (fed[0] is that last token), how many were kept, and the
// session the kept positions now live in.
type inkling2Drafter interface {
	Draft(history []int, k int) []int
	Observe(s *inkling2Resident, fed []int, keep int) error
}

// inkling2CostObserver is a drafter that sizes its drafts from the wall time of
// plain steps (width 1) and verifies of each width.
type inkling2CostObserver interface {
	ObserveCost(width int, took time.Duration)
}

// Inkling2SpecStats extends the greedy stats with the speculative step counts.
type Inkling2SpecStats struct {
	Inkling2LoopGreedyStats
	Verifies, Drafted, Accepted, PlainSteps int
	Draft, Verify                           time.Duration
}

// enableSpec allocates the verify chunk state for up to depth drafted tokens.
func (s *inkling2Resident) enableSpec(depth int) error {
	if s.kvHalf || s.kvFP8 {
		// the verify and draft ops read fp32 caches only
		return fmt.Errorf("inkling2 spec: FAK_INKLING2_KV=%s has no speculative decode; unset FAK_INKLING2_SPEC or FAK_INKLING2_KV", s.kvDType())
	}
	if _, ok := s.be.(compute.Inkling2VerifyDevice); !ok || !s.batched {
		return fmt.Errorf("inkling2 spec: backend %s has no batched verify ops", s.be.Name())
	}
	hp := s.hp
	T := depth + 1
	s.specMax = T
	s.ensurePrefillArenas(T)
	kvDim := hp.KVHeads * hp.HeadDim
	channels := 2*kvDim + 2*hp.Hidden
	L := hp.ConvKernel - 1 + T
	s.convLog = make([]ink2Ref, len(s.hist))
	for slot := range s.convLog {
		s.convLog[slot] = s.alloc(channels * L)
	}
	s.verifyLogits = s.alloc(T * s.output.rows)
	s.verifyIdx = s.alloc(4 * T)
	s.verifyPos, s.verifyKeep = s.alloc(4), s.alloc(4)
	return s.err
}

// verify runs tokens at positions pos..pos+len-1 in one batched forward and returns
// the greedy token after each. Caches and conv histories hold every row until
// commitVerify keeps a prefix.
func (s *inkling2Resident) verify(tokens []int) ([]int, error) {
	hp := s.hp
	T := len(tokens)
	if T < 1 || T > s.specMax {
		return nil, fmt.Errorf("inkling2 spec: verify of %d tokens, max %d", T, s.specMax)
	}
	if s.pos+T > s.ctx {
		return nil, fmt.Errorf("inkling2 spec: %d tokens at %d exceed the %d-token global cache", T, s.pos, s.ctx)
	}
	pd := s.be.(compute.Inkling2PrefillDevice)
	vd := s.be.(compute.Inkling2VerifyDevice)
	rows := make([]float32, T*hp.Hidden)
	for i, tok := range tokens {
		s.w.TokEmbd.Row(tok, rows[i*hp.Hidden:(i+1)*hp.Hidden])
	}
	s.fail(s.dev.Ink2Write(s.prefillIn, rows))
	s.fail(s.dev.Ink2Write(s.verifyPos, []float32{intBits(s.pos)}))
	build := func() {
		// at verify widths the decode expert GEMV over the T*k pairs beats the
		// tensor-core grouped GEMM (11.4 vs 17.5 ms at T=1 on an RTX 5080)
		p := &ink2Prefill{s: s, pd: pd, vd: vd, grouped: inkling2VerifyGrouped(), verify: true}
		p.keepFloats, p.scrFloats = s.prefillArenas(s.prefillCap)
		p.keep, p.scr = s.prefillKeep, s.prefillScr
		p.T, p.pos0 = T, s.pos
		p.forward(true)
	}
	// each width is captured once, on its first run after an eager run of that width
	// and of the full width have grown the lazily sized batched scratch, which a
	// capture cannot allocate
	full := T == s.specMax
	slot, ready, warm := inkling2VerifySlot, &s.verifyReady, &s.verifyWarm
	if !full {
		slot, _ = inkling2TailSlots(T)
		ready, warm = &s.tailReady[T], &s.tailWarm[T]
	}
	switch {
	case *ready:
		s.fail(vd.Ink2GraphLaunchSlot(slot))
	case s.graph && s.verifyWarm && *warm:
		if err := s.dev.Ink2GraphBegin(); err != nil {
			if errors.Is(err, errors.ErrUnsupported) {
				// a device without capture verifies eagerly, as step decodes
				s.graph = false
				build()
				break
			}
			s.fail(err)
			break
		}
		build()
		s.fail(vd.Ink2GraphEndSlot(slot))
		*ready = s.err == nil
		s.fail(vd.Ink2GraphLaunchSlot(slot))
	default:
		build()
		*warm = true
	}
	if s.err != nil {
		return nil, s.err
	}
	got, err := s.dev.Ink2ReadInts(s.verifyIdx, 4*T)
	if err != nil {
		return nil, err
	}
	out := make([]int, T)
	for t := range out {
		out[t] = int(got[4*t])
	}
	return out, nil
}

// commitVerify keeps the first keep of the T tokens the last verify consumed: the
// conv histories rewind to the state after them and the position advances by keep.
// Cache rows of the dropped tokens stay behind the position and are overwritten.
func (s *inkling2Resident) commitVerify(T, keep int) error {
	if keep < 0 || keep > T {
		return fmt.Errorf("inkling2 spec: keep %d of %d", keep, T)
	}
	if keep < T {
		hp := s.hp
		vd := s.be.(compute.Inkling2VerifyDevice)
		kvDim := hp.KVHeads * hp.HeadDim
		channels := 2*kvDim + 2*hp.Hidden
		past := hp.ConvKernel - 1
		// the log row length depends on T, so each width has its own commit graph
		gslot, ready := inkling2CommitSlot, &s.keepReady
		if T != s.specMax {
			_, gslot = inkling2TailSlots(T)
			ready = &s.tailKeepReady[T]
		}
		s.fail(s.dev.Ink2Write(s.verifyKeep, []float32{intBits(keep)}))
		if !*ready && s.graph {
			if err := s.dev.Ink2GraphBegin(); err == nil {
				for slot := range s.hist {
					s.fail(vd.Ink2PConvCommitDev(s.hist[slot], s.convLog[slot], s.verifyKeep, channels, past, past+T))
				}
				s.fail(vd.Ink2GraphEndSlot(gslot))
				*ready = s.err == nil
			}
		}
		if *ready {
			s.fail(vd.Ink2GraphLaunchSlot(gslot))
		} else {
			for slot := range s.hist {
				s.fail(vd.Ink2PConvCommitDev(s.hist[slot], s.convLog[slot], s.verifyKeep, channels, past, past+T))
			}
		}
	}
	s.pos += keep
	if r, ok := s.be.(interface{ Recycle() }); ok {
		r.Recycle()
	}
	return s.err
}

// inkling2ResidentSpecGreedy is greedy decode where each step verifies the
// drafter's proposals in one batched forward and keeps the longest prefix the
// target agrees with, so its tokens are the target's greedy tokens. A step with
// no proposal runs the plain graph step.
func inkling2ResidentSpecGreedy(m *Inkling2LoopModel, be compute.Backend, prompt []int, n, depth int,
	mk func(*inkling2Resident) (inkling2Drafter, error)) ([]int, Inkling2SpecStats, error) {
	return inkling2ResidentSpecGreedyUntil(m, be, prompt, n, depth, -1, mk)
}

// inkling2ResidentSpecGreedyUntil is inkling2ResidentSpecGreedy that stops at eos
// the way inkling2GreedyStepsUntil does, also when eos is an accepted draft token.
func inkling2ResidentSpecGreedyUntil(m *Inkling2LoopModel, be compute.Backend, prompt []int, n, depth, eos int,
	mk func(*inkling2Resident) (inkling2Drafter, error)) ([]int, Inkling2SpecStats, error) {
	var st Inkling2SpecStats
	s, err := newInkling2Resident(m, be)
	if err != nil {
		return nil, st, err
	}
	defer s.close()
	if err := s.enableSpec(depth); err != nil {
		return nil, st, err
	}
	d, err := mk(s)
	if err != nil {
		return nil, st, err
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	start := time.Now()
	if len(prompt) == 0 {
		return nil, st, fmt.Errorf("inkling2 spec: empty prompt")
	}
	next, err := s.prefill(context.Background(), prompt)
	if err != nil {
		return nil, st, err
	}
	if err := d.Observe(s, prompt, len(prompt)); err != nil {
		return nil, st, err
	}
	if s.graph {
		// one full-width verify grows every batched scratch before the decode graph
		// is captured; keep 0 rewinds it. Shorter widths warm and capture on first use.
		warm := make([]int, depth+1)
		for i := range warm {
			warm[i] = next
		}
		if _, err := s.verify(warm); err != nil {
			return nil, st, err
		}
		if err := s.commitVerify(len(warm), 0); err != nil {
			return nil, st, err
		}
	}
	st.Prefill = time.Since(start)
	out := make([]int, 0, n+depth)
	costs, _ := d.(inkling2CostObserver)
	// verify rows that leave the decode GEMV's bits need plain steps of the same
	// arithmetic: a width-1 verify
	verifyPlain := inkling2VerifyMMA(be)
	start = time.Now()
	for len(out) < n && next != eos {
		out = append(out, next)
		if len(out) == n {
			break
		}
		t0 := time.Now()
		props := d.Draft(out, depth)
		st.Draft += time.Since(t0)
		t0 = time.Now()
		if s.err != nil {
			return nil, st, s.err
		}
		plain := props == nil
		if plain && !verifyPlain {
			if next, err = s.step(next); err != nil {
				return nil, st, err
			}
			if costs != nil {
				costs.ObserveCost(1, time.Since(t0))
			}
			st.PlainSteps++
			if err := d.Observe(s, []int{out[len(out)-1]}, 1); err != nil {
				return nil, st, err
			}
			continue
		}
		fed := append([]int{next}, props...)
		t0 = time.Now()
		g, err := s.verify(fed)
		if err != nil {
			return nil, st, err
		}
		a := 0
		for a < len(props) && props[a] == g[a] {
			a++
		}
		if err := s.commitVerify(len(fed), a+1); err != nil {
			return nil, st, err
		}
		took := time.Since(t0)
		if costs != nil {
			costs.ObserveCost(len(fed), took)
		}
		if plain {
			st.PlainSteps++
		} else {
			st.Verify += took
			st.Verifies++
			st.Drafted += len(props)
			st.Accepted += a
		}
		if err := d.Observe(s, fed, a+1); err != nil {
			return nil, st, err
		}
		if stop := slices.Index(props[:a], eos); stop >= 0 {
			out = append(out, props[:stop]...)
			next = eos
			break
		}
		out = append(out, props[:a]...)
		next = g[a]
	}
	st.Decode = time.Since(start)
	if len(out) >= n {
		out = out[:n]
		st.DecodeSteps = n - 1
	} else {
		// every kept token ran one forward, the last of which produced eos
		st.DecodeSteps = len(out)
	}
	return out, st, nil
}

// inkling2OracleDrafter proposes the continuation of a known greedy run, with every
// token whose index hits corrupt replaced, so verify sees both accepts and rejects.
type inkling2OracleDrafter struct {
	ref     []int
	corrupt func(i int) bool
}

func (o *inkling2OracleDrafter) Draft(history []int, k int) []int {
	i := len(history)
	if i >= len(o.ref) {
		return nil
	}
	props := append([]int(nil), o.ref[i:min(len(o.ref), i+k)]...)
	for j := range props {
		if o.corrupt != nil && o.corrupt(i+j) {
			props[j] = (props[j] + 1) % 1000
		}
	}
	return props
}

func (o *inkling2OracleDrafter) Observe(*inkling2Resident, []int, int) error { return nil }

// inkling2SpecGreedy routes Inkling2LoopGreedyTimed through the speculative
// step: FAK_INKLING2_SPEC sets the draft depth, FAK_INKLING2_SPEC_DRAFT the draft
// source, and FAK_INKLING2_MTP_DIR the HF snapshot holding mtp.safetensors.
func inkling2SpecGreedy(m *Inkling2LoopModel, be compute.Backend, prompt []int, n, depth int) ([]int, Inkling2LoopGreedyStats, error) {
	return inkling2SpecGreedyUntil(m, be, prompt, n, depth, -1)
}

// inkling2SpecGreedyUntil is inkling2SpecGreedy stopping at eos.
func inkling2SpecGreedyUntil(m *Inkling2LoopModel, be compute.Backend, prompt []int, n, depth, eos int) ([]int, Inkling2LoopGreedyStats, error) {
	kind, err := inkling2SpecDraftKindFor(m)
	if err != nil {
		return nil, Inkling2LoopGreedyStats{}, err
	}
	dir := os.Getenv("FAK_INKLING2_MTP_DIR")
	var st Inkling2SpecStats
	var out []int
	dflashDir := inkling2DFlash2DirFor(m, be)
	switch inkling2ResolveSpecDraft(kind, dir, dflashDir) {
	case inkling2DraftLookup:
		out, st, err = inkling2ResidentSpecGreedyUntil(m, be, prompt, n, depth, eos, func(*inkling2Resident) (inkling2Drafter, error) {
			return newInkling2LookupDrafter(prompt), nil
		})
	case inkling2DraftDSpark:
		if dir == "" {
			return nil, Inkling2LoopGreedyStats{}, fmt.Errorf("inkling2 spec: the dspark draft needs FAK_INKLING2_MTP_DIR")
		}
		dm, oerr := OpenInkling2DSpark(dir, m)
		if oerr != nil {
			return nil, Inkling2LoopGreedyStats{}, oerr
		}
		defer dm.Close()
		depth = min(depth, dm.HP.Block-1)
		out, st, err = inkling2ResidentSpecGreedyUntil(m, be, prompt, n, depth, eos, func(s *inkling2Resident) (inkling2Drafter, error) {
			d, err := s.attachDraft(dm)
			if err == nil {
				d.steps = depth
			}
			return inkling2DSparkDrafter{d}, err
		})
	case inkling2DraftDFlash:
		if dflashDir == "" {
			return nil, Inkling2LoopGreedyStats{}, fmt.Errorf("inkling2 spec: the dflash2 draft needs %s beside the GGUF and a backend with DFlash2 ops", Inkling2DFlash2DirName)
		}
		dm, oerr := OpenInkling2DFlash2(dflashDir)
		if oerr != nil {
			return nil, Inkling2LoopGreedyStats{}, oerr
		}
		defer dm.Close()
		depth = min(depth, dm.Config.DFlash.Block-1)
		var dd *ink2DFlash2Draft
		out, st, err = inkling2ResidentSpecGreedyUntil(m, be, prompt, n, depth, eos, func(s *inkling2Resident) (inkling2Drafter, error) {
			d, err := s.attachDFlash2(dm)
			dd = d
			return inkling2DFlash2Drafter{d}, err
		})
		dd.close()
	default:
		return nil, Inkling2LoopGreedyStats{}, fmt.Errorf("inkling2 spec: the %s draft has no device drafter in this host loop; use lookup, dspark or dflash2", kind)
	}
	inkling2LastSpec = st
	return out, st.Inkling2LoopGreedyStats, err
}

// inkling2LastSpec holds the speculative counts of the last spec run for tests and
// benches that read them after Inkling2LoopGreedyTimed.
var inkling2LastSpec Inkling2SpecStats

// inkling2VerifyGrouped reports whether verify chunks use the grouped expert GEMM;
// only FAK_INKLING2_VERIFY_GROUPED=1 selects it.
func inkling2VerifyGrouped() bool { return os.Getenv("FAK_INKLING2_VERIFY_GROUPED") == "1" }

// inkling2VerifyMMA reports whether be's verify rows run arithmetic other than
// its decode step's (Metal's FAK_INKLING2_VERIFY_MMA), so a spec session takes
// its plain steps as width-1 verifies.
func inkling2VerifyMMA(be compute.Backend) bool {
	m, ok := be.(interface{ Ink2VerifyMMA() bool })
	return ok && m.Ink2VerifyMMA()
}
