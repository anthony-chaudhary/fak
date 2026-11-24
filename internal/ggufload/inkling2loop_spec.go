package ggufload

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// inkling2SpecMaxDraft bounds the drafted tokens of one speculative step, so a
// verify chunk is at most inkling2SpecMaxDraft+1 tokens.
const inkling2SpecMaxDraft = 8

// Device graph slots of a speculative session beside the decode graph.
const (
	inkling2VerifySlot = 1
	inkling2CommitSlot = 2
)

// intBits carries an int32 through a float32 device word.
func intBits(v int) float32 { return math.Float32frombits(uint32(int32(v))) }

// inkling2SpecDepth is the drafted-token count of a speculative step from
// FAK_INKLING2_SPEC (unset or 0 = plain decode; 1 = the default depth).
func inkling2SpecDepth() int {
	v := os.Getenv("FAK_INKLING2_SPEC")
	if v == "" || v == "0" {
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
// the one the next forward consumes. Observe reports each verify outcome: the
// tokens fed (fed[0] is that last token), how many were kept, and the session the
// kept positions now live in.
type inkling2Drafter interface {
	Draft(history []int, k int) []int
	Observe(s *inkling2Resident, fed []int, keep int) error
}

// Inkling2SpecStats extends the greedy stats with the speculative step counts.
type Inkling2SpecStats struct {
	Inkling2LoopGreedyStats
	Verifies, Drafted, Accepted, PlainSteps int
	Draft, Verify                           time.Duration
}

// enableSpec allocates the verify chunk state for up to depth drafted tokens.
func (s *inkling2Resident) enableSpec(depth int) error {
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
	if s.pos+T > inkling2ResidentCtx {
		return nil, fmt.Errorf("inkling2 spec: %d tokens at %d exceed the %d-token global cache", T, s.pos, inkling2ResidentCtx)
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
	// the full-width verify is captured once, after an eager run has grown every
	// lazily sized batched scratch; shorter tail verifies run eagerly
	full := T == s.specMax
	switch {
	case full && s.verifyReady:
		s.fail(vd.Ink2GraphLaunchSlot(inkling2VerifySlot))
	case full && s.graph && s.verifyWarm:
		if err := s.dev.Ink2GraphBegin(); err != nil {
			s.fail(err)
			break
		}
		build()
		s.fail(vd.Ink2GraphEndSlot(inkling2VerifySlot))
		s.verifyReady = s.err == nil
		s.fail(vd.Ink2GraphLaunchSlot(inkling2VerifySlot))
	default:
		build()
		s.verifyWarm = s.verifyWarm || full
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
		switch {
		case T != s.specMax:
			for slot := range s.hist {
				s.fail(vd.Ink2PConvCommit(s.hist[slot], s.convLog[slot], channels, past, past+T, keep))
			}
		default:
			s.fail(s.dev.Ink2Write(s.verifyKeep, []float32{intBits(keep)}))
			if !s.keepReady && s.graph {
				if err := s.dev.Ink2GraphBegin(); err == nil {
					for slot := range s.hist {
						s.fail(vd.Ink2PConvCommitDev(s.hist[slot], s.convLog[slot], s.verifyKeep, channels, past, past+T))
					}
					s.fail(vd.Ink2GraphEndSlot(inkling2CommitSlot))
					s.keepReady = s.err == nil
				}
			}
			if s.keepReady {
				s.fail(vd.Ink2GraphLaunchSlot(inkling2CommitSlot))
			} else {
				for slot := range s.hist {
					s.fail(vd.Ink2PConvCommitDev(s.hist[slot], s.convLog[slot], s.verifyKeep, channels, past, past+T))
				}
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
	next, err := s.prefill(prompt)
	if err != nil {
		return nil, st, err
	}
	if err := d.Observe(s, prompt, len(prompt)); err != nil {
		return nil, st, err
	}
	// one full-width verify grows every batched scratch before the decode graph is
	// captured; keep 0 rewinds it
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
	st.Prefill = time.Since(start)
	out := make([]int, 0, n+depth)
	start = time.Now()
	for len(out) < n {
		out = append(out, next)
		if len(out) == n {
			break
		}
		t0 := time.Now()
		props := d.Draft(out, depth)
		st.Draft += time.Since(t0)
		if s.err != nil {
			return nil, st, s.err
		}
		if len(props) == 0 {
			if next, err = s.step(next); err != nil {
				return nil, st, err
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
		st.Verify += time.Since(t0)
		st.Verifies++
		st.Drafted += len(props)
		st.Accepted += a
		if err := d.Observe(s, fed, a+1); err != nil {
			return nil, st, err
		}
		out = append(out, props[:a]...)
		next = g[a]
	}
	st.Decode = time.Since(start)
	out = out[:n]
	st.DecodeSteps = n - 1
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

// inkling2SpecGreedy routes Inkling2LoopGreedyTimed through the DSpark speculative
// step: FAK_INKLING2_SPEC sets the draft depth and FAK_INKLING2_MTP_DIR the HF
// snapshot holding mtp.safetensors.
func inkling2SpecGreedy(m *Inkling2LoopModel, be compute.Backend, prompt []int, n, depth int) ([]int, Inkling2LoopGreedyStats, error) {
	dir := os.Getenv("FAK_INKLING2_MTP_DIR")
	if dir == "" {
		return nil, Inkling2LoopGreedyStats{}, fmt.Errorf("inkling2 spec: FAK_INKLING2_SPEC needs FAK_INKLING2_MTP_DIR")
	}
	dm, err := OpenInkling2DSpark(dir, m)
	if err != nil {
		return nil, Inkling2LoopGreedyStats{}, err
	}
	defer dm.Close()
	depth = min(depth, dm.HP.Block-1)
	out, st, err := inkling2ResidentSpecGreedy(m, be, prompt, n, depth, func(s *inkling2Resident) (inkling2Drafter, error) {
		d, err := s.attachDraft(dm)
		if err == nil {
			d.steps = depth
		}
		return inkling2DSparkDrafter{d}, err
	})
	inkling2LastSpec = st
	return out, st.Inkling2LoopGreedyStats, err
}

// inkling2LastSpec holds the speculative counts of the last spec run for tests and
// benches that read them after Inkling2LoopGreedyTimed.
var inkling2LastSpec Inkling2SpecStats

// inkling2VerifyGrouped reports whether verify chunks use the grouped expert GEMM;
// only FAK_INKLING2_VERIFY_GROUPED=1 selects it.
func inkling2VerifyGrouped() bool { return os.Getenv("FAK_INKLING2_VERIFY_GROUPED") == "1" }
