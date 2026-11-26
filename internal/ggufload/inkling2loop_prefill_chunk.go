package ggufload

import (
	"errors"
	"fmt"
	"log"
	"strconv"
	"strings"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkling2PrefillChunkEnv opts into another batched prefill forward width. Unset keeps
// the 512-token chunk and every buffer size that goes with it, except on Metal, whose
// unset default is inkling2PrefillChunkMetal. "1024" (or any multiple
// of 16 in [64, 2048]) sets the width, and "auto" selects 1024.
//
// A wide setting costs memory at bind, long prompt or not: a scalar session widens
// every local sliding-window ring to window + chunk slots and allocates the wide
// prefill arenas there, because a ring cannot be resized once it holds positions. A
// cohort pays only when one of its prompts is longer than 512 tokens. A width the
// device refuses, or a value that does not parse, falls back to 512 with one log line.
const inkling2PrefillChunkEnv = "FAK_INKLING2_PREFILL_CHUNK"

const (
	// inkling2PrefillChunkBase is the chunk of a session that did not opt in.
	inkling2PrefillChunkBase = inkling2PrefillChunk
	// inkling2PrefillChunkWide is the chunk "auto" asks for.
	inkling2PrefillChunkWide = 1024
	inkling2PrefillChunkMin  = 64
	inkling2PrefillChunkMax  = 2048
	// inkling2PrefillReprobeEvery is how many wide-wanting cohorts run at the base
	// chunk after a refusal before the wide chunk is tried again.
	inkling2PrefillReprobeEvery = 16
)

// inkling2PrefillChunkParse reads a FAK_INKLING2_PREFILL_CHUNK value; an unset or
// empty value is the base chunk.
func inkling2PrefillChunkParse(v string, set bool) (int, error) {
	v = strings.ToLower(strings.TrimSpace(v))
	if !set || v == "" {
		return inkling2PrefillChunkBase, nil
	}
	if v == "auto" {
		return inkling2PrefillChunkWide, nil
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < inkling2PrefillChunkMin || n > inkling2PrefillChunkMax || n%16 != 0 {
		return inkling2PrefillChunkBase, fmt.Errorf("%s=%q is not auto or a multiple of 16 in %d..%d",
			inkling2PrefillChunkEnv, v, inkling2PrefillChunkMin, inkling2PrefillChunkMax)
	}
	return n, nil
}

// inkling2PrefillChunkWarned keeps a bad setting to one log line per process.
var inkling2PrefillChunkWarned atomic.Bool

// inkling2PrefillChunkWanted is the chunk width the setting asks for; a value that
// does not parse is the base chunk.
func inkling2PrefillChunkWanted(lookup func(string) (string, bool)) int {
	n, err := inkling2PrefillChunkParse(lookup(inkling2PrefillChunkEnv))
	if err != nil && inkling2PrefillChunkWarned.CompareAndSwap(false, true) {
		log.Printf("inkling2 prefill: %v; using %d", err, n)
	}
	return n
}

// inkling2PrefillChunkMetal is the chunk of a Metal session that did not set
// FAK_INKLING2_PREFILL_CHUNK: with hundreds of experts a 512-token chunk leaves each
// expert's grouped GEMM a handful of rows, so the wider chunk pays for its arenas.
const inkling2PrefillChunkMetal = 2048

// inkling2PrefillChunkWantedOn is inkling2PrefillChunkWanted on backend be: an unset
// setting on Metal asks for inkling2PrefillChunkMetal, still subject to admission.
func inkling2PrefillChunkWantedOn(be any, lookup func(string) (string, bool)) int {
	if v, set := lookup(inkling2PrefillChunkEnv); (!set || strings.TrimSpace(v) == "") && inkling2OnMetal(be) {
		return inkling2PrefillChunkMetal
	}
	return inkling2PrefillChunkWanted(lookup)
}

func inkling2OnMetal(be any) bool {
	n, ok := be.(interface{ Name() string })
	return ok && n.Name() == "metal"
}

// inkling2MetalWorkingSet is swapped by tests that run without a Metal device.
var inkling2MetalWorkingSet = compute.DarwinWorkingSetLimits

// inkling2WideFree is the free device memory a wide chunk is admitted against. Metal
// reports no free figure; on unified memory the working set the driver recommends,
// less what this process already holds on the device, is the room a wide chunk may
// take without the driver paging it.
func inkling2WideFree(be any, free int64, known bool) (int64, bool) {
	if (known && free >= 0) || !inkling2OnMetal(be) {
		return free, known
	}
	lim := inkling2MetalWorkingSet()
	if lim.RecommendedMaxWorkingSet == 0 {
		return free, known
	}
	rec, used := int64(lim.RecommendedMaxWorkingSet), int64(lim.CurrentAllocatedSize)
	return max(0, rec-used), true
}

// inkling2CohortChunk is the prefill chunk of a cohort whose longest prompt is longest
// tokens: a cohort no prompt of which passes the base chunk never widens.
func inkling2CohortChunk(wanted, longest int) int {
	if longest > inkling2PrefillChunkBase {
		return wanted
	}
	return min(wanted, inkling2PrefillChunkBase)
}

// inkling2DeviceRefused marks an error the device raised while allocating, so the
// base-chunk fallback retries only on a refusal and never on a validation error.
type inkling2DeviceRefused struct{ err error }

func (e inkling2DeviceRefused) Error() string { return e.err.Error() }
func (e inkling2DeviceRefused) Unwrap() error { return e.err }

// inkling2ChunkRefusal reports whether err is a device or admission refusal, the only
// failures of a wide chunk the base-chunk fallback recovers from.
func inkling2ChunkRefusal(err error) bool {
	var dr inkling2DeviceRefused
	var mem *Inkling2BatchMemoryError
	return errors.As(err, &dr) || errors.As(err, &mem)
}

// inkling2PrefillChunkFor is the token count of one prefill forward over n tokens on
// a session whose prefill width is chunk.
func inkling2PrefillChunkFor(n, chunk int) int {
	return max(0, min(chunk, n))
}

// inkling2WideRingExtraBytes is the device memory the local rings of one sequence
// gain going from the base chunk to chunk.
func inkling2WideRingExtraBytes(hp model.Inkling2LoopHParams, slots, chunk int) int64 {
	if chunk <= inkling2PrefillChunkBase {
		return 0
	}
	wide := inkling2ResidentCaps(hp, 0, true, chunk)
	base := inkling2ResidentCaps(hp, 0, true, inkling2PrefillChunkBase)
	var positions int64
	for slot := 0; slot < slots; slot++ {
		positions += int64(wide[slot] - base[slot])
	}
	return 4 * 2 * positions * int64(hp.KVHeads*hp.HeadDim)
}

// logInkling2PrefillFallback records one fallback from a wide chunk to the base.
func logInkling2PrefillFallback(why error, chunk int) {
	log.Printf("inkling2 prefill: chunk %d refused (%v); using %d", chunk, why, inkling2PrefillChunkBase)
}

// inkling2ScalarWideHold counts the scalar sessions still to bind at the base chunk
// after a refusal, so a full device does not pay a refused wide build, weights
// included, for every new session.
var inkling2ScalarWideHold atomic.Int32

// inkling2ScalarWideLogged keeps the scalar fallback to one log line per process: a
// server opens a session per request.
var inkling2ScalarWideLogged atomic.Bool

// inkling2ScalarWideHeld reports whether a session that wants a wide chunk binds at
// the base one because a recent session was refused, and spends one of the hold-off.
func inkling2ScalarWideHeld(wanted int) bool {
	if wanted <= inkling2PrefillChunkBase || inkling2ScalarWideHold.Load() <= 0 {
		return false
	}
	inkling2ScalarWideHold.Add(-1)
	return true
}

// inkling2ScalarWideRefused starts the hold-off after a refused wide scalar bind.
func inkling2ScalarWideRefused(why error, chunk int) {
	inkling2ScalarWideHold.Store(inkling2PrefillReprobeEvery)
	if inkling2ScalarWideLogged.CompareAndSwap(false, true) {
		logInkling2PrefillFallback(why, chunk)
	}
}

// wideBindBytes is every byte a scalar session of chunk tokens still has to
// allocate once its weights are bound, past the caches a base session holds: the
// extra ring and score positions, the whole wide arenas (a base session allocates its
// arenas only at its first prefill), and the per-token input, arena and argmax rows.
func (s *inkling2Resident) wideBindBytes(ctx, chunk int) int64 {
	hp := s.hp
	maxCap := func(caps []int) int {
		most := hp.SlidingWindow
		for _, c := range caps {
			most = max(most, c)
		}
		return most
	}
	wide := inkling2ResidentCaps(hp, ctx, true, chunk)
	base := inkling2ResidentCaps(hp, ctx, true, inkling2PrefillChunkBase)
	keep, scr := s.prefillArenas(chunk)
	floats := int64(keep+scr+chunk*hp.Hidden) + int64(2*hp.Heads*(maxCap(wide)-maxCap(base))) +
		int64(hp.Hidden+4+s.arenaFloats+4)
	return inkling2WideRingExtraBytes(hp, len(wide), chunk) + 4*floats
}

// admitWide admits the bind of a scalar session of chunk tokens before any of it is
// allocated.
func (s *inkling2Resident) admitWide(ctx, chunk int) error {
	_, free, known := inkling2DeviceFree(s.be)
	free, known = inkling2WideFree(s.be, free, known)
	return inkling2WideAdmit(1, s.wideBindBytes(ctx, chunk), free, known)
}

// admitWideChunk narrows s.chunkMax, halving it toward the base chunk, to the widest
// chunk the device admits, so a device short of the asked width still gets a wide
// chunk; the error is the refusal of the narrowest wide width tried.
func (s *inkling2Resident) admitWideChunk(ctx int) error {
	var err error
	for c := s.chunkMax; c > inkling2PrefillChunkBase; c = (c / 2) &^ 15 {
		if err = s.admitWide(ctx, c); err == nil {
			s.chunkMax = c
			return nil
		}
	}
	return err
}

// inkling2WideAdmit admits the bytes of a wide chunk against the probed free memory.
// Unlike the base admission it fails closed on an unknown free: the wide chunk is
// opt-in and must not take memory the device cannot be seen to have.
func inkling2WideAdmit(C int, need, free int64, known bool) error {
	if known && free >= 0 && need+inkling2BatchHeadroom <= free {
		return nil
	}
	return &Inkling2BatchMemoryError{Sequences: C, Need: need, Free: free}
}
