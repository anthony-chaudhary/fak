package ggufload

import (
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// Inkling2MetalCohortStateBytes is the device memory the widest cohort state of rows
// sequences at ctx positions takes on a Metal resident binding, plus the headroom
// its admission keeps free. It replays batchShapeChunk without a device so serve
// can price the cohort engine before the load: Metal reports no free figure, so
// the engine's own admission fails open there.
func Inkling2MetalCohortStateBytes(f *File, rows, ctx int) (int64, error) {
	chunk := inkling2PrefillChunkWantedOn(inkling2MetalNamed{}, os.LookupEnv)
	return Inkling2MetalCohortStateBytesFor(f, rows, ctx, chunk, false, os.LookupEnv)
}

// Inkling2MetalCohortStateBytesFor is Inkling2MetalCohortStateBytes with an
// explicit prefill chunk, global-KV half-precision flag, and env lookup.
func Inkling2MetalCohortStateBytesFor(f *File, rows, ctx, chunk int, kvHalf bool, lookup func(string) (string, bool)) (int64, error) {
	if rows < 1 || ctx < 1 {
		return 0, fmt.Errorf("inkling2 cohort price: %d rows at %d positions", rows, ctx)
	}
	hp, vocab, err := inkling2FileShape(f)
	if err != nil {
		return 0, err
	}
	return inkling2MetalCohortBytesFor(hp, vocab, rows, ctx, chunk, kvHalf, lookup)
}

// Inkling2PrefillArenaBytes returns the device bytes of the prefill arenas and
// chunk input buffer for chunk tokens on the model in f.
func Inkling2PrefillArenaBytes(f *File, chunk int) (int64, error) {
	if chunk <= 0 {
		return 0, nil
	}
	hp, _, err := inkling2FileShape(f)
	if err != nil {
		return 0, err
	}
	keep, scr := inkling2PrefillArenaFloats(hp, chunk, hp.ExpertWidth)
	return 4 * int64(keep+scr+chunk*hp.Hidden), nil
}

func inkling2FileShape(f *File) (model.Inkling2LoopHParams, int, error) {
	arch, _ := f.String("general.architecture")
	if arch != inkling2ArchLoop && arch != inkling2ArchBase {
		return model.Inkling2LoopHParams{}, 0, fmt.Errorf("inkling2 cohort price: unsupported general.architecture %q", arch)
	}
	b := &inkling2Binder{f: f, arch: arch, prefix: arch + "."}
	hp, _, err := b.hparams(arch)
	if err != nil {
		return model.Inkling2LoopHParams{}, 0, err
	}
	vocab := 0
	for _, t := range f.Tensors {
		if t.Name == "output.weight" && len(t.Dims) == 2 {
			vocab = int(t.Dims[1])
		}
	}
	if vocab == 0 {
		return model.Inkling2LoopHParams{}, 0, fmt.Errorf("inkling2 cohort price: %s lacks a 2-D output.weight", arch)
	}
	return hp, vocab, nil
}

func inkling2MetalCohortBytes(hp model.Inkling2LoopHParams, vocab, rows, ctx int) (int64, error) {
	chunk := inkling2PrefillChunkWantedOn(inkling2MetalNamed{}, os.LookupEnv)
	return inkling2MetalCohortBytesFor(hp, vocab, rows, ctx, chunk, false, os.LookupEnv)
}

func inkling2MetalCohortBytesFor(hp model.Inkling2LoopHParams, vocab, rows, ctx, chunk int, kvHalf bool, lookup func(string) (string, bool)) (int64, error) {
	if rows < 1 || ctx < 1 {
		return 0, fmt.Errorf("inkling2 cohort price: %d rows at %d positions", rows, ctx)
	}
	if lookup == nil {
		lookup = os.LookupEnv
	}
	// Metal binds every expert unpadded, so padK is the expert input width.
	s := &inkling2Resident{hp: hp, layers: make([]ink2Layer, hp.Layers), output: ink2Mat{rows: vocab}, batchLookup: lookup}
	for i := range s.layers {
		s.layers[i].gateUp.padK, s.layers[i].down.padK = hp.Hidden, hp.ExpertWidth
	}
	// batched rings are the wider ones, so this bounds either binding
	s.baseCaps = inkling2ResidentCaps(hp, 0, true, inkling2PrefillChunkBase)
	sh, err := s.batchShapeChunk(rows, min(ctx, Inkling2LoopResidentContext), chunk)
	if err != nil {
		return 0, err
	}
	need := inkling2BatchBytes(hp, sh.caps, rows, sh.keep, sh.scr, kvHalf)
	// without a device the few-row path's scratch is unseen; Metal may take it
	if few := min(rows, inkling2BatchFewRowsMost(lookup)); few > 0 {
		need += 4 * int64(inkling2BatchFewRowsScratch(hp, few))
	}
	return need + inkling2BatchHeadroom, nil
}

type inkling2MetalNamed struct{}

func (inkling2MetalNamed) Name() string { return "metal" }
