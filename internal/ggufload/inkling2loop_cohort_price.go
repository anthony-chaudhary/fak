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
	if rows < 1 || ctx < 1 {
		return 0, fmt.Errorf("inkling2 cohort price: %d rows at %d positions", rows, ctx)
	}
	arch, _ := f.String("general.architecture")
	if arch != inkling2ArchLoop && arch != inkling2ArchBase {
		return 0, fmt.Errorf("inkling2 cohort price: unsupported general.architecture %q", arch)
	}
	b := &inkling2Binder{f: f, arch: arch, prefix: arch + "."}
	hp, _, err := b.hparams(arch)
	if err != nil {
		return 0, err
	}
	vocab := 0
	for _, t := range f.Tensors {
		if t.Name == "output.weight" && len(t.Dims) == 2 {
			vocab = int(t.Dims[1])
		}
	}
	if vocab == 0 {
		return 0, fmt.Errorf("inkling2 cohort price: %s lacks a 2-D output.weight", arch)
	}
	return inkling2MetalCohortBytes(hp, vocab, rows, ctx)
}

func inkling2MetalCohortBytes(hp model.Inkling2LoopHParams, vocab, rows, ctx int) (int64, error) {
	if rows < 1 || ctx < 1 {
		return 0, fmt.Errorf("inkling2 cohort price: %d rows at %d positions", rows, ctx)
	}
	// Metal binds every expert unpadded, so padK is the expert input width.
	s := &inkling2Resident{hp: hp, layers: make([]ink2Layer, hp.Layers), output: ink2Mat{rows: vocab}}
	for i := range s.layers {
		s.layers[i].gateUp.padK, s.layers[i].down.padK = hp.Hidden, hp.ExpertWidth
	}
	// batched rings are the wider ones, so this bounds either binding
	s.baseCaps = inkling2ResidentCaps(hp, 0, true, inkling2PrefillChunkBase)
	chunk := inkling2PrefillChunkWantedOn(inkling2MetalNamed{}, os.LookupEnv)
	sh, err := s.batchShapeChunk(rows, min(ctx, Inkling2LoopResidentContext), chunk)
	if err != nil {
		return 0, err
	}
	// without a device the few-row path's scratch is unseen; Metal may take it
	if few := min(rows, inkling2BatchFewRowsMost(os.LookupEnv)); few > 0 {
		sh.need += 4 * int64(inkling2BatchFewRowsScratch(hp, few))
	}
	return sh.need + inkling2BatchHeadroom, nil
}

type inkling2MetalNamed struct{}

func (inkling2MetalNamed) Name() string { return "metal" }
