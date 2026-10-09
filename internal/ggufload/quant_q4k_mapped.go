package ggufload

// This loader keeps eligible packed tensors as read-only mapped views while
// preserving ordinary resident-model semantics. Platforms without mapping
// support fall back to owned storage through the existing reader contract.

import (
	"context"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// LoadModelQ4KMappedResident loads path through the mapped-resident Q4_K path and transfers the
// checkpoint lifetime to the returned model: CloseWeights must be called when serving stops, and
// it unmaps the checkpoint only after dropping the tensor views.
//
// Streamed (WithStreamedDenseQ4K / WithStreamedExperts) and expert-sharded option lists are
// refused: those keep their own bounded entry points and working-set contracts.
func LoadModelQ4KMappedResident(path string, p *LoadProfiler, opts ...Q4KLoadOption) (*model.Model, error) {
	return LoadModelQ4KMappedResidentContext(context.Background(), path, p, opts...)
}

// LoadModelQ4KMappedResidentContext is LoadModelQ4KMappedResident with cooperative cancellation.
// On any error the checkpoint is closed before returning.
func LoadModelQ4KMappedResidentContext(ctx context.Context, path string, p *LoadProfiler, opts ...Q4KLoadOption) (*model.Model, error) {
	return loadModelQ4KMappedResidentContext(ctx, path, p, OpenWeightsMapped, opts...)
}

// mappedResidentArchSupported reports whether the mapped-resident route keeps exactly the
// tensor representations of the historical resident loader. The MLA-MoE layout keeps dense
// k-quants off the raw store (computeQ4KTensorWork's !archUsesMLAMoELayout gate), which the
// descriptor route does not re-check, so those architectures take the owned-copy load.
func mappedResidentArchSupported(cfg model.Config) bool {
	return !archUsesMLAMoELayout(cfg.ModelType)
}

func loadModelQ4KMappedResidentContext(ctx context.Context, path string, p *LoadProfiler, open func(string) (*WeightSource, error), opts ...Q4KLoadOption) (*model.Model, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if o := probeQ4KLoadOptions(opts); o.streamedExperts || o.streamedDenseQ4K || o.expertShardSet {
		return nil, fmt.Errorf("gguf: mapped-resident Q4_K load is the ordinary unbounded residency; " +
			"streamed or expert-sharded option lists keep their own load entries")
	}
	ws, err := open(path)
	if err != nil {
		return nil, err
	}
	cfg, err := ws.File.Config()
	if err != nil {
		_ = ws.Close()
		return nil, err
	}
	if !mappedResidentArchSupported(cfg) {
		// Same owned-copy residency as LoadModelQ4KProfileOptions; the mapped reader only
		// changes where TensorBytes copies from.
		defer ws.Close()
		return ws.QuantModelQ4KProfileOptionsContext(ctx, p, opts...)
	}
	streamed := append(append([]Q4KLoadOption(nil), opts...), WithStreamedDenseQ4K(true))
	m, err := ws.QuantModelQ4KProfileOptionsContext(ctx, p, streamed...)
	if err != nil {
		_ = ws.Close()
		return nil, err
	}
	m.SetWeightCloser(ws)
	if err := m.RetainMappedQ4KResidency(); err != nil {
		if closeErr := m.CloseWeights(); closeErr != nil {
			err = fmt.Errorf("%w; checkpoint close: %v", err, closeErr)
		}
		return nil, err
	}
	return m, nil
}
