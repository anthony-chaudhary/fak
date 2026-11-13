package main

import (
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/fitplan"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type fitBand struct {
	Rank       int    `json:"rank"`
	PoolID     string `json:"pool_id"`
	Lo         int    `json:"lo"`
	Hi         int    `json:"hi"`
	LayerBytes int64  `json:"layer_bytes"`
}

func fitLayerBands(s *ggufload.WeightSource, in fitplan.DeploymentInput) ([]fitBand, error) {
	c, err := s.CensusQ4KLayerStorage(fitBandLoaderOptions()...)
	if err != nil {
		return nil, err
	}
	cfg, err := s.File.Config()
	if err != nil {
		return nil, err
	}
	return fitBandsFromCensus(cfg, c, in)
}

func fitBandLoaderOptions() []ggufload.Q4KLoadOption {
	return []ggufload.Q4KLoadOption{
		ggufload.WithMTPRetention(false), ggufload.WithNativeLinearAttnRows(false),
		ggufload.WithQwen35NativeSSMOut(false), ggufload.WithDenseKQuantResident(true),
		ggufload.WithQ2KEmbeddingResident(false), ggufload.WithQ4KEmbeddingResident(false),
		ggufload.WithTiedQ6KEmbeddingResident(false),
	}
}

func fitBandsFromCensus(cfg model.Config, c ggufload.LayerStorageCensus, in fitplan.DeploymentInput) ([]fitBand, error) {
	p := in.Placement
	if len(c.Layers) == 0 || len(c.Layers) != cfg.NumLayers || p.Weights.ReplicatedBytes != 0 || p.Weights.ShardCount != len(c.Layers) {
		return nil, fmt.Errorf("fit bands: declaration requires one shard per layer and no replicated charge")
	}
	r, err := fitplan.EvaluatePlacement(in)
	if err != nil {
		return nil, err
	}
	if r.CapacityStatus != fitplan.StatusFits {
		return nil, fmt.Errorf("fit bands: declared capacity does not fit")
	}
	bands := make([]fitBand, 0, len(p.Ranks))
	cuts := make([]int, 0, len(p.Ranks)-1)
	seen := map[string]bool{}
	lo := 0
	for i, rank := range p.Ranks {
		if rank.Rank != i || rank.Shards <= 0 || seen[rank.PoolID] || rank.Shards > len(c.Layers)-lo {
			return nil, fmt.Errorf("fit bands: invalid pinned rank or pool")
		}
		seen[rank.PoolID] = true
		hi := lo + rank.Shards
		var n int64
		for _, b := range c.Layers[lo:hi] {
			if b < 0 || n > math.MaxInt64-b {
				return nil, fmt.Errorf("fit bands: layer weight overflow")
			}
			n += b
		}
		if n > r.Ranks[i].ShardedBytes {
			return nil, fmt.Errorf("fit bands: rank %d exceeds shard charge", i)
		}
		var fixed int64
		if i == 0 {
			fixed = c.Embedding
		}
		if i == len(p.Ranks)-1 {
			if fixed < 0 || c.Head < 0 || fixed > math.MaxInt64-c.Head {
				return nil, fmt.Errorf("fit bands: head charge invalid")
			}
			fixed += c.Head
		}
		if fixed < 0 || rank.ExtraBytes < fixed {
			return nil, fmt.Errorf("fit bands: rank %d lacks embedding or head charge", i)
		}
		bands = append(bands, fitBand{rank.Rank, rank.PoolID, lo, hi, n})
		if i != len(p.Ranks)-1 {
			cuts = append(cuts, hi)
		}
		lo = hi
	}
	if lo != len(c.Layers) {
		return nil, fmt.Errorf("fit bands: incomplete coverage")
	}
	if _, err := model.NewPartitionPlan(cfg, cuts); err != nil {
		return nil, err
	}
	if cfg.IsQwen35Hybrid() && len(cuts) > 0 {
		costs, err := model.HybridDeviceBandCosts(cfg, c.Layers, c.Embedding, c.Head, 0, 0, compute.KVPrecisionF32)
		if err != nil {
			return nil, err
		}
		for _, cut := range cuts {
			if !costs.Layers[cut-1].FullAttention {
				return nil, fmt.Errorf("fit bands: illegal hybrid cut %d", cut)
			}
		}
	}
	return bands, nil
}
