package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

var ErrV4ExpertRuntime = errors.New("v4 expert runtime failure")

type v4ExpertRuntimeStats struct {
	ExpertOpenCount     int
	ExpertReadCount     int
	HashOpenCount       int
	HashReadCount       int
	HashReadBytes       int64
	SourceReads         int64
	SourceBytes         int64
	PageIns             int
	Hits                int
	Evictions           int
	ResidentBytes       int64
	PeakResident        int64
	RingBudget          int64
	WorldSize           int
	Rank                int
	LocalSelected       int64
	RemoteSelected      int64
	TransportDispatches int64
	TransportPartials   int64
}

type v4ExpertRuntime struct {
	cfg       Config
	experts   *v4ShardedExpertSource
	hashes    *v4HashRouterSource
	ring      *pagedRing
	closed    bool
	stats     v4ExpertRuntimeStats
	placement V4ExpertPlacement
	transport v4ExpertTransport
}

func newV4ExpertRuntime(dir string, cfg Config, be compute.Backend, ringByteCap int64, maxOpen int) (*v4ExpertRuntime, error) {
	if err := AdmitDeepSeekV4Config(cfg); err != nil {
		return nil, fmt.Errorf("%w: config: %v", ErrV4ExpertRuntime, err)
	}
	if be == nil || ringByteCap <= 0 || maxOpen <= 0 {
		return nil, fmt.Errorf("%w: nil backend or invalid ring/open cap", ErrV4ExpertRuntime)
	}
	placement, err := v4ExpertPlacementFromEnv(cfg)
	if err != nil {
		return nil, fmt.Errorf("%w: placement: %v", ErrV4ExpertRuntime, err)
	}
	experts, err := newV4ShardedExpertSource(dir, maxOpen)
	if err != nil {
		return nil, fmt.Errorf("%w: expert source: %v", ErrV4ExpertRuntime, err)
	}
	hashes, err := newV4HashRouterSource(dir)
	if err != nil {
		experts.Close()
		return nil, fmt.Errorf("%w: hash source: %v", ErrV4ExpertRuntime, err)
	}
	return &v4ExpertRuntime{
		cfg: cfg, experts: experts, hashes: hashes,
		ring:      newPagedRing(be, ringByteCap),
		stats:     v4ExpertRuntimeStats{RingBudget: ringByteCap, WorldSize: placement.WorldSize, Rank: placement.Rank},
		placement: placement,
	}, nil
}

func (r *v4ExpertRuntime) Close() error {
	if r == nil || r.closed {
		return nil
	}
	r.closed = true
	if r.ring != nil {
		r.ring.freeAll()
		r.stats.ResidentBytes = 0
	}
	var errs []error
	if err := r.experts.Close(); err != nil {
		errs = append(errs, err)
	}
	if err := r.hashes.Close(); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

func (r *v4ExpertRuntime) forwardScored(layer int, logits, correctionBias []float32, x compute.Tensor) ([]float32, error) {
	if err := r.ready(); err != nil {
		return nil, err
	}
	if layer < v4HashLayers || layer >= r.cfg.NumLayers {
		return nil, fmt.Errorf("%w: scored layer %d outside [%d,%d)", ErrV4ExpertRuntime, layer, v4HashLayers, r.cfg.NumLayers)
	}
	if len(logits) != r.cfg.NumExperts || (len(correctionBias) != 0 && len(correctionBias) != len(logits)) {
		return nil, fmt.Errorf("%w: scored router widths logits=%d bias=%d", ErrV4ExpertRuntime, len(logits), len(correctionBias))
	}
	picks, err := v4ScoredRoute(logits, correctionBias, r.cfg.NumExpertsPerTok, float32(r.cfg.RoutedScalingFactor))
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrV4ExpertRuntime, err)
	}
	return r.forwardSelected(layer, picks, x)
}

func (r *v4ExpertRuntime) forwardHash(layer, tokenID int, logits []float32, x compute.Tensor) ([]float32, error) {
	if err := r.ready(); err != nil {
		return nil, err
	}
	if layer < 0 || layer >= v4HashLayers {
		return nil, fmt.Errorf("%w: hash layer %d outside [0,%d)", ErrV4ExpertRuntime, layer, v4HashLayers)
	}
	if len(logits) != r.cfg.NumExperts {
		return nil, fmt.Errorf("%w: hash router logits width %d, want %d", ErrV4ExpertRuntime, len(logits), r.cfg.NumExperts)
	}
	ids, err := r.hashes.lookup(layer, tokenID)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrV4ExpertRuntime, err)
	}
	picks, err := v4HashRoute(logits, ids, float32(r.cfg.RoutedScalingFactor))
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrV4ExpertRuntime, err)
	}
	return r.forwardSelected(layer, picks, x)
}

func (r *v4ExpertRuntime) forwardSelected(layer int, picks []routePick, x compute.Tensor) ([]float32, error) {
	dispatch, err := r.placement.Dispatch(picks)
	if err != nil {
		return nil, fmt.Errorf("%w: dispatch: %v", ErrV4ExpertRuntime, err)
	}
	local := len(dispatch[r.placement.Rank])
	r.stats.LocalSelected += int64(local)
	r.stats.RemoteSelected += int64(len(picks) - local)
	if r.placement.WorldSize > 1 && local != len(picks) {
		if r.transport == nil {
			return nil, fmt.Errorf("%w: placement rank %d/%d selected %d remote experts; transport is not configured", ErrV4ExpertRuntime, r.placement.Rank, r.placement.WorldSize, len(picks)-local)
		}
		r.stats.TransportDispatches++
		output, partials, err := r.transport.Forward(dispatch, func(rankPicks []routePick) ([]float32, error) {
			return r.forwardLocalSelected(layer, rankPicks, x)
		})
		r.stats.TransportPartials += int64(partials)
		if err != nil {
			return nil, fmt.Errorf("%w: transport: %v", ErrV4ExpertRuntime, err)
		}
		return output, nil
	}
	return r.forwardLocalSelected(layer, picks, x)
}

func (r *v4ExpertRuntime) forwardLocalSelected(layer int, picks []routePick, x compute.Tensor) ([]float32, error) {
	selected := make([]int, len(picks))
	routes := make([]v4RoutedExpert, len(picks))
	for i, pick := range picks {
		selected[i] = pick.expert
		routes[i] = v4RoutedExpert{Expert: pick.expert, Weight: pick.weight}
	}
	plan, err := r.experts.planV4ExpertBatch(layer, selected, r.ring.budget())
	if err != nil {
		return nil, fmt.Errorf("%w: plan: %v", ErrV4ExpertRuntime, err)
	}
	stager, err := newV4ShardedExpertQuantStager(r.experts, r.ring, plan)
	if err != nil {
		return nil, fmt.Errorf("%w: stage: %v", ErrV4ExpertRuntime, err)
	}
	out, err := composeV4RoutedExperts(layer, routes, x, float32(r.cfg.SwigluLimit), stager)
	r.record(stager.Stats())
	if err != nil {
		return nil, fmt.Errorf("%w: compose: %v", ErrV4ExpertRuntime, err)
	}
	return out, nil
}

func (r *v4ExpertRuntime) ready() error {
	if r == nil || r.closed || r.experts == nil || r.hashes == nil || r.ring == nil {
		return fmt.Errorf("%w: closed or uninitialized", ErrV4ExpertRuntime)
	}
	return nil
}

func (r *v4ExpertRuntime) record(s v4ExpertStageStats) {
	r.stats.SourceReads += s.SourceReads
	r.stats.SourceBytes += s.SourceBytes
	r.stats.PageIns += s.PageIn
	r.stats.Hits += s.Hits
	r.stats.Evictions += s.Evictions
	if s.PeakResidentBytes > r.stats.PeakResident {
		r.stats.PeakResident = s.PeakResidentBytes
	}
	r.stats.ResidentBytes = r.ring.used()
	r.stats.ExpertOpenCount = r.experts.openCount
	r.stats.ExpertReadCount = r.experts.readCount
	r.stats.HashOpenCount = r.hashes.openCount
	r.stats.HashReadCount = r.hashes.readCount
	r.stats.HashReadBytes = r.hashes.readBytes
}

func (r *v4ExpertRuntime) Stats() v4ExpertRuntimeStats {
	if r == nil {
		return v4ExpertRuntimeStats{}
	}
	s := r.stats
	if r.experts != nil {
		s.ExpertOpenCount, s.ExpertReadCount = r.experts.openCount, r.experts.readCount
	}
	if r.hashes != nil {
		s.HashOpenCount, s.HashReadCount, s.HashReadBytes = r.hashes.openCount, r.hashes.readCount, r.hashes.readBytes
	}
	if r.ring != nil {
		s.ResidentBytes = r.ring.used()
	}
	return s
}
