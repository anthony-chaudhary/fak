package model

import (
	"errors"
	"fmt"
	"os"
	"strconv"
)

var ErrV4ExpertPlacement = errors.New("model: DeepSeek V4 expert placement refused")

const (
	v4ExpertWorldEnv = "FAK_V4_EXPERT_WORLD_SIZE"
	v4ExpertRankEnv  = "FAK_V4_EXPERT_RANK"
)

// V4ExpertPlacement deterministically partitions the pinned V4 routed-expert
// namespace into contiguous, equally-sized rank-local ranges.
type V4ExpertPlacement struct {
	WorldSize int `json:"world_size"`
	Rank      int `json:"rank"`
	Experts   int `json:"experts"`
}

// V4ExpertDispatch retains a route's original composition position and weight
// while naming the rank that owns its expert payload.
type V4ExpertDispatch struct {
	Rank     int     `json:"rank"`
	Position int     `json:"position"`
	Expert   int     `json:"expert"`
	Weight   float32 `json:"weight"`
}

func NewV4ExpertPlacement(cfg Config, worldSize, rank int) (V4ExpertPlacement, error) {
	if err := AdmitDeepSeekV4Config(cfg); err != nil {
		return V4ExpertPlacement{}, fmt.Errorf("%w: config: %v", ErrV4ExpertPlacement, err)
	}
	if worldSize <= 0 || worldSize > cfg.NumExperts || cfg.NumExperts%worldSize != 0 {
		return V4ExpertPlacement{}, fmt.Errorf("%w: world size %d must divide %d experts", ErrV4ExpertPlacement, worldSize, cfg.NumExperts)
	}
	if rank < 0 || rank >= worldSize {
		return V4ExpertPlacement{}, fmt.Errorf("%w: rank %d outside [0,%d)", ErrV4ExpertPlacement, rank, worldSize)
	}
	return V4ExpertPlacement{WorldSize: worldSize, Rank: rank, Experts: cfg.NumExperts}, nil
}

func v4ExpertPlacementFromEnv(cfg Config) (V4ExpertPlacement, error) {
	world, rank := 1, 0
	var err error
	if raw := os.Getenv(v4ExpertWorldEnv); raw != "" {
		world, err = strconv.Atoi(raw)
		if err != nil {
			return V4ExpertPlacement{}, fmt.Errorf("%w: invalid %s=%q", ErrV4ExpertPlacement, v4ExpertWorldEnv, raw)
		}
	}
	if raw := os.Getenv(v4ExpertRankEnv); raw != "" {
		rank, err = strconv.Atoi(raw)
		if err != nil {
			return V4ExpertPlacement{}, fmt.Errorf("%w: invalid %s=%q", ErrV4ExpertPlacement, v4ExpertRankEnv, raw)
		}
	}
	return NewV4ExpertPlacement(cfg, world, rank)
}

func (p V4ExpertPlacement) Owner(expert int) (int, error) {
	if p.WorldSize <= 0 || p.Experts <= 0 || p.Experts%p.WorldSize != 0 || expert < 0 || expert >= p.Experts {
		return 0, fmt.Errorf("%w: invalid placement or expert %d", ErrV4ExpertPlacement, expert)
	}
	return expert / (p.Experts / p.WorldSize), nil
}

func (p V4ExpertPlacement) Dispatch(picks []routePick) (map[int][]V4ExpertDispatch, error) {
	out := make(map[int][]V4ExpertDispatch)
	for position, pick := range picks {
		owner, err := p.Owner(pick.expert)
		if err != nil {
			return nil, err
		}
		out[owner] = append(out[owner], V4ExpertDispatch{Rank: owner, Position: position, Expert: pick.expert, Weight: pick.weight})
	}
	return out, nil
}
