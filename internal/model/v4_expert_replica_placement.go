package model

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

var ErrV4ExpertReplicaPlacement = errors.New("model: DeepSeek V4 expert replica placement refused")

// V4ExpertReplicaPlacement is a pure logical-to-physical expert plan. Consecutive
// Capacity slots belong to one rank. It carries no migration or runtime policy.
type V4ExpertReplicaPlacement struct {
	PhysicalToLogical []int   `json:"physical_to_logical"`
	LogicalToPhysical [][]int `json:"logical_to_physical"`
	LogicalCount      []int   `json:"logical_count"`
	Capacity          int     `json:"capacity"`
}

// NewV4ExpertReplicaPlacement adapts DeepInfer's EPLB replica selection and
// physical placement at deepseek-ai/DeepInfer revision
// 9bd9cf243fc53ef9afbf8f04d391d989117d6ab9. History is [step][logical expert].
func NewV4ExpertReplicaPlacement(history [][]int64, physicalCount, capacity int) (V4ExpertReplicaPlacement, error) {
	if len(history) == 0 || len(history[0]) == 0 || physicalCount < len(history[0]) || capacity <= 0 || physicalCount%capacity != 0 {
		return V4ExpertReplicaPlacement{}, ErrV4ExpertReplicaPlacement
	}
	logical := len(history[0])
	for _, step := range history {
		if len(step) != logical {
			return V4ExpertReplicaPlacement{}, ErrV4ExpertReplicaPlacement
		}
		for _, n := range step {
			if n < 0 {
				return V4ExpertReplicaPlacement{}, ErrV4ExpertReplicaPlacement
			}
		}
	}

	physical := make([]int, logical, physicalCount)
	counts := make([]int, logical)
	for expert := 0; expert < logical; expert++ {
		physical[expert] = expert
		counts[expert] = 1
	}
	for len(physical) < physicalCount {
		expert := v4ExpertReplicaChoice(history, counts)
		physical = append(physical, expert)
		counts[expert]++
	}

	order := v4ExpertPhysicalOrder(history, physical, counts, capacity)
	placed := make([]int, len(order))
	for slot, oldSlot := range order {
		placed[slot] = physical[oldSlot]
	}
	inverse := make([][]int, logical)
	for slot, expert := range placed {
		inverse[expert] = append(inverse[expert], slot)
	}
	return V4ExpertReplicaPlacement{
		PhysicalToLogical: placed,
		LogicalToPhysical: inverse,
		LogicalCount:      counts,
		Capacity:          capacity,
	}, nil
}

// OwnerRanks returns the sorted, deduplicated ranks holding logicalExpert.
func (p V4ExpertReplicaPlacement) OwnerRanks(logicalExpert int) ([]int, error) {
	if err := p.valid(); err != nil || logicalExpert < 0 || logicalExpert >= len(p.LogicalToPhysical) {
		return nil, fmt.Errorf("%w: logical expert %d", ErrV4ExpertReplicaPlacement, logicalExpert)
	}
	ranks := make([]int, 0, len(p.LogicalToPhysical[logicalExpert]))
	last := -1
	for _, slot := range p.LogicalToPhysical[logicalExpert] {
		rank := slot / p.Capacity
		if rank != last {
			ranks = append(ranks, rank)
			last = rank
		}
	}
	return ranks, nil
}

// RankResidentMask reports which logical experts have at least one slot on rank.
func (p V4ExpertReplicaPlacement) RankResidentMask(rank int) ([]bool, error) {
	if err := p.valid(); err != nil || rank < 0 || rank >= len(p.PhysicalToLogical)/p.Capacity {
		return nil, fmt.Errorf("%w: rank %d", ErrV4ExpertReplicaPlacement, rank)
	}
	mask := make([]bool, len(p.LogicalToPhysical))
	start := rank * p.Capacity
	for _, expert := range p.PhysicalToLogical[start : start+p.Capacity] {
		mask[expert] = true
	}
	return mask, nil
}

// ScoreRank scores one rank's resident logical experts against an exact-width
// histogram by delegating to the existing placement gauge.
func (p V4ExpertReplicaPlacement) ScoreRank(freq []int64, rank, k int) (ExpertPlacementScore, error) {
	if len(freq) != len(p.LogicalToPhysical) {
		return ExpertPlacementScore{}, fmt.Errorf("%w: histogram width %d, want %d", ErrV4ExpertReplicaPlacement, len(freq), len(p.LogicalToPhysical))
	}
	mask, err := p.RankResidentMask(rank)
	if err != nil {
		return ExpertPlacementScore{}, err
	}
	return ScoreExpertPlacement(freq, mask, k), nil
}

func (p V4ExpertReplicaPlacement) valid() error {
	if p.Capacity <= 0 || len(p.PhysicalToLogical) == 0 || len(p.PhysicalToLogical)%p.Capacity != 0 || len(p.LogicalToPhysical) == 0 || len(p.LogicalCount) != len(p.LogicalToPhysical) {
		return ErrV4ExpertReplicaPlacement
	}
	seen := make([]int, len(p.LogicalToPhysical))
	for slot, expert := range p.PhysicalToLogical {
		if expert < 0 || expert >= len(seen) {
			return ErrV4ExpertReplicaPlacement
		}
		seen[expert]++
		slots := p.LogicalToPhysical[expert]
		i := sort.SearchInts(slots, slot)
		if i == len(slots) || slots[i] != slot {
			return ErrV4ExpertReplicaPlacement
		}
	}
	for expert := range seen {
		if seen[expert] == 0 || seen[expert] != p.LogicalCount[expert] || seen[expert] != len(p.LogicalToPhysical[expert]) {
			return ErrV4ExpertReplicaPlacement
		}
	}
	return nil
}

// v4ExpertReplicaChoice preserves policy.py's float32 accumulation and its
// logical-ID*1e-4 tie perturbation.
func v4ExpertReplicaChoice(history [][]int64, counts []int) int {
	scores := make([]float32, len(counts))
	objectives := make([]float32, len(counts))
	for _, step := range history {
		maxExpert := 0
		for expert, n := range step {
			scores[expert] = (float32(n) + float32(expert)*float32(1e-4)) / float32(counts[expert])
			if scores[expert] > scores[maxExpert] {
				maxExpert = expert
			}
		}
		peak := scores[maxExpert]
		for candidate := range objectives {
			candidatePeak := peak
			if candidate == maxExpert {
				scores[maxExpert] = float32(step[maxExpert]) / float32(counts[maxExpert]+1)
				candidatePeak = scores[0]
				for expert := 1; expert < len(scores); expert++ {
					if scores[expert] > candidatePeak {
						candidatePeak = scores[expert]
					}
				}
			}
			objectives[candidate] += candidatePeak
		}
	}
	best := 0
	for candidate := 1; candidate < len(objectives); candidate++ {
		if objectives[candidate] < objectives[best] {
			best = candidate
		}
	}
	return best
}

func v4ExpertPhysicalOrder(history [][]int64, physical, counts []int, capacity int) []int {
	type job struct {
		slot int
		load float32
	}
	jobs := make([]job, len(physical))
	for slot, expert := range physical {
		jobs[slot].slot = slot
		for _, step := range history {
			jobs[slot].load += float32(step[expert]) / float32(counts[expert])
		}
	}
	sort.SliceStable(jobs, func(i, j int) bool { return jobs[i].load > jobs[j].load })

	ranks := len(physical) / capacity
	loads := make([][]float32, ranks)
	occupancy := make([]int, ranks)
	peaks := make([]float32, len(history))
	for rank := range loads {
		loads[rank] = make([]float32, len(history))
	}
	order := make([]int, len(physical))
	for _, job := range jobs {
		expert := physical[job.slot]
		bestRank, bestScore := -1, float32(math.Inf(1))
		for rank := 0; rank < ranks; rank++ {
			if occupancy[rank] == capacity {
				continue
			}
			var score float32
			for step := range history {
				candidate := loads[rank][step] + float32(history[step][expert])/float32(counts[expert])
				if candidate < peaks[step] {
					candidate = peaks[step]
				}
				score += candidate
			}
			if score < bestScore {
				bestRank, bestScore = rank, score
			}
		}
		order[bestRank*capacity+occupancy[bestRank]] = job.slot
		occupancy[bestRank]++
		for step := range history {
			loads[bestRank][step] += float32(history[step][expert]) / float32(counts[expert])
			if loads[bestRank][step] > peaks[step] {
				peaks[step] = loads[bestRank][step]
			}
		}
	}
	return order
}
