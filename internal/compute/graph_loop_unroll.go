package compute

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

// GraphIterationOperation is an operation in an iteration body.
type GraphIterationOperation struct {
	ID        string   `json:"id"`
	Op        string   `json:"op"`
	Value     float64  `json:"value,omitempty"`
	Inputs    []string `json:"inputs,omitempty"`
	EffectTag string   `json:"effect_tag,omitempty"`
}

// GraphIterationBlock represents a structured bounded iteration in the compute IR.
type GraphIterationBlock struct {
	ID         string                    `json:"id"`
	Dynamic    bool                      `json:"dynamic,omitempty"`
	LowerBound int                       `json:"lower_bound"`
	UpperBound int                       `json:"upper_bound"`
	Step       int                       `json:"step"`
	Inits      []string                  `json:"inits"`
	Body       []GraphIterationOperation `json:"body"`
	Yields     []string                  `json:"yields"`
	Outputs    []string                  `json:"outputs"`
}

// GraphIterationUnrollConfig controls unrolling bounds and expansion limits.
type GraphIterationUnrollConfig struct {
	Factor     int `json:"factor"`
	NodeBudget int `json:"node_budget"`
}

// GraphIterationUnrollReceipt records deterministic evidence of unrolling.
type GraphIterationUnrollReceipt struct {
	TripCount        int    `json:"trip_count"`
	UnrollFactor     int    `json:"unroll_factor"`
	MainIterations   int    `json:"main_iterations"`
	TailIterations   int    `json:"tail_iterations"`
	TotalExpanded    int    `json:"total_expanded"`
	BudgetHit        bool   `json:"budget_hit,omitempty"`
	RejectedReason   string `json:"rejected_reason,omitempty"`
	FinalGraphDigest string `json:"final_graph_digest,omitempty"`
}

// UnrolledIterationSequence is the unrolled linear sequence of operations and outputs.
type UnrolledIterationSequence struct {
	Operations   []GraphIterationOperation `json:"operations"`
	FinalOutputs []string                  `json:"final_outputs"`
}

// UnrollStaticGraphIteration unrolls a statically bounded iteration under an explicit budget.
// Dynamic-bound iterations and expansions exceeding NodeBudget are rejected fail-closed.
func UnrollStaticGraphIteration(block GraphIterationBlock, cfg GraphIterationUnrollConfig) (UnrolledIterationSequence, GraphIterationUnrollReceipt, error) {
	receipt := GraphIterationUnrollReceipt{
		UnrollFactor: cfg.Factor,
	}

	if cfg.Factor <= 0 {
		return UnrolledIterationSequence{}, receipt, fmt.Errorf("unroll factor must be positive")
	}
	if cfg.NodeBudget <= 0 {
		return UnrolledIterationSequence{}, receipt, fmt.Errorf("node budget must be positive")
	}
	if block.Step <= 0 {
		return UnrolledIterationSequence{}, receipt, fmt.Errorf("step must be positive")
	}
	if len(block.Inits) != len(block.Yields) {
		return UnrolledIterationSequence{}, receipt, fmt.Errorf("mismatched inits (%d) and yields (%d)", len(block.Inits), len(block.Yields))
	}

	// Dynamic bounds rejection
	if block.Dynamic {
		receipt.RejectedReason = "dynamic-bound iteration cannot be unrolled under static contract"
		return fallbackUnrolledSequence(block), receipt, nil
	}

	if block.UpperBound < block.LowerBound {
		return UnrolledIterationSequence{}, receipt, fmt.Errorf("upper bound (%d) < lower bound (%d)", block.UpperBound, block.LowerBound)
	}

	diff := block.UpperBound - block.LowerBound
	tripCount := (diff + block.Step - 1) / block.Step
	receipt.TripCount = tripCount

	if tripCount == 0 {
		res := UnrolledIterationSequence{
			Operations:   nil,
			FinalOutputs: append([]string(nil), block.Inits...),
		}
		receipt.MainIterations = 0
		receipt.TailIterations = 0
		receipt.TotalExpanded = 0
		receipt.FinalGraphDigest = computeSequenceDigest(res)
		return res, receipt, nil
	}

	mainIters := (tripCount / cfg.Factor) * cfg.Factor
	tailIters := tripCount % cfg.Factor
	receipt.MainIterations = mainIters
	receipt.TailIterations = tailIters

	totalNodes := tripCount * len(block.Body)
	receipt.TotalExpanded = totalNodes

	// Node budget check
	if totalNodes > cfg.NodeBudget {
		receipt.BudgetHit = true
		receipt.RejectedReason = fmt.Sprintf("expansion of %d nodes exceeds declared budget %d", totalNodes, cfg.NodeBudget)
		return fallbackUnrolledSequence(block), receipt, nil
	}

	var ops []GraphIterationOperation
	currentCarried := append([]string(nil), block.Inits...)

	emitIteration := func(iterIndex int) {
		valMap := make(map[string]string)
		for idx, initName := range block.Inits {
			valMap[initName] = currentCarried[idx]
		}

		for _, op := range block.Body {
			newID := fmt.Sprintf("%s$unroll_%d", op.ID, iterIndex)
			valMap[op.ID] = newID

			newInputs := make([]string, len(op.Inputs))
			for inIdx, inName := range op.Inputs {
				if mapped, ok := valMap[inName]; ok {
					newInputs[inIdx] = mapped
				} else {
					newInputs[inIdx] = inName
				}
			}

			emittedOp := GraphIterationOperation{
				ID:        newID,
				Op:        op.Op,
				Value:     op.Value,
				Inputs:    newInputs,
				EffectTag: op.EffectTag,
			}
			ops = append(ops, emittedOp)
		}

		nextCarried := make([]string, len(block.Yields))
		for idx, yieldName := range block.Yields {
			if mapped, ok := valMap[yieldName]; ok {
				nextCarried[idx] = mapped
			} else {
				nextCarried[idx] = yieldName
			}
		}
		currentCarried = nextCarried
	}

	for i := 0; i < mainIters; i++ {
		emitIteration(i)
	}

	for i := mainIters; i < tripCount; i++ {
		emitIteration(i)
	}

	res := UnrolledIterationSequence{
		Operations:   ops,
		FinalOutputs: currentCarried,
	}
	receipt.FinalGraphDigest = computeSequenceDigest(res)

	return res, receipt, nil
}

func fallbackUnrolledSequence(block GraphIterationBlock) UnrolledIterationSequence {
	return UnrolledIterationSequence{
		Operations:   block.Body,
		FinalOutputs: block.Outputs,
	}
}

func computeSequenceDigest(s UnrolledIterationSequence) string {
	b, _ := json.Marshal(s)
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}
