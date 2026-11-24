package ggufload

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// Inkling2MetalTarget stages a causal block. Commit keeps the specified number
// of input rows, including the anchor. Zero discards the entire staged block.
type Inkling2MetalTarget interface {
	Prefill([]int32) ([]float32, error)
	BeginBlockIDs([]int32) ([]int32, error)
	Commit(int) error
}

// Inkling2MetalDraft uses the target's committed feature rows plus the anchor.
type Inkling2MetalDraft interface {
	Draft(anchor int32, depth int) ([]int32, error)
}

// Inkling2MetalTree stages parallel candidate paths and keeps the path to one leaf.
// Each row contains the token, parent row, and depth. Row zero is the anchor.
type Inkling2MetalTree interface {
	DraftTree(anchor int32, depth, width int) ([][3]int32, error)
	BeginTreeIDs(rows [][3]int32) ([]int32, error)
	CommitTree(leaf int) error
}

type inkling2FusedMetalTree interface {
	ProposeVerify(anchor int32, depth, width int) ([][3]int32, []int32, error)
}

// A committed tree returns a bit per retained input row. Its target IDs let
// the host check every retained ancestor before publishing tokens.
type inkling2CommittedMetalTree interface {
	ProposeVerifyCommit(anchor int32, depth, width, maximum int, eos int32) ([][3]int32, []int32, uint32, error)
}

type Inkling2MetalGreedyRequest struct {
	Prompt                            []int
	MaxTokens, Vocab, EOS             int
	VerifyRows, DraftDepth, TreeWidth int
}

func inkling2MetalTokenIDs(ids []int32, vocab int) ([]int, error) {
	out := make([]int, len(ids))
	for i, v := range ids {
		if v < 0 || int(v) >= vocab {
			return nil, fmt.Errorf("Inkling Metal: token %d outside %d-entry vocabulary", v, vocab)
		}
		out[i] = int(v)
	}
	return out, nil
}

// Inkling2MetalGreedy decides which draft tokens to expose and which causal
// rows survive. The target evaluates every proposal, including prompt lookup.
func Inkling2MetalGreedy(ctx context.Context, target Inkling2MetalTarget, draft Inkling2MetalDraft, req Inkling2MetalGreedyRequest) ([]int, Inkling2SpecStats, error) {
	var st Inkling2SpecStats
	if req.MaxTokens < 1 || req.Vocab < 1 || req.Vocab > math.MaxInt32 || req.VerifyRows < 1 || req.DraftDepth < 0 || len(req.Prompt) == 0 {
		return nil, st, errors.New("Inkling Metal: invalid request geometry")
	}
	if err := ctx.Err(); err != nil {
		return nil, st, err
	}
	input := make([]int32, len(req.Prompt))
	for i, v := range req.Prompt {
		if v < 0 || v >= req.Vocab {
			return nil, st, fmt.Errorf("Inkling Metal: invalid prompt token %d", v)
		}
		input[i] = int32(v)
	}
	start := time.Now()
	next, err := inkling2MetalFirstToken(target, input, req.Vocab)
	if err != nil {
		return nil, st, err
	}
	st.Prefill = time.Since(start)
	start = time.Now()
	history := slices.Clone(req.Prompt)
	var out []int
	lookup := model.NgramDrafter{Enabled: true, MinMatch: 3, MaxMatch: 64, MaxDraft: req.VerifyRows - 1}
	// Short matches compete with neural branches in the fused selector. A match
	// spanning a complete draft block can bypass it for a longer verified copy.
	_, fusedPolicy := target.(inkling2FusedMetalTree)
	if fusedPolicy && req.TreeWidth > 1 && draft != nil {
		lookup.MinMatch = max(lookup.MinMatch, req.DraftDepth+1)
	}
	for {
		if err := ctx.Err(); err != nil {
			return nil, st, err
		}
		if next == req.EOS {
			break
		}
		out = append(out, next)
		history = append(history, next)
		remaining := req.MaxTokens - len(out)
		if remaining == 0 {
			break
		}
		depth := min(req.VerifyRows-1, remaining-1)
		var proposal []int
		var treeRows [][3]int32
		var treeScores []int32
		var retained uint32
		var alreadyCommitted bool
		tree, treeOK := target.(Inkling2MetalTree)
		startDraft := time.Now()
		if depth > 0 {
			proposal = lookup.Draft(history)
			proposal = proposal[:min(depth, len(proposal))]
			neuralDepth := min(depth, req.DraftDepth)
			threshold := neuralDepth
			if fusedPolicy && req.TreeWidth > 1 {
				threshold = req.DraftDepth + 1
			}
			if draft != nil && neuralDepth > 0 && len(proposal) < threshold {
				if treeOK && req.TreeWidth > 1 {
					if fused, ok := tree.(inkling2CommittedMetalTree); ok {
						eos := int32(-1)
						if req.EOS >= 0 && req.EOS < req.Vocab {
							eos = int32(req.EOS)
						}
						treeRows, treeScores, retained, err = fused.ProposeVerifyCommit(int32(next), neuralDepth, req.TreeWidth, neuralDepth+1, eos)
						alreadyCommitted = len(treeRows) > 0
						if err == nil && (len(treeScores) != len(treeRows) || len(treeRows) > 32 || !alreadyCommitted && retained != 0) {
							err = errors.New("Inkling Metal: invalid committed tree result")
						}
					} else if fused, ok := tree.(inkling2FusedMetalTree); ok {
						treeRows, treeScores, err = fused.ProposeVerify(int32(next), neuralDepth, req.TreeWidth)
					} else {
						treeRows, err = tree.DraftTree(int32(next), neuralDepth, req.TreeWidth)
					}
				} else {
					var ids []int32
					ids, err = draft.Draft(int32(next), neuralDepth)
					if err == nil && len(ids) > neuralDepth {
						err = errors.New("Inkling Metal: draft exceeded requested depth")
					}
					if err == nil && len(ids) > 0 {
						proposal, err = inkling2MetalTokenIDs(ids, req.Vocab)
					}
				}
				if err != nil {
					return nil, st, err
				}
			}
		}
		st.Draft += time.Since(startDraft)
		startVerify := time.Now()
		var validated []int
		commit := target.Commit
		proposed := len(proposal)
		if len(treeRows) > 1 || len(treeScores) > 0 {
			var path []int
			staged := tree
			if alreadyCommitted {
				staged = nil
			}
			proposal, validated, path, err = inkling2VerifyMetalTree(ctx, staged, treeRows, treeScores, next, depth, req.Vocab)
			if alreadyCommitted {
				commit = func(n int) error {
					var want uint32
					for _, row := range path[:n] {
						want |= uint32(1) << uint(row)
					}
					if retained != want {
						return errors.New("Inkling Metal: retained rows disagree with verified target path")
					}
					return nil
				}
			} else {
				commit = func(n int) error { return tree.CommitTree(path[n-1]) }
			}
			proposed = len(treeRows) - 1
		} else {
			fed := make([]int32, len(proposal)+1)
			fed[0] = int32(next)
			for i, id := range proposal {
				fed[i+1] = int32(id)
			}
			var ids []int32
			ids, err = target.BeginBlockIDs(fed)
			if err == nil {
				validated, err = inkling2MetalTokenIDs(ids, req.Vocab)
				if err == nil && len(validated) != len(fed) {
					err = fmt.Errorf("Inkling Metal: verified %d of %d rows", len(validated), len(fed))
				}
				if err == nil {
					err = ctx.Err()
				}
				if err != nil {
					err = errors.Join(err, target.Commit(0))
				}
			}
		}
		if err != nil {
			return nil, st, err
		}
		accepted := polymodel.AcceptGreedy(proposal, validated).Accepted
		stop := slices.Index(proposal[:accepted], req.EOS)
		if stop >= 0 {
			accepted = stop
		}
		if err = commit(1 + accepted); err != nil {
			return nil, st, err
		}
		st.Verify += time.Since(startVerify)
		if proposed == 0 {
			st.PlainSteps++
		} else {
			st.Verifies++
			st.Drafted += proposed
			st.Accepted += accepted
		}
		out = append(out, proposal[:accepted]...)
		history = append(history, proposal[:accepted]...)
		next = validated[accepted]
		if stop >= 0 {
			break
		}
	}
	st.Decode = time.Since(start)
	st.DecodeSteps = max(0, len(out)-1)
	if next == req.EOS && len(out) > 0 {
		st.DecodeSteps++
	}
	return out, st, nil
}

// Return the verified path as a linear chain so EOS and output limits follow
// the same commit rule for neural trees and prompt-lookup chains.
func inkling2VerifyMetalTree(ctx context.Context, t Inkling2MetalTree, rows [][3]int32, scores []int32, anchor, depth, vocab int) (tokens, argmax, path []int, err error) {
	discard := func(e error) error {
		if t != nil && len(scores) > 0 {
			return errors.Join(e, t.CommitTree(-1))
		}
		return e
	}
	if len(rows) == 0 || rows[0] != [3]int32{int32(anchor), -1, 0} {
		return nil, nil, nil, discard(errors.New("Inkling Metal: invalid root"))
	}
	tree := polymodel.SpecTree{Nodes: make([]polymodel.TreeNode, len(rows))}
	for i, r := range rows {
		if r[0] < 0 || int(r[0]) >= vocab {
			return nil, nil, nil, discard(errors.New("Inkling Metal: invalid proposal token"))
		}
		tree.Nodes[i].Token = int(r[0])
		if i == 0 {
			continue
		}
		parent := int(r[1])
		if parent < 0 || parent >= i || r[2] != rows[parent][2]+1 || int(r[2]) > depth {
			return nil, nil, nil, discard(errors.New("Inkling Metal: invalid proposal ancestry"))
		}
		tree.Nodes[parent].Children = append(tree.Nodes[parent].Children, i)
	}
	ids := scores
	if len(ids) == 0 {
		ids, err = t.BeginTreeIDs(rows)
		if err != nil {
			return nil, nil, nil, err
		}
	}
	valid, err := inkling2MetalTokenIDs(ids, vocab)
	if err == nil && len(valid) != len(rows) {
		err = errors.New("Inkling Metal: missing tree scores")
	}
	if err == nil {
		err = ctx.Err()
	}
	if err != nil {
		if t != nil {
			err = errors.Join(err, t.CommitTree(-1))
		}
		return nil, nil, nil, err
	}
	for i, id := range valid {
		tree.Nodes[i].TargetArgmax = id
	}
	path = append([]int{0}, polymodel.AcceptTree(tree).Path...)
	for i, row := range path {
		if i > 0 {
			tokens = append(tokens, tree.Nodes[row].Token)
		}
		argmax = append(argmax, valid[row])
	}
	return tokens, argmax, path, nil
}

func inkling2MetalFirstToken(target Inkling2MetalTarget, input []int32, vocab int) (int, error) {
	if t, ok := target.(interface{ PrefillID([]int32) (int32, error) }); ok {
		id, err := t.PrefillID(input)
		if err != nil {
			return 0, err
		}
		if id < 0 || int(id) >= vocab {
			return 0, fmt.Errorf("Inkling Metal: prefill selected %d outside vocabulary", id)
		}
		return int(id), nil
	}
	logits, err := target.Prefill(input)
	if err != nil {
		return 0, err
	}
	if len(logits) < vocab {
		return 0, fmt.Errorf("Inkling Metal: got %d first-row scores for %d tokens", len(logits), vocab)
	}
	next := 0
	for i, v := range logits[:vocab] {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return 0, errors.New("Inkling Metal: non-finite first-row logits")
		}
		if v > logits[next] {
			next = i
		}
	}
	return next, nil
}
