package qwen38quantrun

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
)

// StrixHeadToHeadConfig declares one interleaved Strix Halo candidate/reference
// comparison matrix. It is deliberately device-free: the caller supplies the
// exact frozen prompt packet, the concurrency cells, and an ArmRunner.
type StrixHeadToHeadConfig struct {
	// Packet is the frozen prompt-token packet v2 shared by both arms. The
	// scheduler refuses an unfrozen or legacy packet before any arm call.
	Packet PromptTokenPacket
	// LogitTolerance is the scoreboard's selected-token-logprob tolerance.
	LogitTolerance float64
	// ConcurrencyCells is the ordered set of measured concurrency cells. Empty
	// defaults to {1, 4, 8}. Cells must be positive and strictly increasing.
	ConcurrencyCells []int
	// WarmupsPerArm is the number of discarded warmup runs per arm per cell.
	// Zero defaults to 3; negative is refused.
	WarmupsPerArm int
	// MeasuredPairs is the number of measured candidate/reference pairs per
	// cell. Zero defaults to 5; fewer than five is refused.
	MeasuredPairs int
	// LeasePath overrides the canonical GPU lease lockfile. Empty uses
	// gpulease.DefaultPath().
	LeasePath string
	// Scoreboard is the per-cell evaluator. Nil uses BuildAMDScoreboard. It is
	// injectable only so a device-free test can observe the exact per-cell
	// input; production callers leave it nil.
	Scoreboard func(AMDScoreboardInput) AMDScoreboardReport
}

// StrixArmCall identifies one ArmRunner invocation. Role is "candidate" or
// "reference"; Warmup marks a discarded warmup run.
type StrixArmCall struct {
	Role        string
	Concurrency int
	Warmup      bool
	PairIndex   int
	Packet      PromptTokenPacket
}

// StrixArmRunner executes one measured run of one arm at one concurrency and
// returns that run's arm receipt. The scheduler never manufactures concurrency
// by scaling a single request: the runner is responsible for returning accepted
// output tokens divided by the shared first-admission-to-last-completion
// interval for exactly the cell's concurrency requests. The returned receipt
// must carry at least one trial; the scheduler accumulates trials across the
// cell's measured pairs into the cell's scoreboard input.
type StrixArmRunner func(ctx context.Context, call StrixArmCall) (AMDArmReceipt, error)

// StrixHeadToHeadCell is one completed concurrency cell with the evaluated
// scoreboard report for its accumulated measured pairs.
type StrixHeadToHeadCell struct {
	Concurrency int                 `json:"concurrency"`
	Candidate   AMDArmReceipt       `json:"candidate"`
	Reference   AMDArmReceipt       `json:"reference"`
	Report      AMDScoreboardReport `json:"report"`
}

// StrixHeadToHeadResult is the full interleaved matrix. It grants no physical
// comparison credit on its own: credit remains the per-cell scoreboard verdict,
// which is hard-false until authoritative evidence exists.
type StrixHeadToHeadResult struct {
	Cells []StrixHeadToHeadCell `json:"cells"`
}

// ErrStrixHeadToHead is the sentinel wrapped by every scheduler refusal.
var ErrStrixHeadToHead = errors.New("qwen38quantrun: strix head-to-head refused")

func defaultStrixHeadToHeadCells() []int { return []int{1, 4, 8} }

// RunStrixHeadToHead holds one canonical GPU lease across the full predeclared
// matrix, drives both arms with the same frozen packet, alternates pair order
// (odd pairs candidate-then-reference, even pairs reference-then-candidate),
// and evaluates each cell with the AMD scoreboard immediately after its
// measured pairs complete. Any missing or failed arm output stops the matrix
// with no partial ratio or win credit.
func RunStrixHeadToHead(ctx context.Context, cfg StrixHeadToHeadConfig, run StrixArmRunner) (*StrixHeadToHeadResult, error) {
	if run == nil {
		return nil, fmt.Errorf("%w: arm runner is required", ErrStrixHeadToHead)
	}
	if err := VerifyPromptPacket(cfg.Packet); err != nil {
		return nil, fmt.Errorf("%w: prompt packet admission: %v", ErrStrixHeadToHead, err)
	}
	if cfg.Packet.Schema != PromptTokenPacketSchema {
		return nil, fmt.Errorf("%w: packet schema %q is not %q", ErrStrixHeadToHead, cfg.Packet.Schema, PromptTokenPacketSchema)
	}
	if !finitePositive(cfg.LogitTolerance) {
		return nil, fmt.Errorf("%w: logit tolerance must be finite and positive", ErrStrixHeadToHead)
	}
	cells := cfg.ConcurrencyCells
	if len(cells) == 0 {
		cells = defaultStrixHeadToHeadCells()
	}
	for i, c := range cells {
		if c <= 0 {
			return nil, fmt.Errorf("%w: concurrency cell %d is not positive", ErrStrixHeadToHead, c)
		}
		if i > 0 && c <= cells[i-1] {
			return nil, fmt.Errorf("%w: concurrency cells must strictly increase: %v", ErrStrixHeadToHead, cells)
		}
	}
	warmups := cfg.WarmupsPerArm
	if warmups == 0 {
		warmups = StrixComparisonWarmups
	}
	if warmups < 0 {
		return nil, fmt.Errorf("%w: warmups per arm must not be negative", ErrStrixHeadToHead)
	}
	pairs := cfg.MeasuredPairs
	if pairs == 0 {
		pairs = StrixComparisonMinimumMeasuredPairs
	}
	if pairs < StrixComparisonMinimumMeasuredPairs {
		return nil, fmt.Errorf("%w: measured pairs %d is below the required %d", ErrStrixHeadToHead, pairs, StrixComparisonMinimumMeasuredPairs)
	}
	scoreboard := cfg.Scoreboard
	if scoreboard == nil {
		scoreboard = BuildAMDScoreboard
	}

	lease, err := gpulease.Acquire(gpulease.Options{Path: cfg.LeasePath, NoWait: true})
	if err != nil {
		return nil, fmt.Errorf("%w: canonical GPU lease: %v", ErrStrixHeadToHead, err)
	}
	defer lease.Release()

	result := &StrixHeadToHeadResult{}
	for _, c := range cells {
		cell, err := runStrixHeadToHeadCell(ctx, cfg, run, scoreboard, c, warmups, pairs)
		if err != nil {
			return nil, err
		}
		result.Cells = append(result.Cells, cell)
	}
	return result, nil
}

func runStrixHeadToHeadCell(
	ctx context.Context,
	cfg StrixHeadToHeadConfig,
	run StrixArmRunner,
	scoreboard func(AMDScoreboardInput) AMDScoreboardReport,
	concurrency, warmups, pairs int,
) (StrixHeadToHeadCell, error) {
	for i := 0; i < warmups; i++ {
		if err := ctx.Err(); err != nil {
			return StrixHeadToHeadCell{}, fmt.Errorf("%w: context cancelled during warmups: %v", ErrStrixHeadToHead, err)
		}
		for _, role := range []string{"candidate", "reference"} {
			if _, err := run(ctx, StrixArmCall{Role: role, Concurrency: concurrency, Warmup: true, PairIndex: -1, Packet: cfg.Packet}); err != nil {
				return StrixHeadToHeadCell{}, fmt.Errorf("%w: %s warmup %d at c=%d: %v", ErrStrixHeadToHead, role, i+1, concurrency, err)
			}
		}
	}

	var candidate, reference AMDArmReceipt
	for p := 0; p < pairs; p++ {
		if err := ctx.Err(); err != nil {
			return StrixHeadToHeadCell{}, fmt.Errorf("%w: context cancelled during measured pair %d: %v", ErrStrixHeadToHead, p+1, err)
		}
		order := []string{"candidate", "reference"}
		if p%2 == 1 {
			order = []string{"reference", "candidate"}
		}
		_, _, _, candidateSequence, referenceSequence := strixComparisonPairOrder(p)
		for _, role := range order {
			receipt, err := run(ctx, StrixArmCall{Role: role, Concurrency: concurrency, PairIndex: p, Packet: cfg.Packet})
			if err != nil {
				return StrixHeadToHeadCell{}, fmt.Errorf("%w: %s measured pair %d at c=%d: %v", ErrStrixHeadToHead, role, p+1, concurrency, err)
			}
			if len(receipt.Trials) == 0 {
				return StrixHeadToHeadCell{}, fmt.Errorf("%w: %s measured pair %d at c=%d returned no trial", ErrStrixHeadToHead, role, p+1, concurrency)
			}
			// The scheduler, not the runner, owns the canonical pair-order
			// identity: the scoreboard binds each arm's trial to the exact
			// alternating AB/BA sequence it predeclared.
			sequence := candidateSequence
			if role == "reference" {
				sequence = referenceSequence
			}
			receipt.Trials = stampStrixPairOrder(receipt.Trials, p+1, sequence)
			if role == "candidate" {
				candidate = mergeStrixArm(candidate, receipt)
			} else {
				reference = mergeStrixArm(reference, receipt)
			}
		}
	}

	in := AMDScoreboardInput{
		Schema:         AMDScoreboardInputSchema,
		LogitTolerance: cfg.LogitTolerance,
		Concurrency:    concurrency,
		Candidate:      candidate,
		Reference:      reference,
	}
	return StrixHeadToHeadCell{Concurrency: concurrency, Candidate: candidate, Reference: reference, Report: scoreboard(in)}, nil
}

// mergeStrixArm accumulates a fresh run's trials into the cell's arm receipt,
// adopting identity fields from the first run and appending trials in pair
// order. Identity drift between runs of the same cell is refused so a cell can
// never silently mix two different hardware, artifact, or packet identities.
func mergeStrixArm(acc, next AMDArmReceipt) AMDArmReceipt {
	if len(acc.Trials) == 0 && acc.Name == "" {
		adopted := next
		adopted.Trials = slices.Clone(next.Trials)
		return adopted
	}
	if acc.Name != next.Name || acc.Engine != next.Engine || acc.Backend != next.Backend ||
		acc.ArtifactSHA256 != next.ArtifactSHA256 || acc.PromptSHA256 != next.PromptSHA256 ||
		acc.PromptPacketDigest != next.PromptPacketDigest || acc.SoftwareRevision != next.SoftwareRevision ||
		acc.Hardware != next.Hardware || acc.ComparatorOnly != next.ComparatorOnly {
		panic("strix head-to-head: arm identity drift between measured runs")
	}
	acc.Trials = append(acc.Trials, next.Trials...)
	return acc
}

// stampStrixPairOrder rewrites the accumulated trial identity to the canonical
// repetition and alternating sequence the scoreboard requires. Trial observations
// themselves are preserved; only the pair-order identity is normalized.
func stampStrixPairOrder(trials []AMDScoreboardTrial, repetition, sequence int) []AMDScoreboardTrial {
	stamped := slices.Clone(trials)
	for i := range stamped {
		stamped[i].Repetition = repetition
		stamped[i].Sequence = sequence
	}
	return stamped
}
