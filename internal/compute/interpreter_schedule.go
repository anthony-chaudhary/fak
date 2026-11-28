// Adapted from HazyResearch/Megakernels (https://github.com/HazyResearch/Megakernels),
// Copyright (c) 2025 HazyResearch
// SPDX-License-Identifier: MIT
// See THIRD_PARTY_NOTICES.md ("HazyResearch Megakernels").

package compute

import (
	"errors"
	"fmt"
)

const (
	// InterpreterInstructionWidth is the fixed instruction descriptor word width
	// (32 int32 words = 128 bytes per instruction).
	InterpreterInstructionWidth = 32
	// InterpreterMaxInstructionArgs leaves 2 header words (opcode + arg count).
	InterpreterMaxInstructionArgs = InterpreterInstructionWidth - 2
	// InterpreterPipelineStages is the controller instruction ring depth.
	InterpreterPipelineStages = 2
	// InterpreterPipelineStageBits is the number of phase bits tracking ring/page reuse.
	InterpreterPipelineStageBits = 1
	// InterpreterDynamicSemaphores is the per-instruction dynamic semaphore cap.
	InterpreterDynamicSemaphores = 32
	// InterpreterPageSizeBytes is the size of one paged shared-memory pool (16 KiB).
	InterpreterPageSizeBytes = 16384
	// InterpreterNumPages is the fixed 13-page SMEM pool geometry.
	InterpreterNumPages = 13
	// InterpreterTotalPoolBytes is the aggregate paged SMEM pool footprint (208 KiB).
	InterpreterTotalPoolBytes = InterpreterNumPages * InterpreterPageSizeBytes
	// InterpreterTimingWidth is the timing slot count per instruction.
	InterpreterTimingWidth = 128
	// InterpreterOpNoOp is the reserved opcode for no-op barriers/padding.
	InterpreterOpNoOp int32 = 0
)

var (
	// ErrInterpreterInvalidSchedule reports a malformed schedule, instruction word, or pool release order.
	ErrInterpreterInvalidSchedule = errors.New("compute: invalid interpreter schedule")
	// ErrInterpreterDeadlock reports an unsatisfiable or cyclic global-memory counter dependency.
	ErrInterpreterDeadlock = errors.New("compute: interpreter schedule counter-sync deadlock")
)

// InterpreterBarrierWait specifies a global-memory counter threshold that must be
// met before the instruction's work runs.
type InterpreterBarrierWait struct {
	BarrierID int
	Threshold uint32
}

// InterpreterBarrierSignal specifies a global-memory counter increment emitted
// strictly after the instruction finishes writing its outputs.
type InterpreterBarrierSignal struct {
	BarrierID int
	Delta     uint32
}

// InterpreterPageReleaseOrder maps each next-instruction logical page ID (0..12)
// to the previous-instruction logical page slot whose physical page it reuses.
type InterpreterPageReleaseOrder [InterpreterNumPages]int

// InterpreterIdentityReleaseOrder returns the identity page mapping [0..12].
func InterpreterIdentityReleaseOrder() InterpreterPageReleaseOrder {
	var order InterpreterPageReleaseOrder
	for i := 0; i < InterpreterNumPages; i++ {
		order[i] = i
	}
	return order
}

// InterpreterMatvecReleaseOrder returns the 3-stage pipeline release permutation
// (unused pages first, then activation page 0, then weight stages) by iteration count.
func InterpreterMatvecReleaseOrder(iters int) InterpreterPageReleaseOrder {
	if iters <= 0 {
		return InterpreterIdentityReleaseOrder()
	}
	switch {
	case iters == 1:
		return InterpreterPageReleaseOrder{5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4}
	case iters == 2:
		return InterpreterPageReleaseOrder{9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8}
	case iters%3 == 1:
		return InterpreterPageReleaseOrder{0, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4}
	case iters%3 == 2:
		return InterpreterPageReleaseOrder{0, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8}
	default:
		return InterpreterIdentityReleaseOrder()
	}
}

// ValidateInterpreterReleaseOrder verifies that order is a bijection of [0..12].
func ValidateInterpreterReleaseOrder(order InterpreterPageReleaseOrder) error {
	var seen [InterpreterNumPages]bool
	for lid, prevLid := range order {
		if prevLid < 0 || prevLid >= InterpreterNumPages {
			return fmt.Errorf("%w: release_lid[%d]=%d out of [0,%d)",
				ErrInterpreterInvalidSchedule, lid, prevLid, InterpreterNumPages)
		}
		if seen[prevLid] {
			return fmt.Errorf("%w: duplicate release_lid target %d at logical page %d",
				ErrInterpreterInvalidSchedule, prevLid, lid)
		}
		seen[prevLid] = true
	}
	return nil
}

// InterpreterInstructionDesc describes one CPU-scheduled op for an SM worker.
type InterpreterInstructionDesc struct {
	Opcode       int32
	OpName       string
	SMID         int
	SemCount     int
	Args         []int32
	ReleaseOrder *InterpreterPageReleaseOrder
	Waits        []InterpreterBarrierWait
	Signals      []InterpreterBarrierSignal
	Layer        int
	Cost         int
}

// Encode serializes the instruction descriptor into a fixed 32-int32 word
// (128 bytes): [opcode, arg_count, args..., 0-padding].
func (d InterpreterInstructionDesc) Encode() ([InterpreterInstructionWidth]int32, error) {
	var word [InterpreterInstructionWidth]int32
	if d.Opcode < 0 {
		return word, fmt.Errorf("%w: negative opcode %d", ErrInterpreterInvalidSchedule, d.Opcode)
	}
	if len(d.Args) > InterpreterMaxInstructionArgs {
		return word, fmt.Errorf("%w: arg count %d exceeds max %d",
			ErrInterpreterInvalidSchedule, len(d.Args), InterpreterMaxInstructionArgs)
	}
	if d.SemCount < 0 || d.SemCount > InterpreterDynamicSemaphores {
		return word, fmt.Errorf("%w: sem count %d out of [0,%d]",
			ErrInterpreterInvalidSchedule, d.SemCount, InterpreterDynamicSemaphores)
	}
	if d.Opcode == InterpreterOpNoOp && d.SemCount != 0 {
		return word, fmt.Errorf("%w: NoOp must declare 0 dynamic semaphores (got %d)",
			ErrInterpreterInvalidSchedule, d.SemCount)
	}
	word[0] = d.Opcode
	word[1] = int32(len(d.Args))
	copy(word[2:], d.Args)
	return word, nil
}

// DecodeInterpreterWord decodes a fixed 32-int32 instruction word back into its
// opcode and payload arguments.
func DecodeInterpreterWord(word [InterpreterInstructionWidth]int32) (opcode int32, args []int32, err error) {
	opcode = word[0]
	if opcode < 0 {
		return 0, nil, fmt.Errorf("%w: negative encoded opcode %d", ErrInterpreterInvalidSchedule, opcode)
	}
	n := int(word[1])
	if n < 0 || n > InterpreterMaxInstructionArgs {
		return 0, nil, fmt.Errorf("%w: encoded arg count %d out of [0,%d]",
			ErrInterpreterInvalidSchedule, n, InterpreterMaxInstructionArgs)
	}
	out := make([]int32, n)
	copy(out, word[2:2+n])
	return opcode, out, nil
}

// InterpreterSchedule is a CPU-built, tensorized instruction schedule across
// NumSMs worker streams sharing NumBarriers global-memory sync counters.
type InterpreterSchedule struct {
	NumSMs      int
	NumBarriers int
	Streams     [][]InterpreterInstructionDesc
	Words       [][][InterpreterInstructionWidth]int32
}

// BuildInterpreterSchedule validates instruction descriptors, release orders,
// and barrier references, returning a tensorized per-SM schedule ready for
// controller fetch.
func BuildInterpreterSchedule(numSMs, numBarriers int, descs []InterpreterInstructionDesc) (*InterpreterSchedule, error) {
	if numSMs <= 0 {
		return nil, fmt.Errorf("%w: numSMs=%d must be positive", ErrInterpreterInvalidSchedule, numSMs)
	}
	if numBarriers < 0 {
		return nil, fmt.Errorf("%w: numBarriers=%d must be non-negative", ErrInterpreterInvalidSchedule, numBarriers)
	}

	streams := make([][]InterpreterInstructionDesc, numSMs)
	words := make([][][InterpreterInstructionWidth]int32, numSMs)

	for i, d := range descs {
		if d.SMID < 0 || d.SMID >= numSMs {
			return nil, fmt.Errorf("%w: instr[%d] SMID=%d out of [0,%d)",
				ErrInterpreterInvalidSchedule, i, d.SMID, numSMs)
		}
		word, err := d.Encode()
		if err != nil {
			return nil, fmt.Errorf("instr[%d]: %w", i, err)
		}
		rel := InterpreterIdentityReleaseOrder()
		if d.ReleaseOrder != nil {
			if err := ValidateInterpreterReleaseOrder(*d.ReleaseOrder); err != nil {
				return nil, fmt.Errorf("instr[%d]: %w", i, err)
			}
			rel = *d.ReleaseOrder
		}
		for _, w := range d.Waits {
			if w.BarrierID < 0 || w.BarrierID >= numBarriers {
				return nil, fmt.Errorf("%w: instr[%d] wait barrier %d out of [0,%d)",
					ErrInterpreterInvalidSchedule, i, w.BarrierID, numBarriers)
			}
		}
		for _, s := range d.Signals {
			if s.BarrierID < 0 || s.BarrierID >= numBarriers {
				return nil, fmt.Errorf("%w: instr[%d] signal barrier %d out of [0,%d)",
					ErrInterpreterInvalidSchedule, i, s.BarrierID, numBarriers)
			}
			if s.Delta == 0 {
				return nil, fmt.Errorf("%w: instr[%d] signal barrier %d has zero delta",
					ErrInterpreterInvalidSchedule, i, s.BarrierID)
			}
		}
		cloned := d
		cloned.Args = append([]int32(nil), d.Args...)
		cloned.Waits = append([]InterpreterBarrierWait(nil), d.Waits...)
		cloned.Signals = append([]InterpreterBarrierSignal(nil), d.Signals...)
		cloned.ReleaseOrder = &rel
		streams[d.SMID] = append(streams[d.SMID], cloned)
		words[d.SMID] = append(words[d.SMID], word)
	}

	return &InterpreterSchedule{
		NumSMs:      numSMs,
		NumBarriers: numBarriers,
		Streams:     streams,
		Words:       words,
	}, nil
}

// InterpreterExecTraceEntry records one retired instruction in simulation order.
type InterpreterExecTraceEntry struct {
	Step        int
	SMID        int
	InstrIndex  int
	RingSlot    int
	Opcode      int32
	OpName      string
	PIDOrder    [InterpreterNumPages]int
	SemCount    int
	PostSignals []uint32
}

// InterpreterSimReceipt summarizes controller, paging, and counter-sync
// invariants verified by the closed host oracle.
type InterpreterSimReceipt struct {
	TotalInstructions int
	ControllerFetches int
	DrainedSlots      int
	PageAllocChecks   int
	PoolPages         int
	PoolPageBytes     int
	TotalPoolBytes    int
	BarrierChecks     int
	BarrierSignals    int
	FinalBarriers     []uint32
	Trace             []InterpreterExecTraceEntry
}

// InterpreterOpFunc executes one fetched instruction on the host oracle.
// Receives the SM ID, per-SM instruction index, decoded descriptor, and the
// active physical page mapping pidOrder (a verified permutation of 0..12).
type InterpreterOpFunc func(smID, instrIdx int, desc InterpreterInstructionDesc, pidOrder [InterpreterNumPages]int) error

type smControllerState struct {
	pidOrder           [InterpreterPipelineStages][InterpreterNumPages]int
	numSemaphores      [InterpreterPipelineStages]int
	slotOccupied       [InterpreterPipelineStages]bool
	slotInstrIndex     [InterpreterPipelineStages]int
	lastSlotPhaseBit   [InterpreterPipelineStages]int
	pageFinishedPhase  [InterpreterNumPages]int
	nextFetchIdx       int
	nextExecIdx        int
	drainedCount       int
	fetches            int
	pageChecks         int
	semInvalidations   int
	lastControllerStep int
}

// SimulateInterpreterSchedule runs the closed host oracle for sched:
//   - Checks global-memory barrier reachability before execution.
//   - Steps each SM's 2-stage controller ring (wait/invalidate -> ifetch ->
//     ReleaseLID physical page remapping -> semaphore setup -> arrive).
//   - Enforces the 13 x 16 KiB paged SMEM bijection invariant on every instruction.
//   - Enforces counter-in-global-memory ordering (waits block before execOp; signals
//     increment strictly after execOp) and fails closed on deadlock.
func SimulateInterpreterSchedule(sched *InterpreterSchedule, execOp InterpreterOpFunc) (InterpreterSimReceipt, error) {
	if sched == nil || sched.NumSMs <= 0 || len(sched.Streams) != sched.NumSMs || len(sched.Words) != sched.NumSMs {
		return InterpreterSimReceipt{}, fmt.Errorf("%w: nil or inconsistent schedule", ErrInterpreterInvalidSchedule)
	}

	// Pre-flight barrier ceiling check: any wait exceeding total schedule signals cannot resolve.
	totalSignals := make([]uint64, sched.NumBarriers)
	totalInstrs := 0
	for sm := 0; sm < sched.NumSMs; sm++ {
		if len(sched.Streams[sm]) != len(sched.Words[sm]) {
			return InterpreterSimReceipt{}, fmt.Errorf("%w: SM %d stream/word length mismatch",
				ErrInterpreterInvalidSchedule, sm)
		}
		for idx, desc := range sched.Streams[sm] {
			totalInstrs++
			if _, _, err := DecodeInterpreterWord(sched.Words[sm][idx]); err != nil {
				return InterpreterSimReceipt{}, err
			}
			for _, s := range desc.Signals {
				if s.BarrierID < 0 || s.BarrierID >= sched.NumBarriers {
					return InterpreterSimReceipt{}, fmt.Errorf("%w: SM %d instr %d invalid signal barrier %d",
						ErrInterpreterInvalidSchedule, sm, idx, s.BarrierID)
				}
				totalSignals[s.BarrierID] += uint64(s.Delta)
			}
		}
	}
	for sm := 0; sm < sched.NumSMs; sm++ {
		for idx, desc := range sched.Streams[sm] {
			for _, w := range desc.Waits {
				if w.BarrierID < 0 || w.BarrierID >= sched.NumBarriers {
					return InterpreterSimReceipt{}, fmt.Errorf("%w: SM %d instr %d invalid wait barrier %d",
						ErrInterpreterInvalidSchedule, sm, idx, w.BarrierID)
				}
				if uint64(w.Threshold) > totalSignals[w.BarrierID] {
					return InterpreterSimReceipt{}, fmt.Errorf("%w: SM %d instr %d wait barrier[%d]>=%d exceeds total signals %d",
						ErrInterpreterDeadlock, sm, idx, w.BarrierID, w.Threshold, totalSignals[w.BarrierID])
				}
			}
		}
	}

	barriers := make([]uint32, sched.NumBarriers)
	sms := make([]smControllerState, sched.NumSMs)
	receipt := InterpreterSimReceipt{
		TotalInstructions: totalInstrs,
		PoolPages:         InterpreterNumPages,
		PoolPageBytes:     InterpreterPageSizeBytes,
		TotalPoolBytes:    InterpreterTotalPoolBytes,
	}

	// Controller helper: fetch & prepare one instruction into ring slot `k % 2`
	// whenever a slot is free and instructions remain on SM `sm`.
	controllerAdvance := func(sm int) error {
		st := &sms[sm]
		k := st.nextFetchIdx
		if k >= len(sched.Streams[sm]) {
			return nil
		}
		ring := k % InterpreterPipelineStages
		if st.slotOccupied[ring] {
			return nil
		}
		// Step 0: wait for previous instruction in this ring slot to retire and invalidate its semaphores.
		if k >= InterpreterPipelineStages {
			prevK := k - InterpreterPipelineStages
			wantPhase := (prevK / InterpreterPipelineStages) & 1
			if st.lastSlotPhaseBit[ring] != wantPhase {
				return fmt.Errorf("%w: SM %d slot %d phase mismatch got=%d want=%d",
					ErrInterpreterInvalidSchedule, sm, ring, st.lastSlotPhaseBit[ring], wantPhase)
			}
			st.semInvalidations += st.numSemaphores[ring]
			st.numSemaphores[ring] = 0
		}

		// Step 1: fetch 32-int32 word from CPU-built instruction tensor.
		word := sched.Words[sm][k]
		opcode, _, err := DecodeInterpreterWord(word)
		if err != nil {
			return err
		}
		desc := sched.Streams[sm][k]
		if opcode != desc.Opcode {
			return fmt.Errorf("%w: SM %d instr %d word opcode %d != desc opcode %d",
				ErrInterpreterInvalidSchedule, sm, k, opcode, desc.Opcode)
		}
		st.fetches++
		receipt.ControllerFetches++

		// Step 2: compute physical page order pidOrder from previous slot's release order.
		var nextPIDs [InterpreterNumPages]int
		if k == 0 {
			for lid := 0; lid < InterpreterNumPages; lid++ {
				nextPIDs[lid] = lid
			}
		} else {
			prevRing := (ring + InterpreterPipelineStages - 1) % InterpreterPipelineStages
			prevDesc := sched.Streams[sm][k-1]
			rel := InterpreterIdentityReleaseOrder()
			if prevDesc.ReleaseOrder != nil {
				rel = *prevDesc.ReleaseOrder
			}
			if err := ValidateInterpreterReleaseOrder(rel); err != nil {
				return fmt.Errorf("SM %d instr %d: %w", sm, k-1, err)
			}
			for lid := 0; lid < InterpreterNumPages; lid++ {
				nextPIDs[lid] = st.pidOrder[prevRing][rel[lid]]
			}
		}
		if err := verifyPhysicalPageBijection(nextPIDs); err != nil {
			return fmt.Errorf("SM %d instr %d: %w", sm, k, err)
		}
		st.pidOrder[ring] = nextPIDs
		st.pageChecks++
		receipt.PageAllocChecks++

		// Step 3: construct dynamic semaphores (0 for NoOp, desc.SemCount otherwise).
		if opcode == InterpreterOpNoOp {
			st.numSemaphores[ring] = 0
		} else {
			st.numSemaphores[ring] = desc.SemCount
		}

		// Step 4: arrive instruction_arrived[ring].
		st.slotOccupied[ring] = true
		st.slotInstrIndex[ring] = k
		st.nextFetchIdx++
		return nil
	}

	// Preload initial pipeline stages (up to 2 instructions per SM).
	for sm := 0; sm < sched.NumSMs; sm++ {
		for s := 0; s < InterpreterPipelineStages; s++ {
			if err := controllerAdvance(sm); err != nil {
				return receipt, err
			}
		}
	}

	completed := 0
	step := 0
	for completed < totalInstrs {
		progressed := false
		for sm := 0; sm < sched.NumSMs; sm++ {
			st := &sms[sm]
			if st.nextExecIdx >= len(sched.Streams[sm]) {
				continue
			}
			// Keep the 2-stage ring fed if a slot just retired.
			if err := controllerAdvance(sm); err != nil {
				return receipt, err
			}

			k := st.nextExecIdx
			ring := k % InterpreterPipelineStages
			if !st.slotOccupied[ring] || st.slotInstrIndex[ring] != k {
				continue
			}

			desc := sched.Streams[sm][k]
			// Check global-memory counter waits before executing op.
			ready := true
			for _, w := range desc.Waits {
				receipt.BarrierChecks++
				if barriers[w.BarrierID] < w.Threshold {
					ready = false
					break
				}
			}
			if !ready {
				continue
			}

			// Verify page phase reuse bits before worker consumes pages.
			pidOrder := st.pidOrder[ring]
			for _, pid := range pidOrder {
				expectedDoneBefore := k
				if st.pageFinishedPhase[pid] > expectedDoneBefore {
					return receipt, fmt.Errorf("%w: SM %d instr %d physical page %d phase overlap",
						ErrInterpreterInvalidSchedule, sm, k, pid)
				}
				st.pageFinishedPhase[pid] = k + 1
			}

			if execOp != nil {
				if err := execOp(sm, k, desc, pidOrder); err != nil {
					return receipt, err
				}
			}

			// Emit global-memory barrier signals strictly AFTER op execution completes.
			for _, s := range desc.Signals {
				barriers[s.BarrierID] += s.Delta
				receipt.BarrierSignals++
			}

			// Retire slot in 2-stage ring so controller Step 0 can reuse it.
			st.lastSlotPhaseBit[ring] = (k / InterpreterPipelineStages) & 1
			st.slotOccupied[ring] = false
			st.nextExecIdx++
			completed++
			progressed = true

			snap := append([]uint32(nil), barriers...)
			receipt.Trace = append(receipt.Trace, InterpreterExecTraceEntry{
				Step:        step,
				SMID:        sm,
				InstrIndex:  k,
				RingSlot:    ring,
				Opcode:      desc.Opcode,
				OpName:      desc.OpName,
				PIDOrder:    pidOrder,
				SemCount:    st.numSemaphores[ring],
				PostSignals: snap,
			})
			step++

			// Advance controller fetch for the vacated slot.
			if err := controllerAdvance(sm); err != nil {
				return receipt, err
			}
		}
		if !progressed {
			return receipt, fmt.Errorf("%w: stalled after %d/%d instructions (barrier state=%v)",
				ErrInterpreterDeadlock, completed, totalInstrs, barriers)
		}
	}

	// Controller tail drain: confirm final min(len, 2) ring slots retired and
	// invalidate remaining semaphores.
	for sm := 0; sm < sched.NumSMs; sm++ {
		st := &sms[sm]
		n := len(sched.Streams[sm])
		for i := 0; i < InterpreterPipelineStages; i++ {
			idx := n - InterpreterPipelineStages + i
			if idx < 0 {
				continue
			}
			ring := idx % InterpreterPipelineStages
			if st.slotOccupied[ring] {
				return receipt, fmt.Errorf("%w: SM %d slot %d remained occupied at drain",
					ErrInterpreterInvalidSchedule, sm, ring)
			}
			st.semInvalidations += st.numSemaphores[ring]
			st.numSemaphores[ring] = 0
			st.drainedCount++
			receipt.DrainedSlots++
		}
	}

	receipt.FinalBarriers = barriers
	return receipt, nil
}

func verifyPhysicalPageBijection(pidOrder [InterpreterNumPages]int) error {
	var seen [InterpreterNumPages]bool
	for lid, pid := range pidOrder {
		if pid < 0 || pid >= InterpreterNumPages {
			return fmt.Errorf("%w: pid_order[%d]=%d out of [0,%d)",
				ErrInterpreterInvalidSchedule, lid, pid, InterpreterNumPages)
		}
		if seen[pid] {
			return fmt.Errorf("%w: duplicate physical page %d in pid_order",
				ErrInterpreterInvalidSchedule, pid)
		}
		seen[pid] = true
	}
	return nil
}
