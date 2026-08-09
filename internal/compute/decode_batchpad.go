package compute

// decode_batchpad.go — the host-exact slot-token accounting for issue #5852, the follow-up to
// the #5819 negative result. That witness
// (experiments/microcontext/s4e-gcp-inkernel-compat-batch-pass-2026-08-07.json) ran 24 real jobs
// through model.NewBatchFromPrefixReserve + BatchSession.StepBatchActive on the CPU-reference
// path and reached only 0.539209x tuned sequential throughput, allocating 144 slot-token steps
// for 94 useful ones (34.7222% padding). #5852 asks whether length-aware compatible
// sub-bucketing or active-lane compaction recovers that loss.
//
// Both mechanisms are pure SCHEDULING: they change how many lane-slots a decode step allocates,
// never what a slot costs. That makes their effect an exact COUNT on any host — the same honesty
// discipline as prefill.go (exact FLOPs/bytes, never a timer) and fusion_traffic.go (operand
// counts, host-identical). So the padding side of #5852 is decidable here, with no GGUF model,
// no L4, and no wall clock; only the per-slot COST is a device measurement, and it is supplied
// by the caller from an already-captured witness rather than invented.
//
// The load-bearing fact this file makes checkable, and the reason #5852's "keep the change only
// if it grades a net gain" gate can be answered before writing a scheduler: the measured ratio
// factors EXACTLY into two independent terms,
//
//	sequential_wall / batch_wall = 1 / (PaddingFactor × SlotCostFactor)
//	PaddingFactor  = allocated_steps / useful_steps          (scheduling — what #5852 attacks)
//	SlotCostFactor = ms_per_slot_step / ms_per_seq_token     (kernel — what #5852 does NOT touch)
//
// Driving PaddingFactor to its floor of 1.0 therefore buys at most ZeroPaddingRatio, and if that
// ceiling is still below 1.0 no sub-bucketing or compaction policy can turn the result positive.
// A scheduler is then the wrong fix, and the honest outcome is the null result #5852's witness
// clause already admits.

// DecodeSlotSchedule is one batching policy's exact slot-token accounting over a set of
// compatibility classes. UsefulSteps is policy-invariant (it is the work itself); AllocatedSteps
// is what the policy makes the machine pay for it; PaddingFrac is the wasted share. Batches and
// HeadOfLineSlotSteps carry the cost side of the trade, because every padding-reducing policy
// pays for it in MORE, NARROWER batches — #5852 asks for padding and queue tax reported
// separately, and folding them into one number is exactly the conflation that would hide the
// trade.
type DecodeSlotSchedule struct {
	// Policy names the scheduling policy that produced this accounting.
	Policy string
	// Batches is the number of physical batches executed.
	Batches int
	// Lanes is the number of scheduled jobs (one decode lane each).
	Lanes int
	// UsefulSteps is the total real token steps — sum of every lane's decode length. Invariant
	// across policies: scheduling never changes the work, only its packing.
	UsefulSteps int
	// AllocatedSteps is the total lane-slots the machine steps, padding included.
	AllocatedSteps int
	// PaddingFrac is (AllocatedSteps-UsefulSteps)/AllocatedSteps — the witness's
	// "real_padding_tax", computed the same way.
	PaddingFrac float64
	// HeadOfLineSlotSteps is the exact queue tax under the serial batch executor the #5819
	// witness used: the sum over batches of the slot-steps each batch waits behind its
	// predecessors. It RISES as padding falls, which is the whole point of reporting it apart
	// from PaddingFrac.
	HeadOfLineSlotSteps int
}

// StaticSlotSchedule is the #5819 baseline: each compatibility class is chunked, in arrival
// order, into batches of at most width lanes, and every batch steps its FULL width for as many
// steps as its LONGEST lane needs. A short lane keeps consuming a slot after it is done — the
// masked-but-allocated lane that StepBatchActive charges for — which is where the witness's
// 34.7222% went. Classes are never mixed: that is the planner's isolation boundary, and this
// accounting respects it exactly as the executed witness did.
func StaticSlotSchedule(classes [][]int, width int) DecodeSlotSchedule {
	return chunkedSchedule("static-width", classes, width, false)
}

// LengthBucketedSlotSchedule is #5852's first named mechanism, length-aware compatible
// sub-bucketing: WITHIN each compatibility class (never across — sub-bucketing refines
// compatibility, it never relaxes it) lanes are ordered by decode length and then chunked into
// sub-sized batches, so each batch's longest lane is close to its shortest. Padding falls
// because the max-vs-mean gap inside a batch shrinks. sub is clamped to width: a sub-bucket can
// never exceed the physical batch width. sub >= width degenerates to the static baseline's
// batching with length ordering, which is the correct no-op.
func LengthBucketedSlotSchedule(classes [][]int, width, sub int) DecodeSlotSchedule {
	if sub > width {
		sub = width
	}
	return chunkedSchedule("length-bucketed", classes, sub, true)
}

// CompactedSlotSchedule is #5852's second named mechanism, active-lane compaction: the batch is
// formed exactly as the baseline, but at each decode step the retired lanes are compacted out
// and the machine steps only the lanes still active, rounded UP to gran. gran is the honest part
// — a real kernel cannot resize its batch to an arbitrary width for free, so lanes are charged
// in blocks of gran (SIMD/warp/tile granularity); gran <= 1 models ideal per-step compaction,
// whose allocation equals UsefulSteps by construction and is therefore the floor, not a
// prediction. Compaction does not change the batch count, so it costs no extra queue tax — the
// structural reason it dominates sub-bucketing whenever the kernel can actually do it.
func CompactedSlotSchedule(classes [][]int, width, gran int) DecodeSlotSchedule {
	if gran < 1 {
		gran = 1
	}
	s := DecodeSlotSchedule{Policy: "active-lane-compaction"}
	var perBatch []int
	for _, class := range classes {
		for start := 0; start < len(class); start += width {
			end := start + width
			if end > len(class) {
				end = len(class)
			}
			batch := class[start:end]
			longest, alloc := 0, 0
			for _, n := range batch {
				s.Lanes++
				s.UsefulSteps += n
				if n > longest {
					longest = n
				}
			}
			for pos := 0; pos < longest; pos++ {
				active := 0
				for _, n := range batch {
					if pos < n {
						active++
					}
				}
				alloc += ceilMultiple(active, gran)
			}
			perBatch = append(perBatch, alloc)
			s.AllocatedSteps += alloc
			s.Batches++
		}
	}
	s.PaddingFrac = paddingFrac(s.UsefulSteps, s.AllocatedSteps)
	s.HeadOfLineSlotSteps = headOfLine(perBatch)
	return s
}

// chunkedSchedule is the shared body of the two full-width policies: chunk each class into
// batches of at most width lanes and charge longest*width per batch. byLength orders the class
// by descending decode length first, which is the ONLY difference between the static baseline
// and length-aware sub-bucketing — the sorted copy never mutates the caller's slice.
func chunkedSchedule(policy string, classes [][]int, width int, byLength bool) DecodeSlotSchedule {
	s := DecodeSlotSchedule{Policy: policy}
	if width < 1 {
		width = 1
	}
	var perBatch []int
	for _, class := range classes {
		lanes := append([]int(nil), class...)
		if byLength {
			sortDesc(lanes)
		}
		for start := 0; start < len(lanes); start += width {
			end := start + width
			if end > len(lanes) {
				end = len(lanes)
			}
			batch := lanes[start:end]
			longest := 0
			for _, n := range batch {
				s.Lanes++
				s.UsefulSteps += n
				if n > longest {
					longest = n
				}
			}
			alloc := longest * len(batch)
			perBatch = append(perBatch, alloc)
			s.AllocatedSteps += alloc
			s.Batches++
		}
	}
	s.PaddingFrac = paddingFrac(s.UsefulSteps, s.AllocatedSteps)
	s.HeadOfLineSlotSteps = headOfLine(perBatch)
	return s
}

// BatchVsSequential factors a MEASURED batch-vs-sequential decode result into the scheduling
// term #5852 can change and the kernel term it cannot. Nothing here is estimated: the four
// inputs come from an executed witness, and every field is arithmetic over them. Provenance is
// therefore OBSERVED for the measured fields and MODELED only for ZeroPaddingRatio, which is
// labelled as the ceiling it is.
type BatchVsSequential struct {
	UsefulSteps    int
	AllocatedSteps int
	// BatchWallMS and SequentialWallMS are the two measured walls, over the SAME useful steps.
	BatchWallMS, SequentialWallMS float64
	// MSPerSlotStep is BatchWallMS/AllocatedSteps — what one allocated lane-slot actually cost.
	MSPerSlotStep float64
	// MSPerSequentialToken is SequentialWallMS/UsefulSteps — the tuned single-stream cost.
	MSPerSequentialToken float64
	// PaddingFactor is AllocatedSteps/UsefulSteps (>= 1): how many slots the schedule spent per
	// useful step. This is the ONLY term sub-bucketing or compaction moves.
	PaddingFactor float64
	// SlotCostFactor is MSPerSlotStep/MSPerSequentialToken: how much dearer one batched slot is
	// than one tuned sequential token. Below 1 the kernel amortizes; above 1 batching is losing
	// before a single pad step is counted.
	SlotCostFactor float64
	// Ratio is the measured SequentialWallMS/BatchWallMS, identically 1/(PaddingFactor*SlotCostFactor).
	Ratio float64
	// ZeroPaddingRatio is Ratio*PaddingFactor — the ratio a PERFECT scheduler would reach if a
	// slot kept costing what it costs today. It is an OPTIMISTIC ceiling in both directions:
	// perfect packing is unreachable at any nontrivial length spread, and a compacted/narrower
	// batch amortizes its weight stream over FEWER lanes, so its real per-slot cost rises rather
	// than holding. ZeroPaddingRatio < 1 is therefore a hard refusal for the whole family of
	// padding-only fixes, not a close call.
	ZeroPaddingRatio float64
}

// DecomposeBatchVsSequential builds the factoring from an executed witness. It reports the
// zero-valued struct for degenerate input (non-positive steps or walls) rather than dividing by
// zero and emitting a confident nonsense ratio.
func DecomposeBatchVsSequential(usefulSteps, allocatedSteps int, batchWallMS, sequentialWallMS float64) BatchVsSequential {
	if usefulSteps <= 0 || allocatedSteps <= 0 || batchWallMS <= 0 || sequentialWallMS <= 0 {
		return BatchVsSequential{}
	}
	b := BatchVsSequential{
		UsefulSteps: usefulSteps, AllocatedSteps: allocatedSteps,
		BatchWallMS: batchWallMS, SequentialWallMS: sequentialWallMS,
		MSPerSlotStep:        batchWallMS / float64(allocatedSteps),
		MSPerSequentialToken: sequentialWallMS / float64(usefulSteps),
		PaddingFactor:        float64(allocatedSteps) / float64(usefulSteps),
		Ratio:                sequentialWallMS / batchWallMS,
	}
	b.SlotCostFactor = b.MSPerSlotStep / b.MSPerSequentialToken
	b.ZeroPaddingRatio = b.Ratio * b.PaddingFactor
	return b
}

// PaddingFixCanReachParity is #5852's "keep the change only if it grades a net gain" gate,
// decided on the host before any scheduler is written: a padding-only fix can reach tuned
// sequential ONLY if the zero-padding ceiling does. False means the deficit lives in
// SlotCostFactor — the batched kernel's per-slot cost — and no amount of sub-bucketing or
// compaction reaches parity, so the correct outcome is the null result rather than a landed
// scheduler that grades net-false.
func (b BatchVsSequential) PaddingFixCanReachParity() bool { return b.ZeroPaddingRatio >= 1 }

// ProjectRatio is the throughput a schedule would reach if a slot kept costing MSPerSlotStep.
// It is deliberately OPTIMISTIC — see ZeroPaddingRatio — and is bounded above by it, so it is
// usable to refuse a policy and never to claim one. Schedules whose useful steps do not match
// the measured run are rejected as 0: projecting one workload's schedule onto another's wall is
// the exact category error this file exists to prevent.
func (b BatchVsSequential) ProjectRatio(s DecodeSlotSchedule) float64 {
	if b.MSPerSlotStep <= 0 || s.AllocatedSteps <= 0 || s.UsefulSteps != b.UsefulSteps {
		return 0
	}
	return b.SequentialWallMS / (b.MSPerSlotStep * float64(s.AllocatedSteps))
}

func paddingFrac(useful, allocated int) float64 {
	if allocated <= 0 {
		return 0
	}
	return float64(allocated-useful) / float64(allocated)
}

// headOfLine sums each batch's wait behind its predecessors under serial execution — the exact
// queue tax of the executor the #5819 witness ran, in slot-step units.
func headOfLine(perBatch []int) int {
	total, waited := 0, 0
	for _, alloc := range perBatch {
		total += waited
		waited += alloc
	}
	return total
}

func ceilMultiple(n, gran int) int {
	if gran <= 1 {
		return n
	}
	return ((n + gran - 1) / gran) * gran
}

// sortDesc orders decode lengths longest-first in place. Insertion sort keeps this file
// dependency-free and is exact at the batch widths that matter (tens of lanes).
func sortDesc(xs []int) {
	for i := 1; i < len(xs); i++ {
		for j := i; j > 0 && xs[j] > xs[j-1]; j-- {
			xs[j], xs[j-1] = xs[j-1], xs[j]
		}
	}
}
