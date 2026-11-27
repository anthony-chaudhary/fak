package compute

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestInterpreterScheduleWordAndPoolInvariants(t *testing.T) {
	t.Parallel()

	// Pool geometry invariants: 13 paged 16 KiB SMEM pools = 208 KiB.
	if InterpreterNumPages != 13 || InterpreterPageSizeBytes != 16384 || InterpreterTotalPoolBytes != 212992 {
		t.Fatalf("unexpected SMEM pool constants: pages=%d pageBytes=%d total=%d",
			InterpreterNumPages, InterpreterPageSizeBytes, InterpreterTotalPoolBytes)
	}
	if InterpreterInstructionWidth != 32 {
		t.Fatalf("instruction width = %d, want 32 (128B)", InterpreterInstructionWidth)
	}

	// Instruction descriptor round-trips through fixed 32-int32 word encoding.
	desc := InterpreterInstructionDesc{
		Opcode:   5,
		OpName:   "upgate_silu",
		SMID:     0,
		SemCount: 13,
		Args:     []int32{0, 16, 2048, -7, 8192},
	}
	word, err := desc.Encode()
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	gotOpcode, gotArgs, err := DecodeInterpreterWord(word)
	if err != nil {
		t.Fatalf("DecodeInterpreterWord: %v", err)
	}
	if gotOpcode != desc.Opcode || len(gotArgs) != len(desc.Args) {
		t.Fatalf("decoded = (%d, %v), want (%d, %v)", gotOpcode, gotArgs, desc.Opcode, desc.Args)
	}
	for i := range desc.Args {
		if gotArgs[i] != desc.Args[i] {
			t.Fatalf("arg[%d] = %d, want %d", i, gotArgs[i], desc.Args[i])
		}
	}
	// Remaining padding tail must be zeroed.
	for i := 2 + len(desc.Args); i < InterpreterInstructionWidth; i++ {
		if word[i] != 0 {
			t.Fatalf("word[%d] padding = %d, want 0", i, word[i])
		}
	}

	// Overflow args fail closed.
	tooMany := desc
	tooMany.Args = make([]int32, InterpreterMaxInstructionArgs+1)
	if _, err := tooMany.Encode(); !errors.Is(err, ErrInterpreterInvalidSchedule) {
		t.Fatalf("overflow args err = %v, want ErrInterpreterInvalidSchedule", err)
	}

	// Every matvec pipeline release order must be a valid bijection of 0..12.
	for iters := 0; iters <= 6; iters++ {
		rel := InterpreterMatvecReleaseOrder(iters)
		if err := ValidateInterpreterReleaseOrder(rel); err != nil {
			t.Fatalf("InterpreterMatvecReleaseOrder(%d): %v", iters, err)
		}
	}

	// Non-bijective page release order fails closed when building a schedule.
	badRel := InterpreterIdentityReleaseOrder()
	badRel[5] = 0 // duplicate physical page 0, missing page 5
	if _, err := BuildInterpreterSchedule(1, 0, []InterpreterInstructionDesc{
		{Opcode: 1, SMID: 0, ReleaseOrder: &badRel},
	}); !errors.Is(err, ErrInterpreterInvalidSchedule) {
		t.Fatalf("bad release order err = %v, want ErrInterpreterInvalidSchedule", err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestInterpreterScheduleCounterSyncAndDeadlockRefusal(t *testing.T) {
	t.Parallel()

	// Build a 4-SM multi-op fused decode schedule (2 layers, NoOp + QKV/MatVec ->
	// SplitAttention -> AttentionReduction -> UpGate+DownProj) communicating ONLY
	// through global-memory barrier counters (no kernel boundaries), with paged
	// 13 x 16 KiB SMEM release orders between instructions on each SM.
	rel1 := InterpreterMatvecReleaseOrder(1)
	rel2 := InterpreterMatvecReleaseOrder(2)
	rel4 := InterpreterMatvecReleaseOrder(4)

	const (
		barQKV0   = 0
		barAttn0  = 1
		barRed0   = 2
		barUpGate = 3
		barLayer1 = 4
	)

	// Input vector shared across layers; each op transforms state deterministically.
	// Layer 0:
	//   SM 0: NoOp (opcode 0) -> RMS+QKV (writes q[0..3], signals barQKV0 += 1)
	//   SM 1: PartialAttn0 (waits barQKV0 >= 1, writes part0, signals barAttn0 += 1)
	//   SM 2: PartialAttn1 (waits barQKV0 >= 1, writes part1, signals barAttn0 += 1)
	//   SM 3: AttnReduce (waits barAttn0 >= 2, reduces part0+part1 -> attnOut, signals barRed0 += 1)
	//   SM 0: UpGate (waits barRed0 >= 1, computes silu-gate -> upOut, signals barUpGate += 1)
	//   SM 3: DownProj (waits barUpGate >= 1, adds residual -> out, signals barLayer1 += 1)
	//   SM 1: FinalHead (waits barLayer1 >= 1, scales final logits)
	// Deliberately place consumers BEFORE producers in slice/SM order where
	// possible so only counter-sync ordering guarantees correct execution!
	descs := []InterpreterInstructionDesc{
		// SM 0
		{Opcode: InterpreterOpNoOp, OpName: "noop", SMID: 0, SemCount: 0, ReleaseOrder: &rel1},
		{Opcode: 1, OpName: "qkv", SMID: 0, SemCount: 13, Args: []int32{0}, ReleaseOrder: &rel2,
			Signals: []InterpreterBarrierSignal{{BarrierID: barQKV0, Delta: 1}}},
		{Opcode: 5, OpName: "upgate", SMID: 0, SemCount: 13, Args: []int32{0}, ReleaseOrder: &rel4,
			Waits:   []InterpreterBarrierWait{{BarrierID: barRed0, Threshold: 1}},
			Signals: []InterpreterBarrierSignal{{BarrierID: barUpGate, Delta: 1}}},

		// SM 1
		{Opcode: 2, OpName: "partial_attn_0", SMID: 1, SemCount: 10, Args: []int32{0}, ReleaseOrder: &rel1,
			Waits:   []InterpreterBarrierWait{{BarrierID: barQKV0, Threshold: 1}},
			Signals: []InterpreterBarrierSignal{{BarrierID: barAttn0, Delta: 1}}},
		{Opcode: 7, OpName: "lm_head", SMID: 1, SemCount: 13, Args: []int32{0}, ReleaseOrder: &rel2,
			Waits: []InterpreterBarrierWait{{BarrierID: barLayer1, Threshold: 1}}},

		// SM 2
		{Opcode: 2, OpName: "partial_attn_1", SMID: 2, SemCount: 10, Args: []int32{1}, ReleaseOrder: &rel1,
			Waits:   []InterpreterBarrierWait{{BarrierID: barQKV0, Threshold: 1}},
			Signals: []InterpreterBarrierSignal{{BarrierID: barAttn0, Delta: 1}}},

		// SM 3
		{Opcode: 3, OpName: "attn_reduce", SMID: 3, SemCount: 8, Args: []int32{0}, ReleaseOrder: &rel2,
			Waits:   []InterpreterBarrierWait{{BarrierID: barAttn0, Threshold: 2}},
			Signals: []InterpreterBarrierSignal{{BarrierID: barRed0, Delta: 1}}},
		{Opcode: 6, OpName: "down_proj", SMID: 3, SemCount: 13, Args: []int32{0}, ReleaseOrder: &rel4,
			Waits:   []InterpreterBarrierWait{{BarrierID: barUpGate, Threshold: 1}},
			Signals: []InterpreterBarrierSignal{{BarrierID: barLayer1, Delta: 1}}},
	}

	sched, err := BuildInterpreterSchedule(4, 5, descs)
	if err != nil {
		t.Fatalf("BuildInterpreterSchedule: %v", err)
	}

	// Shared in-memory state operated on by the multi-SM interpreter vs sequential oracle.
	type vmMem struct {
		q       [2]float32
		partial [2]float32
		attn    float32
		upGate  float32
		hidden  float32
		logits  float32
	}
	runOp := func(mem *vmMem, desc InterpreterInstructionDesc) {
		switch desc.OpName {
		case "noop":
		case "qkv":
			mem.q[0] = 2.5
			mem.q[1] = -1.25
		case "partial_attn_0":
			mem.partial[0] = mem.q[0]*1.5 + 0.5
		case "partial_attn_1":
			mem.partial[1] = mem.q[1]*2.0 + 3.0
		case "attn_reduce":
			mem.attn = mem.partial[0] + mem.partial[1]
		case "upgate":
			mem.upGate = mem.attn * (mem.attn + 1.0)
		case "down_proj":
			mem.hidden = mem.upGate*0.25 + mem.attn
		case "lm_head":
			mem.logits = mem.hidden * 3.5
		}
	}

	// Sequential CPU reference in exact topological op order.
	var wantMem vmMem
	for _, opName := range []string{"noop", "qkv", "partial_attn_0", "partial_attn_1", "attn_reduce", "upgate", "down_proj", "lm_head"} {
		runOp(&wantMem, InterpreterInstructionDesc{OpName: opName})
	}

	var gotMem vmMem
	stepByOp := map[string]int{}
	receipt, err := SimulateInterpreterSchedule(sched, func(smID, instrIdx int, desc InterpreterInstructionDesc, pidOrder [InterpreterNumPages]int) error {
		// Every executed op must hold an intact 13-page bijection.
		if err := verifyPhysicalPageBijection(pidOrder); err != nil {
			return err
		}
		stepByOp[desc.OpName] = len(stepByOp)
		runOp(&gotMem, desc)
		return nil
	})
	if err != nil {
		t.Fatalf("SimulateInterpreterSchedule: %v", err)
	}

	if receipt.TotalInstructions != len(descs) || receipt.ControllerFetches != len(descs) || receipt.PageAllocChecks != len(descs) {
		t.Fatalf("unexpected controller counts: %+v", receipt)
	}
	if receipt.TotalPoolBytes != InterpreterTotalPoolBytes {
		t.Fatalf("TotalPoolBytes = %d, want %d", receipt.TotalPoolBytes, InterpreterTotalPoolBytes)
	}
	// Verify causal counter-sync ordering across SMs.
	if !(stepByOp["qkv"] < stepByOp["partial_attn_0"] &&
		stepByOp["qkv"] < stepByOp["partial_attn_1"] &&
		stepByOp["partial_attn_0"] < stepByOp["attn_reduce"] &&
		stepByOp["partial_attn_1"] < stepByOp["attn_reduce"] &&
		stepByOp["attn_reduce"] < stepByOp["upgate"] &&
		stepByOp["upgate"] < stepByOp["down_proj"] &&
		stepByOp["down_proj"] < stepByOp["lm_head"]) {
		t.Fatalf("counter-sync violated topological ordering: %v", stepByOp)
	}
	// Output must match sequential reference bit-for-bit.
	if math.Float32bits(gotMem.logits) != math.Float32bits(wantMem.logits) ||
		math.Float32bits(gotMem.hidden) != math.Float32bits(wantMem.hidden) {
		t.Fatalf("interpreter output mismatch: got=%+v want=%+v", gotMem, wantMem)
	}

	// Negative controls: unsatisfied threshold and cyclic waits fail closed with ErrInterpreterDeadlock.
	t.Run("unsatisfied threshold deadlocks", func(t *testing.T) {
		badSched, err := BuildInterpreterSchedule(2, 1, []InterpreterInstructionDesc{
			{Opcode: 1, SMID: 0, Signals: []InterpreterBarrierSignal{{BarrierID: 0, Delta: 1}}},
			{Opcode: 2, SMID: 1, Waits: []InterpreterBarrierWait{{BarrierID: 0, Threshold: 2}}},
		})
		if err != nil {
			t.Fatalf("BuildInterpreterSchedule: %v", err)
		}
		if _, err := SimulateInterpreterSchedule(badSched, nil); !errors.Is(err, ErrInterpreterDeadlock) {
			t.Fatalf("err = %v, want ErrInterpreterDeadlock", err)
		}
	})

	t.Run("cross-SM cycle deadlocks", func(t *testing.T) {
		cycSched, err := BuildInterpreterSchedule(2, 2, []InterpreterInstructionDesc{
			{Opcode: 1, SMID: 0,
				Waits:   []InterpreterBarrierWait{{BarrierID: 1, Threshold: 1}},
				Signals: []InterpreterBarrierSignal{{BarrierID: 0, Delta: 1}}},
			{Opcode: 1, SMID: 1,
				Waits:   []InterpreterBarrierWait{{BarrierID: 0, Threshold: 1}},
				Signals: []InterpreterBarrierSignal{{BarrierID: 1, Delta: 1}}},
		})
		if err != nil {
			t.Fatalf("BuildInterpreterSchedule: %v", err)
		}
		if _, err := SimulateInterpreterSchedule(cycSched, nil); !errors.Is(err, ErrInterpreterDeadlock) {
			t.Fatalf("err = %v, want ErrInterpreterDeadlock", err)
		}
	})

	t.Run("same-SM wait-before-signal inversion deadlocks", func(t *testing.T) {
		invSched, err := BuildInterpreterSchedule(1, 1, []InterpreterInstructionDesc{
			{Opcode: 1, SMID: 0, Waits: []InterpreterBarrierWait{{BarrierID: 0, Threshold: 1}}},
			{Opcode: 2, SMID: 0, Signals: []InterpreterBarrierSignal{{BarrierID: 0, Delta: 1}}},
		})
		if err != nil {
			t.Fatalf("BuildInterpreterSchedule: %v", err)
		}
		if _, err := SimulateInterpreterSchedule(invSched, nil); !errors.Is(err, ErrInterpreterDeadlock) {
			t.Fatalf("err = %v, want ErrInterpreterDeadlock", err)
		}
	})
}
