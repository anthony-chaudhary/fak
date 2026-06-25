//go:build cuda

package compute

import "testing"

func TestCUDABudgetedQ8WeightFreeReleasesDeviceLocalBytes(t *testing.T) {
	cb := cudaOrSkip(t)
	oldBudget, oldUsed, oldManaged := cb.CUDADebugSetResidencyBudget(64)
	defer cb.CUDADebugRestoreResidencyBudget(oldBudget, oldUsed, oldManaged)

	var s lcg = 363
	shape := []int{2, 32} // Q8 code buffer is 64 bytes.
	w := randVec(&s, shape[0]*shape[1])
	dw := cb.Upload(NewF32(cpu(), shape, w), Q8_0)
	db := dw.buf.(*cudaBuf)
	if db.budgetedWeightBytes != 64 {
		t.Fatalf("budget charge=%d want 64", db.budgetedWeightBytes)
	}
	if db.managed || db.managedWeight {
		t.Fatalf("first weight unexpectedly used managed memory: managed=%v managedWeight=%v", db.managed, db.managedWeight)
	}
	_, used, managed := cb.CUDADebugResidencyBudget()
	if used != 64 || managed != 0 {
		t.Fatalf("after first upload used=%d managed=%d, want used=64 managed=0", used, managed)
	}

	cb.Free(dw)
	_, used, managed = cb.CUDADebugResidencyBudget()
	if used != 0 || managed != 0 {
		t.Fatalf("after Free used=%d managed=%d, want both zero", used, managed)
	}
	if db.budgetedWeightBytes != 0 {
		t.Fatalf("freed buffer retained budget charge %d", db.budgetedWeightBytes)
	}

	w2 := append([]float32(nil), w...) // fresh host pointer, so uploadCache cannot mask accounting.
	dw2 := cb.Upload(NewF32(cpu(), shape, w2), Q8_0)
	_, used, managed = cb.CUDADebugResidencyBudget()
	if used != 64 || managed != 0 {
		t.Fatalf("second upload used=%d managed=%d, want used=64 managed=0", used, managed)
	}
	cb.Free(dw2)
}

func TestCUDABudgetSpillsQ8WeightToManaged(t *testing.T) {
	cb := cudaOrSkip(t)
	oldBudget, oldUsed, oldManaged := cb.CUDADebugSetResidencyBudget(64)
	defer cb.CUDADebugRestoreResidencyBudget(oldBudget, oldUsed, oldManaged)

	var s lcg = 0x363
	shape := []int{3, 32} // Q8 code buffer is 96 bytes, over the 64-byte budget.
	w := randVec(&s, shape[0]*shape[1])
	dw := cb.Upload(NewF32(cpu(), shape, w), Q8_0)
	db := dw.buf.(*cudaBuf)
	if !db.managed || !db.managedWeight {
		t.Fatalf("over-budget weight did not spill to managed memory: managed=%v managedWeight=%v", db.managed, db.managedWeight)
	}
	if db.budgetedWeightBytes != 0 {
		t.Fatalf("managed weight charged device-local bytes=%d, want 0", db.budgetedWeightBytes)
	}
	_, used, managed := cb.CUDADebugResidencyBudget()
	if used != 0 || managed != 1 {
		t.Fatalf("after managed spill used=%d managed=%d, want used=0 managed=1", used, managed)
	}

	cb.Free(dw)
	_, used, managed = cb.CUDADebugResidencyBudget()
	if used != 0 || managed != 0 {
		t.Fatalf("after Free used=%d managed=%d, want both zero", used, managed)
	}
	if db.managedWeight {
		t.Fatal("freed managed weight retained accounting flag")
	}
}
