package compute

import "math"

// vulkanF32DecodeOperand is a bounded snapshot of a live owned native binding.
// Shape has no offset: the selected native descriptor binds at byte zero.
// This is not a stale-pointer validator; callers hold vulkanMu and retain owners.
type vulkanF32DecodeOperand struct {
	BufferID                                     int64
	Bytes                                        int64
	Shape                                        []int
	AllocationID                                 uint64
	BindingOffset, BufferBytes, ReservationBytes uint64
}

func vulkanF32DecodeBytes(shape []int) (int64, bool) {
	if len(shape) == 0 || len(shape) > 2 {
		return 0, false
	}
	n := int64(4)
	for _, d := range shape {
		if d <= 0 || int64(d) > math.MaxInt64/n {
			return 0, false
		}
		n *= int64(d)
	}
	return n, true
}

// vulkanF32DecodeMatMulAccesses declares only the ordinary P=1 F32 matmul.
// Failure must be recorded as unknown, not as an empty trusted access list.
func vulkanF32DecodeMatMulAccesses(w, x, y vulkanF32DecodeOperand, out, in int) ([3]VulkanBufferAccess, bool) {
	var accesses [3]VulkanBufferAccess
	if out <= 0 || in <= 0 || int64(out) > math.MaxInt32 || int64(in) > math.MaxInt32 ||
		len(w.Shape) != 2 || w.Shape[0] != out || w.Shape[1] != in {
		return accesses, false
	}
	wantW, ok := vulkanF32DecodeBytes([]int{out, in})
	if !ok {
		return accesses, false
	}
	want := [3]int64{wantW, int64(in) * 4, int64(out) * 4}
	operands := [3]vulkanF32DecodeOperand{w, x, y}
	for i, operand := range operands {
		n, ok := vulkanF32DecodeBytes(operand.Shape)
		if !ok || n != want[i] || operand.BufferID == 0 || operand.AllocationID == 0 ||
			operand.Bytes < n || operand.BufferBytes < uint64(n) ||
			operand.BufferBytes != uint64(operand.Bytes) ||
			operand.BindingOffset > math.MaxInt64 ||
			uint64(n) > uint64(math.MaxInt64)-operand.BindingOffset ||
			operand.BindingOffset > operand.ReservationBytes ||
			operand.BufferBytes > operand.ReservationBytes-operand.BindingOffset {
			return [3]VulkanBufferAccess{}, false
		}
		mode := VulkanAccessRead
		if i == 2 {
			mode = VulkanAccessWrite
		}
		accesses[i] = VulkanBufferAccess{BufferID: operand.BufferID, Size: n, Mode: mode,
			AllocationID: operand.AllocationID, BindingOffset: int64(operand.BindingOffset)}
	}
	// A barrier cannot make in-dispatch read/write aliasing safe. Decline the
	// whole declaration, including overlap through different binding handles.
	for i := 0; i < 2; i++ {
		if _, conflict := accessModesConflict(accesses[i], accesses[2]); conflict {
			return [3]VulkanBufferAccess{}, false
		}
	}
	return accesses, true
}
