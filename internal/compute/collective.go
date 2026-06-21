package compute

import "fmt"

// collective.go — the CPU reference implementation of CollectiveBackend, the tensor-parallel
// cross-rank reduction seam declared in compute.go. It is the device-tensor counterpart of
// model.LocalCollective (which reduces host []float32): single-box, synchronous, and EXACT,
// so a real NCCL/RDMA collective swapped in behind the interface is correct only if it
// reproduces these bytes. AllReduceSum/ReduceScatter reduce, and AllGather concatenates, in
// RANK ORDER (parts[0] first), which is what makes the result deterministic and the model-side
// row-parallel gate exact. ReduceScatter is the dual of AllGather: it reuses the IDENTICAL
// rank-order sum (reduceSumRankOrder) and slices it into equal per-rank shards, so the
// AllReduceSum ≡ AllGather∘ReduceScatter identity holds byte-for-byte by construction.
// AllToAll is the TRANSPOSE collective — it moves a different shard to each peer rather than
// reducing or concatenating one — and is its own inverse (an involution); ReduceScatter is in
// turn recoverable as an AllToAll followed by a local per-rank reduce. Both identities hold
// byte-for-byte by construction, so a real NCCL all-to-all is correct only if it matches them.
//
// Every method routes the parts through collectF32, which enforces the fail-closed contract a
// real communicator needs: no parts, an unready part, a non-F32 part, or a part owned by a
// DIFFERENT backend (the cross-backend reduction NCCL rejects — a CUDA tensor cannot be
// all-reduced against a host tensor) is refused at the boundary, never silently mis-reduced.

// collectF32 validates the per-rank parts against the CollectiveBackend fail-closed contract
// and returns their host f32 views in rank order. requireEqualLen pins the AllReduceSum rule
// that every partial is the same length; AllGather passes false (uneven shards are its point).
// A part must be ready, F32, host-readable, and OWNED BY THIS backend — the affinity check
// that distinguishes a device collective from the host []float32 seam.
//
// Affinity is interface identity (p.Backend() != this backend): a real, stateful backend has a
// distinct instance per device, so a tensor from another communicator/type — the CUDA-tensor-
// vs-host-tensor case a real all-reduce must reject — fails here. (The reference cpuBackend is
// a stateless zero-size value, so all cpu-ref tensors are genuinely interchangeable; the cross-
// backend contract is meaningful precisely for the device backends that carry per-device state.)
func (c *cpuBackend) collectF32(parts []Tensor, requireEqualLen bool) ([][]float32, error) {
	if len(parts) == 0 {
		return nil, fmt.Errorf("compute: collective got no rank parts")
	}
	views := make([][]float32, len(parts))
	for r, p := range parts {
		if p.Backend() != Backend(c) {
			return nil, fmt.Errorf("compute: collective rank %d tensor is owned by a different backend (cross-backend reduction is rejected)", r)
		}
		if p.Dtype != F32 {
			return nil, fmt.Errorf("compute: collective rank %d dtype = %s, want f32", r, p.Dtype)
		}
		if !p.Ready() {
			return nil, fmt.Errorf("compute: collective rank %d tensor is not ready", r)
		}
		v, ok := c.Host(p)
		if !ok {
			return nil, fmt.Errorf("compute: collective rank %d tensor is not host-readable f32", r)
		}
		if requireEqualLen && r > 0 && len(v) != len(views[0]) {
			return nil, fmt.Errorf("compute: AllReduceSum rank %d len = %d, want %d (ragged partials)", r, len(v), len(views[0]))
		}
		views[r] = v
	}
	return views, nil
}

// reduceSumRankOrder is the rank-order element-wise sum shared by AllReduceSum and
// ReduceScatter, so the reduction order is defined ONCE: acc = parts[0]; acc += parts[r] for
// r=1.., over the validated equal-length f32 views. Defining it in a single place is what
// guarantees ReduceScatter reduces byte-identically to AllReduceSum — the property the
// AllReduceSum ≡ AllGather∘ReduceScatter gate relies on. The returned slice is freshly
// allocated, so a caller owns it (AllReduceSum returns it as a tensor; ReduceScatter slices it).
func (c *cpuBackend) reduceSumRankOrder(parts []Tensor) ([]float32, error) {
	views, err := c.collectF32(parts, true)
	if err != nil {
		return nil, err
	}
	n := len(views[0])
	acc := make([]float32, n)
	copy(acc, views[0])
	for r := 1; r < len(views); r++ {
		for i := 0; i < n; i++ {
			acc[i] += views[r][i]
		}
	}
	return acc, nil
}

// AllReduceSum sums the equal-length per-rank partials in rank order and returns a new F32
// tensor. The fixed order makes it bit-identical to model.sumPartialsRankOrder over the same
// slices, so the row-parallel reduction order is pinned. A single part is the identity (a copy
// of its data): the HAL twin of ForwardTP(ranks=1).
func (c *cpuBackend) AllReduceSum(parts []Tensor) (Tensor, error) {
	acc, err := c.reduceSumRankOrder(parts)
	if err != nil {
		return Tensor{}, err
	}
	return c.result([]int{len(acc)}, acc), nil
}

// ReduceScatter performs the rank-order AllReduceSum (via the shared reduceSumRankOrder) and
// then returns, to each of the P = len(parts) ranks, that rank's contiguous 1/P shard of the
// reduced vector — the dual of AllGather, and the sequence-parallel collective (see the
// interface doc in compute.go). The reduced length N must be a multiple of P (real NCCL
// requires sendcount % nranks == 0): an indivisible N fails closed rather than silently
// dropping the remainder. Output shard r is reduced[r*N/P : (r+1)*N/P], copied into its own
// F32 tensor so a rank owns its slice outright. A single part is the identity (one shard equal
// to the lone part). By construction AllGather(ReduceScatter(parts)) == AllReduceSum(parts)
// byte-for-byte, because both consume the SAME rank-order sum.
func (c *cpuBackend) ReduceScatter(parts []Tensor) ([]Tensor, error) {
	acc, err := c.reduceSumRankOrder(parts)
	if err != nil {
		return nil, err
	}
	p := len(parts)
	n := len(acc)
	if n%p != 0 {
		return nil, fmt.Errorf("compute: ReduceScatter reduced length %d is not divisible by rank count %d (real reduce-scatter requires sendcount %% nranks == 0)", n, p)
	}
	shard := n / p
	out := make([]Tensor, p)
	for r := 0; r < p; r++ {
		seg := make([]float32, shard)
		copy(seg, acc[r*shard:(r+1)*shard])
		out[r] = c.result([]int{shard}, seg)
	}
	return out, nil
}

// AllGather concatenates the per-rank shards in rank order (parts[0]‖parts[1]‖…) into a new
// F32 tensor. Shard lengths may differ — gathering uneven output bands is its purpose. A
// single part is the identity (a copy of its data).
func (c *cpuBackend) AllGather(parts []Tensor) (Tensor, error) {
	views, err := c.collectF32(parts, false)
	if err != nil {
		return Tensor{}, err
	}
	total := 0
	for _, v := range views {
		total += len(v)
	}
	out := make([]float32, 0, total)
	for _, v := range views {
		out = append(out, v...)
	}
	return c.result([]int{total}, out), nil
}

// AllToAll redistributes the P = len(parts) equal-length per-rank vectors by the block transpose
// the layout-changing collectives need: each rank's length-N vector is read as P contiguous shards
// of size N/P, and output rank r receives, in rank order, shard r from EVERY input rank —
// out[r] = views[0][r-shard] ‖ views[1][r-shard] ‖ … ‖ views[P-1][r-shard], where view k's
// "r-shard" is its band [r*shard, (r+1)*shard). It is the collective ReduceScatter/AllGather
// cannot express (it moves a DIFFERENT shard to each peer instead of reducing or concatenating
// one): the primitive that turns a sequence-sharded activation into a head-sharded one, and an
// MoE expert dispatch into its combine.
//
// Two identities make it self-checking, the AllToAll twin of AllReduceSum ≡ AllGather∘ReduceScatter:
//   - INVOLUTION: AllToAll(AllToAll(parts)) == parts byte-for-byte. out[r]'s k-th block is
//     parts[k]'s shard r, so out[r][k*shard+j] = parts[k][r*shard+j]; applying the same transpose
//     again swaps the two indices back, restoring parts[k] exactly.
//   - REDUCE-SCATTER-VIA-ALL-TO-ALL: summing out[r]'s P shards elementwise yields
//     Σ_k parts[k][r*shard+j] = the r-th band of AllReduceSum = ReduceScatter shard r — NCCL's
//     own construction of reduce-scatter, and the gate that ties this method to the proven ones.
//
// Like ReduceScatter the per-rank length N must divide evenly by the rank count (real NCCL
// all-to-all requires sendcount % nranks == 0); a ragged or indivisible input fails closed through
// the shared collectF32 boundary plus the divisibility check, never silently mis-routed. Each
// output is copied into its own F32 tensor so a rank owns its slice outright, and a single part is
// the identity (the lone vector unchanged) — the HAL twin of ForwardTP(ranks=1).
func (c *cpuBackend) AllToAll(parts []Tensor) ([]Tensor, error) {
	views, err := c.collectF32(parts, true)
	if err != nil {
		return nil, err
	}
	p := len(views)
	n := len(views[0])
	if n%p != 0 {
		return nil, fmt.Errorf("compute: AllToAll per-rank length %d is not divisible by rank count %d (real all-to-all requires sendcount %% nranks == 0)", n, p)
	}
	shard := n / p
	out := make([]Tensor, p)
	for r := 0; r < p; r++ {
		seg := make([]float32, 0, n)
		for k := 0; k < p; k++ {
			seg = append(seg, views[k][r*shard:(r+1)*shard]...)
		}
		out[r] = c.result([]int{n}, seg)
	}
	return out, nil
}
