package compute

import (
	"fmt"
	"math"
)

// MaxTreeAttentionCandidates is the largest candidate tree representable by one
// uint32 mask row. The compact row is the kernel contract: bit k is set exactly
// when candidate query q may attend candidate key k.
const MaxTreeAttentionCandidates = 32

// TreeVerifyBackend is the optional backend capability for one-pass tree-masked
// speculative verification attention.
type TreeVerifyBackend interface {
	Backend
	TreeVerifyAttention(q, k, v, out *Tensor, maskRows []uint32, qLen, kvLen, nH, nHkv, d int) error
}

// TreeVerifyAttentionBackend exposes tree verification without requiring the
// complete compute Backend interface. Tests and higher-level adapters use this
// narrow seam.
type TreeVerifyAttentionBackend interface {
	TreeVerifyAttention(q, k, v, out *Tensor, maskRows []uint32, qLen, kvLen, nH, nHkv, d int) error
}

// PackTreeAttentionMask packs a topologically ordered KxK causal mask into K
// uint32 rows. Self must be present and future keys must be absent. Exact
// ancestor/sibling validation remains the tree owner's responsibility.
func PackTreeAttentionMask(mask [][]bool) ([]uint32, error) {
	k := len(mask)
	if k == 0 || k > MaxTreeAttentionCandidates {
		return nil, fmt.Errorf("compute: tree attention candidate count %d outside [1,%d]", k, MaxTreeAttentionCandidates)
	}
	rows := make([]uint32, k)
	for q := 0; q < k; q++ {
		if len(mask[q]) != k {
			return nil, fmt.Errorf("compute: tree attention mask row %d length %d != %d", q, len(mask[q]), k)
		}
		for key, allow := range mask[q] {
			if !allow {
				continue
			}
			if key > q {
				return nil, fmt.Errorf("compute: tree attention mask row %d admits future key %d", q, key)
			}
			rows[q] |= uint32(1) << uint(key)
		}
		if rows[q]&(uint32(1)<<uint(q)) == 0 {
			return nil, fmt.Errorf("compute: tree attention mask row %d omits self", q)
		}
	}
	return rows, nil
}

func validateTreeVerifyAttention(q, k, v, out *Tensor, maskRows []uint32, qLen, kvLen, nH, nHkv, d int) error {
	if q == nil || k == nil || v == nil || out == nil {
		return fmt.Errorf("compute: TreeVerifyAttention nil tensor argument")
	}
	if qLen <= 0 || qLen > MaxTreeAttentionCandidates || kvLen < qLen {
		return fmt.Errorf("compute: TreeVerifyAttention invalid lengths qLen=%d kvLen=%d", qLen, kvLen)
	}
	if len(maskRows) != qLen {
		return fmt.Errorf("compute: TreeVerifyAttention mask rows %d != qLen %d", len(maskRows), qLen)
	}
	if nH <= 0 || nHkv <= 0 || nH%nHkv != 0 {
		return fmt.Errorf("compute: TreeVerifyAttention invalid heads nH=%d nHkv=%d", nH, nHkv)
	}
	if d <= 0 || d > 1024 {
		return fmt.Errorf("compute: TreeVerifyAttention head dim %d outside [1,1024]", d)
	}
	validBits := ^uint32(0)
	if qLen < MaxTreeAttentionCandidates {
		validBits = (uint32(1) << uint(qLen)) - 1
	}
	for row, bits := range maskRows {
		if bits&^validBits != 0 {
			return fmt.Errorf("compute: TreeVerifyAttention mask row %d has bits beyond qLen", row)
		}
		if bits&(uint32(1)<<uint(row)) == 0 {
			return fmt.Errorf("compute: TreeVerifyAttention mask row %d omits self", row)
		}
		futureBits := validBits & ^((uint32(1) << uint(row+1)) - 1)
		if row == MaxTreeAttentionCandidates-1 {
			futureBits = 0
		}
		if bits&futureBits != 0 {
			return fmt.Errorf("compute: TreeVerifyAttention mask row %d admits a future key", row)
		}
	}
	expectedQ := qLen * nH * d
	expectedKV := kvLen * nHkv * d
	if q.Numel() != expectedQ {
		return fmt.Errorf("compute: TreeVerifyAttention q numel %d != expected %d", q.Numel(), expectedQ)
	}
	if k.Numel() != expectedKV {
		return fmt.Errorf("compute: TreeVerifyAttention k numel %d != expected %d", k.Numel(), expectedKV)
	}
	if v.Numel() != expectedKV {
		return fmt.Errorf("compute: TreeVerifyAttention v numel %d != expected %d", v.Numel(), expectedKV)
	}
	return nil
}

// treeVerifyAttentionHost is an independent two-pass CPU reference. Prefix
// keys [0,kvLen-qLen) are always visible; candidate key k is visible to query q
// iff bit k of maskRows[q] is set.
func treeVerifyAttentionHost(q, k, v, out []float32, maskRows []uint32, qLen, kvLen, nH, nHkv, d int, scale float32) {
	prefix := kvLen - qLen
	group := nH / nHkv
	kvWidth := nHkv * d
	scores := make([]float32, kvLen)
	for qi := 0; qi < qLen; qi++ {
		for h := 0; h < nH; h++ {
			kvh := h / group
			qh := q[(qi*nH+h)*d : (qi*nH+h+1)*d]
			maxScore := float32(math.Inf(-1))
			for j := 0; j < kvLen; j++ {
				allowed := j < prefix || (maskRows[qi]&(uint32(1)<<uint(j-prefix)) != 0)
				if !allowed {
					scores[j] = float32(math.Inf(-1))
					continue
				}
				kj := k[j*kvWidth+kvh*d : j*kvWidth+(kvh+1)*d]
				scores[j] = dot(qh, kj) * scale
				if scores[j] > maxScore {
					maxScore = scores[j]
				}
			}
			var sum float64
			for j := 0; j < kvLen; j++ {
				if math.IsInf(float64(scores[j]), -1) {
					continue
				}
				scores[j] = float32(math.Exp(float64(scores[j] - maxScore)))
				sum += float64(scores[j])
			}
			oh := out[(qi*nH+h)*d : (qi*nH+h+1)*d]
			for dim := range oh {
				oh[dim] = 0
			}
			for j := 0; j < kvLen; j++ {
				if scores[j] == 0 || math.IsInf(float64(scores[j]), -1) {
					continue
				}
				weight := scores[j] / float32(sum)
				vj := v[j*kvWidth+kvh*d : j*kvWidth+(kvh+1)*d]
				for dim := range oh {
					oh[dim] += weight * vj[dim]
				}
			}
		}
	}
}

// TreeVerifyAttention applies one arbitrary topologically ordered 2D causal
// tree mask to K<=32 speculative queries. The candidate mask is exact integer
// logic; floating-point output is an Approx peer on device because reduction
// order and exp implementations differ from the CPU reference.
func TreeVerifyAttention(q, k, v, out *Tensor, maskRows []uint32, qLen, kvLen, nH, nHkv, d int) error {
	if q == nil {
		return fmt.Errorf("compute: TreeVerifyAttention nil q tensor")
	}
	be := q.Backend()
	if be == nil {
		be = Default()
	}
	if tree, ok := be.(TreeVerifyBackend); ok {
		return tree.TreeVerifyAttention(q, k, v, out, maskRows, qLen, kvLen, nH, nHkv, d)
	}
	ref, ok := Default().(TreeVerifyBackend)
	if !ok {
		return fmt.Errorf("compute: Default backend does not implement TreeVerifyBackend")
	}
	return ref.TreeVerifyAttention(q, k, v, out, maskRows, qLen, kvLen, nH, nHkv, d)
}

// TreeVerifyAttention on cpuBackend is the correctness reference.
func (c *cpuBackend) TreeVerifyAttention(q, k, v, out *Tensor, maskRows []uint32, qLen, kvLen, nH, nHkv, d int) error {
	if err := validateTreeVerifyAttention(q, k, v, out, maskRows, qLen, kvLen, nH, nHkv, d); err != nil {
		return err
	}
	result := make([]float32, qLen*nH*d)
	treeVerifyAttentionHost(c.f32(*q), c.f32(*k), c.f32(*v), result, maskRows, qLen, kvLen, nH, nHkv, d, float32(1/math.Sqrt(float64(d))))
	*out = c.result([]int{qLen, nH, d}, result)
	return nil
}
