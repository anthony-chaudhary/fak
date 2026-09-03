// q2_0_format.go — the ternary (Q2_0) wire-format constants. Deliberately carries NO build tag:
// these describe the PAYLOAD LAYOUT, not the Metal backend, so they are meaningful in every build.
// The Apple-Silicon side (q2_0.go/q2_0.m) uses them for its pointer arithmetic and upload guards;
// the stub build still needs them to size or validate a payload without a Metal device, and the
// format witnesses in q2_0_witness_test.go pin them in exactly that build.
//
// The layout is internal/model/quant_q2.go's g32 form verbatim: a weight is [out, in] row-major
// with in = nblk*Q2_0BlockWeights, and each block is one f32 scale d plus Q2_0BlockBytes of 2-bit
// codes (four per byte, low code first) where code c means d*(c-2). With d = amax the live code
// set is the ternary {-1, 0, +1}·d.

package metalgemm

// Q2_0BlockWeights is the ternary block width: 32 weights share one f32 scale.
const Q2_0BlockWeights = 32

// Q2_0BlockBytes is the packed code size of one ternary block: 32 codes × 2 bits = 8 bytes.
const Q2_0BlockBytes = 8

// Q2_0PayloadBytes returns the resident byte cost of an [out, in] ternary weight: the packed code
// bytes plus the f32 block scales, i.e. 0.375 B per weight. It returns 0 when in is not a multiple
// of Q2_0BlockWeights or a dimension is non-positive — the same shapes UploadQ2_0 refuses. Callers
// use it to budget residency before building a payload, in either build.
func Q2_0PayloadBytes(out, in int) int {
	if out <= 0 || in <= 0 || in%Q2_0BlockWeights != 0 {
		return 0
	}
	nblk := in / Q2_0BlockWeights
	return out * nblk * (Q2_0BlockBytes + 4)
}

// Q2_0G128BlockWeights is the block width for standard GGUF group-128 Q2_0: 128 weights share one f16 scale.
const Q2_0G128BlockWeights = 128

// Q2_0G128BlockBytes is the byte size of one group-128 Q2_0 block: 2 bytes f16 scale + 32 bytes 2-bit codes = 34 bytes.
const Q2_0G128BlockBytes = 34

// Q2_0G128PayloadBytes returns the resident byte cost of an [out, in] group-128 GGUF Q2_0 weight:
// out * (in / 128) * 34 bytes. It returns 0 when in is not a multiple of Q2_0G128BlockWeights or a
// dimension is non-positive — the same shapes UploadQ2_0G128 refuses.
func Q2_0G128PayloadBytes(out, in int) int {
	if out <= 0 || in <= 0 || in%Q2_0G128BlockWeights != 0 {
		return 0
	}
	nblk := in / Q2_0G128BlockWeights
	return out * nblk * Q2_0G128BlockBytes
}
