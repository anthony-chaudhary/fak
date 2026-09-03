// q2k_format.go — Q2_K wire-format constants. Deliberately carries NO build tag:
// these describe the PAYLOAD LAYOUT, not the Metal backend, so they are meaningful in every build.
// The Apple-Silicon side (q2k.go/q2k.m) uses them for pointer arithmetic and upload guards;
// the stub build still needs them to size or validate a payload without a Metal device, and the
// format witnesses in q2k_witness_test.go pin them in every build.
//
// Q2_K is llama.cpp's 2-bit k-quant (GGML_TYPE_Q2_K = 10): each 256-element super-block
// packs into 84 bytes:
//   - 16 bytes: per-subblock scales (low 4 bits = scale multiplier, high 4 bits = min multiplier)
//   - 64 bytes: 2-bit codes (256 codes, packed 4 per byte, 2 bits each)
//   - 2 bytes:  f16 block scale d
//   - 2 bytes:  f16 block min dmin
// Total: 16 + 64 + 2 + 2 = 84 bytes per 256 weights = 0.328125 bytes/weight.

package metalgemm

// Q2KBlockWeights is the super-block element count for Q2_K: 256 weights.
const Q2KBlockWeights = 256

// Q2KBlockBytes is the packed byte size of one Q2_K super-block: 84 bytes.
const Q2KBlockBytes = 84

// Q2KPayloadBytes returns the resident byte cost of an [out, in] Q2_K weight:
// out * (in / 256) * 84 bytes. It returns 0 when in is not a multiple of
// Q2KBlockWeights or a dimension is non-positive.
func Q2KPayloadBytes(out, in int) int {
	if out <= 0 || in <= 0 || in%Q2KBlockWeights != 0 {
		return 0
	}
	nblk := in / Q2KBlockWeights
	return out * nblk * Q2KBlockBytes
}
