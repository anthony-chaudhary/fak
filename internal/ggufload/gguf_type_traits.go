package ggufload

// Block geometry translated from ggml/src/ggml.c type_traits at
// 83078fec0db82d6b5a00d9599062c38c39145755; see THIRD_PARTY_NOTICES.md.
type tensorTraits struct {
	BlockSize, TypeSize uint64
	IsQuantized         bool
}

var ggufTypeTraits = map[TensorType]tensorTraits{
	TensorF32:     {1, 4, false},
	TensorF16:     {1, 2, false},
	TensorBF16:    {1, 2, false},
	TensorQ4_0:    {qk4, blockQ4_0Bytes, true},
	TensorQ4_1:    {qk4, blockQ4_1Bytes, true},
	TensorQ5_0:    {qk5, blockQ5_0Bytes, true},
	TensorQ5_1:    {qk5, blockQ5_1Bytes, true},
	TensorQ8_0:    {qk8_0, blockQ8_0Bytes, true},
	TensorQ2_K:    {qkK, blockQ2KBytes, true},
	TensorQ3_K:    {qkK, blockQ3KBytes, true},
	TensorQ4_K:    {qkK, blockQ4KBytes, true},
	TensorQ5_K:    {qkK, blockQ5KBytes, true},
	TensorQ6_K:    {qkK, blockQ6KBytes, true},
	TensorIQ2_XXS: {qkK, blockIQ2XXSBytes, true},
	TensorIQ2_XS:  {qkK, blockIQ2XSBytes, true},
	TensorIQ1_S:   {qkK, blockIQ1SBytes, true},
	TensorIQ2_S:   {qkK, blockIQ2SBytes, true},
	TensorIQ1_M:   {qkK, blockIQ1MBytes, true},
	TensorMXFP4:   {qkMXFP4, blockMXFP4Bytes, true},
	TensorIQ4_NL:  {qkIQ4NL, blockIQ4NLBytes, true},
	TensorIQ3_S:   {qkIQ3S, blockIQ3SBytes, true},
	TensorIQ4_XS:  {qkK, blockIQ4XSBytes, true},
	TensorIQ3_XXS: {qkK, blockIQ3XXSBytes, true},
	// These format identities retain the loader's existing geometry; upstream Q2_0 uses a different block size.
	TensorQ2_0:  {128, blockQ2_0Bytes, true},
	TensorPQ2_0: {128, blockPQ2_0Bytes, true},
	TensorQ1_0:  {128, blockQ1_0Bytes, true},
	TensorHQQ4:  {qkHQQ4, blockHQQ4Bytes, true},
}

func tensorTypeTraits(t TensorType) (tensorTraits, bool) {
	traits, ok := ggufTypeTraits[t]
	return traits, ok
}
