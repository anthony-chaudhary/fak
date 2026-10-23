package compute

// requireCUDAQ8BlockAlignment follows the positive-block and row-divisibility
// checks. DP4A consumes four codes per aligned word, so an incomplete word would
// drop codes and could misalign later block/row loads. Custom complete words
// remain supported; the canonical Q8_0 block size is not the only valid size.
func requireCUDAQ8BlockAlignment(block int) {
	if block%4 != 0 {
		panic(&CUDAOpError{
			Op: "Upload", Site: "upload-q8-resident", Class: MemoryWeights,
			Msg: "cuda Upload(Q8_0 host) needs QuantSpec.Block divisible by 4 for DP4A",
		})
	}
}
