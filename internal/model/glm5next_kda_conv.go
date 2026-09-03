package model

// GLM5NextKDAConvFilter holds depthwise 1D convolution weights for KDA projections.
// Weight shape: [dim * filterWidth], where filterWidth=4.
type GLM5NextKDAConvFilter struct {
	Dim         int
	FilterWidth int
	// Weight is flattened [Dim * FilterWidth]. For channel d and tap k, index is d*FilterWidth + k.
	Weight []float32
}

// NewGLM5NextKDAConvFilter creates a filter of dimension dim and filterWidth.
func NewGLM5NextKDAConvFilter(dim, filterWidth int) *GLM5NextKDAConvFilter {
	if filterWidth <= 0 {
		filterWidth = 4
	}
	return &GLM5NextKDAConvFilter{
		Dim:         dim,
		FilterWidth: filterWidth,
		Weight:      make([]float32, dim*filterWidth),
	}
}

// Step applies the 1D depthwise causal convolution to a single token input of length filter.Dim,
// updates the history buffer buf (of length (filterWidth-1)*Dim), and returns output with SiLU.
func (f *GLM5NextKDAConvFilter) Step(x []float32, buf []float32) []float32 {
	dim := f.Dim
	kSize := f.FilterWidth
	histLen := kSize - 1
	out := make([]float32, dim)

	for d := 0; d < dim; d++ {
		var sum float32
		wBase := d * kSize
		// Historical taps: k=0 .. histLen-1
		for h := 0; h < histLen; h++ {
			tapWeight := f.Weight[wBase+h]
			pastVal := buf[h*dim+d]
			sum += tapWeight * pastVal
		}
		// Current token tap: k = histLen
		currentWeight := f.Weight[wBase+histLen]
		sum += currentWeight * x[d]

		out[d] = silu(sum)
	}

	// Shift history buffer: drop oldest token, shift left, append current token x at the end
	if histLen > 1 {
		copy(buf[0:(histLen-1)*dim], buf[dim:histLen*dim])
	}
	copy(buf[(histLen-1)*dim:], x)

	return out
}

// Prefill processes a sequence of T tokens (flattened x of length T*Dim) causally,
// updating buf and returning output of length T*Dim with SiLU.
func (f *GLM5NextKDAConvFilter) Prefill(x []float32, T int, buf []float32) []float32 {
	dim := f.Dim
	out := make([]float32, T*dim)
	for t := 0; t < T; t++ {
		tokenX := x[t*dim : (t+1)*dim]
		tokenOut := f.Step(tokenX, buf)
		copy(out[t*dim:(t+1)*dim], tokenOut)
	}
	return out
}
