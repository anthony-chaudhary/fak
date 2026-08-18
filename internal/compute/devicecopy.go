package compute

func readDeviceFloats(length int, copyToHost func(dst []float32)) []float32 {
	out := make([]float32, length)
	if length > 0 {
		copyToHost(out)
	}
	return out
}
