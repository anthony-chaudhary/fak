//go:build cuda && cgo

package compute

func (c *cudaBackend) v41ReadIndexerOwned(original Backend, tensor Tensor) ([]float32, bool) {
	// A promoted method must not bypass a wrapper's Read override or promise
	// ownership of the wrapper's potentially reusable host storage.
	canonical, ok := original.(*cudaBackend)
	if !ok || canonical != c {
		return original.Read(tensor), false
	}
	// Ordinary Read already checks D2H completion and publishes its fence only
	// on success. Its device result is newly allocated; hostBuf reads still alias.
	values := c.Read(tensor)
	_, device := tensor.buf.(*cudaBuf)
	return values, device && tensor.be == c
}
