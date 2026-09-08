//go:build cuda

package compute

// SupportsQ2K advertises native packed Q2_K matrix multiplication.
func (c *cudaBackend) SupportsQ2K() bool { return true }
