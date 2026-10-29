//go:build !cuda || !nccl

// pgsmoke's real main lives behind //go:build cuda && nccl (it witnesses a multi-process
// NCCL communicator, which needs a CUDA+NCCL toolchain). This stub keeps the package
// buildable, vettable, and test-discoverable on GPU-free hosts — go build ./..., go vet,
// and CI all see a well-formed main package — and fails fast with the rebuild recipe
// instead of a link error when someone runs the tag-less binary.
package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "pgsmoke: built without the 'cuda nccl' build tags; rebuild with CGO_ENABLED=1 go build -tags 'cuda nccl' ./cmd/pgsmoke on a CUDA host")
	os.Exit(2)
}
