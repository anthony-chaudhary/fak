//go:build amd64

package strix

//go:noescape
func sfence()

//go:noescape
func mfence()

//go:noescape
func clflushopt(addr uintptr)
