//go:build vulkan && linux && cgo

package compute

/*
#cgo LDFLAGS: -lvulkan -lstdc++
*/
import "C"
