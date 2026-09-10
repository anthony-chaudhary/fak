//go:build vulkan && linux && cgo

package compute

/*
#cgo LDFLAGS: -lvulkan -lstdc++ -lm
*/
import "C"
