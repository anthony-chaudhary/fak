//go:build !(darwin && arm64 && cgo)

package compute

import (
	"errors"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// MetalInkling2Stats reports no counters without the Metal inkling2 device.
func MetalInkling2Stats() (metalgemm.Ink2Stats, bool) { return metalgemm.Ink2Stats{}, false }

// MetalInkling2Profile reports no profile without the Metal inkling2 device.
func MetalInkling2Profile() (metalgemm.Ink2Profile, bool) { return metalgemm.Ink2Profile{}, false }

// MetalInkling2ProfileReset fails without the Metal inkling2 device.
func MetalInkling2ProfileReset() error {
	return errors.New("compute: no Metal inkling2 device on this build")
}

// MetalInkling2MakeResident has nothing to wire without the Metal inkling2 device.
func MetalInkling2MakeResident([]metalgemm.Ink2Buf) error { return nil }

// MetalInkling2Buffers finds no Metal buffers without the Metal inkling2 device.
func MetalInkling2Buffers([]Tensor) []metalgemm.Ink2Buf { return nil }
