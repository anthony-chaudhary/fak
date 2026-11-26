//go:build !darwin || !arm64 || !cgo

package main

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

const inkling2MetalResidentAvailable = false

func openInkling2MetalResident(string) (*ggufload.Inkling2LoopModel, compute.Backend, error) {
	return nil, nil, fmt.Errorf("Inkling 2: the device-resident Metal forward needs a darwin/arm64 cgo build")
}
