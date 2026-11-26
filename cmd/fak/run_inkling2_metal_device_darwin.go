//go:build arm64 && cgo

package main

import (
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

const inkling2MetalResidentAvailable = true

func openInkling2MetalResident(path string) (*ggufload.Inkling2LoopModel, compute.Backend, error) {
	return ggufload.OpenInkling2LoopMetalDevice(path)
}
