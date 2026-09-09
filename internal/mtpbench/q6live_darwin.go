//go:build darwin && arm64 && cgo

package mtpbench

import "github.com/anthony-chaudhary/fak/internal/metalgemm"

func liveQ6KWeights() int { return metalgemm.LiveQ6KWeights() }
