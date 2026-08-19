package main

import (
	"time"

	"github.com/anthony-chaudhary/fak/internal/vcachecalibration"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

var appendVCacheCalibration = vcachecalibration.AppendCalibration

func persistVCacheCalibration(provider, source string, turns []vcacheobserve.Turn) {
	row, ok := vcachecalibration.CalibrationFromTurns(provider, source, turns, time.Now())
	if !ok {
		return
	}
	_ = appendVCacheCalibration(nightrunLedgerPath(vcachecalibration.DefaultCalibrationRel), row)
}
