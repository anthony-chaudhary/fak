package main

import (
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/vcachecalibration"
)

func loadVCacheRuntimeCalibration(provider, model string) *gateway.VCacheRuntimeCalibration {
	cal, ok, _ := vcachecalibration.FreshRuntimeConstants(
		nightrunLedgerPath(vcachecalibration.DefaultCalibrationRel), provider, model,
		time.Now(), vcachecalibration.DefaultCalibrationTTL,
	)
	if !ok {
		return nil
	}
	return &gateway.VCacheRuntimeCalibration{
		Provider: cal.Provider, Model: cal.Model, Source: cal.Source, TS: cal.TS,
		TTLMillis: cal.TTLMillis, TTLMeasured: cal.TTLMeasured,
		MinPrefixTokens: cal.MinPrefixTokens, MinPrefixMeasured: cal.MinPrefixMeasured,
		ReadMult: cal.ReadMult, ReadMultMeasured: cal.ReadMultMeasured,
		Write5mMult: cal.Write5mMult, Write5mMeasured: cal.Write5mMeasured,
		Write1hMult: cal.Write1hMult, Write1hMeasured: cal.Write1hMeasured,
	}
}
