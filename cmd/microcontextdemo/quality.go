package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/microagent"
)

func writeQualityLedger(input, output string, sampleLimit int) error {
	data, err := os.ReadFile(input)
	if err != nil {
		return err
	}
	var identity struct {
		BaseFingerprint string `json:"base_fingerprint"`
	}
	if err := json.Unmarshal(data, &identity); err != nil {
		return err
	}
	ledger, err := microagent.IngestSourceRun(data, "microcontext-ledger-"+identity.BaseFingerprint, identity.BaseFingerprint, microagent.OutcomeCheckFunc(func(string) error { return nil }), sampleLimit)
	if err != nil {
		return err
	}
	encoded, err := json.MarshalIndent(ledger, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(output, append(encoded, '\n'), 0o644)
}

func verifyQualityLedgerArtifact(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var ledger microagent.QualityLedger
	if err := json.Unmarshal(data, &ledger); err != nil {
		return err
	}
	if err := microagent.VerifyQualityLedger(ledger); err != nil {
		return err
	}
	if ledger.ClaimFamilies.UsefulWork.PerWallSecond <= 0 {
		return fmt.Errorf("useful-work rate missing")
	}
	return nil
}
