package qwen38quantrun

import (
	"encoding/json"
	"fmt"
	"os"
)

func BuildAMDScoreboardFile(inputPath, reportPath string) (AMDScoreboardReport, error) {
	raw, err := os.ReadFile(inputPath)
	if err != nil {
		return AMDScoreboardReport{}, fmt.Errorf("scoreboard input: %w", err)
	}
	var input AMDScoreboardInput
	if err := decodeJSONStrict(raw, &input); err != nil {
		return AMDScoreboardReport{}, fmt.Errorf("scoreboard input: %w", err)
	}
	report := BuildAMDScoreboard(input)
	if err := ValidateAMDScoreboardReport(report); err != nil {
		return AMDScoreboardReport{}, err
	}
	encoded, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return AMDScoreboardReport{}, err
	}
	encoded = append(encoded, '\n')
	if err := os.WriteFile(reportPath, encoded, 0o600); err != nil {
		return AMDScoreboardReport{}, fmt.Errorf("scoreboard report: %w", err)
	}
	return report, nil
}
