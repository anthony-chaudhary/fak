//go:build !windows

package main

import (
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/terminalrisk"
)

func gatherTerminalRiskFacts(string) (terminalrisk.Facts, error) {
	return terminalrisk.Facts{}, fmt.Errorf("Windows only")
}
func defaultWTSettingsPath() string { return "" }
func terminalRiskRDPAdvice() string { return "" }
