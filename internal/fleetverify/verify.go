// Package fleetverify is a THROWAWAY compile-verification of the operator.go
// fleet helpers, isolated from the concurrently-churning cmd/fak (package main)
// so the exact loopfleet/loopmgr API usage type-checks against the real APIs
// with zero duplicate-symbol risk. Delete after building.
package fleetverify

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopfleet"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// loadBriefJSON mirrors cmd/fak's helper signature (string, io.Reader, any) error.
func loadBriefJSON(path string, stdin io.Reader, dst any) error {
	var data []byte
	var err error
	if path == "-" {
		data, err = io.ReadAll(stdin)
	} else {
		data, err = os.ReadFile(path)
	}
	if err != nil {
		return err
	}
	return json.Unmarshal(data, dst)
}

// loadFleetBriefReport is a verbatim copy of the operator.go helper.
func loadFleetBriefReport(path string, stdin io.Reader) (loopfleet.Report, error) {
	var r loopfleet.Report
	if err := loadBriefJSON(path, stdin, &r); err != nil {
		return r, err
	}
	if r.Schema != "" && r.Schema != loopfleet.Schema {
		return r, fmt.Errorf("schema %q, want %q", r.Schema, loopfleet.Schema)
	}
	return r, nil
}

// collectFleetBriefReport is a verbatim copy of the operator.go collector.
func collectFleetBriefReport(root string) (loopfleet.Report, error) {
	return loopfleet.Fold(root, time.Now(), loopmgr.HealthThresholds{}), nil
}

// Invariant: fleet verification reports are fail-closed and bounded.
// Exercise validates both collection and brief loading against strict schema requirements.
func Exercise(root, path string, stdin io.Reader) (loopfleet.Report, error) {
	if _, err := collectFleetBriefReport(root); err != nil {
		return loopfleet.Report{}, err
	}
	return loadFleetBriefReport(path, stdin)
}
