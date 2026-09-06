package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/harnessconformance"
)

func TestStockHarnessProbeConforms(t *testing.T) {
	probe := stockHarnessProbe{}
	cert := harnessconformance.Run(probe)
	if !cert.Full {
		t.Fatalf("expected cert.Full=true, got %v", cert.Full)
	}
	if err := cert.Validate(); err != nil {
		t.Fatalf("expected cert.Validate()=nil, got %v", err)
	}
	if len(cert.Checks) != len(harnessconformance.Required) {
		t.Errorf("expected %d checks, got %d", len(harnessconformance.Required), len(cert.Checks))
	}
}
