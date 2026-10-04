package main

import (
	"strings"
	"testing"

	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

func TestServeLMHeadReadyLineAbsentWithoutModel(t *testing.T) {
	if got := serveLMHeadReadyLine(nil); got != "" {
		t.Fatalf("no model must emit no lm_head line, got %q", got)
	}
	if got := residentWeightsLiveView(nil, &fakmodel.ResidentReport{TotalResidentBytes: 1}); got != nil {
		t.Fatalf("no model must omit resident_weights, got %+v", got)
	}
}

func TestServeLMHeadReadyLineEmptyModel(t *testing.T) {
	m := &fakmodel.Model{}
	line := serveLMHeadReadyLine(m)
	if !strings.HasPrefix(line, "lm_head=") || !strings.Contains(line, "resident_total=") {
		t.Fatalf("ready line = %q, want lm_head=<route> ... resident_total=", line)
	}
	if got := residentWeightsLiveView(m, nil); got != nil {
		t.Fatalf("missing startup snapshot must omit resident_weights (no fake zeros), got %+v", got)
	}
	if got := residentWeightsLiveView(m, m.ResidentReport()); got != nil {
		t.Fatalf("empty model must omit resident_weights (no fake zeros), got %+v", got)
	}
}

func TestResidentWeightsLiveViewUsesStartupBytesAndLiveRoute(t *testing.T) {
	m := &fakmodel.Model{}
	startup := &fakmodel.ResidentReport{TotalResidentBytes: 42, Q6KEmbedBytes: 7, LMHead: "stale-route"}
	got := residentWeightsLiveView(m, startup)
	if got == nil || got.TotalResidentBytes != 42 || got.Q6KEmbedBytes != 7 {
		t.Fatalf("resident_weights = %+v, want the startup byte split", got)
	}
	if got.LMHead != m.LMHeadRoute() {
		t.Fatalf("lm_head = %q, want the live route %q", got.LMHead, m.LMHeadRoute())
	}
}
