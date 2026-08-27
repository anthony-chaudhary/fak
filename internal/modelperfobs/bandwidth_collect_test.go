package modelperfobs

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestCollectionBounds(t *testing.T) {
	base := CollectionOptions{Count: 1, Interval: 10 * time.Millisecond, Phase: PhaseDecode, Shape: ShapeSmall}
	if err := ValidateCollectionOptions(base); err != nil {
		t.Fatal(err)
	}
	base.Count = 121
	if err := ValidateCollectionOptions(base); err == nil {
		t.Fatal("expected count bound")
	}
	base.Count = 1
	base.Interval = time.Millisecond
	if err := ValidateCollectionOptions(base); err == nil {
		t.Fatal("expected interval bound")
	}
}
func TestCollectBandwidthPreservesUnavailableDRAM(t *testing.T) {
	r, err := CollectBandwidth(context.Background(), CollectionOptions{Count: 1, Interval: 10 * time.Millisecond, Phase: PhaseDecode, Shape: ShapeSmall, TheoreticalGBS: fp(100)})
	if err != nil {
		t.Fatal(err)
	}
	if r.Engine != "fak-native" || len(r.Report.Observations) != 1 {
		t.Fatalf("%+v", r)
	}
	if r.Availability.DRAMCounters || r.Report.Observations[0].Live.TotalGBS != nil {
		t.Fatal("host/process signals mislabeled as DRAM bandwidth")
	}
	b, _ := json.Marshal(r)
	if strings.Contains(string(b), `"total_gb_s":0`) {
		t.Fatal(string(b))
	}
	if r.Report.Observations[0].Rooflines.SelectedSource != "theoretical" {
		t.Fatal("roofline not selected")
	}
}
