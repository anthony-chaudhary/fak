package metrics

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func dwellAt(second int) time.Time { return time.Date(2026, 8, 8, 12, 0, second, 0, time.UTC) }

func TestMeasureOperatorDwellBanksFocusBoundariesExactlyOnce(t *testing.T) {
	got := MeasureOperatorDwell([]DwellObservation{
		{At: dwellAt(0), Visible: true, Focused: true},
		{At: dwellAt(5), Visible: true, Focused: false},
		{At: dwellAt(5), Visible: false, Focused: false},
		{At: dwellAt(9), Visible: true, Focused: false},
		{At: dwellAt(10), Visible: true, Focused: true},
		{At: dwellAt(13), Visible: false, Focused: true},
	})
	if !got.Accepted || !got.Complete || got.ObservableFocusNanos != int64(8*time.Second) {
		t.Fatalf("result = %+v, want accepted complete 8s observable focus", got)
	}
	if err := ValidateOperatorDwellResult(got); err != nil {
		t.Fatal(err)
	}
}

func TestMeasureOperatorDwellMarksOpenIntervalIncomplete(t *testing.T) {
	got := MeasureOperatorDwell([]DwellObservation{{At: dwellAt(0)}, {At: dwellAt(1), Visible: true, Focused: true}})
	if !got.Accepted || got.Complete {
		t.Fatalf("result = %+v, want accepted but visibly incomplete", got)
	}
}

func TestMeasureOperatorDwellTypedRefusals(t *testing.T) {
	tests := []struct {
		name         string
		observations []DwellObservation
		want         DwellRefusal
	}{
		{"absent", nil, DwellRefusalNoObservations},
		{"zero timestamp", []DwellObservation{{Visible: true, Focused: true}}, DwellRefusalInvalidTimestamp},
		{"time reverses", []DwellObservation{{At: dwellAt(2)}, {At: dwellAt(1)}}, DwellRefusalNonMonotonic},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MeasureOperatorDwell(tt.observations)
			if got.Accepted || got.Refusal != tt.want {
				t.Fatalf("result = %+v, want refusal %q", got, tt.want)
			}
		})
	}
}

func TestOperatorDwellCommittedReceipts(t *testing.T) {
	for _, name := range []string{"operator_dwell_accepted.json", "operator_dwell_refused.json"} {
		data, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		var receipt OperatorDwellResult
		if err := json.Unmarshal(data, &receipt); err != nil {
			t.Fatal(err)
		}
		if err := ValidateOperatorDwellResult(receipt); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
	}
}
