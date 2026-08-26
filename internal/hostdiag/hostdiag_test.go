package hostdiag

import (
	"testing"
	"time"
)

func TestCorrelateHistoricalUnresolved(t *testing.T) {
	event := ResourceEvent{TimeMS: 1000, Source: "Windows Error Reporting", EventID: 1001, RecordID: "1", Name: "RADAR_PRE_LEAK_64", ReportID: "r", App: "fak.exe"}
	got, ok := Correlate(event, nil)
	if !ok || got.Status != "historical_unresolved" || !got.Observational || got.CorrelationID == "" {
		t.Fatalf("%+v ok=%v", got, ok)
	}
	again, _ := Correlate(event, nil)
	if again.CorrelationID != got.CorrelationID {
		t.Fatal("unstable identity")
	}
}

func TestCorrelateIdentifiedAndAmbiguous(t *testing.T) {
	at := time.UnixMilli(2000)
	one := NewProcessSample(at, 42, time.UnixMilli(500), `C:\bin\fak.exe`, "sha", "rev", "guard", "g1", 10, 20, 3, 4)
	event := ResourceEvent{TimeMS: 1000, EventID: 1001, Name: "RADAR_PRE_LEAK_64", App: "fak.exe"}
	got, _ := Correlate(event, []ProcessSample{one})
	if got.Status != "identified" || len(got.Candidates) != 1 || got.Candidates[0].CommandClass != "guard" {
		t.Fatalf("%+v", got)
	}
	two := one
	two.PID = 43
	got, _ = Correlate(event, []ProcessSample{one, two})
	if got.Status != "ambiguous" || len(got.Candidates) != 2 {
		t.Fatalf("%+v", got)
	}
}

func TestCorrelateRejectsUnrelated(t *testing.T) {
	for _, event := range []ResourceEvent{{TimeMS: 1, EventID: 1001, Name: "APPCRASH", App: "fak.exe"}, {TimeMS: 1, EventID: 1001, Name: "RADAR_PRE_LEAK_64", App: "other.exe"}, {Name: "RADAR_PRE_LEAK_64", App: "fak.exe"}} {
		if _, ok := Correlate(event, nil); ok {
			t.Fatalf("accepted %+v", event)
		}
	}
}

func TestClassifyCommandDoesNotRetainArguments(t *testing.T) {
	if got := ClassifyCommand(`C:\bin\fak.exe guard --api-key secret`); got != "guard" {
		t.Fatalf("got %q", got)
	}
	if got := ClassifyCommand(`C:\bin\fak.exe unusual --token secret`); got != "other" {
		t.Fatalf("got %q", got)
	}
}
