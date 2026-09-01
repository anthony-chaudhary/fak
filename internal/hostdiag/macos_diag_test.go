package hostdiag

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"
)

func TestParseMacOSResourceIncidentProjectsWitnessedFieldsAndBoundedProvenance(t *testing.T) {
	data, err := os.ReadFile("testdata/macos-resource-incident.diag")
	if err != nil {
		t.Fatal(err)
	}
	event, err := ParseMacOSResourceIncident(`/private/users/example/fak_2026-08-25_private-host.diag`, data)
	if err != nil {
		t.Fatal(err)
	}
	start, _ := time.Parse(macOSDiagTimestampLayout, "2026-08-25 14:35:52.769 -0700")
	end, _ := time.Parse(macOSDiagTimestampLayout, "2026-08-25 14:46:34.171 -0700")
	if event.TimeMS != end.UTC().UnixMilli() ||
		event.Source != MacOSDiagnosticReportsSource ||
		event.Name != MacOSResourceIncidentEventName ||
		event.EventID != 0 ||
		event.Fault != nil ||
		event.Hang != nil ||
		event.ResourceIncident == nil {
		t.Fatalf("event = %+v", event)
	}
	incident := event.ResourceIncident
	if incident.IncidentType != MacOSResourceIncidentDiskWrites ||
		incident.ReportStartMS != start.UTC().UnixMilli() ||
		incident.ReportEndMS != end.UTC().UnixMilli() ||
		incident.Classification != "disk writes" ||
		incident.ActionTaken != "none" ||
		incident.DirtiedMB != 8589.95 ||
		incident.DurationSeconds != 641 ||
		incident.AverageMBPerSecond != 13.39 ||
		incident.Process != "fak" ||
		incident.PID != 20870 ||
		incident.FootprintMB != 16.36 ||
		incident.BinaryUUID != "7D5CFA95-4B8D-3CEF-DD58-D6BC242B7AA1" ||
		incident.SampledStackEnd != "write(2)" {
		t.Fatalf("incident = %+v", incident)
	}
	sum := sha256.Sum256(data)
	if incident.Artifact.Basename != "macos-resource-incident.diag" ||
		incident.Artifact.ByteCount != int64(len(data)) ||
		incident.Artifact.SHA256 != "sha256:"+hex.EncodeToString(sum[:]) {
		t.Fatalf("artifact = %+v", incident.Artifact)
	}
	encoded, err := json.Marshal(event)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{
		`"windows_event_id"`, `"application_fault"`, `"application_hang"`,
		"private/users", "private-host", "worker_loop", "flush_buffer", "system limit",
	} {
		if strings.Contains(string(encoded), forbidden) {
			t.Fatalf("normalized event retained %q: %s", forbidden, encoded)
		}
	}
}

func TestParseMacOSResourceIncidentRejectsMalformedReports(t *testing.T) {
	data, err := os.ReadFile("testdata/macos-resource-incident.diag")
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string][]byte{
		"missing action": []byte(strings.Replace(string(data), "Action taken:    none\n", "", 1)),
		"duplicate pid":  []byte(strings.Replace(string(data), "PID:             20870\n", "PID:             20870\nPID:             20870\n", 1)),
		"wrong event":    []byte(strings.Replace(string(data), "disk writes", "application hang", 1)),
		"wrong action":   []byte(strings.Replace(string(data), "Action taken:    none", "Action taken:    terminate", 1)),
		"bad uuid":       []byte(strings.Replace(string(data), "7D5CFA95-4B8D-3CEF-DD58-D6BC242B7AA1", "not-a-uuid", 1)),
		"wrong stack end": []byte(strings.Replace(
			string(data), "write + 8 (libsystem_kernel.dylib + 1234)", "close + 8 (libsystem_kernel.dylib + 1234)", 1,
		)),
		"duration mismatch": []byte(strings.Replace(string(data), "over 641 seconds", "over 300 seconds", 1)),
	}
	for name, fixture := range tests {
		t.Run(name, func(t *testing.T) {
			if event, err := ParseMacOSResourceIncident("macos-resource-incident.diag", fixture); err == nil {
				t.Fatalf("accepted malformed report: %+v", event)
			}
		})
	}
}

func TestParseMacOSResourceIncidentRejectsOversize(t *testing.T) {
	oversize := make([]byte, MacOSDiagFixtureMaxBytes+1)
	if _, err := ParseMacOSResourceIncident("macos-resource-incident.diag", oversize); err == nil ||
		!strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("oversize error = %v", err)
	}
}
