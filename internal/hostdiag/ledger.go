package hostdiag

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"
)

func LoadCensus(path string) ([]ProcessSample, error) {
	file, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer file.Close()
	var samples []ProcessSample
	scan := bufio.NewScanner(file)
	scan.Buffer(make([]byte, 4096), 1<<20)
	for scan.Scan() {
		var envelope struct {
			Schema string `json:"schema"`
		}
		if err := json.Unmarshal(scan.Bytes(), &envelope); err != nil {
			return nil, fmt.Errorf("parse hostdiag ledger: %w", err)
		}
		if envelope.Schema != CensusSchema {
			continue
		}
		var sample ProcessSample
		if err := json.Unmarshal(scan.Bytes(), &sample); err != nil || sample.SampleID == "" || sample.PID <= 0 || sample.SampledAtMS <= 0 || sample.ProcessStartMS <= 0 {
			return nil, fmt.Errorf("invalid hostdiag census row")
		}
		samples = append(samples, sample)
	}
	if err := scan.Err(); err != nil {
		return nil, err
	}
	return samples, nil
}

func CorrelateAll(events []ResourceEvent, samples []ProcessSample) []Correlation {
	out := make([]Correlation, 0, len(events))
	for _, event := range events {
		if correlation, ok := Correlate(event, samples); ok {
			out = append(out, correlation)
		}
	}
	return out
}

func NewResourceEvent(at time.Time, source string, eventID int, recordID, name, reportID, app, message string) ResourceEvent {
	return ResourceEvent{TimeMS: at.UTC().UnixMilli(), Source: source, EventID: eventID, RecordID: recordID, Name: name, ReportID: reportID, App: app, Message: message}
}
