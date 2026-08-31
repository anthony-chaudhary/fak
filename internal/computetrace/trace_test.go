package computetrace

import (
	"bytes"
	"encoding/json"
	"testing"
	"time"
)

func TestRecorderIsDisabledAndBounded(t *testing.T) {
	disabled := New(0)
	disabled.Record(Event{Operation: "matmul"})
	if got := disabled.Artifact(); len(got.Events) != 0 || got.Dropped != 0 {
		t.Fatalf("disabled recorder changed: %+v", got)
	}
	r := New(1)
	e := Event{RunID: "run", RequestID: "request", Operation: "matmul", Phase: "kernel", Backend: "cuda", Device: "cuda:0", Kernel: "sgemm", StartedAt: time.Unix(1, 2).UTC(), DurationNS: 3, TimerDomain: "cuda_event", Bytes: 4, Shapes: [][]int{{2, 2}}, ProvenanceDigest: Digest("cuda", "sgemm")}
	r.Record(e)
	r.Record(e)
	var b bytes.Buffer
	if err := r.Write(&b); err != nil {
		t.Fatal(err)
	}
	got, err := Read(&b)
	if err != nil {
		t.Fatal(err)
	}
	if got.Schema != Schema || len(got.Events) != 1 || got.Dropped != 1 || got.Events[0].Sequence != 1 || got.Events[0].TimerDomain != "cuda_event" {
		t.Fatalf("bad artifact: %+v", got)
	}
	if got.ObserverOverheadNS < 0 {
		t.Fatalf("negative overhead: %d", got.ObserverOverheadNS)
	}
}

func TestGlobalRecorderIsOptIn(t *testing.T) {
	Record(Event{Operation: "ignored"})
	r, disable := Enable(1, "run", "request")
	Record(Event{Operation: "matmul"})
	disable()
	got := r.Artifact()
	if len(got.Events) != 1 || got.Events[0].RunID != "run" || got.Events[0].RequestID != "request" {
		t.Fatalf("global recorder: %+v", got)
	}
}

func TestReadRejectsUnknownSchema(t *testing.T) {
	if _, err := Read(bytes.NewBufferString(`{"schema":"future"}`)); err == nil {
		t.Fatal("accepted unknown schema")
	}
}

func TestEventMetalFieldsJSONRoundTrip(t *testing.T) {
	want := Event{Route: "metal_command_buffer", DeviceDurationNS: 1234, InputDType: "f32", WeightDType: "f16", OutputDType: "f32", BytesRead: 4096, BytesWritten: 1024, EstimatedFLOPs: 8192}
	encoded, err := json.Marshal(want)
	if err != nil {
		t.Fatalf("marshal Metal trace fields: %v", err)
	}
	for _, field := range []string{`"route":"metal_command_buffer"`, `"device_duration_ns":1234`, `"input_dtype":"f32"`, `"weight_dtype":"f16"`, `"output_dtype":"f32"`, `"bytes_read":4096`, `"bytes_written":1024`, `"estimated_flops":8192`} {
		if !bytes.Contains(encoded, []byte(field)) {
			t.Fatalf("encoded event missing %s: %s", field, encoded)
		}
	}
	var got Event
	if err := json.Unmarshal(encoded, &got); err != nil {
		t.Fatalf("unmarshal Metal trace fields: %v", err)
	}
	if got.Route != want.Route || got.DeviceDurationNS != want.DeviceDurationNS || got.InputDType != want.InputDType || got.WeightDType != want.WeightDType || got.OutputDType != want.OutputDType || got.BytesRead != want.BytesRead || got.BytesWritten != want.BytesWritten || got.EstimatedFLOPs != want.EstimatedFLOPs {
		t.Fatalf("Metal trace fields changed after round trip: got %+v want %+v", got, want)
	}
	empty, err := json.Marshal(Event{})
	if err != nil {
		t.Fatalf("marshal empty event: %v", err)
	}
	for _, field := range []string{`"route"`, `"device_duration_ns"`, `"input_dtype"`, `"weight_dtype"`, `"output_dtype"`, `"bytes_read"`, `"bytes_written"`, `"estimated_flops"`} {
		if bytes.Contains(empty, []byte(field)) {
			t.Fatalf("omitempty field %s present in empty event: %s", field, empty)
		}
	}
}
