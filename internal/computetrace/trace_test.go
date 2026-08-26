package computetrace

import (
	"bytes"
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
