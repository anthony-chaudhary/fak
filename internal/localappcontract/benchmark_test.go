package localappcontract

import "testing"

// BenchmarkLocalAppContract measures the composite contract pipeline:
// receipt validation and serialization, compatibility checks, sensitive field detection, and event replay.
func BenchmarkLocalAppContract(b *testing.B) {
	receipt := Receipt{
		Schema:           Schema,
		TaskID:           "task-bench-42",
		Engine:           "fak-native",
		Location:         "local",
		Revision:         "r100",
		AdmittedEnvelope: map[string]int64{"tokens": 4096, "ms": 500},
		ObservedEnvelope: map[string]int64{"tokens": 1280, "ms": 142},
		Quality:          "high",
		Attempts:         1,
		Authority:        "app-supervisor",
		Terminal:         Completed,
	}

	oldContract := []byte(`{"schema":"fak.local-app-contract/1","revision":"r1","engine":"fak-native"}`)
	newContract := []byte(`{"schema":"fak.local-app-contract/1","revision":"r2","engine":"fak-native","tasks":["t1"]}`)

	events := []Event{
		{Schema: Schema, Sequence: 1, TaskID: "task-bench-42", Kind: "admitted"},
		{Schema: Schema, Sequence: 2, TaskID: "task-bench-42", Kind: "executing"},
		{Schema: Schema, Sequence: 3, TaskID: "task-bench-42", Kind: "completed"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		raw, err := receipt.Marshal()
		if err != nil {
			b.Fatalf("receipt marshal failed: %v", err)
		}
		if ContainsSensitiveField(raw) {
			b.Fatal("unexpected sensitive field in receipt")
		}
		if err := CheckAdditiveCompatibility(oldContract, newContract); err != nil {
			b.Fatalf("compatibility check failed: %v", err)
		}
		if err := Replay(events); err != nil {
			b.Fatalf("replay failed: %v", err)
		}
	}
}

// BenchmarkReceiptMarshal measures validation and serialization throughput of receipts.
func BenchmarkReceiptMarshal(b *testing.B) {
	receipt := Receipt{
		Schema:           Schema,
		TaskID:           "task-bench-42",
		Engine:           "fak-native",
		Location:         "local",
		Revision:         "r100",
		AdmittedEnvelope: map[string]int64{"tokens": 4096},
		ObservedEnvelope: map[string]int64{"tokens": 1280},
		Quality:          "high",
		Attempts:         1,
		Authority:        "app-supervisor",
		Terminal:         Completed,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := receipt.Marshal(); err != nil {
			b.Fatalf("marshal failed: %v", err)
		}
	}
}

// BenchmarkCheckAdditiveCompatibility measures schema field retention validation.
func BenchmarkCheckAdditiveCompatibility(b *testing.B) {
	oldRaw := []byte(`{"schema":"v1","revision":"r1","engine":"fak-native","tasks":["t1"]}`)
	newRaw := []byte(`{"schema":"v1","revision":"r2","engine":"fak-native","tasks":["t1"],"vcache":true}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := CheckAdditiveCompatibility(oldRaw, newRaw); err != nil {
			b.Fatalf("compatibility failed: %v", err)
		}
	}
}

// BenchmarkContainsSensitiveField measures inspection throughput for sensitive key patterns.
func BenchmarkContainsSensitiveField(b *testing.B) {
	payload := []byte(`{"schema":"fak.local-app-contract/1","task_id":"t1","metrics":{"tokens":120},"status":"ok"}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if ContainsSensitiveField(payload) {
			b.Fatal("unexpected sensitive match")
		}
	}
}

// BenchmarkReplay measures monotonicity and schema verification for event streams.
func BenchmarkReplay(b *testing.B) {
	events := make([]Event, 16)
	for i := range events {
		events[i] = Event{
			Schema:   Schema,
			Sequence: uint64(i + 1),
			TaskID:   "task-1",
			Kind:     "step",
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Replay(events); err != nil {
			b.Fatalf("replay failed: %v", err)
		}
	}
}
