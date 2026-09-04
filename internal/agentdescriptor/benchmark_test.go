package agentdescriptor

import (
	"encoding/json"
	"testing"
)

var (
	benchDescriptorSink Descriptor
	benchBytesSink      []byte
	benchReceiptSink    OperationReceipt
)

func BenchmarkValidateDescriptor(b *testing.B) {
	descriptors := []Descriptor{
		New("macro:release-steward", "micro", "frontier", "f", 1, "single"),
		New("macro:planner", "macro", "frontier", "claude-3-7-sonnet", 4, "parallel"),
		New("worker:executor", "worker", "small", "qwen-3.8-7b", 16, "fanout"),
		New("sentinel:guard", "daemon", "fast", "local-guard", 1, "pipeline"),
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		d := descriptors[i%len(descriptors)]
		if err := d.Validate(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkParseDescriptor(b *testing.B) {
	samples := []Descriptor{
		New("macro:release-steward", "micro", "frontier", "f", 1, "single"),
		New("macro:planner", "macro", "frontier", "claude-3-7-sonnet", 4, "parallel"),
		New("worker:executor", "worker", "small", "qwen-3.8-7b", 16, "fanout"),
		New("sentinel:guard", "daemon", "fast", "local-guard", 1, "pipeline"),
	}

	payloads := make([][]byte, len(samples))
	for i, s := range samples {
		raw, err := s.Marshal()
		if err != nil {
			b.Fatalf("setup marshal failed: %v", err)
		}
		payloads[i] = raw
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		d, err := Decode(payloads[i%len(payloads)])
		if err != nil {
			b.Fatal(err)
		}
		benchDescriptorSink = d
	}
}

func BenchmarkMarshalDescriptor(b *testing.B) {
	descriptors := []Descriptor{
		New("macro:release-steward", "micro", "frontier", "f", 1, "single"),
		New("macro:planner", "macro", "frontier", "claude-3-7-sonnet", 4, "parallel"),
		New("worker:executor", "worker", "small", "qwen-3.8-7b", 16, "fanout"),
		New("sentinel:guard", "daemon", "fast", "local-guard", 1, "pipeline"),
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		raw, err := descriptors[i%len(descriptors)].Marshal()
		if err != nil {
			b.Fatal(err)
		}
		benchBytesSink = raw
	}
}

func BenchmarkNewDescriptor(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchDescriptorSink = New("macro:planner", "macro", "frontier", "claude-3-7-sonnet", 4, "parallel")
	}
}

func BenchmarkOperationReceiptMarshal(b *testing.B) {
	receipt := OperationReceipt{
		Schema:      Schema,
		OperationID: "op-benchmark-42",
		Descriptor:  New("macro:release-steward", "micro", "frontier", "f", 1, "single"),
		RouteRule:   "fast",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		raw, err := json.Marshal(receipt)
		if err != nil {
			b.Fatal(err)
		}
		benchBytesSink = raw
	}
}

func BenchmarkOperationReceiptUnmarshal(b *testing.B) {
	receipt := OperationReceipt{
		Schema:      Schema,
		OperationID: "op-benchmark-42",
		Descriptor:  New("macro:release-steward", "micro", "frontier", "f", 1, "single"),
		RouteRule:   "fast",
	}
	raw, err := json.Marshal(receipt)
	if err != nil {
		b.Fatalf("setup marshal failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var got OperationReceipt
		if err := json.Unmarshal(raw, &got); err != nil {
			b.Fatal(err)
		}
		benchReceiptSink = got
	}
}
