package providerjobaccounting

import (
	"bufio"
	"bytes"
	"encoding/json"
	"os"
	"testing"
)

func TestBenchmarkProviderJobAccountingSanity(t *testing.T) {
	fixturePath := docsPath("standards", "fixtures", "provider-job-accounting-gpt56-sol-api.jsonl")
	data, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	scanner := bufio.NewScanner(bytes.NewReader(data))
	var records []map[string]any
	for scanner.Scan() {
		var record map[string]any
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		records = append(records, record)
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scanner error: %v", err)
	}
	if err := validateLedger(records); err != nil {
		t.Fatalf("validate ledger error: %v", err)
	}
}

func BenchmarkProviderJobAccounting(b *testing.B) {
	fixturePath := docsPath("standards", "fixtures", "provider-job-accounting-gpt56-sol-api.jsonl")
	data, err := os.ReadFile(fixturePath)
	if err != nil {
		b.Fatalf("read fixture: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		scanner := bufio.NewScanner(bytes.NewReader(data))
		var records []map[string]any
		for scanner.Scan() {
			var record map[string]any
			if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
				b.Fatalf("unmarshal: %v", err)
			}
			records = append(records, record)
		}
		if err := scanner.Err(); err != nil {
			b.Fatalf("scanner error: %v", err)
		}
		if err := validateLedger(records); err != nil {
			b.Fatalf("validate ledger error: %v", err)
		}
	}
}
