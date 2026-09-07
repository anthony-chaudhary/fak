package configsurface

import (
	"encoding/json"
	"testing"
)

var (
	benchReportSink Report
	benchBytesSink  []byte
	benchErrSink    error
)

// BenchmarkAudit measures the complete config-surface discoverability and default coverage audit.
func BenchmarkAudit(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := Audit()
		if !r.Discoverable {
			b.Fatalf("Audit reported undiscoverable surface: %+v", r.Findings)
		}
		benchReportSink = r
	}
}

// BenchmarkReportCheck_Discoverable measures the zero-error fast path for discoverable reports.
func BenchmarkReportCheck_Discoverable(b *testing.B) {
	report := Audit()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := report.Check(); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkReportCheck_WithFindings measures error generation when findings are present.
func BenchmarkReportCheck_WithFindings(b *testing.B) {
	report := Report{
		Discoverable: false,
		Findings: []Finding{
			{Key: "auth.unspecified_secret", Reason: "missing built-in default"},
			{Reason: "config key budget exceeded: 34 > 33"},
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := report.Check()
		if err == nil {
			b.Fatal("expected non-nil error")
		}
		benchErrSink = err
	}
}

// BenchmarkReportJSONMarshal measures JSON serialization of an audit report (fak config audit --json).
func BenchmarkReportJSONMarshal(b *testing.B) {
	report := Audit()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, err := json.Marshal(report)
		if err != nil {
			b.Fatal(err)
		}
		benchBytesSink = data
	}
}

// BenchmarkReportJSONUnmarshal measures JSON deserialization of an audit report.
func BenchmarkReportJSONUnmarshal(b *testing.B) {
	report := Audit()
	data, err := json.Marshal(report)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var decoded Report
		if err := json.Unmarshal(data, &decoded); err != nil {
			b.Fatal(err)
		}
		benchReportSink = decoded
	}
}
