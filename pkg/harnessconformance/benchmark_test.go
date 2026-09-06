package harnessconformance_test

import (
	"encoding/json"
	"testing"

	hc "github.com/anthony-chaudhary/fak/pkg/harnessconformance"
)

var (
	sinkCert hc.Certificate
	sinkErr  error
	sinkJSON []byte
)

func BenchmarkProbeSuite(b *testing.B) {
	full := adapter{caps: all()}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkCert = hc.Run(full)
	}
}

func BenchmarkProbeSuiteVariations(b *testing.B) {
	b.Run("FullPass", func(b *testing.B) {
		a := adapter{caps: all()}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkCert = hc.Run(a)
		}
	})
	b.Run("PartialSkip", func(b *testing.B) {
		a := adapter{caps: all(), skip: hc.Backpressure}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkCert = hc.Run(a)
		}
	})
	b.Run("Failure", func(b *testing.B) {
		a := adapter{caps: all(), broken: hc.Policy}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkCert = hc.Run(a)
		}
	})
}

func BenchmarkEvaluateCertificate(b *testing.B) {
	cert := hc.Run(adapter{caps: all()})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkErr = cert.Validate()
	}
}

func BenchmarkEvaluateCertificateInvalid(b *testing.B) {
	cert := hc.Run(adapter{caps: all(), broken: hc.Lifecycle})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkErr = cert.Validate()
	}
}

func BenchmarkCertificateJSONMarshal(b *testing.B) {
	cert := hc.Run(adapter{caps: all()})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var err error
		sinkJSON, err = cert.JSON()
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCertificateJSONUnmarshal(b *testing.B) {
	cert := hc.Run(adapter{caps: all()})
	data, err := cert.JSON()
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var decoded hc.Certificate
		if err := json.Unmarshal(data, &decoded); err != nil {
			b.Fatal(err)
		}
		sinkCert = decoded
	}
}
