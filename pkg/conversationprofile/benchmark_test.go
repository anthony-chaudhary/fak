package conversationprofile_test

import (
	"context"
	"encoding/json"
	"testing"

	cp "github.com/anthony-chaudhary/fak/pkg/conversationprofile"
	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

var (
	benchSinkProfile cp.Profile
	benchSinkReceipt cp.Receipt
	benchSinkBinding cp.Binding
	benchSinkBool    bool
	benchSinkErr     error
	benchSinkBytes   []byte
)

type benchService struct {
	result harnesskit.Result
}

func (s benchService) Invoke(_ context.Context, _ harnesskit.Invocation) (harnesskit.Result, error) {
	return s.result, nil
}

func BenchmarkProfileValidation(b *testing.B) {
	rawValid := []byte(`{
		"schema": "fak.conversation-profile/v1",
		"id": "support-default",
		"settings": {
			"response.detail": {"value": "brief", "fidelity": "required"},
			"interaction.questions": {"value": "when_blocked", "fidelity": "required"},
			"tone": {"value": "warm", "fidelity": "optional"}
		}
	}`)

	b.ReportAllocs()
	b.SetBytes(int64(len(rawValid)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p, err := cp.Parse(rawValid)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkProfile = p
	}
}

func BenchmarkProfileValidationVariations(b *testing.B) {
	valid := []byte(`{"schema":"fak.conversation-profile/v1","id":"bench","settings":{"response.detail":{"value":"brief","fidelity":"required"}}}`)
	badSchema := []byte(`{"schema":"wrong","id":"bench","settings":{"response.detail":{"value":"brief","fidelity":"required"}}}`)
	badValue := []byte(`{"schema":"fak.conversation-profile/v1","id":"bench","settings":{"response.detail":{"value":"unknown","fidelity":"required"}}}`)

	b.Run("ValidMinimal", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(valid)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			p, err := cp.Parse(valid)
			if err != nil {
				b.Fatal(err)
			}
			benchSinkProfile = p
		}
	})

	b.Run("BadSchema", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(badSchema)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := cp.Parse(badSchema)
			benchSinkErr = err
		}
	})

	b.Run("BadValue", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(badValue)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := cp.Parse(badValue)
			benchSinkErr = err
		}
	})
}

func BenchmarkAdapterResolve(b *testing.B) {
	supported := map[string]bool{
		"response.detail=brief":              true,
		"interaction.questions=when_blocked": true,
		"tone=warm":                          true,
	}
	adapter := mapAdapter{name: "bench-adapter", prefix: "prompt", supported: supported}

	b.Run("Match", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			binding, ok := adapter.Resolve("response.detail", "brief")
			benchSinkBinding = binding
			benchSinkBool = ok
		}
	})

	b.Run("Miss", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			binding, ok := adapter.Resolve("unsupported.key", "value")
			benchSinkBinding = binding
			benchSinkBool = ok
		}
	})
}

func BenchmarkProfileRun(b *testing.B) {
	p, err := cp.Parse([]byte(portable))
	if err != nil {
		b.Fatal(err)
	}

	supported := map[string]bool{
		"response.detail=brief":              true,
		"interaction.questions=when_blocked": true,
		"tone=warm":                          true,
	}
	fullAdapter := mapAdapter{name: "bench-full", prefix: "bench", supported: supported}
	partialAdapter := mapAdapter{name: "bench-partial", prefix: "bench", supported: map[string]bool{
		"response.detail=brief":              true,
		"interaction.questions=when_blocked": true,
	}}
	refusalAdapter := mapAdapter{name: "bench-refuse", prefix: "bench", supported: map[string]bool{
		"tone=warm": true,
	}}

	ctx := context.Background()
	svc := benchService{result: harnesskit.Result{Content: []byte(`{"applied":{"response.detail":"brief"}}`)}}

	b.Run("FullMatch", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			r, err := cp.Run(ctx, p, fullAdapter, svc)
			if err != nil {
				b.Fatal(err)
			}
			benchSinkReceipt = r
		}
	})

	b.Run("OptionalGap", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			r, err := cp.Run(ctx, p, partialAdapter, svc)
			if err != nil {
				b.Fatal(err)
			}
			benchSinkReceipt = r
		}
	})

	b.Run("RequiredRefusal", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := cp.Run(ctx, p, refusalAdapter, svc)
			benchSinkErr = err
		}
	})
}

func BenchmarkReceiptJSONMarshal(b *testing.B) {
	p, err := cp.Parse([]byte(portable))
	if err != nil {
		b.Fatal(err)
	}
	supported := map[string]bool{
		"response.detail=brief":              true,
		"interaction.questions=when_blocked": true,
		"tone=warm":                          true,
	}
	adapter := mapAdapter{name: "bench-full", prefix: "bench", supported: supported}
	svc := benchService{result: harnesskit.Result{Content: []byte(`{"applied":{"response.detail":"brief"}}`)}}
	receipt, err := cp.Run(context.Background(), p, adapter, svc)
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, err := json.Marshal(receipt)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkBytes = data
	}
}

func BenchmarkDirectConfigMarshal(b *testing.B) {
	cfg := cp.DirectConfig{
		Adapter: "runtime-controls",
		Raw:     json.RawMessage(`{"detail":1,"mode":"strict"}`),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, err := json.Marshal(cfg)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkBytes = data
	}
}

func TestBenchmarkOperationsSanity(t *testing.T) {
	rawValid := []byte(`{
		"schema": "fak.conversation-profile/v1",
		"id": "support-default",
		"settings": {
			"response.detail": {"value": "brief", "fidelity": "required"}
		}
	}`)
	p, err := cp.Parse(rawValid)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	adapter := mapAdapter{
		name:   "sanity",
		prefix: "prompt",
		supported: map[string]bool{
			"response.detail=brief": true,
		},
	}
	b, ok := adapter.Resolve("response.detail", "brief")
	if !ok || b.Key != "response.detail" {
		t.Fatalf("resolve failed: got %+v, ok=%v", b, ok)
	}

	svc := benchService{result: harnesskit.Result{Content: []byte(`{"applied":{"response.detail":"brief"}}`)}}
	r, err := cp.Run(context.Background(), p, adapter, svc)
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}
	if len(r.Bindings) != 1 || r.Adapter != "sanity" {
		t.Fatalf("unexpected receipt: %+v", r)
	}

	cfg := cp.DirectConfig{
		Adapter: "runtime-controls",
		Raw:     json.RawMessage(`{"detail":1}`),
	}
	data, err := json.Marshal(cfg)
	if err != nil || len(data) == 0 {
		t.Fatalf("direct config marshal failed: %v", err)
	}
}
