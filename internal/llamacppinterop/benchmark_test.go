package llamacppinterop

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/quantmeta"
)

var (
	benchResultSink Result
	benchBoolSink   bool
)

type benchRunner struct {
	versionOut []byte
	helpOut    []byte
	devicesOut []byte
}

func (r benchRunner) Output(_ context.Context, _ string, args ...string) ([]byte, error) {
	if len(args) > 0 {
		switch args[0] {
		case "--help":
			return r.helpOut, nil
		case "--list-devices":
			return r.devicesOut, nil
		}
	}
	return r.versionOut, nil
}

func BenchmarkLlamaCppInterop(b *testing.B) {
	ctx := context.Background()
	runner := benchRunner{
		versionOut: []byte("llama-server version: 0.0.6123 (commit 8144f31)"),
		helpOut:    []byte("--spec-type none,draft-mtp"),
		devicesOut: []byte("CUDA0: NVIDIA A100"),
	}
	desc := quantmeta.Descriptor{
		Artifact: &quantmeta.ArtifactSpec{ContainerID: "gguf"},
		Extra: map[string]json.RawMessage{
			"gguf_architecture": json.RawMessage(`"qwen35"`),
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		discovered := Discover(ctx, runner, "llama-server")
		planned := PlanQwen38MTP(discovered.Capability, "models/qwen38-mtp.gguf", desc, 18080, 4096)
		witnessed := WitnessedQwen38MTP(discovered.Capability)

		benchResultSink = planned
		benchBoolSink = witnessed
	}
}

func BenchmarkDiscover(b *testing.B) {
	ctx := context.Background()
	runner := benchRunner{
		versionOut: []byte("llama-server version: 0.0.6123 (commit 8144f31)"),
		helpOut:    []byte("--spec-type none,draft-mtp"),
		devicesOut: []byte("CUDA0: NVIDIA A100"),
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchResultSink = Discover(ctx, runner, "llama-server")
	}
}

func BenchmarkPlan(b *testing.B) {
	cap := Capability{
		Binary:  "llama-server",
		Version: "0.0.6123",
		Commit:  "8144f31",
		Server:  true,
	}
	desc := quantmeta.Descriptor{
		Artifact: &quantmeta.ArtifactSpec{ContainerID: "gguf"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchResultSink = Plan(cap, "models/base.gguf", desc)
	}
}

func BenchmarkPlanQwen38MTP(b *testing.B) {
	cap := Capability{
		Binary:   "llama-server",
		Version:  "0.0.6123",
		Commit:   "8144f31",
		Server:   true,
		DraftMTP: true,
		CUDA:     true,
	}
	desc := quantmeta.Descriptor{
		Artifact: &quantmeta.ArtifactSpec{ContainerID: "gguf"},
		Extra: map[string]json.RawMessage{
			"gguf_architecture": json.RawMessage(`"qwen35"`),
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchResultSink = PlanQwen38MTP(cap, "models/qwen38-mtp.gguf", desc, 18080, 4096)
	}
}

func BenchmarkWitnessedQwen38MTP(b *testing.B) {
	cap := Capability{
		Commit:   "8144f3192e5a3131cd043f284525e6ceebf82d0f",
		Server:   true,
		DraftMTP: true,
		CUDA:     true,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBoolSink = WitnessedQwen38MTP(cap)
	}
}

func BenchmarkCheckHealth(b *testing.B) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ready"}`))
	}))
	defer server.Close()

	client := server.Client()
	ctx := context.Background()
	url := server.URL

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchResultSink = CheckHealth(ctx, client, url)
	}
}
