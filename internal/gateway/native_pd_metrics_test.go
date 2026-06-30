package gateway

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
)

func TestNativePDMetricsRenderIntoLiveMetrics(t *testing.T) {
	srv := newTestServer(t)
	if pre := srv.renderMetrics(); strings.Contains(pre, "fak_native_pd_") {
		t.Fatalf("native P/D metrics present before SetNativePDMetrics:\n%s", pre)
	}

	m := model.NewSynthetic(modelengine.SyntheticConfig())
	cluster := modelengine.NewNativePDCluster(m, 1)
	defer cluster.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	admit, err := cluster.Admit(ctx, modelengine.NativePDRequest{
		Call:             inlineGatewayCall("issue28_pd_metrics", `{"prompt":"shared"}`),
		Prompt:           []int{3, 1, 4, 1, 5},
		Taint:            abi.TaintTrusted,
		Scope:            abi.ScopeFleet,
		AdmissionVerdict: cachemeta.AdmissionAllow,
	})
	if err != nil {
		t.Fatalf("Admit native P/D: %v", err)
	}
	for range admit.Request.Tokens() {
	}
	if _, err := admit.Request.Result(); err != nil {
		t.Fatalf("native P/D result: %v", err)
	}

	srv.SetNativePDMetrics(cluster)
	out := srv.renderMetrics()
	for _, want := range []string{
		`# TYPE fak_native_pd_worker_requests_total counter`,
		`fak_native_pd_worker_requests_total{role="prefill",worker="prefill-0"} 1`,
		`fak_native_pd_worker_requests_total{role="decode",worker="decode-0"} 1`,
		`fak_native_pd_worker_bytes_moved_total{role="decode",worker="decode-0"}`,
		`fak_native_pd_route_total{result="cold_receive"} 1`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("live /metrics surface missing %q\n--- got ---\n%s", want, out)
		}
	}

	srv.SetNativePDMetrics(nil)
	if post := srv.renderMetrics(); strings.Contains(post, "fak_native_pd_") {
		t.Fatalf("native P/D metrics still present after detaching:\n%s", post)
	}
}

func inlineGatewayCall(tool, args string) *abi.ToolCall {
	return &abi.ToolCall{
		Tool: tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args), Len: int64(len(args))},
	}
}
