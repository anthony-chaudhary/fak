package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

type deviceOperationRecordingBackend struct {
	compute.Backend
	uploads int
}

func (b *deviceOperationRecordingBackend) Name() string { return "cpu-policy-recording" }
func (b *deviceOperationRecordingBackend) Caps() compute.Caps {
	return compute.Caps{DeviceMemory: true, UploadDtype: true}
}
func (b *deviceOperationRecordingBackend) Upload(t compute.Tensor, dtype compute.Dtype) compute.Tensor {
	b.uploads++
	return b.Backend.Upload(t, dtype)
}

// SW-VERIFIED only: the actual native planner must report its existing architecture
// policy refusal; this synthetic CPU fixture does not qualify DeepSeek device compute.
// fak-test:runtime fast est=500ms
func TestDeviceOnlyExpertRingNativePolicyFailureReachesDefaultPerfLedger(t *testing.T) {
	cfg := kvmmuSynthCfg()
	cfg.ModelType = "deepseek41"
	if !cfg.IsDeepSeekV41() {
		t.Fatal("fixture lost DeepSeek V4.1 architecture identity")
	}
	m := model.NewSynthetic(cfg)
	m.Quantize()
	be := &deviceOperationRecordingBackend{Backend: compute.Default()}
	const modelID = "synthetic-deepseek41-policy"
	p := agent.NewInKernelPlannerWithConfig(m, newByteLevelTokenizer(t), modelID, false, be, false,
		agent.InKernelPlannerConfig{RequireDeviceExecution: true})
	srv := newTestServerWithConfig(t, Config{EngineID: "test", Model: modelID, VDSO: true})
	srv.planner = p
	srv.servedSide = localitySelfHosted
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	client := ts.Client()
	client.Timeout = 2 * time.Second
	resp, err := client.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(
		`{"model":"`+modelID+`","messages":[{"role":"user","content":"hello"}],"max_tokens":1}`))
	if err != nil {
		t.Fatalf("native policy refusal dropped HTTP request: %T", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadGateway {
		t.Errorf("native policy failure HTTP status=%d, want %d", resp.StatusCode, http.StatusBadGateway)
	}
	resp, err = client.Get(ts.URL + "/v1/fak/perf/recent")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var report perfledger.Report
	if err := json.NewDecoder(resp.Body).Decode(&report); err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK || len(report.Records) != 1 || report.Summary.Errors != 1 {
		t.Fatalf("default recent status=%d records=%d errors=%d", resp.StatusCode, len(report.Records), report.Summary.Errors)
	}
	row := report.Records[0]
	if row.Model != modelID || row.Status != http.StatusBadGateway || row.Error != perfledger.ErrorUpstream || row.FinishReason != perfledger.FinishReasonError {
		t.Fatalf("policy refusal ledger identity/status/class/finish=%q/%d/%q/%q", row.Model, row.Status, row.Error, row.FinishReason)
	}
}
