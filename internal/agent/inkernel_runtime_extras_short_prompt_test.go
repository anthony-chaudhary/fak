package agent

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestQwen35RuntimeExtrasShortRequestIsAdmitted reproduces the strix2 Qwen3.8-27B Vulkan
// regression: with the default 512-token prefill chunk, every request whose P+O is below the
// chunk (the one-token serve warmup is ~10 prompt tokens + 1) tripped the runtime-extras
// estimator's panel<=planned invariant and was refused as runtime-extras-unknown ("plan needs
// 0 bytes, available budget is 0 bytes"), so `fak serve` never became ready. A prompt shorter
// than the chunk prefills in one pass of len(prompt) rows, so the panel must be priced at
// min(chunk, P+O) — never refused.
func TestQwen35RuntimeExtrasShortRequestIsAdmitted(t *testing.T) {
	for _, tc := range []struct {
		name          string
		prompt, out   int
		mtp           bool
		wantPanelRows int
	}{
		{name: "serve-warmup", prompt: 12, out: 1, wantPanelRows: 13},
		{name: "short-chat", prompt: 200, out: 100, wantPanelRows: 300},
		{name: "short-chat-mtp-history", prompt: 200, out: 100, mtp: true, wantPanelRows: 300},
		{name: "long-prompt-keeps-chunk", prompt: 4096, out: 256, wantPanelRows: 512},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Default production chunk (512) on a Vulkan-like backend: device memory known
			// and the Qwen3.5 sequence-prefill path advertised (no GPU needed).
			p := qwen35RuntimeExtraPlanner(512)
			if tc.mtp {
				p.vulkanMTP = true
				p.speculativeEngine = model.NewSpeculativeEngine(nil, nil, model.SpeculativeEngineConfig{})
			}
			p.backend = runtimeExtraBackend{Backend: compute.Default(), total: 1 << 40, free: 1 << 40, known: true}

			if err := p.refuseOversizeRequest(tc.prompt, tc.out); err != nil {
				t.Fatalf("refuseOversizeRequest(%d,%d) on a roomy device = %v, want admitted", tc.prompt, tc.out, err)
			}
			plan, err := p.requestMemoryPlanWithExtras(tc.prompt, tc.out, p.requestMTPHistoryEligible(tc.out))
			if err != nil {
				t.Fatalf("requestMemoryPlanWithExtras(%d,%d) = %v, want a bounded plan", tc.prompt, tc.out, err)
			}
			// The panel term is the excess of min(chunk, P+O) rows over already-priced
			// transient scratch: it may be zero (fully covered) but never above that peak.
			peak := int64(tc.wantPanelRows) * int64(p.m.Cfg.HiddenSize) * 4
			for _, d := range plan {
				if d.Detail == "qwen35-vulkan-prefill-panel-additional" && d.Bytes > peak {
					t.Fatalf("panel term %d bytes exceeds the %d-row peak %d bytes", d.Bytes, tc.wantPanelRows, peak)
				}
			}
			if tc.mtp {
				found := false
				for _, d := range plan {
					found = found || d.Detail == "qwen35-mtp-retained-hidden-history"
				}
				if !found {
					t.Fatalf("MTP-eligible short request lost its retained-history reservation: %#v", plan)
				}
			}
		})
	}
}

// TestInKernelCapacityErrorRuntimeExtrasUnknownNamesSite pins that a runtime-extras-unknown
// refusal names its site instead of the misleading "plan needs 0 bytes, available budget is 0
// bytes" wording, while keeping the stable refusal prefix callers match on.
func TestInKernelCapacityErrorRuntimeExtrasUnknownNamesSite(t *testing.T) {
	e := &InKernelCapacityError{Class: compute.MemoryUnknown, Scope: compute.MemoryScopeDevice, Site: "runtime-extras-unknown"}
	msg := e.Error()
	if !strings.HasPrefix(msg, "in-kernel GPU capacity precheck refused request (") {
		t.Fatalf("Error() = %q, want the stable refusal prefix", msg)
	}
	if !strings.Contains(msg, "runtime-extras-unknown") {
		t.Fatalf("Error() = %q, want the runtime-extras-unknown site named", msg)
	}
	if strings.Contains(msg, "needs 0 bytes") {
		t.Fatalf("Error() = %q must not report a fake zero-byte demand", msg)
	}
	var capErr *InKernelCapacityError
	if !errors.As(error(e), &capErr) {
		t.Fatalf("errors.As lost the typed capacity error")
	}
}
