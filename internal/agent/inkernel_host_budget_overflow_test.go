package agent

import (
	"context"
	"errors"
	"math"
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// The request is priced through the actual pre-allocation host admission seam.
// Large token COUNTS never become token slices, sessions, or KV allocations.
// fak-test:runtime medium est=2s lane=default
func TestInKernelHostDemandOverflow(t *testing.T) {
	if strconv.IntSize != 64 {
		t.Skip("the overflow witness uses the serving 64-bit int contract")
	}
	for _, name := range []string{"session", "retained", "combined", "panel", "width", "ordinary", "unknown", "unarmed"} {
		t.Run(name, func(t *testing.T) {
			cfg := tinyConcurrencyConfig()
			m := model.NewSynthetic(cfg)
			t.Cleanup(func() {
				if err := m.CloseWeights(); err != nil {
					t.Error(err)
				}
			})
			// Metadata declares a large window; model allocations above remain tiny.
			m.Cfg.MaxPositionEmbeddings = math.MaxInt
			p := &InKernelPlanner{m: m, modelID: "host-demand-overflow"}
			prompt, maxNew := 1, math.MaxInt/2
			ceiling := int64(1 << 30)
			known := true
			switch name {
			case "retained", "combined":
				p.tree = radixkv.New(1)
				perToken := compute.EstimateKVStoreBytes(m.Cfg.ContextSizeConfig().KV, 1)
				if perToken <= 0 {
					t.Fatal("fixture must have positive per-token demand")
				}
				if name == "retained" {
					maxNew = int(math.MaxInt64 / perToken / 2)
					ceiling = math.MaxInt64 / 4
				} else {
					maxNew = int(math.MaxInt64/perToken/3) + 1
					ceiling = math.MaxInt64 - 1
				}
			case "panel":
				prompt, maxNew = math.MaxInt/2, 0
			case "width":
				m.Cfg.HiddenSize = math.MaxInt
				prompt, maxNew = 1, 1
			case "ordinary":
				prompt, maxNew = 2, 2
				p.tree = radixkv.New(1)
			case "unknown":
				known = false
			}
			if err := p.refuseContextLength(prompt, maxNew); err != nil {
				t.Fatalf("fixture must reach capacity admission: %v", err)
			}
			if name != "unarmed" {
				p.SetHostMemoryBudget(ceiling, func() (int64, bool) { return 0, known })
			}
			session, retained, plan := p.hostRequestDemand(prompt, maxNew)
			ctx, release, err := p.admitHostMemory(context.Background(), prompt, maxNew)
			defer release()
			switch name {
			case "session", "panel", "width", "retained":
				var capacity *InKernelCapacityError
				if !errors.As(err, &capacity) || capacity.Want <= 0 || capacity.Want != session || capacity.Avail != ceiling || capacity.Scope != compute.MemoryScopeHost || capacity.Site != "host-memory-precheck" || hostReserved(p) != 0 {
					t.Fatalf("host admission err=%v session=%d retained=%d reserved=%d; want typed capacity refusal before allocation", err, session, retained, hostReserved(p))
				}
				if name == "retained" {
					if retained != math.MaxInt64 {
						t.Fatalf("retained=%d, want saturated demand", retained)
					}
				} else {
					if plan.Total() != math.MaxInt64 || session != math.MaxInt64 {
						t.Fatalf("plan=%d session=%d, want saturated demand", plan.Total(), session)
					}
				}
				if (name == "panel" || name == "width") && p.hostPrefillPanelBytes(prompt) != math.MaxInt64 {
					t.Fatal("panel demand must saturate")
				}
			case "combined":
				if err != nil || session <= 0 || retained != math.MaxInt64 || !inKernelSkipPrefixAdmission(ctx) || hostReserved(p) != session {
					t.Fatalf("combined err=%v session=%d retained=%d reserved=%d skip=%v", err, session, retained, hostReserved(p), inKernelSkipPrefixAdmission(ctx))
				}
			case "ordinary":
				if err != nil || session <= 0 || retained <= 0 || session+retained > ceiling || hostReserved(p) != session+retained || inKernelSkipPrefixAdmission(ctx) {
					t.Fatalf("ordinary err=%v session=%d retained=%d reserved=%d", err, session, retained, hostReserved(p))
				}
			default:
				if err != nil || hostReserved(p) != 0 {
					t.Fatalf("unchanged %s policy err=%v reserved=%d", name, err, hostReserved(p))
				}
			}
			release()
			release()
			if hostReserved(p) != 0 {
				t.Fatalf("release left reserved=%d", hostReserved(p))
			}
		})
	}
}
