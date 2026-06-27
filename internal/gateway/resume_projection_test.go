package gateway

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

// newResumeProjServer returns a minimal Server carrying only the surfaces resume_projection.go
// touches: the self-contained resumeProj accumulator (zero-value ready) and a real metrics sink for
// the unrelated families. observeResumeProjection reads/writes ONLY s.resumeProj and logs via
// s.logf — it never touches the request, the response, the kernel, or any session state. That is the
// structural reason SHADOW mode is inert: there is no code path here that resumes/cuts/resets a
// session, so the live turn is byte-identical whatever the residual says.
func newResumeProjServer() *Server {
	return &Server{metrics: newGatewayMetrics(time.Now())}
}

// opusPricing is the model base price used across these tests (Opus 4.8 = {5, 25}); a per-token
// input price of 5e-6 makes the dollar arithmetic checkable by hand.
func opusPricing() resume.Pricing {
	return resume.Pricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25}
}

func TestComputeResumeProjectionResidual_ColdMatchesCalibratedProjection(t *testing.T) {
	// A 250k session idle 2h on a 5-minute cache: idle >> TTL, so the projection is COLD and (being
	// far above the shed budget) recommends CUT. resume.Plan prices the cold re-prefill at the
	// calibrated ColdWriteShare; an observed first turn whose cold-write share is EXACTLY the
	// calibrated constant must yield ~0 residual on both axes — the projection tracked the bill.
	const resident = 250_000
	in := resume.Input{ResidentTokens: resident, IdleSeconds: 7200, TTL: resume.TTL5m, Pricing: opusPricing()}

	// Build an observed bill whose cold-write share == resume.ColdWriteShare and whose remainder is
	// uncached input (mirroring the cold blend the projection models). Compute creation with integer
	// math (0.852 == 852/1000) so the observed share is bit-exact 0.852, not a float-rounded near-miss.
	creation := resident * 852 / 1000 // == 213000 for resident == 250000
	input := resident - creation
	observed := CacheUsage{InputTokens: input, CacheCreationTokens: creation, CacheReadTokens: 0, WriteTTL: CacheTTL5m}

	r := computeResumeProjectionResidual(in, observed)

	if r.Posture != resume.PostureCold {
		t.Fatalf("idle 7200s on a 300s TTL must project COLD, got %s", r.Posture)
	}
	if r.Recommended != resume.StrategyCut {
		t.Fatalf("a 250k cold resume must recommend CUT (cut-by-default), got %s", r.Recommended)
	}
	if r.ProjectedColdWriteShare != resume.ColdWriteShare {
		t.Fatalf("projected cold-write share must echo the calibrated constant %v, got %v", resume.ColdWriteShare, r.ProjectedColdWriteShare)
	}
	// Observed share == calibrated share -> share residual ~0.
	if abs(r.ColdWriteShareResidual) > 1e-6 {
		t.Fatalf("share residual must be ~0 when observed share equals the calibrated share, got %v", r.ColdWriteShareResidual)
	}
	// Cost residual ~0: the observed prompt bill priced on the same axis equals the projected cold
	// re-prefill within float tolerance (the projection's cold-write multiplier IS the share blend).
	if abs(r.PromptCostDeltaUSD) > 1e-6 {
		t.Fatalf("cost residual must be ~0 when the observed bill matches the calibrated projection, got %v (projected=%v observed=%v)",
			r.PromptCostDeltaUSD, r.ProjectedPromptCostUSD, r.ObservedPromptCostUSD)
	}
	if r.ProjectedPromptCostUSD <= 0 || r.ObservedPromptCostUSD <= 0 {
		t.Fatalf("a 250k cold re-prefill must carry a positive prompt cost, got projected=%v observed=%v", r.ProjectedPromptCostUSD, r.ObservedPromptCostUSD)
	}
}

func TestComputeResumeProjectionResidual_WarmPostureProjectsCacheRead(t *testing.T) {
	// Idle WITHIN the TTL: the projection is WARM, so the modeled first-turn prompt cost is a cache
	// read (0.1x), far below a cold re-prefill. An observed bill that DID re-prefill cold (high
	// creation) then reads a large POSITIVE residual — the live "proj=WARM obs=COLD" miss, surfaced.
	const resident = 200_000
	in := resume.Input{ResidentTokens: resident, IdleSeconds: 60, TTL: resume.TTL5m, Pricing: opusPricing()}
	observed := CacheUsage{InputTokens: 30_000, CacheCreationTokens: 170_000, WriteTTL: CacheTTL5m}

	r := computeResumeProjectionResidual(in, observed)
	if r.Posture != resume.PostureWarm {
		t.Fatalf("idle 60s within a 300s TTL must project WARM, got %s", r.Posture)
	}
	// Projected prompt cost is a warm read of the resident: resident * perTok * 0.1.
	wantWarm := float64(resident) * perToken(in.Pricing.InputPerMTokUSD) * CacheReadMultiplier
	if abs(r.ProjectedPromptCostUSD-wantWarm) > 1e-9 {
		t.Fatalf("warm projected prompt cost must be a 0.1x read of the resident (%v), got %v", wantWarm, r.ProjectedPromptCostUSD)
	}
	if r.PromptCostDeltaUSD <= 0 {
		t.Fatalf("a cold re-prefill against a WARM projection must read a positive cost residual (live proj=warm obs=cold), got %v", r.PromptCostDeltaUSD)
	}
}

func TestObserveResumeProjection_RecordsRowMetricAndLog(t *testing.T) {
	var sb strings.Builder
	s := newResumeProjServer()
	s.logf = func(format string, args ...any) {
		if len(args) == 1 {
			if b, ok := args[0].([]byte); ok {
				sb.Write(b)
				sb.WriteByte('\n')
				return
			}
		}
		sb.WriteString(strings.TrimSpace(format))
		sb.WriteByte('\n')
	}

	in := resume.Input{ResidentTokens: 250_000, IdleSeconds: 7200, TTL: resume.TTL5m, Pricing: opusPricing()}
	observed := CacheUsage{InputTokens: 37_000, CacheCreationTokens: 213_000, WriteTTL: CacheTTL5m}

	r := s.observeResumeProjection("trace-r1", in, observed)
	if r.Recommended != resume.StrategyCut || r.Posture != resume.PostureCold {
		t.Fatalf("expected the audit row to carry a COLD/CUT plan, got posture=%s rec=%s", r.Posture, r.Recommended)
	}

	// The metric recorded exactly one boundary in the cold posture / cut strategy buckets.
	snap := s.resumeProj.snapshot()
	if snap.boundaries != 1 {
		t.Fatalf("want 1 recorded boundary, got %d", snap.boundaries)
	}
	if snap.postures[string(resume.PostureCold)] != 1 {
		t.Fatalf("cold posture bucket must be 1, got %d", snap.postures[string(resume.PostureCold)])
	}
	if snap.strategies[string(resume.StrategyCut)] != 1 {
		t.Fatalf("cut strategy bucket must be 1, got %d", snap.strategies[string(resume.StrategyCut)])
	}

	// The content-free audit row carries the event marker, the plan recommendation, and BOTH the
	// projected and observed cost side by side — never a prompt byte (none was supplied).
	out := sb.String()
	for _, want := range []string{
		`"event":"gateway_resume_projection"`,
		`"recommended":"cut"`,
		`"projected_prompt_cost_usd"`,
		`"observed_prompt_cost_usd"`,
		`"prompt_cost_delta_usd"`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("audit row missing %q:\n%s", want, out)
		}
	}
}

func TestWriteResumeProjectionMetrics_DefaultOffThenRenders(t *testing.T) {
	s := newResumeProjServer()

	// DEFAULT-OFF: a Server whose resume hook never fired renders every posture/strategy bucket at 0
	// and zero boundaries — the panel exists but records nothing (the opt-in contract the acceptance
	// criteria require). No log either: observeResumeProjection was never called.
	var off strings.Builder
	s.resumeProj.writeMetrics(&off)
	offOut := off.String()
	for _, want := range []string{
		`fak_gateway_resume_projection_boundaries_total 0`,
		`fak_gateway_resume_projection_posture_total{posture="cold"} 0`,
		`fak_gateway_resume_projection_posture_total{posture="warm"} 0`,
		`fak_gateway_resume_projection_posture_total{posture="unknown"} 0`,
		`fak_gateway_resume_projection_recommendation_total{strategy="resume_full"} 0`,
		`fak_gateway_resume_projection_recommendation_total{strategy="cut"} 0`,
		`fak_gateway_resume_projection_recommendation_total{strategy="reset"} 0`,
		`fak_gateway_resume_projection_cost_delta_usd `,
		`fak_gateway_resume_projection_cold_write_share_residual `,
	} {
		if !strings.Contains(offOut, want) {
			t.Fatalf("default-off render missing %q:\n%s", want, offOut)
		}
	}

	// After one observed boundary the cold/cut buckets and the boundary counter tick, and the cost
	// delta gauge reflects the live residual.
	in := resume.Input{ResidentTokens: 250_000, IdleSeconds: 7200, TTL: resume.TTL5m, Pricing: opusPricing()}
	observed := CacheUsage{InputTokens: 37_000, CacheCreationTokens: 213_000, WriteTTL: CacheTTL5m}
	s.observeResumeProjection("trace-m", in, observed)

	var on strings.Builder
	s.resumeProj.writeMetrics(&on)
	onOut := on.String()
	if !strings.Contains(onOut, `fak_gateway_resume_projection_boundaries_total 1`) {
		t.Fatalf("after one boundary the counter must read 1:\n%s", onOut)
	}
	if !strings.Contains(onOut, `fak_gateway_resume_projection_posture_total{posture="cold"} 1`) {
		t.Fatalf("after one cold boundary the cold bucket must read 1:\n%s", onOut)
	}
	if !strings.Contains(onOut, `fak_gateway_resume_projection_recommendation_total{strategy="cut"} 1`) {
		t.Fatalf("after one cut recommendation the cut bucket must read 1:\n%s", onOut)
	}
}

func TestObserveResumeProjection_NilServerIsSafeNoop(t *testing.T) {
	var s *Server
	if r := s.observeResumeProjection("t", resume.Input{ResidentTokens: 1000, Pricing: opusPricing()}, CacheUsage{}); r != (ResumeProjectionResidual{}) {
		t.Fatalf("a nil server must be a safe no-op returning the zero residual, got %+v", r)
	}
}

// abs is a tiny float helper kept local to this test file.
func abs(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}
