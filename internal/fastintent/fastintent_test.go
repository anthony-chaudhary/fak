package fastintent

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/orchestration"
	"github.com/anthony-chaudhary/fak/internal/ultracodebench"
)

func fixture(t *testing.T) (orchestration.FastExecutionPlan, []ProviderRealization, ultracodebench.FastProfileReport) {
	t.Helper()
	var plan orchestration.FastExecutionPlan
	b, err := os.ReadFile("../orchestration/testdata/fast-claude.json")
	if err != nil {
		t.Fatal(err)
	}
	var task orchestration.TaskSpec
	if err := json.Unmarshal(b, &task); err != nil {
		t.Fatal(err)
	}
	res, err := orchestration.Resolve(orchestration.OrchestrationProfile{Name: orchestration.ProfileFast}, task, orchestration.HarnessCapabilities{Concurrency: orchestration.SupportNative, TaskMessaging: orchestration.SupportNative, Cancellation: orchestration.SupportNative, Leases: orchestration.SupportNative, IndependentWitness: orchestration.SupportNative, ClaudeSpeed: orchestration.SupportNative})
	if err != nil {
		t.Fatal(err)
	}
	plan = *res.Resolved.Fast

	mk := func(provider string, realized modelroute.ServiceMode, rewarm int64) ProviderRealization {
		c, _ := modelroute.LookupProviderContract(provider)
		_, r, err := modelroute.BindServiceTier(c, modelroute.ServiceTierRequest{Mode: modelroute.ServiceModeFast, Policy: modelroute.ServiceAllowDeclaredDowngrade})
		if err != nil {
			t.Fatal(err)
		}
		r, err = modelroute.RealizeServiceTier(r, realized, rewarm)
		if err != nil {
			t.Fatal(err)
		}
		return ProviderRealization{Provider: provider, Receipt: r}
	}
	raw, err := os.ReadFile("../ultracodebench/testdata/issue8963-fast-profile-replay.json")
	if err != nil {
		t.Fatal(err)
	}
	var input ultracodebench.FastProfileBundle
	if err := json.Unmarshal(raw, &input); err != nil {
		t.Fatal(err)
	}
	return plan, []ProviderRealization{mk("openai", modelroute.ServiceModeFast, 0), mk("anthropic", modelroute.ServiceModeStandard, 73)}, ultracodebench.EvaluateFastProfile(input)
}

func TestJoinDeterministicPortableReceipt(t *testing.T) {
	plan, providers, evaluation := fixture(t)
	a, err := Join(plan, providers, evaluation)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Join(plan, providers, evaluation)
	if err != nil {
		t.Fatal(err)
	}
	if a.EvidenceDigest != b.EvidenceDigest || a.EvidenceDigest == "" {
		t.Fatalf("digests %q %q", a.EvidenceDigest, b.EvidenceDigest)
	}
	if a.Verdict != evaluation.Verdict || len(a.Providers) != 2 {
		t.Fatalf("receipt %+v", a)
	}
	if a.Providers[1].Receipt.DowngradeReason == "" || !a.Providers[1].Receipt.CacheInvalidated {
		t.Fatalf("downgrade hidden: %+v", a.Providers[1])
	}
}

func TestJoinRejectsMissingQualityAndSilentFallback(t *testing.T) {
	plan, providers, evaluation := fixture(t)
	plan.Requested.QualityFloor = ""
	if _, err := Join(plan, providers, evaluation); err == nil {
		t.Fatal("missing quality accepted")
	}
	plan, providers, evaluation = fixture(t)
	providers[1].Receipt.DowngradeReason = ""
	if _, err := Join(plan, providers, evaluation); err == nil {
		t.Fatal("silent fallback accepted")
	}
	plan, providers, evaluation = fixture(t)
	evaluation.BundleDigest = ""
	if _, err := Join(plan, providers, evaluation); err == nil {
		t.Fatal("unreplayable evaluation accepted")
	}
}

func BenchmarkJoin(b *testing.B) {
	plan, providers, evaluation := fixture(&testing.T{})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundle, err := Join(plan, providers, evaluation)
		if err != nil {
			b.Fatal(err)
		}
		if bundle.EvidenceDigest == "" {
			b.Fatal("missing digest")
		}
	}
}

func BenchmarkJoinParallel(b *testing.B) {
	plan, providers, evaluation := fixture(&testing.T{})
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			bundle, err := Join(plan, providers, evaluation)
			if err != nil {
				b.Fatal(err)
			}
			if bundle.EvidenceDigest == "" {
				b.Fatal("missing digest")
			}
		}
	})
}
