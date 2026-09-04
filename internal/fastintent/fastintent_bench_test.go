package fastintent

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/orchestration"
	"github.com/anthony-chaudhary/fak/internal/ultracodebench"
)

func benchFixture(b *testing.B) (orchestration.FastExecutionPlan, []ProviderRealization, ultracodebench.FastProfileReport) {
	b.Helper()
	rawPlan, err := os.ReadFile("../orchestration/testdata/fast-claude.json")
	if err != nil {
		b.Fatal(err)
	}
	var task orchestration.TaskSpec
	if err := json.Unmarshal(rawPlan, &task); err != nil {
		b.Fatal(err)
	}
	res, err := orchestration.Resolve(
		orchestration.OrchestrationProfile{Name: orchestration.ProfileFast},
		task,
		orchestration.HarnessCapabilities{
			Concurrency:        orchestration.SupportNative,
			TaskMessaging:      orchestration.SupportNative,
			Cancellation:       orchestration.SupportNative,
			Leases:             orchestration.SupportNative,
			IndependentWitness: orchestration.SupportNative,
			ClaudeSpeed:        orchestration.SupportNative,
		},
	)
	if err != nil {
		b.Fatal(err)
	}
	plan := *res.Resolved.Fast

	mk := func(provider string, realized modelroute.ServiceMode, rewarm int64) ProviderRealization {
		c, _ := modelroute.LookupProviderContract(provider)
		_, r, err := modelroute.BindServiceTier(c, modelroute.ServiceTierRequest{Mode: modelroute.ServiceModeFast, Policy: modelroute.ServiceAllowDeclaredDowngrade})
		if err != nil {
			b.Fatal(err)
		}
		r, err = modelroute.RealizeServiceTier(r, realized, rewarm)
		if err != nil {
			b.Fatal(err)
		}
		return ProviderRealization{Provider: provider, Receipt: r}
	}

	rawEval, err := os.ReadFile("../ultracodebench/testdata/issue8963-fast-profile-replay.json")
	if err != nil {
		b.Fatal(err)
	}
	var input ultracodebench.FastProfileBundle
	if err := json.Unmarshal(rawEval, &input); err != nil {
		b.Fatal(err)
	}

	return plan, []ProviderRealization{
		mk("openai", modelroute.ServiceModeFast, 0),
		mk("anthropic", modelroute.ServiceModeStandard, 73),
	}, ultracodebench.EvaluateFastProfile(input)
}

// BenchmarkFastIntent exercises the complete Join fast intent extraction and bundle digest generation.
func BenchmarkFastIntent(b *testing.B) {
	plan, providers, evaluation := benchFixture(b)
	b.ResetTimer()
	b.ReportAllocs()

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

// BenchmarkFastIntentParallel exercises Join under concurrent execution.
func BenchmarkFastIntentParallel(b *testing.B) {
	plan, providers, evaluation := benchFixture(b)
	b.ResetTimer()
	b.ReportAllocs()

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
