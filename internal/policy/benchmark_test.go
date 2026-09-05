package policy

import (
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

var (
	benchTierDecisionSink   TierDecision
	benchRuntimeSink        Runtime
	benchPolicySink         adjudicator.Policy
	benchCombinedPolicySink CombinedPolicy
	benchAllowListSink      []string
	benchDenyListSink       []string
	benchPrivacySink        PrivacyPolicy
	benchLintDefectsSink    []string
	benchDangerRulesSink    []ArgRule
	benchPrecedenceSink     OrgPrecedenceResolution
	benchManifestBytesSink  []byte
	benchBoolSink           bool
	benchErrSink            error
)

// BenchmarkPolicyEvaluation measures decoupled tier evaluation across common production call shapes:
// allowed convenience tools, frozen-core SSRF blocks, destructive command checks, and unknown commands.
func BenchmarkPolicyEvaluation(b *testing.B) {
	eval := NewTieredEvaluator()
	eval.AllowTool("read_file")
	eval.AllowTool("write_file")
	eval.AllowTool("bash")
	eval.AllowCommand("git status")
	eval.AllowCommand("go test ./...")

	b.Run("AllowedTool_Read", func(b *testing.B) {
		args := map[string]any{"path": "internal/policy/policy.go"}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchTierDecisionSink = eval.Evaluate("read_file", args)
		}
	})

	b.Run("AllowedTool_Command", func(b *testing.B) {
		args := map[string]any{"command": "git status"}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchTierDecisionSink = eval.Evaluate("bash", args)
		}
	})

	b.Run("FrozenCore_SSRF", func(b *testing.B) {
		args := map[string]any{"url": "http://169.254.169.254/latest/meta-data/"}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchTierDecisionSink = eval.Evaluate("curl", args)
		}
	})

	b.Run("FrozenCore_DestructiveRoot", func(b *testing.B) {
		args := map[string]any{"command": "rm -rf / --no-preserve-root"}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchTierDecisionSink = eval.Evaluate("bash", args)
		}
	})

	b.Run("UnknownCommand_Guidance", func(b *testing.B) {
		args := map[string]any{"command": "custom_internal_cli --dump"}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchTierDecisionSink = eval.Evaluate("bash", args)
		}
	})

	b.Run("Manifest_EvaluateAgainstTiers", func(b *testing.B) {
		m := FromPolicy(adjudicator.DefaultPolicy())
		args := map[string]any{"path": "internal/policy/policy.go"}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchTierDecisionSink = m.EvaluateAgainstTiers("read_file", args)
		}
	})
}

// BenchmarkPolicyCombine measures hierarchical multi-tenant policy combination:
// intersecting allowlists, unioning denylists, combining least-privilege privacy policies,
// evaluating budgets against recorded usage, and full policy merging.
func BenchmarkPolicyCombine(b *testing.B) {
	parentAllow := []string{
		"bash", "edit_file", "git", "glob", "grep", "ls", "make", "python",
		"read_file", "test", "view", "write_file",
	}
	childAllow := []string{
		"bash", "docker", "git", "go", "grep", "kubectl", "read_file", "write_file",
	}

	parentDeny := []string{"curl", "nc", "nmap", "sudo", "tcpdump"}
	childDeny := []string{"nmap", "rm", "sudo", "wireshark"}

	parentPrivacy := PrivacyPolicy{
		ZeroRetention:   false,
		MaskPII:         true,
		LogContent:      true,
		StoreTranscript: false,
	}
	childPrivacy := PrivacyPolicy{
		ZeroRetention:   true,
		MaskPII:         false,
		LogContent:      false,
		StoreTranscript: false,
	}

	budgets := []Budget{
		{MaxTokens: 200_000, MaxCostUSD: 10.0, MaxCalls: 500},
		{MaxTokens: 100_000, MaxCostUSD: 5.0, MaxCalls: 250},
	}
	usage := Usage{Tokens: 45_000, CostUSD: 2.25, Calls: 120}

	parentPolicy := CombinedPolicy{
		Allow:   parentAllow,
		Deny:    parentDeny,
		Privacy: parentPrivacy,
		Budgets: budgets[:1],
	}
	childPolicy := CombinedPolicy{
		Allow:   childAllow,
		Deny:    childDeny,
		Privacy: childPrivacy,
		Budgets: budgets[1:],
	}

	b.Run("CombineAllowlists", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchAllowListSink = CombineAllowlists(parentAllow, childAllow)
		}
	})

	b.Run("CombineDenylists", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchDenyListSink = CombineDenylists(parentDeny, childDeny)
		}
	})

	b.Run("CombinePrivacy", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchPrivacySink = CombinePrivacy(parentPrivacy, childPrivacy)
		}
	})

	b.Run("EvaluateBudgets", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchErrSink = EvaluateBudgets(budgets, usage)
		}
	})

	b.Run("CombinePolicies", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchCombinedPolicySink = CombinePolicies(parentPolicy, childPolicy)
		}
	})
}

// BenchmarkHarnessProfiles measures harness profile compliance linting, shell danger rule
// query resolution, and attaching shell danger rules onto runtime policies.
func BenchmarkHarnessProfiles(b *testing.B) {
	floorBytes := FromPolicy(adjudicator.DefaultPolicy()).JSON()

	b.Run("LintHarnessProfiles", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchLintDefectsSink = LintHarnessProfiles(floorBytes)
		}
	})

	b.Run("ShellDangerRuleSetsFor", func(b *testing.B) {
		tools := []string{"bash", "powershell", "exec_command", "shell_command", "read_file"}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for _, t := range tools {
				_ = ShellDangerRuleSetsFor(t)
			}
		}
	})

	b.Run("ShellDangerRulesFor", func(b *testing.B) {
		tools := []string{"bash", "powershell", "exec_command", "shell_command"}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for _, t := range tools {
				benchDangerRulesSink, _ = ShellDangerRulesFor(t)
			}
		}
	})

	b.Run("AttachShellDangerRules", func(b *testing.B) {
		rt, err := ParseRuntime(floorBytes)
		if err != nil {
			b.Fatalf("ParseRuntime: %v", err)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			copyRT := rt
			_ = AttachShellDangerRules(&copyRT, "bash")
		}
	})
}

// BenchmarkPresetResolution measures least-agency preset ladder resolution:
// manifest extraction and full runtime parsing for read-only, propose-only,
// bounded-write, and autonomous tiers.
func BenchmarkPresetResolution(b *testing.B) {
	names := PresetNames()
	for _, name := range names {
		rung := name
		b.Run(fmt.Sprintf("PresetManifest/%s", rung), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchManifestBytesSink, benchErrSink = PresetManifest(rung)
			}
		})

		b.Run(fmt.Sprintf("LoadPreset/%s", rung), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchRuntimeSink, benchErrSink = LoadPreset(rung)
			}
		})
	}
}

// BenchmarkPolicyParseAndToPolicy measures parsing of declarative JSON policy manifests
// and resolution into runtime adjudicator.Policy structures.
func BenchmarkPolicyParseAndToPolicy(b *testing.B) {
	defaultManifest := FromPolicy(adjudicator.DefaultPolicy())
	rawBytes := defaultManifest.JSON()

	b.Run("Parse", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchPolicySink, benchErrSink = Parse(rawBytes)
		}
	})

	b.Run("ParseRuntime", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchRuntimeSink, benchErrSink = ParseRuntime(rawBytes)
		}
	})

	b.Run("ManifestToPolicy", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchPolicySink, benchErrSink = defaultManifest.ToPolicy()
		}
	})

	b.Run("DeniesToolUnconditionally", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBoolSink = defaultManifest.DeniesToolUnconditionally("rm_rf")
		}
	})
}

// BenchmarkOrgPrecedenceResolution measures folding authority precedence across
// compiled-in, central org, and local operator policy channels.
func BenchmarkOrgPrecedenceResolution(b *testing.B) {
	ratchetInput := OrgPrecedenceInput{
		Class:      AmendRatchet,
		CompiledIn: OrgKnobContribution{Deny: true},
		Central:    OrgKnobContribution{Set: true, Cap: 100},
		Operator:   OrgKnobContribution{Set: true, Cap: 50},
	}
	gatedWidenInput := OrgPrecedenceInput{
		Class:      AmendGatedWiden,
		CompiledIn: OrgKnobContribution{Set: true, Cap: 100},
		Central:    OrgKnobContribution{Set: true, Cap: 80},
		Operator:   OrgKnobContribution{Set: true, Cap: 60},
	}

	b.Run("Ratchet", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchPrecedenceSink = ResolveOrgPrecedence(ratchetInput)
		}
	})

	b.Run("GatedWiden", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchPrecedenceSink = ResolveOrgPrecedence(gatedWidenInput)
		}
	})
}
