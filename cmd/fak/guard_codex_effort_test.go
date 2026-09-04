package main

import (
	"strings"
	"testing"
)

// guard_codex_effort_test.go — the #10669 red/green witness. Goal-continuation and guarded
// Codex sessions must run at the user-configured reasoning effort (`high`), never a silently
// forced `xhigh` (measured to roughly double the post-tool wait). `xhigh` is opt-in only,
// via $FAK_GUARD_CODEX_REASONING_EFFORT, and the resolved effort stays auditable in the
// guard install receipt and the banner.

// guardCodexReasoningEffortEnvOptInName pins the opt-in env spelling: the operator contract
// in docs and runbooks names this exact variable.
const guardCodexReasoningEffortEnvOptInName = "FAK_GUARD_CODEX_REASONING_EFFORT"

// The default guarded Codex session config must pin the configured effort (high), not the
// xhigh escalation the guard used to force onto every goal-continuation turn (#10669).
func TestGuardCodexConfigArgsEffortIsConfiguredNotXhigh(t *testing.T) {
	t.Setenv(guardCodexReasoningEffortEnvOptInName, "")
	got := guardCodexConfigArgs("http://127.0.0.1:8137", "", "")
	if containsArg(got, `model_reasoning_effort="xhigh"`) {
		t.Errorf("#10669: guarded Codex config args still force xhigh (doubles post-tool wait): %v", got)
	}
	if !containsArg(got, `model_reasoning_effort="high"`) {
		t.Errorf("#10669: guarded Codex config args must default to the configured high effort: %v", got)
	}
}

// The opt-in env is the ONLY path to xhigh: unset defers to the configured default, and an
// explicit value wins over everything — including a custom model the default-effort pin
// deliberately never touches.
func TestGuardCodexReasoningEffortEnvOptInControlsSessionEffort(t *testing.T) {
	const gw = "http://127.0.0.1:8137"

	// The opt-in knob is a named operator contract: the guard's constant must keep the
	// spelling the docs and runbooks name.
	if guardCodexReasoningEffortEnv != guardCodexReasoningEffortEnvOptInName {
		t.Errorf("guardCodexReasoningEffortEnv = %q, want %q", guardCodexReasoningEffortEnv, guardCodexReasoningEffortEnvOptInName)
	}

	t.Run("unset defers to the configured default", func(t *testing.T) {
		t.Setenv(guardCodexReasoningEffortEnvOptInName, "")
		got := guardCodexConfigArgs(gw, "", "")
		if !containsArg(got, `model_reasoning_effort="high"`) || containsArg(got, `model_reasoning_effort="xhigh"`) {
			t.Errorf("#10669: unset opt-in must keep the configured high effort: %v", got)
		}
	})

	t.Run("explicit high opt-in overrides the forced xhigh", func(t *testing.T) {
		t.Setenv(guardCodexReasoningEffortEnvOptInName, "high")
		got := guardCodexConfigArgs(gw, "", "")
		if !containsArg(got, `model_reasoning_effort="high"`) || containsArg(got, `model_reasoning_effort="xhigh"`) {
			t.Errorf("#10669: explicit high opt-in must win over any forced effort: %v", got)
		}
	})

	t.Run("explicit xhigh opt-in is honored", func(t *testing.T) {
		t.Setenv(guardCodexReasoningEffortEnvOptInName, "xhigh")
		got := guardCodexConfigArgs(gw, "", "")
		if !containsArg(got, `model_reasoning_effort="xhigh"`) {
			t.Errorf("#10669: explicit xhigh opt-in must be honored: %v", got)
		}
	})

	t.Run("opt-in value is trimmed not case-mangled", func(t *testing.T) {
		t.Setenv(guardCodexReasoningEffortEnvOptInName, "  xhigh  ")
		got := guardCodexConfigArgs(gw, "", "")
		if !containsArg(got, `model_reasoning_effort="xhigh"`) {
			t.Errorf("#10669: whitespace-padded opt-in must resolve to xhigh: %v", got)
		}
	})

	t.Run("explicit opt-in applies to a custom model too", func(t *testing.T) {
		t.Setenv(guardCodexReasoningEffortEnvOptInName, "xhigh")
		got := guardCodexConfigArgs(gw, "", "gpt-custom")
		if !containsArg(got, `model_reasoning_effort="xhigh"`) {
			t.Errorf("#10669: an explicit opt-in must pin the effort even for a custom model: %v", got)
		}
	})

	t.Run("custom model without opt-in keeps no effort pin", func(t *testing.T) {
		t.Setenv(guardCodexReasoningEffortEnvOptInName, "")
		got := guardCodexConfigArgs(gw, "", "gpt-custom")
		if containsArg(got, "model_reasoning_effort=") {
			t.Errorf("#10669: custom model without opt-in must keep its own effort contract: %v", got)
		}
	})
}

// The install receipt must carry the RESOLVED effort so per-turn escalation is auditable
// from the guard's own record, not just from Codex's turn_context.
func TestInstallGuardCodexReasoningEffortReceiptAuditable(t *testing.T) {
	const gw = "http://127.0.0.1:8137"

	t.Run("default install receipts the configured high effort", func(t *testing.T) {
		t.Setenv(guardCodexReasoningEffortEnvOptInName, "")
		_, info := installGuardCodexConfig([]string{"codex", "exec", "task"}, true, gw, "")
		if !info.Applied {
			t.Fatal("#10669: codex install not applied")
		}
		if info.Reasoning != "high" {
			t.Errorf("#10669: install receipt Reasoning = %q, want the configured high (got forced xhigh?)", info.Reasoning)
		}
		if info.ReasoningOptIn {
			t.Errorf("#10669: default install must not mark the effort as operator-opted-in: %+v", info)
		}
	})

	t.Run("opted-in install receipts the opted-in effort", func(t *testing.T) {
		t.Setenv(guardCodexReasoningEffortEnvOptInName, "xhigh")
		out, info := installGuardCodexConfig([]string{"codex", "exec", "task"}, true, gw, "")
		if !info.Applied {
			t.Fatal("#10669: codex install not applied")
		}
		if info.Reasoning != "xhigh" {
			t.Errorf("#10669: opted-in install receipt Reasoning = %q, want xhigh", info.Reasoning)
		}
		if !info.ReasoningOptIn {
			t.Errorf("#10669: opted-in install must attribute the effort to the explicit opt-in: %+v", info)
		}
		if !containsArg(out, `model_reasoning_effort="xhigh"`) {
			t.Errorf("#10669: opted-in argv must carry the xhigh pin: %v", out)
		}
	})
}

// The banner names the resolved effort, and when it came from the explicit opt-in the banner
// says so — an operator reading the startup report can tell a default high from an opted-in
// xhigh without reverse-engineering the environment.
func TestPrintGuardCodexNoteNamesEffortSource(t *testing.T) {
	t.Run("default effort is named plainly", func(t *testing.T) {
		var b strings.Builder
		printGuardCodexNote(&b, guardCodexInstall{
			Applied:    true,
			ProviderID: "fak",
			EnvKey:     "OPENAI_API_KEY",
			BaseURL:    "http://127.0.0.1:8137/v1",
			Model:      "gpt-5.6-sol",
			Reasoning:  "high",
		})
		out := b.String()
		if !strings.Contains(out, `model_reasoning_effort=high`) {
			t.Errorf("#10669: banner must name the resolved effort: %s", out)
		}
		if strings.Contains(out, "opt-in") {
			t.Errorf("#10669: a default-effort install must not claim an operator opt-in: %s", out)
		}
	})

	t.Run("opted-in effort is attributed to the opt-in env", func(t *testing.T) {
		var b strings.Builder
		printGuardCodexNote(&b, guardCodexInstall{
			Applied:        true,
			ProviderID:     "fak",
			EnvKey:         "OPENAI_API_KEY",
			BaseURL:        "http://127.0.0.1:8137/v1",
			Model:          "gpt-5.6-sol",
			Reasoning:      "xhigh",
			ReasoningOptIn: true,
		})
		out := b.String()
		if !strings.Contains(out, `model_reasoning_effort=xhigh`) {
			t.Errorf("#10669: banner must name the resolved effort: %s", out)
		}
		if !strings.Contains(out, "$"+guardCodexReasoningEffortEnv) {
			t.Errorf("#10669: an opted-in xhigh banner must name the opt-in env so escalation is attributable: %s", out)
		}
	})
}

func TestGuardCodexReasoningEffortGPT6AstraAliases(t *testing.T) {
	for _, m := range []string{"gpt-6-astra", "gpt 6 astra", "astra", "gpt-6", "gpt6astra", "openai/gpt-6-astra"} {
		if got := guardCodexReasoningEffort(m); got != "high" {
			t.Errorf("guardCodexReasoningEffort(%q) = %q, want high", m, got)
		}
	}
}
