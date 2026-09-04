package agent

import (
	"bytes"
	"context"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
)

func TestOwnedSystemBlockComposesWorkAndResponseProfiles(t *testing.T) {
	t.Setenv(syspromptmmu.WorkProfileEnvVar, "ponytail:medium")
	t.Setenv(syspromptmmu.StyleEnvVar, "caveman:high")
	block := BuildOwnedSystemBlock(nil, func(syspromptmmu.BaseEdit) bool { return true })
	if !block.CacheStable() {
		t.Fatalf("cache audit = %+v", block.Audit)
	}
	if block.WorkProfile != "ponytail:native:medium" || block.Style != "caveman:native:high" {
		t.Fatalf("profile readout = work %q style %q", block.WorkProfile, block.Style)
	}
	work := []byte("Work profile: Ponytail-inspired, native, medium intensity.")
	style := []byte("Signal-first level 3 (compressed)")
	wi, si := bytes.Index(block.Value, work), bytes.Index(block.Value, style)
	if wi < 0 || si < 0 || wi >= si {
		t.Fatalf("work/style overlays absent or wrong precedence: work=%d style=%d", wi, si)
	}
}

func TestOwnedSystemBlockDefaultsToPonytailMedium(t *testing.T) {
	_ = os.Unsetenv(syspromptmmu.WorkProfileEnvVar)
	block := BuildOwnedSystemBlock(nil, func(syspromptmmu.BaseEdit) bool { return true })
	if block.WorkProfile != "ponytail:native:medium" || block.WorkProfileIntensity != "medium" {
		t.Fatalf("default work profile = %+v", block)
	}
}

func TestOwnedAgentLoopMediatesDefaultPonytailMedium(t *testing.T) {
	t.Setenv(syspromptmmu.WorkProfileEnvVar, "")
	messages := (runConfig{}).seedMessages("task")
	if len(messages) != 2 || messages[0].Role != RoleSystem {
		t.Fatalf("seed messages = %+v", messages)
	}
	for _, want := range []string{SystemPrompt, "fak:work-profile", "Ponytail-inspired, native, medium intensity."} {
		if !strings.Contains(messages[0].Content, want) {
			t.Errorf("mediated default system prompt missing %q: %q", want, messages[0].Content)
		}
	}
}

func TestOwnedAgentLoopCanDisableDefaultPonytail(t *testing.T) {
	t.Setenv(syspromptmmu.WorkProfileEnvVar, "standard")
	got := (runConfig{}).seedMessages("task")[0].Content
	if strings.Contains(got, "<fak:work-profile>") {
		t.Fatalf("standard opt-out retained work profile: %q", got)
	}
}

type workProfileCapturePlanner struct {
	messages []Message
}

func (p *workProfileCapturePlanner) Model() string { return "work-profile-capture" }

func (p *workProfileCapturePlanner) Complete(_ context.Context, messages []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	p.messages = append([]Message(nil), messages...)
	return &Completion{Message: Message{Role: RoleAssistant, Content: "done"}}, nil
}

func TestRunReportCapturesDefaultWorkProfileWitness(t *testing.T) {
	t.Setenv(syspromptmmu.WorkProfileEnvVar, "")
	planner := &workProfileCapturePlanner{}
	result, _, err := Run(context.Background(), planner, "task", 1)
	if err != nil {
		t.Fatal(err)
	}
	if result.WorkProfile != syspromptmmu.WorkProfilePonytailNativeMed || result.WorkProfileWitness == "" {
		t.Fatalf("work profile report = %+v", result)
	}
	if len(planner.messages) == 0 || !strings.Contains(planner.messages[0].Content, "Ponytail-inspired, native, medium intensity.") {
		t.Fatalf("planner did not receive mediated Ponytail prompt: %+v", planner.messages)
	}
}

func TestSeedMessagesWithSystemPromptAndMemoryDigest(t *testing.T) {
	t.Setenv(syspromptmmu.WorkProfileEnvVar, "standard")
	cfg := resolveRunConfig([]RunOption{
		WithSystemPrompt(CodeAgentSystemPrompt),
		WithMemoryDigest("# Workspace memory\nFact 1"),
		WithConversation([]Message{
			{Role: RoleUser, Content: "turn 1"},
			{Role: RoleAssistant, Content: "ans 1"},
		}),
	})
	msgs := cfg.seedMessages("turn 2")
	if len(msgs) != 4 {
		t.Fatalf("expected 4 messages (sys + mem + 2 conv), got %d: %+v", len(msgs), msgs)
	}
	if msgs[0].Role != RoleSystem || !strings.Contains(msgs[0].Content, "software engineering agent") {
		t.Errorf("msgs[0] expected CodeAgentSystemPrompt, got: %+v", msgs[0])
	}
	if msgs[1].Role != RoleSystem || !strings.Contains(msgs[1].Content, "Fact 1") {
		t.Errorf("msgs[1] expected memory digest, got: %+v", msgs[1])
	}
	if msgs[2].Role != RoleUser || msgs[2].Content != "turn 1" {
		t.Errorf("msgs[2] expected user turn 1, got: %+v", msgs[2])
	}
	if msgs[3].Role != RoleAssistant || msgs[3].Content != "ans 1" {
		t.Errorf("msgs[3] expected assistant ans 1, got: %+v", msgs[3])
	}
}
