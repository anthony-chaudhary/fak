package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// This fixture witnesses converter text bytes, not authenticated model admission
// or the official Python encoder. Reversed results retain arrival order. Missing
// oracle evidence must fail rather than turn the contract into a successful skip.
// fak-test:justify why=contract when=changed:internal/agent/**
// fak-test:runtime fast est=5ms lane=default
func TestDeepSeekV411017GGUFTextContract(t *testing.T) {
	var fixture struct {
		ArtifactRevision   string    `json:"artifact_revision"`
		ArtifactFile       string    `json:"artifact_file"`
		TemplateSource     string    `json:"template_source"`
		TemplateSHA256     string    `json:"template_sha256"`
		Template           string    `json:"template"`
		RendererID         string    `json:"renderer_id"`
		OracleVerified     bool      `json:"oracle_verified"`
		OracleOutputSHA256 string    `json:"oracle_output_sha256"`
		Messages           []Message `json:"messages"`
		Expected           string    `json:"expected"`
	}
	raw, err := os.ReadFile(filepath.Join("testdata", "deepseek41_1017_gguf_chat.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	const revision = "1df54e7d1b1e442051ad393ce871870a929f243e"
	const artifact = "Q4_K_M/DeepSeek-V4.1-Flash-1017-Q4_K_M-00001-of-00010.gguf"
	const source = "https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF/blob/" + revision + "/" + artifact
	const templateHash = "7bd4fbdf0c06cd7398ddb6913de6e4a5839b9d06569e74e5b2ec246b375ca9c9"
	const outputHash = "2cabf014a2eb41d8fe0223834e1a1e2a735ff9d9147b41b3565bc11cc37daa96"
	sum := sha256.Sum256([]byte(fixture.Template))
	if fixture.ArtifactRevision != revision || fixture.ArtifactFile != artifact || fixture.TemplateSource != source ||
		fixture.TemplateSHA256 != templateHash || hex.EncodeToString(sum[:]) != templateHash {
		t.Fatal("1017 candidate template provenance drift")
	}
	sum = sha256.Sum256([]byte(fixture.Expected))
	if !fixture.OracleVerified || fixture.OracleOutputSHA256 != outputHash || hex.EncodeToString(sum[:]) != outputHash {
		t.Fatal("1017 candidate requires the independently witnessed oracle output")
	}
	if fixture.RendererID != deepSeekV41ChatRendererID {
		t.Fatalf("fixture renderer = %q, want %q", fixture.RendererID, deepSeekV41ChatRendererID)
	}
	// Display names cannot select a release contract. These checks intentionally
	// preserve the existing converter fallback until authenticated admission exists.
	for _, name := range []string{"", "DeepSeek-V4.1-Flash", "DeepSeek-V4.1-Flash-1017", "unverified-1017"} {
		cfg := model.Config{ModelType: "deepseek41", Name: name}
		if got := inKernelPromptRendererID(cfg, SampleParams{}); got != fixture.RendererID {
			t.Fatalf("name %q renderer = %q, want %q", name, got, fixture.RendererID)
		}
		if got := inKernelRenderPrompt(fixture.Messages, nil, cfg, SampleParams{}); got != fixture.Expected {
			t.Fatalf("name %q converter text differs:\ngot  %q\nwant %q", name, got, fixture.Expected)
		}
		sp := SampleParams{RawPrompt: true}
		if got := inKernelPromptRendererID(cfg, sp); got != "deepseek41-raw-bos-v1" {
			t.Fatalf("name %q raw renderer = %q", name, got)
		}
		if got := inKernelRenderPrompt([]Message{{Role: RoleUser, Content: "raw"}}, nil, cfg, sp); got != "raw" {
			t.Fatalf("name %q raw text = %q", name, got)
		}
	}
}

// This synthetic builder exercises trusted-loader receipt transfer and prompt
// admission only. No fixture claims to authenticate the real checkpoint bytes.
// fak-test:justify why=regression when=changed:internal/agent/**
// fak-test:runtime fast est=20ms lane=default
func TestDeepSeekV411017ExplicitPromptAdmission(t *testing.T) {
	identity := model.AuthenticatedArtifactIdentity{
		ManifestSHA256:     strings.Repeat("a", 64),
		ShardSetSHA256:     deepSeekV411017ShardSetSHA256,
		ChatTemplateSHA256: deepSeekV411017TemplateSHA256,
	}
	newPlanner := func(id *model.AuthenticatedArtifactIdentity, profile string) *InKernelPlanner {
		t.Helper()
		builder := model.NewQuantBuilder(model.Config{ModelType: "deepseek41", HiddenSize: 256, MaxPositionEmbeddings: 8192}, false)
		if id != nil {
			if err := builder.BindAuthenticatedArtifactIdentity(*id); err != nil {
				t.Fatal(err)
			}
		}
		if err := builder.AddResidentQ4K("model.layers.0.mlp.down_proj.weight", []int{1, 256}, make([]byte, 144)); err != nil {
			t.Fatal(err)
		}
		m, err := builder.Build()
		if err != nil {
			t.Fatal(err)
		}
		return NewInKernelPlannerWithConfig(m, loadProbeTok(t), "same-display-name", true, nil, false,
			InKernelPlannerConfig{DeepSeekV41ChatProfile: profile, ContextTokens: 8192})
	}
	messages := []Message{{Role: RoleUser, Content: "hello"}}
	p := newPlanner(&identity, DeepSeekV411017GGUFTextProfile)
	legacy := newPlanner(&identity, "")
	got, err := p.EncodePrompt(context.Background(), messages, nil)
	if err != nil {
		t.Fatal(err)
	}
	old, err := legacy.EncodePrompt(context.Background(), messages, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.TokenIDs, old.TokenIDs) || got.RenderedSHA256 != old.RenderedSHA256 || got.RendererID == old.RendererID {
		t.Fatal("explicit profile must preserve converter bytes and distinguish prompt provenance")
	}
	if p.RuntimeConfig().DeepSeekV41ChatProfile != DeepSeekV411017GGUFTextProfile || p.warmDiskExecutionIdentity() == legacy.warmDiskExecutionIdentity() {
		t.Fatal("explicit profile lost runtime readback or restart-cache identity")
	}
	for _, tc := range []struct {
		name    string
		id      *model.AuthenticatedArtifactIdentity
		profile string
	}{
		{"missing", nil, DeepSeekV411017GGUFTextProfile},
		{"unknown-profile", &identity, "1017"},
		{"wrong-shards", &model.AuthenticatedArtifactIdentity{ManifestSHA256: identity.ManifestSHA256, ShardSetSHA256: strings.Repeat("b", 64), ChatTemplateSHA256: identity.ChatTemplateSHA256}, DeepSeekV411017GGUFTextProfile},
		{"wrong-template", &model.AuthenticatedArtifactIdentity{ManifestSHA256: identity.ManifestSHA256, ShardSetSHA256: identity.ShardSetSHA256, ChatTemplateSHA256: strings.Repeat("b", 64)}, DeepSeekV411017GGUFTextProfile},
	} {
		t.Run(tc.name, func(t *testing.T) {
			candidate := newPlanner(tc.id, tc.profile)
			// Nil tokenizer proves refusal happens before rendering/tokenization.
			candidate.tok = nil
			if _, err := candidate.preparePrompt(context.Background(), messages, nil, SampleParams{}); !errors.Is(err, ErrDeepSeekV41ChatProfile) {
				t.Fatalf("invalid explicit identity did not fail closed: %v", err)
			}
		})
	}
	for _, raw := range []string{
		`{"role":"user","content":[{"type":"text","text":"hello"}]}`,
		`{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.invalid/image"}}]}`,
		`{"role":"user","content":"hello","content_blocks":[{"type":"image"}]}`,
	} {
		var message Message
		if err := json.Unmarshal([]byte(raw), &message); err != nil {
			t.Fatal(err)
		}
		if !message.ContentWasStructured {
			t.Fatal("original structured shape was lost")
		}
		if _, err := p.preparePrompt(context.Background(), []Message{message}, nil, SampleParams{}); !errors.Is(err, ErrDeepSeekV41ChatProfile) {
			t.Fatalf("structured input admitted after flattening: %v", err)
		}
		if err := legacy.admitDeepSeekV41ChatProfile([]Message{message}, SampleParams{}); err != nil {
			t.Fatalf("legacy behavior changed: %v", err)
		}
		if err := json.Unmarshal([]byte(`{"role":"user","content":"hello"}`), &message); err != nil || message.ContentWasStructured {
			t.Fatalf("reused decode retained stale shape: %v", err)
		}
	}
	if _, err := p.EncodePrompt(context.Background(), messages, nil, WithReasoningEffort(EffortTierHigh)); !errors.Is(err, ErrDeepSeekV41ChatProfile) {
		t.Fatalf("thinking request admitted: %v", err)
	}
	sp := SampleParams{RawPrompt: true}
	if err := p.admitDeepSeekV41ChatProfile(messages, sp); err != nil || p.promptRendererID(sp) != legacy.promptRendererID(sp) || inKernelRenderPrompt(messages, nil, p.m.Cfg, sp) != "hello" {
		t.Fatalf("raw completion changed: %v", err)
	}
}
