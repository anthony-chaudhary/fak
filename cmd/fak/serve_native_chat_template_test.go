package main

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func jinjaServeFlagsForTest(t *testing.T, args ...string) *serveFlags {
	t.Helper()
	fs, sf := newServeFlagSet()
	fs.SetOutput(io.Discard)
	if err := fs.Parse(args); err != nil {
		t.Fatal(err)
	}
	sf.explicit = explicitFlagNames(fs)
	return sf
}
func jinjaServeSelectedArgs(path, digest string) []string {
	return []string{"--gguf", "owned-unused-model.gguf", "--model", "synthetic-model",
		"--native-chat-template-file", path, "--native-chat-template-sha256", digest,
		"--native-chat-template-model", "synthetic-model"}
}

// fak-test:runtime fast est=1s lane=default
func TestServeNativeChatTemplateFlagContract(t *testing.T) {
	sf := jinjaServeFlagsForTest(t)
	untouched := agent.InKernelPlannerConfig{}
	if err := validateServeNativeChatTemplateFlags(sf); err != nil {
		t.Fatal(err)
	}
	if err := prepareServeNativeChatTemplate(sf); err != nil {
		t.Fatal(err)
	}
	if err := applyServeNativeChatTemplate(sf, &untouched, false); err != nil || untouched.ChatTemplate != nil {
		t.Fatal("unset flags changed ordinary planner")
	}
	digest := strings.Repeat("a", 64)
	selected := jinjaServeFlagsForTest(t, jinjaServeSelectedArgs("deliberately-absent-template.jinja", digest)...)
	// Pure validation does not open the selected file, including inspection paths.
	if err := validateServeNativeChatTemplateFlags(selected); err != nil {
		t.Fatal(err)
	}
	if err := prepareServeNativeChatTemplate(selected); !errors.Is(err, agent.ErrNativeChatTemplate) {
		t.Fatal("missing selected source did not refuse")
	}
	if err := applyServeNativeChatTemplate(selected, &untouched, true); !errors.Is(err, agent.ErrNativeChatTemplate) {
		t.Fatal("unprepared selection silently fell back")
	}
	cases := [][]string{
		{"--native-chat-template-file", ""},
		{"--native-chat-template-file", "private-template.jinja"},
		{"--gguf", "unused", "--model", "synthetic-model", "--native-chat-template-file", "private-template.jinja", "--native-chat-template-sha256", digest, "--native-chat-template-model", "other-model"},
		append(jinjaServeSelectedArgs("private-template.jinja", digest), "--mock"),
		append(jinjaServeSelectedArgs("private-template.jinja", digest), "--base-url", "https://example.invalid"),
		append(jinjaServeSelectedArgs("private-template.jinja", digest), "--qwen38-runtime", "llama-mtp"),
		jinjaServeSelectedArgs("private-template.jinja", strings.Repeat("A", 64)),
	}
	for _, args := range cases {
		got := jinjaServeFlagsForTest(t, args...)
		err := validateServeNativeChatTemplateFlags(got)
		if !errors.Is(err, agent.ErrNativeChatTemplate) {
			t.Fatal("invalid selected flags accepted")
		}
		if strings.Contains(err.Error(), "private-template") || strings.Contains(err.Error(), digest) {
			t.Fatal("selection diagnostic leaked input")
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestServeNativeChatTemplateOrdinaryConfigConsumer(t *testing.T) {
	const source = "literal {{ messages[0].content }}"
	sum := sha256.Sum256([]byte(source))
	digest := hex.EncodeToString(sum[:])
	path := filepath.Join(t.TempDir(), "selected.jinja")
	if err := os.WriteFile(path, []byte(source), 0600); err != nil {
		t.Fatal(err)
	}
	sf := jinjaServeFlagsForTest(t, jinjaServeSelectedArgs(path, digest)...)
	if err := prepareServeNativeChatTemplate(sf); err != nil {
		t.Fatal(err)
	}
	selected := sf.nativeChatTemplate
	if selected == nil || selected.SHA256() != digest {
		t.Fatal("explicit file source not bound")
	}
	cfg := serveNativePlannerConfigWithContext(sf, 512)
	if err := applyServeNativeChatTemplate(sf, &cfg, false); !errors.Is(err, agent.ErrNativeChatTemplate) {
		t.Fatal("missing native consumer accepted")
	}
	if err := applyServeNativeChatTemplate(sf, &cfg, true); err != nil {
		t.Fatal(err)
	}
	if cfg.ChatTemplate != selected {
		t.Fatal("ordinary serving config lost typed selection")
	}
	if err := os.WriteFile(path, []byte("changed after construction"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := applyServeNativeChatTemplate(sf, &cfg, true); err != nil || cfg.ChatTemplate != selected {
		t.Fatal("prepared source unexpectedly reloaded")
	}
	fresh := jinjaServeFlagsForTest(t, jinjaServeSelectedArgs(path, digest)...)
	if err := prepareServeNativeChatTemplate(fresh); !errors.Is(err, agent.ErrNativeChatTemplate) {
		t.Fatal("new startup accepted changed source")
	}
	otherSource := "other"
	otherSum := sha256.Sum256([]byte(otherSource))
	other, err := agent.NewNativeChatTemplate(otherSource, hex.EncodeToString(otherSum[:]), "synthetic-model")
	if err != nil {
		t.Fatal(err)
	}
	cfg.ChatTemplate = other
	if err := applyServeNativeChatTemplate(sf, &cfg, true); !errors.Is(err, agent.ErrNativeChatTemplate) {
		t.Fatal("conflicting typed config overwritten")
	}
}
