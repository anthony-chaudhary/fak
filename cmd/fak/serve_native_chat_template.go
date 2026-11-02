package main

import (
	"encoding/hex"
	"flag"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func registerServeNativeChatTemplateFlags(fs *flag.FlagSet, sf *serveFlags) {
	sf.nativeChatTemplateFile = fs.String("native-chat-template-file", "", "explicit local text Jinja template; requires --native-chat-template-sha256 and --native-chat-template-model; native local-model serving only")
	sf.nativeChatTemplateSHA256 = fs.String("native-chat-template-sha256", "", "independently selected lowercase SHA-256 of the exact native chat template bytes")
	sf.nativeChatTemplateModel = fs.String("native-chat-template-model", "", "exact --model identity bound to the selected native text Jinja template")
}
func serveNativeChatTemplateSelected(sf *serveFlags) bool {
	if sf == nil {
		return false
	}
	for _, name := range []string{"native-chat-template-file", "native-chat-template-sha256", "native-chat-template-model"} {
		if sf.isExplicitFlag(name) {
			return true
		}
	}
	for _, p := range []*string{sf.nativeChatTemplateFile, sf.nativeChatTemplateSHA256, sf.nativeChatTemplateModel} {
		if p != nil && *p != "" {
			return true
		}
	}
	return false
}
func serveNativeChatTemplateRefusal(reason string) error {
	return fmt.Errorf("%w: %s", agent.ErrNativeChatTemplate, reason)
}

// Validation is flag-only. Print-features/effective-config do not read the file.
func validateServeNativeChatTemplateFlags(sf *serveFlags) error {
	if !serveNativeChatTemplateSelected(sf) {
		return nil
	}
	for _, name := range []string{"native-chat-template-file", "native-chat-template-sha256", "native-chat-template-model"} {
		if !sf.isExplicitFlag(name) {
			return serveNativeChatTemplateRefusal("incomplete explicit CLI selection")
		}
	}
	if sf.nativeChatTemplateFile == nil || sf.nativeChatTemplateSHA256 == nil || sf.nativeChatTemplateModel == nil ||
		*sf.nativeChatTemplateFile == "" || len(*sf.nativeChatTemplateFile) > 4096 || len(*sf.nativeChatTemplateModel) > 4096 || strings.TrimSpace(*sf.nativeChatTemplateModel) == "" {
		return serveNativeChatTemplateRefusal("incomplete explicit CLI selection")
	}
	digest := *sf.nativeChatTemplateSHA256
	if len(digest) != 64 {
		return serveNativeChatTemplateRefusal("digest binding")
	}
	raw, err := hex.DecodeString(digest)
	if err != nil || len(raw) != 32 || digest != strings.ToLower(digest) {
		return serveNativeChatTemplateRefusal("digest binding")
	}
	if sf.model == nil || *sf.nativeChatTemplateModel != *sf.model {
		return serveNativeChatTemplateRefusal("model binding")
	}
	if sf.ggufPath == nil || strings.TrimSpace(*sf.ggufPath) == "" || sf.mock == nil || *sf.mock ||
		sf.baseURL == nil || strings.TrimSpace(*sf.baseURL) != "" || len(sf.replicaBaseURLs.Values()) != 0 ||
		sf.qwen38Runtime == nil || *sf.qwen38Runtime != qwen38RuntimeNative {
		return serveNativeChatTemplateRefusal("local native model route required")
	}
	return nil
}

// The selected file is compiled once after flag-only exits and before model load.
func prepareServeNativeChatTemplate(sf *serveFlags) error {
	if err := validateServeNativeChatTemplateFlags(sf); err != nil {
		return err
	}
	if !serveNativeChatTemplateSelected(sf) {
		return nil
	}
	if sf.nativeChatTemplate != nil {
		return serveNativeChatTemplateRefusal("selection already prepared")
	}
	selected, err := agent.LoadNativeChatTemplateFile(*sf.nativeChatTemplateFile, *sf.nativeChatTemplateSHA256, *sf.nativeChatTemplateModel)
	if err != nil {
		return err
	}
	sf.nativeChatTemplate = selected
	return nil
}

// This is the ordinary buildGateway configuration seam, not a standalone loader.
func applyServeNativeChatTemplate(sf *serveFlags, cfg *agent.InKernelPlannerConfig, hasNativeModelAndTokenizer bool) error {
	if err := validateServeNativeChatTemplateFlags(sf); err != nil {
		return err
	}
	if !serveNativeChatTemplateSelected(sf) {
		if sf != nil && sf.nativeChatTemplate != nil {
			return serveNativeChatTemplateRefusal("unexpected prepared selection")
		}
		return nil
	}
	if cfg == nil || sf.nativeChatTemplate == nil || !hasNativeModelAndTokenizer {
		return serveNativeChatTemplateRefusal("selected native consumer unavailable")
	}
	if cfg.ChatTemplate != nil && cfg.ChatTemplate != sf.nativeChatTemplate {
		return serveNativeChatTemplateRefusal("conflicting typed selections")
	}
	cfg.ChatTemplate = sf.nativeChatTemplate
	return nil
}
