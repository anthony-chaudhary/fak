package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/projectassets"
)

const (
	guardOpenCodeLegacyProviderID = "fak"
	guardOpenCodeRouterProviderID = "fak-router"
)

type openCodeConfigInstall struct {
	Applied    bool   `json:"applied"`
	ProviderID string `json:"provider_id"`
	Model      string `json:"model"`
	BaseURL    string `json:"base_url"`
	Reason     string `json:"reason,omitempty"`
	// Command is the child argv to launch: the input with `--standalone` added
	// to a bare `opencode run` so the injected config reaches the provider call.
	Command []string `json:"command,omitempty"`
}

type guardOpenCodeModelLimits struct {
	ID      string `json:"id"`
	Context int    `json:"context_length"`
	Output  int    `json:"max_output_tokens"`
}

func guardIsOpencode(command string) bool {
	return guardAgentBaseName(command) == "opencode"
}

// guardOpenCodeStandaloneCommand inserts `--standalone` after a leading `run`
// in an OpenCode child's argv. OpenCode v2's `run` otherwise hands the prompt to
// its background service, which keeps the environment it started with, so the
// guard's OPENCODE_CONFIG_CONTENT never applies and the provider call bypasses
// the guard proxy. A `--standalone` or `--server` before `--` is left alone.
func guardOpenCodeStandaloneCommand(command []string) []string {
	if len(command) < 2 || !guardIsOpencode(command[0]) || command[1] != "run" {
		return command
	}
	for _, arg := range command[2:] {
		if arg == "--" {
			break
		}
		if arg == "--standalone" || arg == "--server" || strings.HasPrefix(arg, "--standalone=") || strings.HasPrefix(arg, "--server=") {
			return command
		}
	}
	out := make([]string, 0, len(command)+1)
	out = append(out, command[0], "run", "--standalone")
	return append(out, command[2:]...)
}

func extractModelFromCommand(command []string) string {
	model := ""
	for i := 0; i < len(command); i++ {
		arg := command[i]
		if arg == "--" {
			break
		}
		switch {
		case arg == "-m" || arg == "--model":
			if i+1 < len(command) && command[i+1] != "--" {
				model = command[i+1]
				i++
			}
		case strings.HasPrefix(arg, "-m="):
			model = strings.TrimPrefix(arg, "-m=")
		case strings.HasPrefix(arg, "--model="):
			model = strings.TrimPrefix(arg, "--model=")
		}
	}
	return model
}

func installGuardOpenCodeConfig(command []string, gwURL, modelID string, getenv func(string) string, advertised ...guardOpenCodeModelLimits) ([][2]string, openCodeConfigInstall) {
	if getenv == nil {
		getenv = func(string) string { return "" }
	}

	if len(command) == 0 || !guardIsOpencode(command[0]) {
		return nil, openCodeConfigInstall{Reason: "non-opencode-child"}
	}
	if gwURL == "" {
		return nil, openCodeConfigInstall{Reason: "empty-gateway-url"}
	}

	targetModel := strings.TrimSpace(modelID)
	commandModel := strings.TrimSpace(extractModelFromCommand(command))
	if targetModel == "" {
		targetModel = commandModel
	}
	if targetModel == "" {
		targetModel = projectassets.DefaultOpenCodeModelID
	}
	providerID := guardOpenCodeLegacyProviderID
	if commandProvider, ok := guardOpenCodeProviderPrefix(commandModel); ok {
		providerID = commandProvider
	} else if targetProvider, ok := guardOpenCodeProviderPrefix(targetModel); ok {
		providerID = targetProvider
	}
	cleanModel := guardOpenCodeModelName(targetModel)
	if cleanModel == "" {
		cleanModel = projectassets.DefaultOpenCodeModelID
	}

	baseURL := strings.TrimRight(gwURL, "/") + "/v1"

	config := make(map[string]any)
	if raw := strings.TrimSpace(getenv("OPENCODE_CONFIG_CONTENT")); raw != "" {
		var parsed map[string]any
		if err := json.Unmarshal([]byte(raw), &parsed); err == nil && parsed != nil {
			config = parsed
		}
	}

	var providerMap map[string]any
	if existing, ok := config["provider"].(map[string]any); ok && existing != nil {
		providerMap = existing
	} else {
		providerMap = make(map[string]any)
	}

	var fakProvider map[string]any
	if existing, ok := providerMap[providerID].(map[string]any); ok && existing != nil {
		fakProvider = existing
	} else {
		fakProvider = make(map[string]any)
	}

	var modelsMap map[string]any
	if existing, ok := fakProvider["models"].(map[string]any); ok && existing != nil {
		modelsMap = existing
	} else {
		modelsMap = make(map[string]any)
	}

	defaultAliases := []string{
		"fak-local",
		"qwen38",
		"qwen38:27b",
		"qwen2.5-coder:7b",
		"qwen2.5-coder:32b",
		"qwen-2.5-coder-32b-instruct",
		"glm-5.2",
		"glm-5.3-flash",
	}
	if providerID == guardOpenCodeLegacyProviderID {
		for _, alias := range defaultAliases {
			mergeOpenCodeModel(modelsMap, alias)
		}
	}

	if providerID == guardOpenCodeLegacyProviderID {
		if cmdModel := strings.TrimSpace(extractModelFromCommand(command)); cmdModel != "" {
			cleanCmd := guardOpenCodeModelName(cmdModel)
			if cleanCmd != "" {
				mergeOpenCodeModel(modelsMap, cleanCmd)
			}
		}
	}

	mergeOpenCodeModel(modelsMap, cleanModel)
	if len(advertised) > 0 && advertised[0].Context > 0 {
		// Advertised output may be 0 (the gateway /v1/models roster sets context_length
		// but not max_output_tokens) or nonsensically >= the context window. Letting it
		// through unset makes OpenCode default max_tokens to 32000, which then overflows a
		// 32768-token window on any prompt >768 tokens ("in-kernel request exceeds the
		// context window"). Derive a bounded fraction of the remaining window instead.
		outputLimit := advertised[0].Output
		if outputLimit <= 0 || outputLimit >= advertised[0].Context {
			outputLimit = advertised[0].Context / 4
			if outputLimit > 8192 {
				outputLimit = 8192
			}
			if outputLimit < 512 {
				outputLimit = 512
			}
		}
		model := modelsMap[cleanModel].(map[string]any)
		if _, explicit := model["limit"]; !explicit {
			model["limit"] = map[string]any{"context": advertised[0].Context, "output": outputLimit}
		}
	}

	fakProvider["npm"] = "@ai-sdk/openai-compatible"
	fakProvider["name"] = "fak (kernel-adjudicated)"
	options, _ := fakProvider["options"].(map[string]any)
	if options == nil {
		options = make(map[string]any)
	}
	options["baseURL"] = "{env:OPENAI_BASE_URL}"
	options["apiKey"] = "{env:OPENAI_API_KEY}"
	fakProvider["options"] = options
	fakProvider["models"] = modelsMap
	providerMap[providerID] = fakProvider
	config["provider"] = providerMap

	guardedModel := providerID + "/" + cleanModel
	config["model"] = guardedModel
	config["small_model"] = guardedModel
	// The guard owns the child model plane. OpenCode gives agent.<name>.model
	// precedence over the session default, so a subagent left on its own pin
	// (especially a local router opt-in) would clear the guarded provider and
	// escape the kernel adjudication. Rewrite every explicit child pin onto the
	// guarded model; an agent with no model still inherits the guarded parent.
	guardOpenCodeSubagentOverlay(config, guardedModel)

	encodedJSON, err := json.Marshal(config)
	if err != nil {
		encodedJSON = []byte("{}")
	}

	injected := [][2]string{
		{"OPENCODE_CONFIG_CONTENT", string(encodedJSON)},
	}
	if strings.TrimSpace(getenv("OPENAI_API_KEY")) == "" {
		injected = append(injected, [2]string{"OPENAI_API_KEY", "fak-guard-placeholder"})
	}

	return injected, openCodeConfigInstall{
		Applied:    true,
		ProviderID: providerID,
		Model:      guardedModel,
		BaseURL:    baseURL,
		Command:    guardOpenCodeCommand(command, guardedModel),
	}
}

func guardOpenCodeModelName(model string) string {
	model = strings.TrimSpace(model)
	if _, ok := guardOpenCodeProviderPrefix(model); ok {
		return model[strings.IndexByte(model, '/')+1:]
	}
	return model
}

func guardOpenCodeProviderPrefix(model string) (string, bool) {
	model = strings.TrimSpace(model)
	for _, providerID := range []string{guardOpenCodeRouterProviderID, guardOpenCodeLegacyProviderID} {
		if strings.HasPrefix(model, providerID+"/") {
			return providerID, true
		}
	}
	return "", false
}

func guardOpenCodeCommand(command []string, guardedModel string) []string {
	out := guardOpenCodeStandaloneCommand(command)
	return pinOpenCodeModelArgs(out, 2, guardedModel)
}

func pinOpenCodeModelArgs(args []string, start int, guardedModel string) []string {
	out := args
	cloned := false
	clone := func() {
		if !cloned {
			out = append([]string(nil), args...)
			cloned = true
		}
	}
	for i := start; i < len(out); i++ {
		arg := out[i]
		if arg == "--" {
			break
		}
		switch {
		case arg == "-m" || arg == "--model":
			if i+1 < len(out) && out[i+1] != "--" {
				clone()
				out[i+1] = guardedModel
				i++
			}
		case strings.HasPrefix(arg, "-m="):
			clone()
			out[i] = "-m=" + guardedModel
		case strings.HasPrefix(arg, "--model="):
			clone()
			out[i] = "--model=" + guardedModel
		}
	}
	return out
}

func discoverGuardOpenCodeModelLimits(gwURL, modelID string) guardOpenCodeModelLimits {
	client := &http.Client{Timeout: 500 * time.Millisecond}
	resp, err := client.Get(strings.TrimRight(gwURL, "/") + "/v1/models")
	if err != nil {
		return guardOpenCodeModelLimits{}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return guardOpenCodeModelLimits{}
	}
	var roster struct {
		Data []guardOpenCodeModelLimits `json:"data"`
	}
	if json.NewDecoder(resp.Body).Decode(&roster) != nil {
		return guardOpenCodeModelLimits{}
	}
	for _, row := range roster.Data {
		if row.ID == modelID {
			return row
		}
	}
	if modelID == "" && len(roster.Data) > 0 {
		return roster.Data[0]
	}
	return guardOpenCodeModelLimits{}
}

// guardOpenCodeSubagentOverlay forces every explicitly pinned OpenCode subagent
// onto the guard-selected model. Unlike the private failover repair (which
// preserves local router opt-ins), the guard overlay is a security boundary and
// must repoint local pins too: a child session that kept a local pin would
// speak to a different endpoint than the guarded gateway. Agents with no model
// are left to inherit the guarded parent model.
func guardOpenCodeSubagentOverlay(config map[string]any, guardedModel string) {
	agents, ok := config["agent"].(map[string]any)
	if !ok {
		return
	}
	for _, spec := range agents {
		entry, ok := spec.(map[string]any)
		if !ok {
			continue
		}
		if _, hasModel := entry["model"]; hasModel {
			entry["model"] = guardedModel
		}
	}
}

func mergeOpenCodeModel(models map[string]any, id string) {
	model, _ := models[id].(map[string]any)
	if model == nil {
		model = make(map[string]any)
	}
	if _, exists := model["name"]; !exists {
		model["name"] = id
	}
	models[id] = model
}

func printGuardOpenCodeNote(w io.Writer, in openCodeConfigInstall) {
	if !in.Applied {
		return
	}
	fmt.Fprintf(w, "fak guard: OpenCode session wired via OPENCODE_CONFIG_CONTENT (provider=%s, base_url=%s, model=%s) — root settings unaffected\n", in.ProviderID, in.BaseURL, in.Model)
}
