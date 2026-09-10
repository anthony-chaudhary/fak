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

type openCodeConfigInstall struct {
	Applied    bool   `json:"applied"`
	ProviderID string `json:"provider_id"`
	Model      string `json:"model"`
	BaseURL    string `json:"base_url"`
	Reason     string `json:"reason,omitempty"`
}

type guardOpenCodeModelLimits struct {
	ID      string `json:"id"`
	Context int    `json:"context_length"`
	Output  int    `json:"max_output_tokens"`
}

func guardIsOpencode(command string) bool {
	return guardAgentBaseName(command) == "opencode"
}

func extractModelFromCommand(command []string) string {
	for i := 0; i < len(command); i++ {
		arg := command[i]
		switch {
		case arg == "-m" || arg == "--model":
			if i+1 < len(command) {
				return command[i+1]
			}
		case strings.HasPrefix(arg, "-m="):
			return strings.TrimPrefix(arg, "-m=")
		case strings.HasPrefix(arg, "--model="):
			return strings.TrimPrefix(arg, "--model=")
		}
	}
	return ""
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
	if targetModel == "" {
		targetModel = strings.TrimSpace(extractModelFromCommand(command))
	}
	if targetModel == "" {
		targetModel = projectassets.DefaultOpenCodeModelID
	}
	cleanModel := strings.TrimPrefix(targetModel, "fak/")
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
	if existing, ok := providerMap["fak"].(map[string]any); ok && existing != nil {
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
	for _, alias := range defaultAliases {
		mergeOpenCodeModel(modelsMap, alias)
	}

	if cmdModel := strings.TrimSpace(extractModelFromCommand(command)); cmdModel != "" {
		cleanCmd := strings.TrimPrefix(cmdModel, "fak/")
		if cleanCmd != "" {
			mergeOpenCodeModel(modelsMap, cleanCmd)
		}
	}

	mergeOpenCodeModel(modelsMap, cleanModel)
	if len(advertised) > 0 && advertised[0].Context > 0 && advertised[0].Output > 0 && advertised[0].Output <= advertised[0].Context {
		model := modelsMap[cleanModel].(map[string]any)
		if _, explicit := model["limit"]; !explicit {
			model["limit"] = map[string]any{"context": advertised[0].Context, "output": advertised[0].Output}
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
	providerMap["fak"] = fakProvider
	config["provider"] = providerMap

	config["model"] = "fak/" + cleanModel
	config["small_model"] = "fak/" + cleanModel

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
		ProviderID: "fak",
		Model:      "fak/" + cleanModel,
		BaseURL:    baseURL,
	}
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
