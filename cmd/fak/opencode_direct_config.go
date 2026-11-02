package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"runtime"
	"sort"
	"strings"
)

// openCodeDirectConfigEnvironment preserves the winning inherited inline document
// before replacing its environment entry. Like os/exec, the last duplicate wins;
// Windows environment names are case-insensitive. Errors contain no config data.
func openCodeDirectConfigEnvironment(env []string, generated []byte) ([]string, error) {
	const name = "OPENCODE_CONFIG_CONTENT"
	matches := func(key string) bool {
		return key == name || (runtime.GOOS == "windows" && strings.EqualFold(key, name))
	}
	inherited := ""
	for _, entry := range env {
		key, value, ok := strings.Cut(entry, "=")
		if ok && matches(key) {
			inherited = value
		}
	}
	config, err := mergeOpenCodeDirectSessionConfig(generated, inherited)
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, len(env)+1)
	for _, entry := range env {
		key, _, ok := strings.Cut(entry, "=")
		if !ok || !matches(key) {
			out = append(out, entry)
		}
	}
	return append(out, name+"="+string(config)), nil
}

// mergeOpenCodeDirectSessionConfig changes only the legacy direct route already
// owned by this launcher: model, snapshot=false, provider.fak's compatible adapter
// and baseURL, and the existing launch session header. All other inherited leaves
// retain their JSON bytes, including numeric limits, credentials and permissions.
//
// This is not an effective-config loader. It reads no global/project files,
// starts no OpenCode service, and installs no catalog caps or child-model pins.
// OpenCode v2.0.15 normalizes a native providers.fak entry over the whole legacy
// provider.fak entry, so that ambiguous route is refused rather than erased or
// modified through a legacy field the client would ignore.
func mergeOpenCodeDirectSessionConfig(generated []byte, inherited string) ([]byte, error) {
	if inherited == "" {
		return generated, nil
	}
	config, err := openCodeDirectObject([]byte(inherited), "inline configuration")
	if err != nil {
		return nil, err
	}
	defaults, err := openCodeDirectObject(generated, "generated configuration")
	if err != nil {
		return nil, err
	}
	if raw, ok := config["providers"]; ok {
		native, err := openCodeDirectObject(raw, "providers")
		if err != nil {
			return nil, err
		}
		if _, ok := native["fak"]; ok {
			return nil, fmt.Errorf("direct inline configuration has native providers.fak; explicit native-route composition is not supported")
		}
	}

	providers, err := openCodeDirectObject(config["provider"], "provider")
	if err != nil {
		return nil, err
	}
	provider, err := openCodeDirectObject(providers["fak"], "provider.fak")
	if err != nil {
		return nil, err
	}
	defaultProviders, err := openCodeDirectObject(defaults["provider"], "generated provider")
	if err != nil {
		return nil, err
	}
	defaultProvider, err := openCodeDirectObject(defaultProviders["fak"], "generated provider.fak")
	if err != nil {
		return nil, err
	}
	if raw, ok := provider["npm"]; ok {
		var existing, required string
		if json.Unmarshal(raw, &existing) != nil || json.Unmarshal(defaultProvider["npm"], &required) != nil || existing != required {
			return nil, fmt.Errorf("direct inline provider.fak uses an incompatible adapter")
		}
	}
	// Provider-wide API/environment aliases have independent routing/credential
	// semantics. This leaf does not expand or translate them.
	for _, key := range []string{"api", "env", "id"} {
		if _, ok := provider[key]; ok {
			return nil, fmt.Errorf("direct inline provider.fak has an unsupported routing or credential selector")
		}
	}
	for key, value := range defaultProvider {
		if key != "options" && key != "models" {
			if _, ok := provider[key]; !ok {
				provider[key] = value
			}
		}
	}

	options, err := openCodeDirectObject(provider["options"], "provider.fak.options")
	if err != nil {
		return nil, err
	}
	defaultOptions, err := openCodeDirectObject(defaultProvider["options"], "generated provider.fak.options")
	if err != nil {
		return nil, err
	}
	// Never carry inherited credentials/options to a different or unknown
	// endpoint. Compare literals only: env/file indirection is not expanded.
	if raw, ok := options["baseURL"]; ok {
		var existing, required string
		if json.Unmarshal(raw, &existing) != nil || json.Unmarshal(defaultOptions["baseURL"], &required) != nil ||
			existing != required || strings.Contains(existing, "{env:") || strings.Contains(existing, "{file:") || strings.Contains(existing, "${") {
			return nil, fmt.Errorf("direct inline provider.fak endpoint does not match the selected literal endpoint")
		}
	} else if len(options) != 0 {
		return nil, fmt.Errorf("direct inline provider.fak options have no explicit bound endpoint")
	}
	options["baseURL"] = defaultOptions["baseURL"]
	headers, err := openCodeDirectObject(options["headers"], "provider.fak.options.headers")
	if err != nil {
		return nil, err
	}
	defaultHeaders, err := openCodeDirectObject(defaultOptions["headers"], "generated provider.fak.options.headers")
	if err != nil {
		return nil, err
	}
	for key, value := range defaultHeaders {
		present := false
		for existing := range headers {
			if strings.EqualFold(existing, key) {
				present = true
				break
			}
		}
		if !present {
			headers[key] = value
		}
	}
	if len(headers) != 0 || options["headers"] != nil {
		options["headers"] = encodeOpenCodeDirectObject(headers)
	}
	provider["options"] = encodeOpenCodeDirectObject(options)

	models, err := openCodeDirectObject(provider["models"], "provider.fak.models")
	if err != nil {
		return nil, err
	}
	defaultModels, err := openCodeDirectObject(defaultProvider["models"], "generated provider.fak.models")
	if err != nil {
		return nil, err
	}
	var selected string
	if json.Unmarshal(defaults["model"], &selected) != nil || !strings.HasPrefix(selected, "fak/") {
		return nil, fmt.Errorf("generated direct configuration has no selected fak model")
	}
	selected = strings.TrimPrefix(selected, "fak/")
	if raw, ok := models[selected]; ok {
		entry, err := openCodeDirectObject(raw, "selected provider.fak model")
		if err != nil {
			return nil, err
		}
		// Model-level routing, HTTP and variant overrides can supersede the
		// provider binding. Refuse instead of moving or erasing those settings.
		for _, key := range []string{"provider", "options", "headers", "variants"} {
			if _, ok := entry[key]; ok {
				return nil, fmt.Errorf("direct inline selected model has an unsupported routing or HTTP override")
			}
		}
		if rawID, ok := entry["id"]; ok {
			var id string
			if json.Unmarshal(rawID, &id) != nil || id != selected {
				return nil, fmt.Errorf("direct inline selected model has a different wire model id")
			}
		}
		// Preserve the entire selected entry, not just recognized limit fields.
	} else {
		value, ok := defaultModels[selected]
		if !ok {
			return nil, fmt.Errorf("generated direct configuration lacks its selected model")
		}
		models[selected] = value
	}
	// No new limits or unselected model aliases are merged into inherited
	// configuration. The no-inheritance path above remains byte-identical.
	provider["models"] = encodeOpenCodeDirectObject(models)
	providers["fak"] = encodeOpenCodeDirectObject(provider)
	config["provider"] = encodeOpenCodeDirectObject(providers)
	for key, value := range defaults {
		if key != "provider" {
			if _, ok := config[key]; !ok {
				config[key] = value
			}
		}
	}
	config["model"] = defaults["model"]
	config["snapshot"] = defaults["snapshot"]
	return encodeOpenCodeDirectObject(config), nil
}

// Absent owned objects are created; explicit null, arrays, strings, malformed
// JSON and JSONC are not silently replaced. Diagnostics never echo their bytes.
func openCodeDirectObject(raw json.RawMessage, field string) (map[string]json.RawMessage, error) {
	if len(raw) == 0 {
		return make(map[string]json.RawMessage), nil
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(raw, &object); err != nil || object == nil {
		return nil, fmt.Errorf("direct %s must be a JSON object; unsupported input was not discarded", field)
	}
	return object, nil
}

// Only modified object boundaries are serialized. RawMessage avoids float64
// rounding and retains unmodified values (including nested JSON) byte-for-byte.
func encodeOpenCodeDirectObject(object map[string]json.RawMessage) json.RawMessage {
	keys := make([]string, 0, len(object))
	for key := range object {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var out bytes.Buffer
	out.WriteByte('{')
	for i, key := range keys {
		if i != 0 {
			out.WriteByte(',')
		}
		encodedKey, _ := json.Marshal(key)
		out.Write(encodedKey)
		out.WriteByte(':')
		out.Write(object[key])
	}
	out.WriteByte('}')
	return out.Bytes()
}
