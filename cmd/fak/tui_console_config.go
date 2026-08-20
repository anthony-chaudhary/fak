package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/tuiplugin"
)

// tuiConsoleConfig is the persisted user-control layer for fak console. Keep it
// deliberately small until a control is wired end-to-end; accepting unused config
// keys would make a typo look like a working preference.
type tuiConsoleConfig struct {
	OverviewPanes []string                              `json:"overview_panes,omitempty"`
	PaneDefaults  map[string]map[string]json.RawMessage `json:"pane_defaults,omitempty"`
}

// defaultTUIConsoleFile is the optional user override file. FAK_CONSOLE_FILE
// wins; otherwise ~/.fak/console.json.
func defaultTUIConsoleFile() string {
	return envOrHomePath("FAK_CONSOLE_FILE", ".fak", "console.json")
}

func loadTUIConsoleConfig(path string) (tuiConsoleConfig, string, error) {
	path = strings.TrimSpace(pathutil.ExpandTilde(path))
	if path == "" {
		return tuiConsoleConfig{}, "", nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return tuiConsoleConfig{}, "", nil
		}
		return tuiConsoleConfig{}, "", fmt.Errorf("read console config %s: %w", path, err)
	}
	var cfg tuiConsoleConfig
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&cfg); err != nil {
		return tuiConsoleConfig{}, "", fmt.Errorf("parse console config %s: %w", path, err)
	}
	cfg.OverviewPanes = normalizeTUIOverviewPaneList(cfg.OverviewPanes)
	cfg.PaneDefaults = normalizeTUIConsolePaneDefaults(cfg.PaneDefaults)
	return cfg, path, nil
}

func saveTUIConsoleConfig(path string, cfg tuiConsoleConfig) (string, error) {
	path = strings.TrimSpace(pathutil.ExpandTilde(path))
	if path == "" {
		return "", fmt.Errorf("console config path is empty")
	}
	cfg.OverviewPanes = normalizeTUIOverviewPaneList(cfg.OverviewPanes)
	cfg.PaneDefaults = normalizeTUIConsolePaneDefaults(cfg.PaneDefaults)
	if err := validateTUIConsoleConfig(cfg); err != nil {
		return "", err
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return "", fmt.Errorf("encode console config: %w", err)
	}
	data = append(data, '\n')
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return "", fmt.Errorf("create console config dir %s: %w", dir, err)
		}
	}
	if err := writeTUIConsoleConfigFile(path, data, os.Rename); err != nil {
		return "", fmt.Errorf("write console config %s: %w", path, err)
	}
	return path, nil
}

func writeTUIConsoleConfigFile(path string, data []byte, replace func(string, string) error) error {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, filepath.Base(path)+".tmp-*")
	if err != nil {
		return fmt.Errorf("create temporary file: %w", err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if err := tmp.Chmod(0o600); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("restrict temporary file: %w", err)
	}
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("write temporary file: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("sync temporary file: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close temporary file: %w", err)
	}
	if err := replace(tmpName, path); err != nil {
		return fmt.Errorf("replace destination: %w", err)
	}
	return nil
}

func prepareTUIPaneArgs(pane tuiplugin.Pane, argv []string) ([]string, error) {
	clean, cfgPath, explicit, err := extractTUIConsoleConfigArg(argv)
	if err != nil {
		return nil, err
	}
	if pane.ID == "settings" {
		if explicit {
			return append([]string{"--path", cfgPath}, clean...), nil
		}
		return clean, nil
	}
	if !explicit {
		cfgPath = defaultTUIConsoleFile()
	}
	cfg, cfgSource, err := loadTUIConsoleConfig(cfgPath)
	if err != nil {
		return nil, err
	}
	return applyTUIConsoleConfigDefaults(pane, cfg, cfgSource, clean)
}

func extractTUIConsoleConfigArg(argv []string) (clean []string, path string, explicit bool, err error) {
	clean = make([]string, 0, len(argv))
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		if arg == "--" {
			clean = append(clean, argv[i:]...)
			return clean, path, explicit, nil
		}
		if arg == "--console-config" {
			if i+1 >= len(argv) {
				return nil, "", false, fmt.Errorf("--console-config requires a file path")
			}
			path = argv[i+1]
			explicit = true
			i++
			continue
		}
		if strings.HasPrefix(arg, "--console-config=") {
			path = strings.TrimPrefix(arg, "--console-config=")
			explicit = true
			continue
		}
		clean = append(clean, arg)
	}
	return clean, path, explicit, nil
}

func applyTUIConsoleConfigDefaults(pane tuiplugin.Pane, cfg tuiConsoleConfig, cfgSource string, argv []string) ([]string, error) {
	if len(cfg.PaneDefaults) == 0 && (pane.ID != "overview" || len(cfg.OverviewPanes) == 0) {
		return argv, nil
	}
	if err := validateTUIConsoleDefaults(cfg); err != nil {
		return nil, err
	}
	defaults := []string{}
	if pane.ID == "overview" && len(cfg.OverviewPanes) > 0 && !hasTUIFlag(argv, "--pane") {
		defaults = append(defaults, "--pane-source", cfgSource)
		for _, name := range cfg.OverviewPanes {
			defaults = append(defaults, "--pane", name)
		}
	}
	paneDefaults := cfg.PaneDefaults[pane.ID]
	if len(paneDefaults) > 0 {
		controls := map[string]tuiplugin.Control{}
		for _, c := range pane.Controls {
			controls[c.ID] = c
		}
		ids := make([]string, 0, len(paneDefaults))
		for id := range paneDefaults {
			ids = append(ids, id)
		}
		sort.Strings(ids)
		for _, id := range ids {
			control := controls[id]
			if control.Flag == "" || hasTUIFlag(argv, control.Flag) {
				continue
			}
			args, err := defaultArgsForTUIControl(control, paneDefaults[id])
			if err != nil {
				return nil, fmt.Errorf("pane_defaults.%s.%s: %w", pane.ID, id, err)
			}
			defaults = append(defaults, args...)
		}
	}
	if len(defaults) == 0 {
		return argv, nil
	}
	out := make([]string, 0, len(defaults)+len(argv))
	out = append(out, defaults...)
	out = append(out, argv...)
	return out, nil
}

func validateTUIConsoleConfig(cfg tuiConsoleConfig) error {
	if err := validateTUIConsoleOverviewPanes(cfg); err != nil {
		return err
	}
	return validateTUIConsoleDefaults(cfg)
}

func validateTUIConsoleOverviewPanes(cfg tuiConsoleConfig) error {
	if len(cfg.OverviewPanes) == 0 {
		return nil
	}
	descs := map[string]tuiplugin.Descriptor{}
	for _, d := range tuiplugin.Descriptors() {
		descs[d.ID] = d
	}
	for _, paneID := range cfg.OverviewPanes {
		desc, ok := descs[paneID]
		if !ok {
			return fmt.Errorf("overview_panes.%s: unknown pane", paneID)
		}
		if !desc.Overview {
			return fmt.Errorf("overview_panes.%s: pane has no overview adapter", paneID)
		}
	}
	return nil
}

func validateTUIConsoleDefaults(cfg tuiConsoleConfig) error {
	if len(cfg.PaneDefaults) == 0 {
		return nil
	}
	descs := tuiplugin.Descriptors()
	panes := map[string]tuiplugin.Descriptor{}
	for _, d := range descs {
		panes[d.ID] = d
	}
	paneIDs := make([]string, 0, len(cfg.PaneDefaults))
	for id := range cfg.PaneDefaults {
		paneIDs = append(paneIDs, id)
	}
	sort.Strings(paneIDs)
	for _, paneID := range paneIDs {
		if paneID == "config" || paneID == "settings" {
			return fmt.Errorf("pane_defaults.%s: settings pane cannot be defaulted", paneID)
		}
		desc, ok := panes[paneID]
		if !ok {
			return fmt.Errorf("pane_defaults.%s: unknown pane", paneID)
		}
		controls := map[string]tuiplugin.Control{}
		for _, c := range desc.Controls {
			controls[c.ID] = c
		}
		controlIDs := make([]string, 0, len(cfg.PaneDefaults[paneID]))
		for id := range cfg.PaneDefaults[paneID] {
			controlIDs = append(controlIDs, id)
		}
		sort.Strings(controlIDs)
		for _, controlID := range controlIDs {
			c, ok := controls[controlID]
			if !ok {
				return fmt.Errorf("pane_defaults.%s.%s: unknown control", paneID, controlID)
			}
			if c.Flag == "--console-config" {
				return fmt.Errorf("pane_defaults.%s.%s: dispatcher-only control cannot be defaulted", paneID, controlID)
			}
			if c.Flag == "" {
				return fmt.Errorf("pane_defaults.%s.%s: control has no CLI flag", paneID, controlID)
			}
			if _, err := defaultArgsForTUIControl(c, cfg.PaneDefaults[paneID][controlID]); err != nil {
				return fmt.Errorf("pane_defaults.%s.%s: %w", paneID, controlID, err)
			}
		}
	}
	return nil
}

func defaultArgsForTUIControl(c tuiplugin.Control, raw json.RawMessage) ([]string, error) {
	if len(bytes.TrimSpace(raw)) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	switch strings.ToLower(strings.TrimSpace(c.Kind)) {
	case "toggle", "action":
		on, err := rawTUIBool(raw)
		if err != nil {
			return nil, err
		}
		if !on {
			return nil, nil
		}
		return []string{c.Flag}, nil
	default:
		value, err := rawTUIScalar(raw)
		if err != nil {
			return nil, err
		}
		if value == "" {
			return nil, nil
		}
		if len(c.Options) > 0 && !tuiControlAllowsOption(c, value) {
			return nil, fmt.Errorf("must be one of %s", strings.Join(c.Options, ", "))
		}
		return []string{c.Flag, value}, nil
	}
}

func tuiControlAllowsOption(c tuiplugin.Control, value string) bool {
	for _, option := range c.Options {
		if value == option {
			return true
		}
	}
	return false
}

func rawTUIBool(raw json.RawMessage) (bool, error) {
	var b bool
	if err := json.Unmarshal(raw, &b); err == nil {
		return b, nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		v, err := strconv.ParseBool(strings.TrimSpace(s))
		if err != nil {
			return false, fmt.Errorf("must be a boolean")
		}
		return v, nil
	}
	return false, fmt.Errorf("must be a boolean")
}

func rawTUIScalar(raw json.RawMessage) (string, error) {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return strings.TrimSpace(s), nil
	}
	var n json.Number
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	if err := dec.Decode(&n); err == nil {
		return n.String(), nil
	}
	return "", fmt.Errorf("must be a string or number")
}

func hasTUIFlag(argv []string, flagName string) bool {
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		if arg == "--" {
			return false
		}
		if arg == flagName || strings.HasPrefix(arg, flagName+"=") {
			return true
		}
	}
	return false
}

func normalizeTUIConsolePaneDefaults(in map[string]map[string]json.RawMessage) map[string]map[string]json.RawMessage {
	if len(in) == 0 {
		return nil
	}
	out := map[string]map[string]json.RawMessage{}
	for pane, controls := range in {
		pane = strings.TrimSpace(strings.ToLower(pane))
		if pane == "" {
			continue
		}
		dst := map[string]json.RawMessage{}
		for control, raw := range controls {
			control = strings.TrimSpace(strings.ToLower(control))
			if control == "" {
				continue
			}
			dst[control] = raw
		}
		out[pane] = dst
	}
	return out
}

func normalizeTUIOverviewPaneList(values []string) []string {
	out := make([]string, 0, len(values))
	seen := map[string]bool{}
	for _, v := range values {
		name := strings.TrimSpace(strings.ToLower(v))
		if name == "config" {
			name = "settings"
		}
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		out = append(out, name)
	}
	return out
}
