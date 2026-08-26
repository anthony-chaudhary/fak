package fleetpane

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
)

func FindRepoRoot(start string) (string, error) {
	if start == "" {
		start = "."
	}
	abs, err := filepath.Abs(start)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(abs)
	if err == nil && !info.IsDir() {
		abs = filepath.Dir(abs)
	}
	for {
		if _, err := os.Stat(filepath.Join(abs, "VERSION")); err == nil {
			return abs, nil
		}
		next := filepath.Dir(abs)
		if next == abs {
			return "", fmt.Errorf("could not find repo root from %s", start)
		}
		abs = next
	}
}

func LoadConfig(root string) (Config, error) {
	if root == "" {
		var err error
		root, err = FindRepoRoot(".")
		if err != nil {
			return Config{}, err
		}
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return Config{}, err
	}
	paths := ConfigPaths(abs)
	doc := defaultConfigDoc(abs)
	for _, key := range []string{"example", "loop_catalog", "local"} {
		overlay := readJSONMap(paths[key])
		doc = deepMerge(doc, overlay)
	}
	applyEnvOverrides(doc)
	cfg := normalizeConfig(doc, abs)
	cfg.Paths = paths
	_, err = os.Stat(paths["local"])
	cfg.LocalExists = err == nil
	return cfg, nil
}

func ConfigPaths(root string) map[string]string {
	return map[string]string{
		"example":      filepath.Join(root, "tools", "control_pane.example.json"),
		"loop_catalog": filepath.Join(root, "tools", "control_pane.loops.json"),
		"local":        filepath.Join(root, "tools", "_registry", "control_pane.local.json"),
	}
}

func defaultConfigDoc(root string) map[string]any {
	host, _ := os.Hostname()
	python := "python"
	if path, err := exec.LookPath("python"); err == nil {
		python = path
	}
	return map[string]any{
		"schema":                ConfigSchema,
		"session_window_h":      float64(6),
		"target":                float64(4),
		"python":                python,
		"user_home":             userHome(),
		"job_dir":               siblingJob(root),
		"claude_exe":            "",
		"registry_dir":          "tools/_registry",
		"machine_dir":           "tools/_registry/machines",
		"machine_id":            host,
		"machine_stale_min":     float64(15),
		"watchdog_log_dir":      "tools/_watchdog",
		"refresh_timeout_s":     float64(90),
		"git_remote":            "origin",
		"loops":                 map[string]any{},
		"supervisor_status_cmd": []any{python, "tools/dos_supervisor_status.py", "--json"},
		"monitor_status_cmd":    []any{"fak", "fleet", "monitor", "--json"},
		"control_tick": map[string]any{
			"task_name":             "FleetControlPaneTick",
			"register_script":       "tools/register_control_pane_tick.ps1",
			"posix_register_script": "tools/register_control_pane_tick.sh",
			"interval_min":          float64(5),
		},
		"watchdogs": map[string]any{
			"supervisor": map[string]any{
				"task_name":       "FleetSupervisorWatchdog",
				"script":          "tools/fleet_supervisor_watchdog.ps1",
				"register_script": "tools/register_supervisor_watchdog.ps1",
				"interval_min":    float64(5),
			},
			"resume": map[string]any{
				"task_name":       "FleetResumeWatchdog",
				"script":          "tools/fleet_resume_watchdog.ps1",
				"register_script": "tools/register_resume_watchdog.ps1",
				"interval_min":    float64(10),
			},
		},
	}
}

func userHome() string {
	if home, err := os.UserHomeDir(); err == nil {
		return home
	}
	return ""
}

func siblingJob(root string) string {
	job := filepath.Join(filepath.Dir(root), "job")
	if info, err := os.Stat(job); err == nil && info.IsDir() {
		return job
	}
	return ""
}

func readJSONMap(path string) map[string]any {
	raw, err := os.ReadFile(path)
	if err != nil {
		return map[string]any{}
	}
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		return map[string]any{}
	}
	return doc
}

func deepMerge(base, overlay map[string]any) map[string]any {
	out := make(map[string]any, len(base)+len(overlay))
	for k, v := range base {
		out[k] = v
	}
	for k, v := range overlay {
		if bv, ok := out[k].(map[string]any); ok {
			if ov, ok := v.(map[string]any); ok {
				out[k] = deepMerge(bv, ov)
				continue
			}
		}
		out[k] = v
	}
	return out
}

func applyEnvOverrides(doc map[string]any) {
	envMap := map[string]string{
		"FLEET_PYTHON":      "python",
		"FLEET_USER_HOME":   "user_home",
		"FLEET_JOB_DIR":     "job_dir",
		"FLEET_CLAUDE_EXE":  "claude_exe",
		"FLEET_REG_DIR":     "registry_dir",
		"FLEET_MACHINE_DIR": "machine_dir",
		"FLEET_MACHINE_ID":  "machine_id",
	}
	for env, key := range envMap {
		if val := os.Getenv(env); val != "" {
			doc[key] = val
		}
	}
}

func normalizeConfig(doc map[string]any, root string) Config {
	cfg := Config{
		Root:             root,
		Python:           stringValue(doc["python"]),
		UserHome:         resolvePathString(stringValue(doc["user_home"]), root),
		JobDir:           resolvePathString(stringValue(doc["job_dir"]), root),
		ClaudeExe:        stringValue(doc["claude_exe"]),
		RegistryDir:      resolvePathString(stringValue(doc["registry_dir"]), root),
		MachineDir:       resolvePathString(stringValue(doc["machine_dir"]), root),
		MachineID:        stringValue(doc["machine_id"]),
		WatchdogLogDir:   resolvePathString(stringValue(doc["watchdog_log_dir"]), root),
		GitRemote:        stringValueDefault(doc["git_remote"], "origin"),
		Target:           intValueDefault(doc["target"], 4),
		SessionWindowH:   floatValueDefault(doc["session_window_h"], 6),
		RefreshTimeoutS:  intValueDefault(doc["refresh_timeout_s"], 90),
		MachineStaleMin:  floatValueDefault(doc["machine_stale_min"], 15),
		Loops:            parseLoops(doc["loops"]),
		SupervisorStatus: stringSlice(doc["supervisor_status_cmd"]),
		MonitorStatus:    stringSlice(doc["monitor_status_cmd"]),
		ControlTick:      mapValue(doc["control_tick"]),
		Watchdogs:        nestedMap(doc["watchdogs"]),
	}
	if cfg.Python == "" {
		cfg.Python = "python"
	}
	return cfg
}

func parseLoops(raw any) map[string]LoopSpec {
	loops := map[string]LoopSpec{}
	for name, val := range mapValue(raw) {
		specMap, ok := val.(map[string]any)
		if !ok {
			continue
		}
		spec := LoopSpec{
			Enabled:         true,
			StatusCmd:       stringSlice(specMap["status_cmd"]),
			RecoverCmd:      stringSlice(specMap["recover_cmd"]),
			AutoRecover:     boolValue(specMap["auto_recover"]),
			Action:          stringValue(specMap["action"]),
			TimeoutS:        intValueDefault(specMap["timeout_s"], 0),
			RecoverTimeoutS: intValueDefault(specMap["recover_timeout_s"], 0),
			Cwd:             stringValue(specMap["cwd"]),
			OKReturnCodes:   intSlice(specMap["ok_returncodes"]),
		}
		if rawEnabled, exists := specMap["enabled"]; exists && !boolValueDefault(rawEnabled, true) {
			spec.Enabled = false
		}
		loops[name] = spec
	}
	return loops
}

func SourceEntries(root string) map[string][]SourceEntry {
	paths := ConfigPaths(root)
	order := []struct {
		source string
		key    string
	}{
		{"example", "example"},
		{"repo", "loop_catalog"},
		{"local", "local"},
	}
	out := map[string][]SourceEntry{}
	for _, item := range order {
		doc := readJSONMap(paths[item.key])
		for name, val := range mapValue(doc["loops"]) {
			spec, ok := val.(map[string]any)
			if !ok {
				continue
			}
			out[name] = append(out[name], SourceEntry{
				Source:        item.source,
				Path:          paths[item.key],
				DisplayPath:   displayPath(paths[item.key], root),
				Enabled:       boolValueDefault(spec["enabled"], true),
				HasStatusCmd:  len(stringSlice(spec["status_cmd"])) > 0,
				HasRecoverCmd: len(stringSlice(spec["recover_cmd"])) > 0,
				AutoRecover:   boolValue(spec["auto_recover"]),
			})
		}
	}
	for name := range out {
		sort.Slice(out[name], func(i, j int) bool {
			return sourceRank(out[name][i].Source) < sourceRank(out[name][j].Source)
		})
	}
	return out
}

func sourceRank(source string) int {
	switch source {
	case "example":
		return 0
	case "repo":
		return 1
	case "local":
		return 2
	default:
		return 3
	}
}
