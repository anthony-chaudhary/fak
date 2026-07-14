package terminalrisk

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const Direct2D = "direct2d"

type Facts struct {
	AMDPresent         bool
	PriorWTRenderCrash bool
	SettingsPath       string
	Settings           []byte
}

type Report struct {
	Risk               bool   `json:"risk"`
	AMDPresent         bool   `json:"amd_present"`
	PriorWTRenderCrash bool   `json:"prior_wt_render_crash"`
	GraphicsAPI        string `json:"graphics_api,omitempty"`
	GPUPath            bool   `json:"gpu_path"`
	Reason             string `json:"reason"`
	SettingsPath       string `json:"settings_path"`
}

func Assess(f Facts) (Report, error) {
	r := Report{AMDPresent: f.AMDPresent, PriorWTRenderCrash: f.PriorWTRenderCrash, SettingsPath: f.SettingsPath}
	var root map[string]any
	if err := json.Unmarshal(f.Settings, &root); err != nil {
		return r, fmt.Errorf("parse Windows Terminal settings: %w", err)
	}
	if v, ok := root["graphicsAPI"].(string); ok {
		r.GraphicsAPI = strings.ToLower(strings.TrimSpace(v))
	}
	gpuPath := r.GraphicsAPI == ""
	if experimental, ok := root["experimental"].(map[string]any); ok {
		atlas, _ := experimental["useAtlasEngine"].(bool)
		software := false
		if rendering, ok := experimental["rendering"].(map[string]any); ok {
			software, _ = rendering["software"].(bool)
		}
		if atlas && !software {
			gpuPath = true
		}
	}
	r.GPUPath = gpuPath
	r.Risk = f.AMDPresent && f.PriorWTRenderCrash && gpuPath
	switch {
	case r.Risk:
		r.Reason = "AMD + prior WT render AV + GPU render path"
	case r.GraphicsAPI == Direct2D:
		r.Reason = "clean: global graphicsAPI=direct2d"
	case !f.AMDPresent:
		r.Reason = "clean: no AMD adapter"
	case !f.PriorWTRenderCrash:
		r.Reason = "clean: no prior WT render AV"
	default:
		r.Reason = "clean: non-GPU render path"
	}
	return r, nil
}

func ApplyDirect2D(path string, raw []byte, now time.Time) (string, error) {
	var root map[string]any
	if err := json.Unmarshal(raw, &root); err != nil {
		return "", err
	}
	root["graphicsAPI"] = Direct2D
	out, err := json.MarshalIndent(root, "", "    ")
	if err != nil {
		return "", err
	}
	out = append(out, '\n')
	backup := path + ".fak-backup-" + now.UTC().Format("20060102T150405Z")
	if err := os.WriteFile(backup, raw, 0o600); err != nil {
		return "", fmt.Errorf("backup: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".wt-settings-*.tmp")
	if err != nil {
		return "", err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return "", err
	}
	if _, err := tmp.Write(out); err != nil {
		tmp.Close()
		return "", err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return "", err
	}
	if err := tmp.Close(); err != nil {
		return "", err
	}
	if err := replace(name, path); err != nil {
		return "", err
	}
	return backup, nil
}
