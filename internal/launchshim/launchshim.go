// Package launchshim owns the persisted, reversible zero-adoption launcher configuration.
package launchshim

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

type Provider struct {
	Command     string   `json:"command"`
	Canonical   string   `json:"canonical,omitempty"`
	Args        []string `json:"args,omitempty"`
	InstallShim bool     `json:"install_shim,omitempty"`
}

const ConfigSchema = "fak.launch.v2"

type Config struct {
	Schema    string              `json:"schema,omitempty"`
	Default   string              `json:"default,omitempty"`
	Disabled  bool                `json:"disabled,omitempty"`
	Providers map[string]Provider `json:"providers,omitempty"`
}

var saveMu sync.Mutex
var saveReplace = replaceConfig

func Path() (string, error) {
	if p := strings.TrimSpace(os.Getenv("FAK_LAUNCH_CONFIG")); p != "" {
		return p, nil
	}
	d, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(d, "fak", "launch.json"), nil
}

// BinDir returns the install directory for launcher shims.
func BinDir() (string, error) {
	if path := strings.TrimSpace(os.Getenv("FAK_LAUNCH_BIN")); path != "" {
		return path, nil
	}
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "fak", "bin"), nil
}

func Load() (Config, error) {
	p, err := Path()
	if err != nil {
		return Config{}, err
	}
	b, err := os.ReadFile(p)
	if errors.Is(err, os.ErrNotExist) {
		return Config{Providers: map[string]Provider{}}, nil
	}
	if err != nil {
		return Config{}, err
	}
	var c Config
	if err := json.Unmarshal(b, &c); err != nil {
		return Config{}, fmt.Errorf("parse %s: %w", p, err)
	}
	if c.Schema != "" && c.Schema != ConfigSchema {
		return Config{}, fmt.Errorf("unsupported launch config schema %q", c.Schema)
	}
	c.Schema = ConfigSchema
	if c.Providers == nil {
		c.Providers = map[string]Provider{}
	}
	for name, provider := range c.Providers {
		canonical, err := NormalizeProvider(name)
		if err != nil || canonical != name || strings.TrimSpace(provider.Command) == "" {
			return Config{}, fmt.Errorf("invalid launch provider %q", name)
		}
	}
	if c.Default != "" {
		canonical, err := NormalizeProvider(c.Default)
		if err != nil || canonical != c.Default {
			return Config{}, fmt.Errorf("invalid default provider %q", c.Default)
		}
	}
	return c, nil
}

func Save(c Config) error {
	saveMu.Lock()
	defer saveMu.Unlock()
	p, err := Path()
	if err != nil {
		return err
	}
	if c.Providers == nil {
		c.Providers = map[string]Provider{}
	}
	b, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	tmpFile, err := os.CreateTemp(filepath.Dir(p), filepath.Base(p)+".tmp-*")
	if err != nil {
		return err
	}
	tmp := tmpFile.Name()
	defer os.Remove(tmp)
	if err := tmpFile.Chmod(0o600); err != nil {
		tmpFile.Close()
		return err
	}
	if _, err := tmpFile.Write(b); err != nil {
		tmpFile.Close()
		return err
	}
	if err := tmpFile.Close(); err != nil {
		return err
	}
	if err := os.Chmod(tmp, 0o600); err != nil {
		return err
	}
	return saveReplace(tmp, p)
}

// CanonicalCommand resolves a provider executable to a stable absolute identity.
// EvalSymlinks catches indirect shim recursion on Unix; Abs/Clean and case-folded
// comparisons retain the same contract for Windows PATHEXT-resolved paths.

func replaceConfig(tmp, dst string) error {
	// os.Rename replaces atomically on Unix. Windows refuses replacement, so retain
	// the prior file as a rollback witness until the complete temp becomes live.
	if err := os.Rename(tmp, dst); err == nil {
		return nil
	}
	backup := dst + ".previous"
	_ = os.Remove(backup)
	if err := os.Rename(dst, backup); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if err := os.Rename(tmp, dst); err != nil {
		_ = os.Rename(backup, dst)
		return err
	}
	_ = os.Remove(backup)
	return nil
}

func CanonicalCommand(path string) (string, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return "", errors.New("empty provider command")
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	resolved, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(resolved)
	if err != nil {
		return "", err
	}
	if info.IsDir() {
		return "", fmt.Errorf("provider command %s is a directory", filepath.Base(path))
	}
	return filepath.Clean(resolved), nil
}

func SameCommand(a, b string) bool {
	ca, ea := CanonicalCommand(a)
	cb, eb := CanonicalCommand(b)
	if ea == nil && eb == nil {
		return strings.EqualFold(ca, cb)
	}
	aa, _ := filepath.Abs(a)
	bb, _ := filepath.Abs(b)
	return strings.EqualFold(filepath.Clean(aa), filepath.Clean(bb))
}

func NormalizeProvider(name string) (string, error) {
	name = strings.ToLower(strings.TrimSpace(name))
	if name == "" {
		return "", fmt.Errorf("provider name is empty")
	}
	if name == "." || name == ".." || strings.ContainsAny(name, `/\`) {
		return "", fmt.Errorf("provider name %q is path-like", name)
	}
	for i, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9' && i > 0) || (r == '-' && i > 0 && i < len(name)-1) {
			continue
		}
		return "", fmt.Errorf("provider name %q must match [a-z][a-z0-9-]*", name)
	}
	return name, nil
}

func EffectiveDirect(c Config, flag bool) bool {
	if flag || c.Disabled {
		return true
	}
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_DIRECT"))) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}
