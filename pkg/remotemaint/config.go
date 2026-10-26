package remotemaint

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

const maxConfigBytes = 1 << 20

type Config struct {
	Schema   string    `json:"schema"`
	Profiles []Profile `json:"profiles"`
}

type Profile struct {
	ProfileID     string       `json:"profile_id"`
	Target        string       `json:"target"`
	HostKey       string       `json:"host_key"`
	Operation     string       `json:"operation"`
	PayloadSHA256 string       `json:"payload_sha256"`
	Holder        string       `json:"holder"`
	Paths         []string     `json:"paths"`
	PathPrefixes  []PathPrefix `json:"path_prefixes"`
	Units         []string     `json:"units"`
	LeaseNodes    []string     `json:"lease_nodes"`
	ExpiresAt     time.Time    `json:"expires_at"`
	Authority     Command      `json:"authority"`
}

type Command struct {
	Executable string   `json:"executable"`
	SHA256     string   `json:"sha256"`
	Args       []string `json:"args"`
}

func DefaultConfigPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return "", fmt.Errorf("%w: user home unavailable", ErrUnavailable)
	}
	configPath, err := filepath.Abs(filepath.Join(home, ".fak", "guard", "remote-maintenance.json"))
	if err != nil {
		return "", fmt.Errorf("%w: user context path unavailable", ErrUnavailable)
	}
	return configPath, nil
}

func ProtectedPaths(configPath string) ([]string, error) {
	abs, err := filepath.Abs(configPath)
	if err != nil {
		return nil, fmt.Errorf("%w: invalid config path", ErrUnavailable)
	}
	if _, err := os.Lstat(abs); err != nil {
		if os.IsNotExist(err) {
			return []string{abs}, nil
		}
		return []string{abs}, fmt.Errorf("%w: context unreadable", ErrUnavailable)
	}
	cfg, _, err := loadConfig(abs, time.Now().UTC())
	if err != nil {
		return []string{abs}, err
	}
	paths := []string{abs}
	for _, p := range cfg.Profiles {
		paths = append(paths, p.Authority.Executable)
	}
	sort.Strings(paths)
	return compactStrings(paths), nil
}

func ParseConfig(raw []byte) (Config, error) {
	if len(raw) == 0 || len(raw) > maxConfigBytes {
		return Config{}, fmt.Errorf("%w: invalid context size", ErrAuthority)
	}
	var cfg Config
	if err := decodeStrict(raw, &cfg); err != nil {
		return Config{}, fmt.Errorf("%w: malformed context", ErrAuthority)
	}
	if err := ValidateConfig(cfg); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func ValidateConfig(cfg Config) error {
	return validateConfigAt(cfg, time.Now().UTC())
}

func loadConfig(configPath string, now time.Time) (Config, string, error) {
	info, err := os.Lstat(configPath)
	if err != nil {
		return Config{}, "", fmt.Errorf("%w: context unreadable", ErrUnavailable)
	}
	if !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
		return Config{}, "", fmt.Errorf("%w: context is not a regular file", ErrAuthority)
	}
	if runtime.GOOS != "windows" && info.Mode().Perm()&0o077 != 0 {
		return Config{}, "", fmt.Errorf("%w: context permissions are not private", ErrAuthority)
	}
	f, err := os.Open(configPath)
	if err != nil {
		return Config{}, "", fmt.Errorf("%w: context unreadable", ErrUnavailable)
	}
	defer f.Close()
	raw, err := io.ReadAll(io.LimitReader(f, maxConfigBytes+1))
	if err != nil || len(raw) > maxConfigBytes {
		return Config{}, "", fmt.Errorf("%w: invalid context size", ErrAuthority)
	}
	var cfg Config
	if err := decodeStrict(raw, &cfg); err != nil {
		return Config{}, "", fmt.Errorf("%w: malformed context", ErrAuthority)
	}
	if err := validateConfigAt(cfg, now); err != nil {
		return Config{}, "", err
	}
	sum := sha256.Sum256(raw)
	return cfg, hex.EncodeToString(sum[:]), nil
}

func validateConfigAt(cfg Config, now time.Time) error {
	if cfg.Schema != ConfigSchema || len(cfg.Profiles) == 0 || len(cfg.Profiles) > 128 {
		return fmt.Errorf("%w: invalid context schema", ErrAuthority)
	}
	seen := make(map[string]struct{}, len(cfg.Profiles))
	for _, p := range cfg.Profiles {
		if _, ok := seen[p.ProfileID]; ok {
			return fmt.Errorf("%w: duplicate profile", ErrAuthority)
		}
		seen[p.ProfileID] = struct{}{}
		if !validName(p.ProfileID, 128) || !validName(p.Target, 512) || !validHostKey(p.HostKey) ||
			p.Operation != OperationSystemdProfileInstallReload || !isDigest(p.PayloadSHA256) || !validName(p.Holder, 256) {
			return fmt.Errorf("%w: invalid profile identity", ErrAuthority)
		}
		if !p.ExpiresAt.Equal(p.ExpiresAt.UTC()) || !p.ExpiresAt.After(now) {
			return fmt.Errorf("%w: profile", ErrExpired)
		}
		if !canonicalStrings(p.Paths) || !canonicalPathPrefixes(p.PathPrefixes) || !canonicalStrings(p.Units) ||
			!canonicalStrings(p.LeaseNodes) || p.PathPrefixes == nil || len(p.Paths) == 0 || len(p.Units) == 0 || len(p.LeaseNodes) == 0 {
			return fmt.Errorf("%w: non-canonical profile scope", ErrAuthority)
		}
		for _, item := range p.Paths {
			if !validAbsolutePOSIX(item) {
				return fmt.Errorf("%w: invalid configured path", ErrAuthority)
			}
		}
		for _, item := range p.PathPrefixes {
			if !validPathPrefix(item) {
				return fmt.Errorf("%w: invalid configured prefix", ErrAuthority)
			}
		}
		for _, item := range p.Units {
			if !validName(item, 256) || strings.Contains(item, "/") {
				return fmt.Errorf("%w: invalid configured unit", ErrAuthority)
			}
		}
		for _, item := range p.LeaseNodes {
			if !validName(item, 256) {
				return fmt.Errorf("%w: invalid configured lease node", ErrAuthority)
			}
		}
		if err := validateCommand(p.Authority); err != nil {
			return err
		}
	}
	return nil
}

func compactStrings(v []string) []string {
	if len(v) < 2 {
		return v
	}
	out := v[:1]
	for _, s := range v[1:] {
		if s != out[len(out)-1] {
			out = append(out, s)
		}
	}
	return out
}
