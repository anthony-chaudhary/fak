package qwen38quantrun

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/qwen38quant"
)

type EndpointConfig struct {
	Endpoint    string        `json:"endpoint"`
	APIKey      string        `json:"api_key,omitempty"`
	Model       string        `json:"model"`
	Repetitions int           `json:"repetitions,omitempty"`
	Timeout     time.Duration `json:"timeout,omitempty"`
}

func (c EndpointConfig) runnerConfig() Config {
	return Config{Endpoint: c.Endpoint, APIKey: c.APIKey, Model: c.Model, Repetitions: c.Repetitions, Timeout: c.Timeout}
}

// AdapterConfig is the file-backed operator contract for a live campaign.
// ObservationCommand must emit one Observation JSON object on stdout; lifecycle
// commands are argv arrays and are never interpreted by a shell.
type AdapterConfig struct {
	Endpoint EndpointConfig `json:"endpoint"`
	// ExecutionEngine carries the model-math runtime identity used for evidence
	// promotion, distinct from endpoint, planner, transport, and hardware backend identity.
	ExecutionEngine    string               `json:"execution_engine"`
	Arm                string               `json:"arm"`
	Expected           qwen38quant.Identity `json:"expected"`
	Command            []string             `json:"command"`
	RequireDevice      string               `json:"require_device"`
	StaleAfter         string               `json:"stale_after"`
	RollbackThreshold  string               `json:"rollback_threshold"`
	ObservationCommand []string             `json:"observation_command"`
	RestartCommand     []string             `json:"restart_command"`
	ReadyCommand       []string             `json:"ready_command"`
	CleanupCommand     []string             `json:"cleanup_command"`
}

// RunAdapter loads the frozen corpus and operator config, executes the real
// endpoint campaign, validates it, and atomically writes the archive and report.
func RunAdapter(ctx context.Context, configPath, corpusPath, reportPath, archivePath string) error {
	var cfg AdapterConfig
	if err := decodeFile(configPath, &cfg); err != nil {
		return fmt.Errorf("config: %w", err)
	}
	corpusBytes, err := os.ReadFile(corpusPath)
	if err != nil {
		return fmt.Errorf("corpus: %w", err)
	}
	corpus, err := qwen38quant.DecodeCorpus(corpusBytes)
	if err != nil {
		return fmt.Errorf("corpus: %w", err)
	}
	if len(cfg.ObservationCommand) == 0 {
		return errors.New("observation_command is required")
	}
	if len(cfg.RestartCommand) == 0 || len(cfg.ReadyCommand) == 0 || len(cfg.CleanupCommand) == 0 {
		return errors.New("restart_command, ready_command, and cleanup_command are required")
	}
	campaign, err := (Runner{}).RunCampaign(ctx, CampaignConfig{
		Endpoint: cfg.Endpoint.runnerConfig(), ExecutionEngine: cfg.ExecutionEngine, Arm: cfg.Arm, Expected: cfg.Expected, Command: cfg.Command,
		RequireDevice: cfg.RequireDevice, StaleAfter: cfg.StaleAfter,
		RollbackThreshold: cfg.RollbackThreshold,
		Probe:             commandProbe{argv: cfg.ObservationCommand},
		Lifecycle:         commandLifecycle{restart: cfg.RestartCommand, ready: cfg.ReadyCommand, cleanup: cfg.CleanupCommand},
	}, corpus)
	if err != nil {
		return err
	}
	if err := qwen38quant.Validate(campaign.Report, corpus); err != nil {
		return fmt.Errorf("validate report: %w", err)
	}
	report, err := json.MarshalIndent(campaign.Report, "", "  ")
	if err != nil {
		return err
	}
	report = append(report, '\n')
	if err := writeAtomic(archivePath, campaign.Archive, 0o600); err != nil {
		return fmt.Errorf("archive: %w", err)
	}
	if err := writeAtomic(reportPath, report, 0o644); err != nil {
		_ = os.Remove(archivePath)
		return fmt.Errorf("report: %w", err)
	}
	return nil
}

func decodeFile(path string, dst any) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	d := json.NewDecoder(f)
	d.DisallowUnknownFields()
	if err := d.Decode(dst); err != nil {
		return err
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		if err == nil {
			return errors.New("multiple JSON values")
		}
		return err
	}
	return nil
}

type commandProbe struct{ argv []string }

func (p commandProbe) Observe(ctx context.Context) (Observation, error) {
	out, err := runArgv(ctx, p.argv)
	if err != nil {
		return Observation{}, err
	}
	var got Observation
	d := json.NewDecoder(strings.NewReader(string(out)))
	d.DisallowUnknownFields()
	if err := d.Decode(&got); err != nil {
		return Observation{}, fmt.Errorf("decode observation: %w", err)
	}
	return got, nil
}

type commandLifecycle struct{ restart, ready, cleanup []string }

func (l commandLifecycle) Restart(ctx context.Context) error {
	_, err := runArgv(ctx, l.restart)
	return err
}
func (l commandLifecycle) Ready(ctx context.Context) error {
	_, err := runArgv(ctx, l.ready)
	return err
}
func (l commandLifecycle) Cleanup(ctx context.Context) error {
	_, err := runArgv(ctx, l.cleanup)
	return err
}

func runArgv(ctx context.Context, argv []string) ([]byte, error) {
	if len(argv) == 0 {
		return nil, errors.New("empty command")
	}
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("%s: %w: %s", argv[0], err, strings.TrimSpace(string(out)))
	}
	return out, nil
}

func writeAtomic(path string, data []byte, mode os.FileMode) error {
	if path == "" {
		return errors.New("output path is required")
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	f, err := os.CreateTemp(dir, ".qwen38-*")
	if err != nil {
		return err
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	if _, err = f.Write(data); err == nil {
		err = f.Chmod(mode)
	}
	if err == nil {
		err = f.Sync()
	}
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	for i := 0; i < 5; i++ {
		err = os.Rename(tmp, path)
		if err == nil {
			return nil
		}
		time.Sleep(time.Duration(i+1) * 10 * time.Millisecond)
	}
	return err
}
