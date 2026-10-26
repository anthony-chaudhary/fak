// Command pdbench proves process-separated prefill/decode state handoff over loopback TCP.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type cliConfig struct {
	mode         string
	gguf         string
	backend      string
	address      string
	prompt       []int
	steps        int
	timeout      time.Duration
	softwareOnly bool
}

func main() {
	if err := realMain(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func realMain() error {
	cfg, err := parseFlags(os.Args[1:])
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	be, ok := compute.Lookup(cfg.backend)
	if !ok {
		return fmt.Errorf("pdbench: backend %q is not registered (available: %s)", cfg.backend, strings.Join(compute.Registered(), ","))
	}
	if err := validateBackend(be, cfg.softwareOnly); err != nil {
		return err
	}
	// The current Q8 loader has no context-aware entrypoint. The timeout starts
	// after loading, sets socket deadlines, and is checked between model calls;
	// one synchronous model call cannot be preempted.
	m, quantFormat, err := loadModelForBackend(cfg.gguf, be)
	if err != nil {
		return fmt.Errorf("pdbench: load %s GGUF: %w", backendLoadLabel(be), err)
	}
	defer m.CloseWeights()
	ctx, cancel := context.WithTimeout(context.Background(), cfg.timeout)
	defer cancel()
	artifact, err := identifyArtifact(cfg.gguf, m)
	if err != nil {
		return err
	}
	artifact.QuantFormat = quantFormat
	configID, err := identifyConfig(m.Cfg)
	if err != nil {
		return err
	}
	runCfg := runConfig{artifact: artifact, backend: identifyBackend(be), configID: configID, expectedPrompt: cfg.prompt, expectedSteps: cfg.steps}
	if cfg.mode == "producer" {
		conn, err := (&net.Dialer{}).DialContext(ctx, "tcp", cfg.address)
		if err != nil {
			return fmt.Errorf("pdbench: connect decoder: %w", err)
		}
		defer conn.Close()
		setDeadline(conn, ctx)
		r, runErr := runProducer(ctx, runCfg, m, be, conn, cfg.prompt, cfg.steps)
		emitErr := emitReceiptIfPresent(r)
		if runErr != nil {
			return runErr
		}
		return emitErr
	}

	lc := net.ListenConfig{}
	ln, err := lc.Listen(ctx, "tcp", cfg.address)
	if err != nil {
		return fmt.Errorf("pdbench: listen: %w", err)
	}
	defer ln.Close()
	stopListenCancel := context.AfterFunc(ctx, func() { _ = ln.Close() })
	defer stopListenCancel()
	if deadline, ok := ctx.Deadline(); ok {
		if tcp, ok := ln.(*net.TCPListener); ok {
			_ = tcp.SetDeadline(deadline)
		}
	}
	conn, err := ln.Accept()
	if err != nil {
		return fmt.Errorf("pdbench: accept: %w", err)
	}
	defer conn.Close()
	setDeadline(conn, ctx)
	r, runErr := runConsumer(ctx, runCfg, m, be, conn)
	emitErr := emitReceiptIfPresent(r)
	if runErr != nil {
		return runErr
	}
	return emitErr
}

func parseFlags(args []string) (cliConfig, error) {
	fs := flag.NewFlagSet("pdbench", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	mode := fs.String("mode", "", "producer or consumer")
	gguf := fs.String("gguf", "", "GGUF model path (CUDA Q8 loader; software-only CPU f32 loader)")
	backend := fs.String("backend", "cuda", "exact compute backend name")
	address := fs.String("address", "127.0.0.1:19191", "loopback TCP address")
	promptIDs := fs.String("prompt-ids", "", "same comma-separated integer token IDs pinned on both roles")
	steps := fs.Int("steps", 8, "greedy continuation tokens, minimum 2")
	timeout := fs.Duration("timeout", 2*time.Minute, "socket deadline and between-stage cancellation (load/hash excluded)")
	softwareOnly := fs.Bool("software-only", false, "permit cpu-ref; receipt is software-only")
	if err := fs.Parse(args); err != nil {
		return cliConfig{}, err
	}
	if *mode != "producer" && *mode != "consumer" {
		return cliConfig{}, errors.New("pdbench: -mode must be producer or consumer")
	}
	if *gguf == "" {
		return cliConfig{}, errors.New("pdbench: -gguf is required")
	}
	if *timeout <= 0 || *timeout > 30*time.Minute {
		return cliConfig{}, errors.New("pdbench: -timeout must be >0 and <=30m")
	}
	if err := requireLoopback(*address); err != nil {
		return cliConfig{}, err
	}
	prompt, err := parseTokenIDs(*promptIDs)
	if err != nil {
		return cliConfig{}, err
	}
	if *steps < 2 || *steps > maxDecodeSteps {
		return cliConfig{}, fmt.Errorf("pdbench: -steps must be 2..%d", maxDecodeSteps)
	}
	return cliConfig{*mode, *gguf, *backend, *address, prompt, *steps, *timeout, *softwareOnly}, nil
}

func parseTokenIDs(s string) ([]int, error) {
	if strings.TrimSpace(s) == "" {
		return nil, errors.New("pdbench: -prompt-ids is required and must match on both roles")
	}
	parts := strings.Split(s, ",")
	if len(parts) > maxPromptTokens {
		return nil, fmt.Errorf("pdbench: at most %d prompt IDs", maxPromptTokens)
	}
	out := make([]int, len(parts))
	for i, part := range parts {
		n, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil || n < 0 {
			return nil, fmt.Errorf("pdbench: invalid token ID %q", part)
		}
		out[i] = n
	}
	return out, nil
}

func requireLoopback(address string) error {
	host, _, err := net.SplitHostPort(address)
	if err != nil {
		return fmt.Errorf("pdbench: invalid TCP address: %w", err)
	}
	ip := net.ParseIP(host)
	if ip == nil || !ip.IsLoopback() {
		return errors.New("pdbench: address must use a loopback IP literal")
	}
	return nil
}

func validateBackend(be compute.Backend, softwareOnly bool) error {
	if be.Name() == "cpu-ref" {
		if !softwareOnly {
			return errors.New("pdbench: cpu-ref requires explicit -software-only")
		}
		return nil
	}
	if softwareOnly {
		return errors.New("pdbench: -software-only is valid only with cpu-ref")
	}
	if be.Name() != "cuda" || !be.Caps().DeviceMemory || !be.Caps().UploadDtype {
		return fmt.Errorf("pdbench: hardware proof requires exact cuda device backend, got %q", be.Name())
	}
	device := os.Getenv("CUDA_VISIBLE_DEVICES")
	if !validGPUUUID(device) {
		return errors.New("pdbench: CUDA_VISIBLE_DEVICES must be one exact GPU UUID (GPU-...), not an ordinal or list")
	}
	return nil
}

func loadModelForBackend(path string, be compute.Backend) (*model.Model, string, error) {
	quantFormat := backendQuantFormat(be)
	if quantFormat == "f32-dequant-on-load" {
		m, err := ggufload.LoadModel(path)
		return m, quantFormat, err
	}
	m, err := ggufload.LoadModelQuant(path)
	return m, quantFormat, err
}

func backendLoadLabel(be compute.Backend) string {
	return backendQuantFormat(be)
}

func backendQuantFormat(be compute.Backend) string {
	if !be.Caps().UploadDtype {
		return "f32-dequant-on-load"
	}
	return "q8_0-on-load"
}

func validGPUUUID(s string) bool {
	if len(s) != 40 || !strings.HasPrefix(s, "GPU-") {
		return false
	}
	for i, c := range s[4:] {
		if i == 8 || i == 13 || i == 18 || i == 23 {
			if c != '-' {
				return false
			}
			continue
		}
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F') {
			return false
		}
	}
	return true
}

func identifyBackend(be compute.Backend) backendIdentity {
	return backendIdentity{
		Name: be.Name(), Tier: be.Tier(), Class: fmt.Sprint(be.Class()), DeviceMemory: be.Caps().DeviceMemory,
		QuantFormat: backendQuantFormat(be), CUDAVisibleDevices: os.Getenv("CUDA_VISIBLE_DEVICES"),
	}
}

func evidenceScope(be compute.Backend) string {
	if be.Name() == "cpu-ref" {
		return "software-only"
	}
	return "declared-distinct-device-process-evidence-no-hardware-qualification"
}

func identifyArtifact(path string, m *model.Model) (artifactIdentity, error) {
	f, err := os.Open(path)
	if err != nil {
		return artifactIdentity{}, fmt.Errorf("pdbench: open GGUF for identity: %w", err)
	}
	defer f.Close()
	stat, err := f.Stat()
	if err != nil {
		return artifactIdentity{}, err
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return artifactIdentity{}, fmt.Errorf("pdbench: hash GGUF: %w", err)
	}
	id := artifactIdentity{Bytes: stat.Size(), RawSHA256: hex.EncodeToString(h.Sum(nil))}
	if auth, ok := m.AuthenticatedArtifactIdentity(); ok {
		id.ManifestSHA256 = auth.ManifestSHA256
		id.ShardSetSHA256 = auth.ShardSetSHA256
		id.ChatTemplateSHA256 = auth.ChatTemplateSHA256
	}
	return id, nil
}

func identifyConfig(cfg model.Config) (string, error) {
	b, err := json.Marshal(cfg)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:]), nil
}

func setDeadline(conn net.Conn, ctx context.Context) {
	if deadline, ok := ctx.Deadline(); ok {
		_ = conn.SetDeadline(deadline)
	}
}

func emitReceipt(r receipt) error {
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintln(os.Stdout, string(b))
	return err
}

func emitReceiptIfPresent(r receipt) error {
	if r.Protocol == "" {
		return nil
	}
	return emitReceipt(r)
}
