package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/zaitask"
)

var (
	errUsage          = errors.New("usage")
	errUnsuitable     = errors.New("task not suitable for bounded GLM-5")
	errFlashOnlyFlags = errors.New("flag requires the GLM-5.3-Flash model")
)

type classedError struct {
	class error
	msg   string
}

func (e classedError) Error() string { return e.msg }
func (e classedError) Unwrap() error { return e.class }

func main() {
	err := run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr, os.Getenv)
	if err != nil && !errors.Is(err, errUsage) && !errors.Is(err, flag.ErrHelp) {
		fmt.Fprintln(os.Stderr, "zaitask:", err)
	}
	os.Exit(exitCode(err))
}

func exitCode(err error) int {
	switch {
	case err == nil, errors.Is(err, flag.ErrHelp):
		return 0
	case errors.Is(err, errUsage):
		return 2
	default:
		return 1
	}
}

func run(args []string, stdin io.Reader, stdout, stderr io.Writer, getenv func(string) string) error {
	fs := flag.NewFlagSet("zaitask", flag.ContinueOnError)
	fs.SetOutput(stderr)
	model := fs.String("model", zaitask.DefaultModel, "Z.AI model")
	base := fs.String("base-url", zaitask.DefaultBaseURL, "Z.AI API base URL (GLM-5.3-Flash resolves the direct general endpoint)")
	reasoningEffort := fs.String("reasoning-effort", "max", "GLM-5.3-Flash reasoning effort: low, high, or max")
	stream := fs.Bool("stream", false, "request and assemble an SSE response (GLM-5.3-Flash)")
	max := fs.Int("max-tokens", 4000, "maximum completion tokens")
	timeout := fs.Duration("timeout", 3*time.Minute, "request timeout")
	jsonOut := fs.Bool("json", false, "emit content plus usage receipt as JSON")
	taskClass := fs.String("task-class", "auto", "routing class: auto, light, gardening, tier3, hard, engineering, or apex")
	force := fs.Bool("force", false, "run even when fleet classification says the task is unsuitable")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return err
		}
		return fmt.Errorf("%w: %v", errUsage, err)
	}
	prompt := strings.Join(fs.Args(), " ")
	if prompt == "" {
		raw, err := io.ReadAll(io.LimitReader(stdin, 2<<20))
		if err != nil {
			return err
		}
		prompt = string(raw)
	}
	classification := zaitask.Classify(prompt, *taskClass)
	if !classification.Suitable && !*force {
		return classedError{errUnsuitable, fmt.Sprintf("task is %s/tier%d, not suitable for bounded GLM-5: %s (use --force to override)", classification.Class, classification.TargetTier, classification.Reason)}
	}
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	client := zaitask.Client{BaseURL: *base, APIKey: getenv("ZAI_API_KEY")}
	var got zaitask.Result
	var err error
	if *model == zaitask.GLM53FlashModel {
		got, err = client.RunChat(ctx, zaitask.Request{
			Model: *model, Messages: []zaitask.Message{{Role: "user", Content: prompt}},
			MaxTokens: *max, Stream: *stream, ReasoningEffort: *reasoningEffort,
		})
	} else {
		if *stream || *reasoningEffort != "max" {
			return classedError{errFlashOnlyFlags, "--stream and --reasoning-effort are supported only with --model " + zaitask.GLM53FlashModel}
		}
		got, err = client.Run(ctx, prompt, *model, *max)
	}
	if err != nil {
		return err
	}
	if *jsonOut {
		payload := struct {
			zaitask.Result
			Routing zaitask.Suitability `json:"routing"`
		}{got, classification}
		return json.NewEncoder(stdout).Encode(payload)
	}
	fmt.Fprint(stdout, got.Content)
	receipt := struct {
		Model     string        `json:"model"`
		Provider  string        `json:"provider"`
		Engine    string        `json:"engine"`
		FakNative bool          `json:"fak_native"`
		RequestID string        `json:"request_id"`
		Usage     zaitask.Usage `json:"usage"`
		LatencyMS int64         `json:"latency_ms"`
	}{got.Model, got.Provider, got.Engine, got.FakNative, got.RequestID, got.Usage, got.LatencyMS}
	raw, _ := json.Marshal(receipt)
	fmt.Fprintf(stderr, "\n[zaitask receipt] %s\n", raw)
	return nil
}
