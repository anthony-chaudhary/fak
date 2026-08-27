package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/zaitask"
)

func main() {
	fs := flag.NewFlagSet("zaitask", flag.ExitOnError)
	model := fs.String("model", zaitask.DefaultModel, "Z.AI model")
	base := fs.String("base-url", zaitask.DefaultBaseURL, "Z.AI API base URL (GLM-5.3-Flash resolves the direct general endpoint)")
	reasoningEffort := fs.String("reasoning-effort", "max", "GLM-5.3-Flash reasoning effort: low, high, or max")
	stream := fs.Bool("stream", false, "request and assemble an SSE response (GLM-5.3-Flash)")
	max := fs.Int("max-tokens", 4000, "maximum completion tokens")
	timeout := fs.Duration("timeout", 3*time.Minute, "request timeout")
	jsonOut := fs.Bool("json", false, "emit content plus usage receipt as JSON")
	taskClass := fs.String("task-class", "auto", "routing class: auto, light, gardening, tier3, hard, engineering, or apex")
	force := fs.Bool("force", false, "run even when fleet classification says the task is unsuitable")
	_ = fs.Parse(os.Args[1:])
	prompt := strings.Join(fs.Args(), " ")
	if prompt == "" {
		raw, err := io.ReadAll(io.LimitReader(os.Stdin, 2<<20))
		if err != nil {
			fail(err)
		}
		prompt = string(raw)
	}
	classification := zaitask.Classify(prompt, *taskClass)
	if !classification.Suitable && !*force {
		fail(fmt.Errorf("task is %s/tier%d, not suitable for bounded GLM-5: %s (use --force to override)", classification.Class, classification.TargetTier, classification.Reason))
	}
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	client := zaitask.Client{BaseURL: *base, APIKey: os.Getenv("ZAI_API_KEY")}
	var got zaitask.Result
	var err error
	if *model == zaitask.GLM53FlashModel {
		got, err = client.RunChat(ctx, zaitask.Request{
			Model: *model, Messages: []zaitask.Message{{Role: "user", Content: prompt}},
			MaxTokens: *max, Stream: *stream, ReasoningEffort: *reasoningEffort,
		})
	} else {
		if *stream || *reasoningEffort != "max" {
			fail(fmt.Errorf("--stream and --reasoning-effort are supported only with --model %s", zaitask.GLM53FlashModel))
		}
		got, err = client.Run(ctx, prompt, *model, *max)
	}
	if err != nil {
		fail(err)
	}
	if *jsonOut {
		payload := struct {
			zaitask.Result
			Routing zaitask.Suitability `json:"routing"`
		}{got, classification}
		if err := json.NewEncoder(os.Stdout).Encode(payload); err != nil {
			fail(err)
		}
		return
	}
	fmt.Print(got.Content)
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
	fmt.Fprintf(os.Stderr, "\n[zaitask receipt] %s\n", raw)
}
func fail(err error) { fmt.Fprintln(os.Stderr, "zaitask:", err); os.Exit(1) }
