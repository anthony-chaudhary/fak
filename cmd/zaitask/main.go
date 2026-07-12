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
	base := fs.String("base-url", zaitask.DefaultBaseURL, "Z.AI Coding Plan API base URL")
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
	got, err := (zaitask.Client{BaseURL: *base, APIKey: os.Getenv("ZAI_API_KEY")}).Run(ctx, prompt, *model, *max)
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
		RequestID string        `json:"request_id"`
		Usage     zaitask.Usage `json:"usage"`
		LatencyMS int64         `json:"latency_ms"`
	}{got.Model, got.RequestID, got.Usage, got.LatencyMS}
	raw, _ := json.Marshal(receipt)
	fmt.Fprintf(os.Stderr, "\n[zaitask receipt] %s\n", raw)
}
func fail(err error) { fmt.Fprintln(os.Stderr, "zaitask:", err); os.Exit(1) }
