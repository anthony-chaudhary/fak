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
	_ = fs.Parse(os.Args[1:])
	prompt := strings.Join(fs.Args(), " ")
	if prompt == "" {
		raw, err := io.ReadAll(io.LimitReader(os.Stdin, 2<<20))
		if err != nil {
			fail(err)
		}
		prompt = string(raw)
	}
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	got, err := (zaitask.Client{BaseURL: *base, APIKey: os.Getenv("ZAI_API_KEY")}).Run(ctx, prompt, *model, *max)
	if err != nil {
		fail(err)
	}
	if *jsonOut {
		if err := json.NewEncoder(os.Stdout).Encode(got); err != nil {
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
