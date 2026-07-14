package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardcompile"
)

const (
	guardCompileDefaultEndpoint = "https://api.openai.com/v1/chat/completions"
	guardCompileDefaultModel    = "gpt-4.1-mini"
	guardCompileMaxBody         = 1 << 20
)

type guardCompileHTTPDoer interface {
	Do(*http.Request) (*http.Response, error)
}

type guardCompileOptions struct {
	transcript     string
	transcriptFile string
	intent         string
	tool           string
	field          string
	policyPath     string
	model          string
	endpoint       string
	apiKeyEnv      string
}

func runGuardCompile(stdout, stderr io.Writer, argv []string) int {
	return runGuardCompileWithDoer(stdout, stderr, argv, &http.Client{Timeout: 45 * time.Second})
}

func runGuardCompileWithDoer(stdout, stderr io.Writer, argv []string, doer guardCompileHTTPDoer) int {
	fs := flag.NewFlagSet("fak guard compile", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var opts guardCompileOptions
	fs.StringVar(&opts.transcript, "transcript", "", "transcript excerpt")
	fs.StringVar(&opts.transcriptFile, "transcript-file", "", "path to transcript excerpt")
	fs.StringVar(&opts.intent, "intent", "", "one-line guard intent")
	fs.StringVar(&opts.tool, "tool", "Bash", "policy tool name")
	fs.StringVar(&opts.field, "field", "command", "tool argument field")
	fs.StringVar(&opts.policyPath, "policy", "examples/repo-guard-policy.json", "policy manifest to propose changing")
	fs.StringVar(&opts.model, "model", guardCompileDefaultModel, "authoring-time extraction model")
	fs.StringVar(&opts.endpoint, "endpoint", guardCompileDefaultEndpoint, "OpenAI-compatible chat completions endpoint")
	fs.StringVar(&opts.apiKeyEnv, "api-key-env", "OPENAI_API_KEY", "environment variable containing API key")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "guard compile: unexpected arguments: %s\n", strings.Join(fs.Args(), " "))
		return 2
	}
	if opts.transcript != "" && opts.transcriptFile != "" {
		fmt.Fprintln(stderr, "guard compile: choose one of --transcript or --transcript-file")
		return 2
	}
	if opts.transcriptFile != "" {
		b, err := os.ReadFile(opts.transcriptFile)
		if err != nil {
			fmt.Fprintf(stderr, "guard compile: read transcript: %v\n", err)
			return 1
		}
		opts.transcript = string(b)
	}
	if strings.TrimSpace(opts.transcript) == "" || strings.TrimSpace(opts.intent) == "" {
		fmt.Fprintln(stderr, "guard compile: --transcript or --transcript-file and --intent are required")
		return 2
	}
	before, err := os.ReadFile(opts.policyPath)
	if err != nil {
		fmt.Fprintf(stderr, "guard compile: read policy: %v\n", err)
		return 1
	}
	key := strings.TrimSpace(os.Getenv(opts.apiKeyEnv))
	if key == "" {
		fmt.Fprintf(stderr, "guard compile: %s is not set\n", opts.apiKeyEnv)
		return 1
	}

	extractor := &guardCompileOpenAIExtractor{
		doer: doer, endpoint: opts.endpoint, model: opts.model, apiKey: key,
	}
	candidate, err := guardcompile.Compile(guardcompile.Request{
		Transcript: opts.transcript, Intent: opts.intent, Tool: opts.tool, Field: opts.field,
	}, before, extractor)
	if err != nil {
		fmt.Fprintf(stderr, "guard compile: %v\n", err)
		return 1
	}
	fmt.Fprint(stdout, guardcompile.ProposedDiff(opts.policyPath, before, candidate.Manifest))
	fmt.Fprintln(stdout, "REVIEW REQUIRED: proposal only; policy was not applied.")
	return 0
}

type guardCompileOpenAIExtractor struct {
	doer     guardCompileHTTPDoer
	endpoint string
	model    string
	apiKey   string
}

func (e *guardCompileOpenAIExtractor) Extract(prompt string) ([]byte, error) {
	requestBody := struct {
		Model    string `json:"model"`
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
		ResponseFormat struct {
			Type string `json:"type"`
		} `json:"response_format"`
		Temperature int `json:"temperature"`
	}{Model: e.model}
	requestBody.Messages = append(requestBody.Messages, struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}{Role: "user", Content: prompt})
	requestBody.ResponseFormat.Type = "json_object"
	body, err := json.Marshal(requestBody)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, e.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+e.apiKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := e.doer.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	responseBody, err := io.ReadAll(io.LimitReader(resp.Body, guardCompileMaxBody+1))
	if err != nil {
		return nil, err
	}
	if len(responseBody) > guardCompileMaxBody {
		return nil, errors.New("model response exceeds 1 MiB")
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("model HTTP %s: %s", resp.Status, strings.TrimSpace(string(responseBody)))
	}
	var wire struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(responseBody, &wire); err != nil {
		return nil, fmt.Errorf("decode model response: %w", err)
	}
	if len(wire.Choices) != 1 || strings.TrimSpace(wire.Choices[0].Message.Content) == "" {
		return nil, fmt.Errorf("model response has %d choices, want one non-empty choice", len(wire.Choices))
	}
	return []byte(wire.Choices[0].Message.Content), nil
}
