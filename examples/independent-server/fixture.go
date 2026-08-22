package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessserver"
	"github.com/anthony-chaudhary/fak/internal/serveradapter"
	"github.com/anthony-chaudhary/fak/internal/serverlifecycle"
	"github.com/anthony-chaudhary/fak/internal/serverproduct"
)

func runServerController(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("server-controller", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	operation := fs.String("operation", "", "lifecycle operation")
	dir := fs.String("dir", "", "server product root")
	model := fs.String("model", "", "fixture model")
	digest := fs.String("sha256", "", "fixture model digest")
	adapter := fs.String("adapter-executable", "", "fixture adapter")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return errors.New("unexpected server controller arguments")
	}
	var result serverlifecycle.Result
	var err error
	switch *operation {
	case "init":
		result, err = serverlifecycle.Init(context.Background(), serverlifecycle.InitOptions{
			InstanceDirectory: *dir,
			ServerName:        modelAlias,
			ModelPath:         *model,
			ArtifactSHA256:    *digest,
			AdapterExecutable: *adapter,
			ModelAlias:        modelAlias,
			TokenWindow:       128,
			Threads:           1,
			VersionConstraint: "fixture-v1",
			ProtocolRevision:  protocolRev,
		})
	case "up":
		result, err = serverlifecycle.Up(context.Background(), *dir, serverlifecycle.Options{
			ReadinessTimeout: 10 * time.Second,
			ProbeInterval:    20 * time.Millisecond,
			StopTimeout:      5 * time.Second,
		})
	case "down":
		result, err = serverlifecycle.Down(context.Background(), *dir, serverlifecycle.Options{StopTimeout: 5 * time.Second})
	default:
		return fmt.Errorf("unknown server controller operation %q", *operation)
	}
	if err != nil {
		return err
	}
	if err := appendEvent(filepath.Join(*dir, eventFilename), eventRecord{
		Schema: eventSchema, Kind: "lifecycle", Actor: "server-product", Operation: *operation, Success: true,
	}); err != nil {
		return err
	}
	return json.NewEncoder(stdout).Encode(result)
}

func runServerProduct(executable, dir, operation string, extra ...string) (serverlifecycle.Result, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	args := []string{"--server-controller", "--operation", operation, "--dir", dir}
	args = append(args, extra...)
	cmd := exec.CommandContext(ctx, executable, args...)
	cmd.Dir = dir
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	var result serverlifecycle.Result
	if stdout.Len() > 0 {
		if decodeErr := json.Unmarshal(stdout.Bytes(), &result); decodeErr != nil {
			return result, fmt.Errorf("decode server %s result: %w", operation, decodeErr)
		}
	}
	if err != nil {
		return result, fmt.Errorf("server product %s: %w: %s", operation, err, strings.TrimSpace(stderr.String()))
	}
	return result, nil
}

func runFixtureServer(args []string) error {
	fs := flag.NewFlagSet("llama-server-fixture", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	model := fs.String("model", "", "fixture model")
	alias := fs.String("alias", "", "fixture alias")
	host := fs.String("host", "", "fixture host")
	port := fs.Int("port", 0, "fixture port")
	_ = fs.Int("ctx-size", 0, "fixture context size")
	_ = fs.Int("threads", 0, "fixture threads")
	_ = fs.Int("n-gpu-layers", 0, "fixture GPU layers")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 || *host != serveradapter.LocalBindIP || *port < 1 || *port > 65535 || *alias != modelAlias {
		return errors.New("fixture received an invalid llama-server invocation")
	}
	modelPath, err := filepath.Abs(*model)
	if err != nil {
		return err
	}
	eventPath := filepath.Join(filepath.Dir(modelPath), eventFilename)
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		if !recordHTTPEvent(w, eventPath, "probe", "server-product", "health", "/health") {
			return
		}
		writeFixtureJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, _ *http.Request) {
		if !recordHTTPEvent(w, eventPath, "probe", "server-product", "models", "/v1/models") {
			return
		}
		writeFixtureJSON(w, http.StatusOK, map[string]any{
			"object": "list",
			"data":   []map[string]string{{"id": *alias}},
		})
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, request *http.Request) {
		body, err := io.ReadAll(io.LimitReader(request.Body, 1<<20))
		if err != nil {
			http.Error(w, "read request", http.StatusBadRequest)
			return
		}
		var input struct {
			Model    string `json:"model"`
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
			MaxTokens int  `json:"max_tokens"`
			Stream    bool `json:"stream"`
		}
		decoder := json.NewDecoder(bytes.NewReader(body))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&input); err != nil || input.Model != *alias || len(input.Messages) == 0 {
			http.Error(w, "invalid chat request", http.StatusBadRequest)
			return
		}
		actor, operation, response := "server-product", "readiness-chat", "OK"
		if strings.Contains(input.Messages[len(input.Messages)-1].Content, "HANDOFF_OK") {
			actor, operation, response = "harness-product", "chat", "HANDOFF_OK"
		}
		kind := "probe"
		if actor == "harness-product" {
			kind = "chat"
		}
		if !recordHTTPEvent(w, eventPath, kind, actor, operation, "/v1/chat/completions") {
			return
		}
		writeFixtureJSON(w, http.StatusOK, map[string]any{
			"object": "chat.completion",
			"model":  *alias,
			"choices": []map[string]any{{
				"message": map[string]string{"role": "assistant", "content": response},
			}},
		})
	})
	server := &http.Server{Addr: net.JoinHostPort(*host, strconv.Itoa(*port)), Handler: mux, ReadHeaderTimeout: 2 * time.Second}
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

func recordHTTPEvent(w http.ResponseWriter, path, kind, actor, operation, route string) bool {
	err := appendEvent(path, eventRecord{Schema: eventSchema, Kind: kind, Actor: actor, Operation: operation, Path: route, Success: true})
	if err != nil {
		http.Error(w, "record fixture event", http.StatusInternalServerError)
		return false
	}
	return true
}

func writeFixtureJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func runHarnessProduct(executable, harnessDir, receiptPath string) (harnessEvidence, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, executable,
		"--harness-consumer",
		"--harness-dir", harnessDir,
		"--receipt", receiptPath,
	)
	cmd.Dir = harnessDir
	output, err := cmd.CombinedOutput()
	if err != nil {
		return harnessEvidence{}, fmt.Errorf("harness product: %w: %s", err, strings.TrimSpace(string(output)))
	}
	var evidence harnessEvidence
	decoder := json.NewDecoder(bytes.NewReader(output))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&evidence); err != nil {
		return harnessEvidence{}, fmt.Errorf("decode harness evidence: %w", err)
	}
	return evidence, nil
}

func runHarnessConsumer(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("harness-consumer", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	harnessDir := fs.String("harness-dir", "", "harness product root")
	receiptPath := fs.String("receipt", "", "immutable external server receipt")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return errors.New("unexpected harness consumer arguments")
	}
	harnessRoot, err := filepath.Abs(*harnessDir)
	if err != nil {
		return err
	}
	receipt, err := filepath.Abs(*receiptPath)
	if err != nil {
		return err
	}
	workingDir, err := os.Getwd()
	if err != nil {
		return err
	}
	if !samePath(workingDir, harnessRoot) {
		return errors.New("harness consumer must run from its own product root")
	}
	binding, err := harnessserver.Import(harnessRoot, receipt, harnessserver.Requirements{
		ModelAlias:           modelAlias,
		ProtocolFamily:       serverproduct.ProtocolOpenAIHTTP,
		ProtocolRevision:     protocolRev,
		RequiredCapabilities: []string{"chat.completions", "models.list"},
		MinimumGeneration:    1,
	})
	if err != nil {
		return err
	}
	bindingPath := filepath.Join(harnessRoot, harnessserver.BindingFileName)
	if _, err := harnessserver.WriteBinding(bindingPath, binding); err != nil {
		return err
	}
	verified, err := harnessserver.VerifyFile(bindingPath)
	if err != nil {
		return err
	}
	response, err := callHarnessChat(verified)
	if err != nil {
		return err
	}
	bindingRaw, err := os.ReadFile(bindingPath)
	if err != nil {
		return err
	}
	evidence := harnessEvidence{
		Schema:              harnessserver.ResolutionSchema,
		BindingDigest:       digestBytes(bindingRaw),
		PinnedReceiptDigest: verified.ReceiptDigest,
		Generation:          verified.Generation,
		ModelAlias:          verified.ModelAlias,
		ProtocolFamily:      verified.ProtocolFamily,
		ProtocolRevision:    verified.ProtocolRevision,
		ChatCalls:           1,
		ChatResponse:        response,
		LifecycleCalls:      0,
	}
	encoder := json.NewEncoder(stdout)
	encoder.SetEscapeHTML(false)
	return encoder.Encode(evidence)
}

func callHarnessChat(server harnessserver.Verified) (string, error) {
	endpoint, err := url.Parse(server.BaseURL)
	if err != nil {
		return "", err
	}
	host := endpoint.Hostname()
	ip := net.ParseIP(host)
	if endpoint.Scheme != "http" || ip == nil || !ip.IsLoopback() {
		return "", errors.New("verified receipt endpoint must remain loopback HTTP")
	}
	body, err := json.Marshal(map[string]any{
		"model": server.ModelAlias,
		"messages": []map[string]string{{
			"role": "user", "content": "Use the imported receipt and reply with HANDOFF_OK.",
		}},
		"max_tokens": 4,
		"stream":     false,
	})
	if err != nil {
		return "", err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(server.BaseURL, "/")+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := (&http.Client{Timeout: 5 * time.Second}).Do(request)
	if err != nil {
		return "", err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return "", fmt.Errorf("harness chat returned HTTP %d", response.StatusCode)
	}
	var result struct {
		Object  string `json:"object"`
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	decoder := json.NewDecoder(io.LimitReader(response.Body, 1<<20))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&result); err != nil {
		return "", err
	}
	if result.Object != "chat.completion" || result.Model != server.ModelAlias || len(result.Choices) != 1 ||
		result.Choices[0].Message.Role != "assistant" || result.Choices[0].Message.Content != "HANDOFF_OK" {
		return "", errors.New("harness chat response did not match the imported receipt")
	}
	return result.Choices[0].Message.Content, nil
}

func samePath(left, right string) bool {
	rel, err := filepath.Rel(left, right)
	return err == nil && rel == "."
}
