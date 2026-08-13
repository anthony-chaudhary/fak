package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func runSessionMove(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		sessionMoveUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "checkpoint":
		return runSessionMoveCheckpoint(os.Stdin, stdout, stderr, argv[1:])
	case "admit":
		return runSessionMoveAdmit(os.Stdin, stdout, stderr, argv[1:])
	case "restore":
		return runSessionMoveRestore(os.Stdin, stdout, stderr, argv[1:])
	case "verify":
		return runSessionMoveVerify(os.Stdin, stdout, stderr, argv[1:])
	default:
		sessionMoveUsage(stderr)
		return 2
	}
}

func runSessionMoveCheckpoint(stdin io.Reader, stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("session move checkpoint", flag.ContinueOnError)
	fs.SetOutput(stderr)
	sessionID := fs.String("session-id", "", "stable logical session ID")
	sourceEpoch := fs.String("source-epoch", "", "current source execution epoch")
	eventHead := fs.Uint64("event-head", 0, "source journal event head")
	sourceProvider := fs.String("source-provider", "", "source provider")
	sourceAccount := fs.String("source-account-ref", "", "source account reference")
	sourceModel := fs.String("source-model", "", "source model")
	sourceCompute := fs.String("source-compute", "", "source compute placement")
	provider, account, model, compute, capabilities, contextLimit, budget, cache := sessionMoveDestinationFlags(fs)
	if err := fs.Parse(argv); err != nil || fs.NArg() != 0 {
		return 2
	}
	terminal, err := io.ReadAll(stdin)
	if err != nil {
		fmt.Fprintf(stderr, "session move checkpoint: %v\n", err)
		return 1
	}
	checkpoint := gateway.FinalizeSessionMoveCheckpoint(gateway.SessionMoveCheckpoint{
		SessionID: *sessionID, SourceEpoch: *sourceEpoch, EventHead: *eventHead,
		Source:      gateway.SessionPlacement{Provider: *sourceProvider, AccountRef: *sourceAccount, Model: *sourceModel, Compute: *sourceCompute, ComputeAvailable: true},
		Destination: sessionMoveDestination(*provider, *account, *model, *compute, *capabilities, *contextLimit, *budget, *cache),
		Terminal:    terminal, CreatedAt: time.Now().UTC(),
	})
	if checkpoint.SessionID == "" || checkpoint.SourceEpoch == "" || checkpoint.Source.Provider == "" || checkpoint.Source.AccountRef == "" || checkpoint.Source.Model == "" || checkpoint.Source.Compute == "" {
		fmt.Fprintln(stderr, "session move checkpoint: source identity and placement flags are required")
		return 2
	}
	if _, err := gateway.AdmitSessionMoveCheckpoint(checkpoint, checkpoint.Destination); err != nil {
		fmt.Fprintf(stderr, "session move checkpoint: %v\n", err)
		return 1
	}
	return writeSessionMoveJSON(stdout, checkpoint)
}

func sessionMoveDestinationFlags(fs *flag.FlagSet) (*string, *string, *string, *string, *string, *int64, *int64, *string) {
	return fs.String("provider", "", "destination provider"), fs.String("account-ref", "", "destination account reference"), fs.String("model", "", "destination model"), fs.String("compute", "", "destination compute placement"), fs.String("capabilities", "", "comma-separated destination capabilities"), fs.Int64("context-limit", 0, "destination context limit"), fs.Int64("budget", 0, "destination available budget"), fs.String("cache-lineage", "", "destination cache lineage")
}

func sessionMoveDestination(provider, account, model, compute, capabilities string, contextLimit int64, budget int64, cache string) gateway.SessionPlacement {
	return gateway.SessionPlacement{Provider: provider, AccountRef: account, Model: model, Compute: compute, Capabilities: splitSessionMoveCSV(capabilities), ContextLimit: contextLimit, BudgetAvailable: budget, ComputeAvailable: true, CacheLineage: cache}
}

func runSessionMoveAdmit(stdin io.Reader, stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("session move admit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	provider := fs.String("provider", "", "destination provider")
	account := fs.String("account-ref", "", "destination account reference")
	model := fs.String("model", "", "destination model")
	compute := fs.String("compute", "", "destination compute identity")
	caps := fs.String("capabilities", "", "comma-separated destination capabilities")
	contextLimit := fs.Int64("context-limit", 0, "destination context limit")
	budget := fs.Int64("budget", 0, "destination available budget")
	cache := fs.String("cache-lineage", "", "destination cache lineage")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		sessionMoveUsage(stderr)
		return 2
	}
	var checkpoint gateway.SessionMoveCheckpoint
	if err := json.NewDecoder(io.LimitReader(stdin, 16<<20)).Decode(&checkpoint); err != nil {
		fmt.Fprintln(stderr, "fak session move admit:", err)
		return 1
	}
	destination := gateway.SessionPlacement{Provider: *provider, AccountRef: *account, Model: *model, Compute: *compute, Capabilities: splitSessionMoveCSV(*caps), ContextLimit: *contextLimit, BudgetAvailable: *budget, ComputeAvailable: true, CacheLineage: *cache}
	receipt, err := gateway.AdmitSessionMoveCheckpoint(checkpoint, destination)
	if err != nil {
		fmt.Fprintln(stderr, "fak session move admit:", err)
		return 1
	}
	if err := json.NewEncoder(stdout).Encode(receipt); err != nil {
		fmt.Fprintln(stderr, "fak session move admit:", err)
		return 1
	}
	return 0
}

func runSessionMoveRestore(stdin io.Reader, stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("session move restore", flag.ContinueOnError)
	fs.SetOutput(stderr)
	provider, account, model, compute, capabilities, contextLimit, budget, cache := sessionMoveDestinationFlags(fs)
	endpoint := fs.String("provider-endpoint", "http://127.0.0.1:11434", "destination Ollama endpoint (credentials must not be embedded)")
	stateFile := fs.String("state-file", "", "destination materialized checkpoint path")
	if err := fs.Parse(argv); err != nil || fs.NArg() != 0 || *stateFile == "" {
		return 2
	}
	var checkpoint gateway.SessionMoveCheckpoint
	if err := json.NewDecoder(stdin).Decode(&checkpoint); err != nil {
		fmt.Fprintf(stderr, "session move restore: decode checkpoint: %v\n", err)
		return 1
	}
	destination := sessionMoveDestination(*provider, *account, *model, *compute, *capabilities, *contextLimit, *budget, *cache)
	if _, err := gateway.AdmitSessionMoveCheckpoint(checkpoint, destination); err != nil {
		fmt.Fprintf(stderr, "session move restore: %v\n", err)
		return 1
	}
	materialized, err := materializeSessionMoveCheckpoint(*stateFile, checkpoint)
	if err != nil {
		fmt.Fprintf(stderr, "session move restore: materialize: %v\n", err)
		return 1
	}
	providerWitness, err := continueSessionMoveProvider(*endpoint, checkpoint, destination)
	if err != nil {
		fmt.Fprintf(stderr, "session move restore: provider continuation: %v\n", err)
		return 1
	}
	receipt, err := gateway.RestoreSessionMoveCheckpoint(checkpoint, materialized, destination, providerWitness)
	if err != nil {
		fmt.Fprintf(stderr, "session move restore: %v\n", err)
		return 1
	}
	return writeSessionMoveJSON(stdout, receipt)
}

func materializeSessionMoveCheckpoint(path string, checkpoint gateway.SessionMoveCheckpoint) (gateway.SessionMoveCheckpoint, error) {
	body, err := json.MarshalIndent(checkpoint, "", "  ")
	if err != nil {
		return gateway.SessionMoveCheckpoint{}, err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return gateway.SessionMoveCheckpoint{}, err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(body, '\n'), 0600); err != nil {
		return gateway.SessionMoveCheckpoint{}, err
	}
	if err := os.Rename(tmp, path); err != nil {
		return gateway.SessionMoveCheckpoint{}, err
	}
	var restored gateway.SessionMoveCheckpoint
	f, err := os.Open(path)
	if err != nil {
		return restored, err
	}
	defer f.Close()
	if err := json.NewDecoder(f).Decode(&restored); err != nil {
		return restored, err
	}
	return restored, nil
}

func continueSessionMoveProvider(endpoint string, checkpoint gateway.SessionMoveCheckpoint, destination gateway.SessionPlacement) (gateway.SessionMoveProviderWitness, error) {
	parsedEndpoint, err := url.Parse(endpoint)
	if err != nil || parsedEndpoint.Scheme == "" || parsedEndpoint.Host == "" || parsedEndpoint.User != nil {
		return gateway.SessionMoveProviderWitness{}, errors.New("provider endpoint must be an absolute credential-free URL")
	}
	prompt := fmt.Sprintf("Continue logical session %s from checkpoint %s. Reply with exactly RESTORED.", checkpoint.SessionID, checkpoint.Digest)
	request := map[string]any{"model": destination.Model, "prompt": prompt, "stream": false, "options": map[string]any{"temperature": 0}}
	body, _ := json.Marshal(request)
	req, err := http.NewRequest(http.MethodPost, strings.TrimRight(parsedEndpoint.String(), "/")+"/api/generate", bytes.NewReader(body))
	if err != nil {
		return gateway.SessionMoveProviderWitness{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 2 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return gateway.SessionMoveProviderWitness{}, err
	}
	defer resp.Body.Close()
	responseBody, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return gateway.SessionMoveProviderWitness{}, err
	}
	if resp.StatusCode != http.StatusOK {
		return gateway.SessionMoveProviderWitness{}, fmt.Errorf("status %s", resp.Status)
	}
	var decoded struct {
		Model    string `json:"model"`
		Response string `json:"response"`
		Done     bool   `json:"done"`
	}
	if err := json.Unmarshal(responseBody, &decoded); err != nil {
		return gateway.SessionMoveProviderWitness{}, err
	}
	if decoded.Model == "" || strings.TrimSpace(decoded.Response) == "" || !decoded.Done {
		return gateway.SessionMoveProviderWitness{}, fmt.Errorf("incomplete model response")
	}
	requestSum, responseSum := sha256.Sum256(body), sha256.Sum256(responseBody)
	return gateway.SessionMoveProviderWitness{Provider: destination.Provider, Model: decoded.Model, RequestDigest: "sha256:" + hex.EncodeToString(requestSum[:]), ResponseDigest: "sha256:" + hex.EncodeToString(responseSum[:]), CompletedAt: time.Now().UTC()}, nil
}

func writeSessionMoveJSON(stdout io.Writer, value any) int {
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(value); err != nil {
		return 1
	}
	return 0
}

func runSessionMoveVerify(stdin io.Reader, stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("session move verify", flag.ContinueOnError)
	fs.SetOutput(stderr)
	receiptPath := fs.String("receipt", "", "destination receipt JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 || *receiptPath == "" {
		sessionMoveUsage(stderr)
		return 2
	}
	var checkpoint gateway.SessionMoveCheckpoint
	if err := json.NewDecoder(io.LimitReader(stdin, 16<<20)).Decode(&checkpoint); err != nil {
		fmt.Fprintln(stderr, "fak session move verify:", err)
		return 1
	}
	f, err := os.Open(*receiptPath)
	if err != nil {
		fmt.Fprintln(stderr, "fak session move verify:", err)
		return 1
	}
	defer f.Close()
	var receipt gateway.SessionMoveDestinationReceipt
	if err := json.NewDecoder(io.LimitReader(f, 4<<20)).Decode(&receipt); err != nil {
		fmt.Fprintln(stderr, "fak session move verify:", err)
		return 1
	}
	if err := gateway.VerifySessionMoveDestinationReceipt(checkpoint, receipt); err != nil {
		fmt.Fprintln(stderr, "fak session move verify:", err)
		return 1
	}
	fmt.Fprintf(stdout, "MOVE RECEIPT PASS session=%s source_epoch=%s checkpoint=%s destination=%s/%s/%s/%s receipt=%s\n", receipt.SessionID, receipt.SourceEpoch, receipt.CheckpointHash, receipt.Destination.Provider, receipt.Destination.AccountRef, receipt.Destination.Model, receipt.Destination.Compute, receipt.ReceiptDigest)
	return 0
}

func splitSessionMoveCSV(s string) []string {
	var out []string
	for _, v := range strings.Split(s, ",") {
		if v = strings.TrimSpace(v); v != "" {
			out = append(out, v)
		}
	}
	return out
}
func sessionMoveUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak session move admit --provider P --account-ref R --model M --compute C [--capabilities A,B] [--context-limit N] [--budget N] < checkpoint.json\n       fak session move verify --receipt receipt.json < checkpoint.json")
}
