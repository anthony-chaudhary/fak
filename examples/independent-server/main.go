package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessserver"
	"github.com/anthony-chaudhary/fak/internal/serverlifecycle"
	"github.com/anthony-chaudhary/fak/internal/serverproduct"
)

const (
	selfcheckSchema = "fak.independent-server-selfcheck/v1"
	eventSchema     = "fak.independent-server-event/v1"
	modelAlias      = "local-code"
	protocolRev     = "2026-02"
	eventFilename   = "events.jsonl"
)

var modelFixture = []byte("FAK independent-server fixture artifact\n")

type phaseEvidence struct {
	Name      string `json:"name"`
	ElapsedMS int64  `json:"elapsed_ms"`
}

type boundaryEvidence struct {
	CleanProductDirectories int      `json:"clean_product_directories"`
	ServerRoot              string   `json:"server_root"`
	HarnessRoot             string   `json:"harness_root"`
	DistinctRoots           bool     `json:"distinct_roots"`
	ReceiptOnlyHandoff      bool     `json:"receipt_only_handoff"`
	CrossedPaths            []string `json:"crossed_paths"`
	SharedMutablePaths      []string `json:"shared_mutable_paths"`
}

type receiptEvidence struct {
	Schema           string   `json:"schema"`
	Digest           string   `json:"digest"`
	SpecDigest       string   `json:"spec_digest"`
	Generation       uint64   `json:"generation"`
	ArtifactName     string   `json:"artifact_name"`
	ArtifactDigest   string   `json:"artifact_digest"`
	AdapterName      string   `json:"adapter_name"`
	AdapterVersion   string   `json:"adapter_version"`
	ExecutableDigest string   `json:"executable_digest"`
	ReadinessProbe   string   `json:"readiness_probe"`
	ReadinessDigest  string   `json:"readiness_digest"`
	Capabilities     []string `json:"capabilities"`
	EventLogDigest   string   `json:"event_log_digest"`
	ReceiptUnchanged bool     `json:"receipt_unchanged"`
}

type harnessEvidence struct {
	Schema              string `json:"schema"`
	BindingDigest       string `json:"binding_digest"`
	PinnedReceiptDigest string `json:"pinned_receipt_digest"`
	Generation          uint64 `json:"generation"`
	ModelAlias          string `json:"model_alias"`
	ProtocolFamily      string `json:"protocol_family"`
	ProtocolRevision    string `json:"protocol_revision"`
	ChatCalls           int    `json:"chat_calls"`
	ChatResponse        string `json:"chat_response"`
	LifecycleCalls      int    `json:"lifecycle_calls"`
}

type teardownEvidence struct {
	State                 serverlifecycle.State `json:"state"`
	Owned                 bool                  `json:"owned"`
	InstanceIDMatch       bool                  `json:"instance_id_match"`
	GenerationMatch       bool                  `json:"generation_match"`
	ProcessIDMatch        bool                  `json:"process_id_match"`
	OwnedTeardowns        int                   `json:"owned_teardowns"`
	HarnessLifecycleCalls int                   `json:"harness_lifecycle_calls"`
}

type selfcheckResult struct {
	Schema                  string           `json:"schema"`
	Verdict                 string           `json:"verdict"`
	Boundary                boundaryEvidence `json:"boundary"`
	Receipt                 receiptEvidence  `json:"receipt"`
	Harness                 harnessEvidence  `json:"harness"`
	Teardown                teardownEvidence `json:"teardown"`
	ReadinessProbes         int              `json:"readiness_probes"`
	ExternalNetworkRequests int              `json:"external_network_requests"`
	Phases                  []phaseEvidence  `json:"phases"`
}

type eventRecord struct {
	Schema    string `json:"schema"`
	Kind      string `json:"kind"`
	Actor     string `json:"actor"`
	Operation string `json:"operation"`
	Path      string `json:"path,omitempty"`
	Success   bool   `json:"success"`
}

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	switch {
	case len(args) == 1 && args[0] == "--version":
		fmt.Fprintln(stdout, "llama-server fixture version 1")
		return 0
	case len(args) > 0 && args[0] == "--model":
		if err := runFixtureServer(args); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		return 0
	case len(args) > 0 && args[0] == "--server-controller":
		if err := runServerController(args[1:], stdout); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		return 0
	case len(args) > 0 && args[0] == "--harness-consumer":
		if err := runHarnessConsumer(args[1:], stdout); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		return 0
	}

	fs := flag.NewFlagSet("independent-server", flag.ContinueOnError)
	fs.SetOutput(stderr)
	selfcheck := fs.Bool("selfcheck", false, "run the deterministic independent-products witness")
	workRoot := fs.String("work-root", "", "retain artifacts beneath an empty directory (test/debug only)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if !*selfcheck || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: go run ./examples/independent-server -selfcheck")
		return 2
	}
	result, err := runSelfcheck(*workRoot)
	if err != nil {
		fmt.Fprintf(stderr, "independent-server selfcheck: %v\n", err)
		return 1
	}
	encoder := json.NewEncoder(stdout)
	encoder.SetEscapeHTML(false)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(result); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	return 0
}

func runSelfcheck(requestedRoot string) (result selfcheckResult, retErr error) {
	root, removeRoot, err := prepareRoot(requestedRoot)
	if err != nil {
		return result, err
	}
	if removeRoot {
		defer os.RemoveAll(root)
	}
	serverDir := filepath.Join(root, "server-product")
	harnessDir := filepath.Join(root, "harness-product")
	if err := ensureDistinctSiblingRoots(root, serverDir, harnessDir); err != nil {
		return result, err
	}
	for _, dir := range []string{serverDir, harnessDir} {
		if err := os.Mkdir(dir, 0o700); err != nil {
			return result, fmt.Errorf("create clean product directory: %w", err)
		}
	}

	currentExecutable, err := os.Executable()
	if err != nil {
		return result, err
	}
	serverExecutable := filepath.Join(serverDir, executableName("llama-server"))
	serverController := filepath.Join(serverDir, executableName("server-controller"))
	harnessExecutable := filepath.Join(harnessDir, executableName("harness-consumer"))
	for _, destination := range []string{serverExecutable, serverController, harnessExecutable} {
		if err := copyExecutable(currentExecutable, destination); err != nil {
			return result, err
		}
	}
	modelPath := filepath.Join(serverDir, "fixture.gguf")
	if err := os.WriteFile(modelPath, modelFixture, 0o600); err != nil {
		return result, err
	}
	modelDigest := digestBytes(modelFixture)
	eventPath := filepath.Join(serverDir, eventFilename)

	var initResult, upResult, downResult serverlifecycle.Result
	phases := make([]phaseEvidence, 0, 4)
	started := time.Now()
	initResult, err = runServerProduct(serverController, serverDir, "init",
		"--model", modelPath,
		"--sha256", modelDigest,
		"--adapter-executable", serverExecutable,
	)
	phases = append(phases, elapsedPhase("init", started))
	if err != nil || initResult.State != serverlifecycle.StateConfigured {
		return result, fmt.Errorf("server init: state=%s: %w", initResult.State, err)
	}

	serverReady := false
	serverStopped := false
	defer func() {
		if serverReady && !serverStopped {
			_, _ = runServerProduct(serverController, serverDir, "down")
		}
	}()
	started = time.Now()
	upResult, err = runServerProduct(serverController, serverDir, "up")
	phases = append(phases, elapsedPhase("up", started))
	if err != nil || upResult.State != serverlifecycle.StateReady {
		return result, fmt.Errorf("server up: state=%s: %w", upResult.State, err)
	}
	serverReady = true

	receiptPath := filepath.Join(serverDir, serverlifecycle.ReceiptFilename)
	receiptBefore, receipt, err := readReceipt(receiptPath)
	if err != nil {
		return result, err
	}
	started = time.Now()
	harness, err := runHarnessProduct(harnessExecutable, harnessDir, receiptPath)
	phases = append(phases, elapsedPhase("harness", started))
	if err != nil {
		return result, err
	}

	started = time.Now()
	downResult, err = runServerProduct(serverController, serverDir, "down")
	phases = append(phases, elapsedPhase("down", started))
	if err != nil || downResult.State != serverlifecycle.StateStopped {
		return result, fmt.Errorf("server down: state=%s: %w", downResult.State, err)
	}
	serverStopped = true

	receiptAfter, rereadReceipt, err := readReceipt(receiptPath)
	if err != nil {
		return result, err
	}
	if digestBytes(receiptBefore) != digestBytes(receiptAfter) {
		return result, errors.New("server receipt changed after immutable handoff")
	}
	if rereadReceipt.Identity.InstanceID != receipt.Identity.InstanceID || rereadReceipt.Generation != receipt.Generation {
		return result, errors.New("reread receipt identity changed")
	}
	bindingPath := filepath.Join(harnessDir, harnessserver.BindingFileName)
	bindingRaw, err := os.ReadFile(bindingPath)
	if err != nil {
		return result, fmt.Errorf("reread harness binding: %w", err)
	}
	if digestBytes(bindingRaw) != harness.BindingDigest {
		return result, errors.New("harness binding digest changed after consumer exit")
	}
	eventRaw, events, err := readEvents(eventPath)
	if err != nil {
		return result, err
	}
	readinessProbes := countEvents(events, "probe", "server-product", "")
	harnessChatCalls := countEvents(events, "chat", "harness-product", "chat")
	ownedTeardowns := countEvents(events, "lifecycle", "server-product", "down")
	harnessLifecycleCalls := countEvents(events, "lifecycle", "harness-product", "")
	if readinessProbes != 3 || harnessChatCalls != 1 || ownedTeardowns != 1 || harnessLifecycleCalls != 0 {
		return result, fmt.Errorf("event witness mismatch: readiness=%d chat=%d teardowns=%d harness_lifecycle=%d", readinessProbes, harnessChatCalls, ownedTeardowns, harnessLifecycleCalls)
	}
	receiptDigest := digestBytes(receiptAfter)
	owned := downResult.Evidence.ProcessIdentityMatch && downResult.Evidence.ReceiptValid &&
		downResult.InstanceID == receipt.Identity.InstanceID && downResult.Generation == receipt.Generation &&
		downResult.ProcessID == receipt.Ownership.ProcessID
	if !owned {
		return result, errors.New("teardown did not target the receipt-owned process identity")
	}
	if harness.PinnedReceiptDigest != receiptDigest || harness.ChatResponse != "HANDOFF_OK" || harness.LifecycleCalls != 0 {
		return result, errors.New("harness receipt/chat evidence did not match the server receipt")
	}

	capabilities := append([]string(nil), receipt.Protocol.Capabilities...)
	sort.Strings(capabilities)
	return selfcheckResult{
		Schema:  selfcheckSchema,
		Verdict: "pass",
		Boundary: boundaryEvidence{
			CleanProductDirectories: 2,
			ServerRoot:              "server-product",
			HarnessRoot:             "harness-product",
			DistinctRoots:           true,
			ReceiptOnlyHandoff:      true,
			CrossedPaths:            []string{serverlifecycle.ReceiptFilename},
			SharedMutablePaths:      []string{},
		},
		Receipt: receiptEvidence{
			Schema:           receipt.Schema,
			Digest:           receiptDigest,
			SpecDigest:       receipt.SpecDigest,
			Generation:       receipt.Generation,
			ArtifactName:     filepath.Base(receipt.Artifact.Reference),
			ArtifactDigest:   receipt.Artifact.Digest,
			AdapterName:      receipt.Adapter.Name,
			AdapterVersion:   receipt.Adapter.Version,
			ExecutableDigest: receipt.Adapter.ExecutableDigest,
			ReadinessProbe:   receipt.Readiness.Probe,
			ReadinessDigest:  receipt.Readiness.ProbeDigest,
			Capabilities:     capabilities,
			EventLogDigest:   digestBytes(eventRaw),
			ReceiptUnchanged: true,
		},
		Harness: harness,
		Teardown: teardownEvidence{
			State:                 downResult.State,
			Owned:                 true,
			InstanceIDMatch:       downResult.InstanceID == receipt.Identity.InstanceID,
			GenerationMatch:       downResult.Generation == receipt.Generation,
			ProcessIDMatch:        downResult.ProcessID == receipt.Ownership.ProcessID,
			OwnedTeardowns:        ownedTeardowns,
			HarnessLifecycleCalls: harnessLifecycleCalls,
		},
		ReadinessProbes:         readinessProbes,
		ExternalNetworkRequests: 0,
		Phases:                  phases,
	}, nil
}

func prepareRoot(requested string) (string, bool, error) {
	if requested == "" {
		root, err := os.MkdirTemp("", "fak-independent-server-")
		return root, true, err
	}
	root, err := filepath.Abs(requested)
	if err != nil {
		return "", false, err
	}
	root = filepath.Clean(root)
	if err := os.MkdirAll(root, 0o700); err != nil {
		return "", false, err
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		return "", false, err
	}
	if len(entries) != 0 {
		return "", false, errors.New("work root must be empty")
	}
	return root, false, nil
}

func ensureDistinctSiblingRoots(root, serverDir, harnessDir string) error {
	for _, dir := range []string{serverDir, harnessDir} {
		rel, err := filepath.Rel(root, dir)
		if err != nil || rel == "." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			return errors.New("product directory escapes the selfcheck root")
		}
	}
	leftToRight, err := filepath.Rel(serverDir, harnessDir)
	if err != nil || leftToRight == "." || !strings.HasPrefix(leftToRight, ".."+string(filepath.Separator)) {
		return errors.New("server and harness roots must be distinct siblings")
	}
	return nil
}

func executableName(base string) string {
	if runtime.GOOS == "windows" {
		return base + ".exe"
	}
	return base
}

func copyExecutable(source, destination string) error {
	src, err := os.Open(source)
	if err != nil {
		return err
	}
	defer src.Close()
	dst, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o755)
	if err != nil {
		return err
	}
	closed := false
	defer func() {
		if !closed {
			_ = dst.Close()
		}
	}()
	if _, err := io.Copy(dst, src); err != nil {
		return err
	}
	if err := dst.Close(); err != nil {
		return err
	}
	closed = true
	return nil
}

func readReceipt(path string) ([]byte, serverproduct.ServerReceipt, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, serverproduct.ServerReceipt{}, fmt.Errorf("reread server receipt: %w", err)
	}
	receipt, err := serverproduct.DecodeReceipt(raw)
	if err != nil {
		return nil, serverproduct.ServerReceipt{}, err
	}
	return raw, receipt, nil
}

func appendEvent(path string, event eventRecord) error {
	raw, err := json.Marshal(event)
	if err != nil {
		return err
	}
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	if _, err := file.Write(append(raw, '\n')); err != nil {
		_ = file.Close()
		return err
	}
	return file.Close()
}

func readEvents(path string) ([]byte, []eventRecord, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	var events []eventRecord
	scanner := bufio.NewScanner(strings.NewReader(string(raw)))
	for scanner.Scan() {
		var event eventRecord
		decoder := json.NewDecoder(strings.NewReader(scanner.Text()))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&event); err != nil {
			return nil, nil, err
		}
		if event.Schema != eventSchema || !event.Success {
			return nil, nil, errors.New("event log contains invalid or unsuccessful evidence")
		}
		events = append(events, event)
	}
	if err := scanner.Err(); err != nil {
		return nil, nil, err
	}
	return raw, events, nil
}

func countEvents(events []eventRecord, kind, actor, operation string) int {
	count := 0
	for _, event := range events {
		if event.Kind == kind && event.Actor == actor && (operation == "" || event.Operation == operation) {
			count++
		}
	}
	return count
}

func elapsedPhase(name string, started time.Time) phaseEvidence {
	return phaseEvidence{Name: name, ElapsedMS: time.Since(started).Milliseconds()}
}

func digestBytes(raw []byte) string {
	sum := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(sum[:])
}
