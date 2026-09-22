package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/canon"
	"github.com/anthony-chaudhary/fak/internal/childprocess"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// Native launch evidence vocabulary. The shared receipt may only carry what the
// child's own Fak-arm receipt can evidence: a missing enforcement block stays
// unknown, while an unsupported posture or a tool surface that contradicts the
// requested floor is refused and can never inherit OpenCode qualification.
const (
	opsNativeHarness = "native"

	opsNativeGuardEvidenceQualified = "qualified"
	opsNativeGuardEvidenceUnknown   = "unknown"
	opsNativeGuardEvidenceRefused   = "refused"

	opsNativeReceiptStoreEnv    = "FAK_OPS_NATIVE_RECEIPT_STORE"
	opsNativeConfigPolicySource = "native-capability-floor"
)

// opsNativeCapabilityFloor is the exact tool surface this launch asks the child
// to apply. The native arm runs fail_closed with sys/MCP/skills/memory disabled
// (see the argv built in runOpsNative), so the receipt states the requested
// toggles instead of inferring a guarded:true stub from ambient state.
type opsNativeCapabilityFloor struct {
	Posture string
	System  bool
	MCP     bool
	Skills  bool
	Memory  bool
}

func requestedOpsNativeCapabilityFloor() opsNativeCapabilityFloor {
	return opsNativeCapabilityFloor{Posture: "fail_closed"}
}

func (f opsNativeCapabilityFloor) digest() string {
	return opsRunDigest(f.mediation())
}

// mediation renders the requested toggles for the shared launch-identity field
// so the receipt records the ACTUAL capability surface, not a presence claim.
func (f opsNativeCapabilityFloor) mediation() string {
	return actualOpsNativeToolMediation(nativeAgentToolCapabilities{System: f.System, MCP: f.MCP, Skills: f.Skills, Memory: f.Memory})
}

func (f opsNativeCapabilityFloor) configPolicy() opsRunConfigPolicyReceipt {
	return opsRunConfigPolicyReceipt{
		Source: opsNativeConfigPolicySource,
		Digest: f.digest(),
		Status: opsNativeGuardEvidenceQualified,
	}
}

func actualOpsNativeToolMediation(tools nativeAgentToolCapabilities) string {
	return fmt.Sprintf("native:sys=%t,mcp=%t,skills=%t,memory=%t", tools.System, tools.MCP, tools.Skills, tools.Memory)
}

func opsNativePostureSupported(posture string) bool {
	switch posture {
	case "fail_closed", "admit_and_log", "default_open":
		return true
	default:
		return false
	}
}

// qualifyOpsNativeEnforcement binds the child-owned enforcement evidence to the
// capability floor this launch requested. Unknown and refused are distinct: a
// receipt that predates the evidence block leaves the run unqualified but usable,
// while a receipt that DECLARES an unsupported posture or a surface contradicting
// the requested floor is refused outright.
func qualifyOpsNativeEnforcement(native *nativeAgentReceipt, floor opsNativeCapabilityFloor) (string, string) {
	if native == nil || native.Enforcement == nil {
		return opsNativeGuardEvidenceUnknown, "missing_child_enforcement_evidence"
	}
	evidence := native.Enforcement
	if evidence.Schema != nativeAgentEnforcementSchema {
		return opsNativeGuardEvidenceRefused, "unsupported_enforcement_schema"
	}
	if !opsNativePostureSupported(evidence.GuardPosture) {
		return opsNativeGuardEvidenceRefused, "unsupported_guard_posture"
	}
	if evidence.GuardPosture != floor.Posture {
		return opsNativeGuardEvidenceRefused, "guard_posture_mismatch"
	}
	if evidence.Tools.System != floor.System || evidence.Tools.MCP != floor.MCP || evidence.Tools.Skills != floor.Skills || evidence.Tools.Memory != floor.Memory {
		return opsNativeGuardEvidenceRefused, "tool_capability_mismatch"
	}
	return opsNativeGuardEvidenceQualified, ""
}

var errOpsNativeReceiptStillSecret = errors.New("native receipt still scans as secret after redaction")

// opsNativeReceiptStoreRoot is the out-of-repository, content-addressed store
// for redacted native receipts. The child receipt carries task text, so its
// durable copy must never become repository scratch.
func opsNativeReceiptStoreRoot() (string, error) {
	if root := strings.TrimSpace(os.Getenv(opsNativeReceiptStoreEnv)); root != "" {
		return root, nil
	}
	cache, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(cache, "fak", "ops-native-receipts"), nil
}

// preserveOpsNativeReceipt writes a redacted, content-addressed copy of the
// private child receipt and returns its "sha256:<hex>" reference. It must run
// before nativePath is deleted: the durable reference is the only thing the
// shared receipt may point at, never the soon-deleted temporary path.
func preserveOpsNativeReceipt(nativePath string) (string, error) {
	raw, err := os.ReadFile(nativePath)
	if err != nil {
		return "", err
	}
	if len(bytes.TrimSpace(raw)) == 0 {
		return "", errOpsNativeReceiptStillSecret
	}
	redacted, _ := canon.RedactSecrets(raw)
	if canon.Scan(redacted).Secret {
		return "", errOpsNativeReceiptStillSecret
	}
	sum := sha256.Sum256(redacted)
	hexSum := hex.EncodeToString(sum[:])
	root, err := opsNativeReceiptStoreRoot()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(root, "sha256")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	target := filepath.Join(dir, hexSum+".json")
	if _, statErr := os.Stat(target); statErr != nil {
		temp, createErr := os.CreateTemp(dir, ".native-receipt-*.json")
		if createErr != nil {
			return "", createErr
		}
		name := temp.Name()
		defer os.Remove(name)
		if _, writeErr := temp.Write(redacted); writeErr != nil {
			_ = temp.Close()
			return "", writeErr
		}
		if syncErr := temp.Sync(); syncErr != nil {
			_ = temp.Close()
			return "", syncErr
		}
		if closeErr := temp.Close(); closeErr != nil {
			return "", closeErr
		}
		if renameErr := os.Rename(name, target); renameErr != nil {
			return "", renameErr
		}
	}
	return "sha256:" + hexSum, nil
}

// opsNativeFinalizeEvidence preserves the redacted child receipt, then binds the
// shared launch-identity fields to the child-attested enforcement evidence. It
// returns the guard evidence status the shared receipt must report.
func opsNativeFinalizeEvidence(identity *opsRunLaunchIdentityReceipt, nativePath string, floor opsNativeCapabilityFloor) (string, string) {
	var native *nativeAgentReceipt
	if data, readErr := os.ReadFile(nativePath); readErr == nil {
		var parsed nativeAgentReceipt
		if json.Unmarshal(data, &parsed) == nil {
			native = &parsed
		}
	}
	if identity != nil {
		if ref, preserveErr := preserveOpsNativeReceipt(nativePath); preserveErr == nil {
			identity.CapabilityEvidenceRef = ref
			identity.GuardEvidenceRef = ref
		}
	}
	status, reason := qualifyOpsNativeEnforcement(native, floor)
	if identity == nil {
		return status, reason
	}
	identity.GuardEffective = status
	identity.InferenceGuard = status
	identity.NativeToolMediation = floor.mediation()
	if status == opsNativeGuardEvidenceQualified {
		identity.GuardModeEffective = "enforce"
		identity.NativeToolMediation = actualOpsNativeToolMediation(native.Enforcement.Tools)
		if identity.PolicyDigest == "unknown" && native.Enforcement.PolicyDigest != "" {
			// The parent has no --policy digest to attest, so the child's own
			// receipt is the only evidence for the built-in floor. Keep the
			// provenance explicit rather than presenting it as operator-supplied.
			identity.PolicyDigest = native.Enforcement.PolicyDigest
			identity.PolicySource = "child_receipt"
		}
	}
	return status, reason
}

func runOpsNative(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("ops run --harness native", flag.ContinueOnError)
	fs.SetOutput(stderr)
	harness := fs.String("harness", "native", "native FAK chat harness")
	provider := fs.String("provider", "openai", "native provider wire")
	model := fs.String("model", "", "explicit upstream model")
	baseURL := fs.String("base-url", "", "explicit upstream endpoint (required)")
	keyEnv := fs.String("api-key-env", "", "environment variable holding upstream key")
	codexAuth := fs.Bool("codex-auth", false, "explicitly reuse a Codex-managed ChatGPT login read-only; Codex owns renewal")
	codexHome := fs.String("codex-home", "", "Codex credential home for --codex-auth (default: existing discovery)")
	promptFile := fs.String("prompt-file", "", "UTF-8 task file")
	receiptPath := fs.String("receipt", "", "metadata-only execution receipt")
	policy := fs.String("policy", "", "native capability floor policy file")
	workspace := fs.String("workspace", "", "root for bounded code tools (default current directory)")
	fs.StringVar(workspace, "code-workspace", "", "alias for --workspace")
	maxTurns := fs.Int("max-turns", 10, "positive native model turn ceiling")
	effort := fs.String("effort", "", "native reasoning effort")
	timeout := fs.Duration("timeout", 5*time.Minute, "positive wall-clock deadline")
	dryRun := fs.Bool("dry-run", false, "validate without launching")
	if err := fs.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if *harness != "native" || fs.NArg() != 0 || strings.TrimSpace(*model) == "" || strings.TrimSpace(*baseURL) == "" || *promptFile == "" || *timeout <= 0 || *maxTurns <= 0 || (!*dryRun && *receiptPath == "") {
		fmt.Fprintln(stderr, "ops run native: require --model, --base-url, --prompt-file, --receipt, positive --timeout and --max-turns; no positional arguments")
		return 2
	}
	if *effort != "" && *effort != "none" && *effort != "low" && *effort != "medium" && *effort != "balanced" && *effort != "adaptive" && *effort != "high" {
		fmt.Fprintln(stderr, "ops run native: unsupported reasoning effort")
		return 2
	}
	switch *provider {
	case "openai", "openai-responses", "astra", "anthropic", "gemini", "xai":
	default:
		fmt.Fprintln(stderr, "ops run native: unsupported provider wire")
		return 2
	}
	if *codexHome != "" && !*codexAuth {
		fmt.Fprintln(stderr, "ops run native: --codex-home requires --codex-auth")
		return 2
	}
	if *codexAuth && (*provider != "openai-responses" || strings.TrimRight(*baseURL, "/") != guardCodexChatGPTBackendBaseURL || *keyEnv != "") {
		fmt.Fprintf(stderr, "ops run native: --codex-auth requires --provider openai-responses, --base-url %s and no API key environment\n", guardCodexChatGPTBackendBaseURL)
		return 2
	}
	prompt, err := os.ReadFile(*promptFile)
	if err != nil || len(strings.TrimSpace(string(prompt))) == 0 {
		fmt.Fprintln(stderr, "ops run native: prompt file must be readable and nonempty")
		return 2
	}
	if *policy != "" {
		if _, err := os.ReadFile(*policy); err != nil {
			fmt.Fprintln(stderr, "ops run native: policy must be readable")
			return 2
		}
	}
	for _, input := range []string{*promptFile, *policy} {
		if input != "" && *receiptPath != "" && opsRunSamePath(input, *receiptPath) {
			fmt.Fprintln(stderr, "ops run native: receipt must be distinct from prompt and policy")
			return 2
		}
	}
	if *dryRun {
		_ = json.NewEncoder(stdout).Encode(map[string]any{"schema": "fak-ops-run-plan/1", "harness": opsNativeHarness, "provider": *provider, "guarded": false, "guard_requested": "fail_closed", "guard_effective": "unknown", "prompt_delivery": "file", "timeout": timeout.String(), "max_turns": *maxTurns, "native_tool_mediation": requestedOpsNativeCapabilityFloor().mediation()})
		return 0
	}
	// The native receipt contains task text. Keep it private and ephemeral; the
	// durable Ops receipt intentionally retains execution metadata only.
	nativeReceipt, err := os.CreateTemp("", "fak-ops-native-*.json")
	if err != nil {
		fmt.Fprintln(stderr, "ops run native: create private receipt:", err)
		return 1
	}
	nativePath := nativeReceipt.Name()
	_ = nativeReceipt.Close()
	defer os.Remove(nativePath)
	promptPath, err := filepath.Abs(*promptFile)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	argv := []string{"chat", "--task-file", promptPath, "--provider", *provider, "--model", *model, "--base-url", *baseURL, "--api-key-env", *keyEnv, "--receipt", nativePath, "--max-turns", fmt.Sprint(*maxTurns), "--posture", "fail_closed", "--sys-tools=false", "--mcp-tools=false", "--skills=false", "--memory=false"}
	for _, pair := range [][2]string{{"--policy", *policy}, {"--code-workspace", *workspace}, {"--effort", *effort}} {
		if pair[1] != "" {
			argv = append(argv, pair[0], pair[1])
		}
	}
	if *codexAuth {
		argv = append(argv, "--codex-auth")
		selectedCodexHome := strings.TrimSpace(*codexHome)
		if selectedCodexHome == "" {
			selectedCodexHome, _ = resolveCodexHome("", true)
		}
		if selectedCodexHome != "" {
			argv = append(argv, "--codex-home", selectedCodexHome)
		}
	}
	env, cleanupEnv, err := opsRunChildEnvironment("{}", *keyEnv)
	if err != nil {
		fmt.Fprintln(stderr, "ops run native: create isolated child environment:", err)
		return 1
	}
	defer cleanupEnv()
	ctx, stop := signal.NotifyContext(context.Background(), terminatingSignals()...)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	// The shared receipt must carry the same canonical workspace and the same
	// requested capability floor the child applies; resolve both before launch.
	resolvedWorkspace := strings.TrimSpace(*workspace)
	if resolvedWorkspace == "" {
		resolvedWorkspace, _ = os.Getwd()
	}
	if canonical, resolveErr := resolveOpsRunWorkspace(resolvedWorkspace); resolveErr == nil {
		resolvedWorkspace = canonical
	}
	floor := requestedOpsNativeCapabilityFloor()
	configPolicy := floor.configPolicy()
	var nonce [12]byte
	runID := "ops-native-unknown"
	if _, randErr := rand.Read(nonce[:]); randErr == nil {
		runID = "ops-" + hex.EncodeToString(nonce[:])
	}
	identity := newOpsRunLaunchIdentity(runID, opsNativeHarness, resolvedWorkspace, *provider, *baseURL, *model, tuiExecutable(), "", *policy, "enforce", false, false)
	identity.NativeToolMediation = floor.mediation()
	receipt := opsRunReceipt{Schema: "fak-ops-run/1", Harness: opsNativeHarness, Workspace: resolvedWorkspace, LaunchIdentity: &identity, Status: "running", Started: time.Now().UTC(), ConfigPolicy: &configPolicy}
	if err := writeOpsRunReceipt(*receiptPath, receipt); err != nil {
		fmt.Fprintln(stderr, "ops run native:", err)
		return 1
	}
	if *provider != "openai" {
		preflight := opsRunInferenceRefusal(*provider, *baseURL, *model, "unsupported_provider_protocol")
		return failOpsRunInferencePreflight(stderr, *receiptPath, receipt, preflight)
	}
	preflight, err := opsRunInferencePreflight(ctx, *baseURL, *model)
	receipt.InferencePreflight = &preflight
	if err != nil {
		return failOpsRunInferencePreflight(stderr, *receiptPath, receipt, preflight)
	}
	opsRunAddCapabilityEvidence(receipt.LaunchIdentity, preflight.ReceiptRef)
	// Persist the qualifying reference before launch. A receipt write failure
	// cannot produce an unqualified native child process.
	if err := writeOpsRunReceipt(*receiptPath, receipt); err != nil {
		fmt.Fprintln(stderr, "ops run native:", err)
		return 1
	}
	cmd := exec.CommandContext(ctx, tuiExecutable(), argv...)
	cmd.Stdout, cmd.Stderr = stdout, stderr
	cmd.Env = env
	cmd.WaitDelay = 5 * time.Second
	procguard.ConfigureProcessTreeCancel(cmd)
	windowgate.ConfigureBackgroundCommand(cmd)
	err = cmd.Run()
	code := childprocess.ExitCode(err, 1)
	if err == nil {
		code = 0
	}
	receipt.Status = "failed"
	if ctx.Err() != nil {
		code, receipt.Status = 130, "cancelled"
		if ctx.Err() == context.DeadlineExceeded {
			code, receipt.Status = 124, "timed_out"
		}
	} else if code == 0 {
		data, readErr := os.ReadFile(nativePath)
		var native nativeAgentReceipt
		if readErr == nil && json.Unmarshal(data, &native) == nil && native.Schema == nativeAgentReceiptSchema && native.Status == "completed" && native.Metrics.Arm == "fak" && !native.Metrics.HitTurnCap && !native.Metrics.CircuitBreakerTripped {
			receipt.Status = "succeeded"
		} else {
			code = 1
			fmt.Fprintln(stderr, "ops run native: child did not report a completed native turn")
		}
	}
	// The private child receipt is deleted when this function returns, so the
	// durable redacted copy must be preserved and referenced first.
	evidenceStatus, evidenceReason := opsNativeFinalizeEvidence(receipt.LaunchIdentity, nativePath, floor)
	configPolicy.Status = evidenceStatus
	configPolicy.Reason = evidenceReason
	receipt.ConfigPolicy = &configPolicy
	if evidenceStatus == opsNativeGuardEvidenceRefused && receipt.Status != "cancelled" && receipt.Status != "timed_out" {
		code, receipt.Status = 1, "refused"
		fmt.Fprintf(stderr, "ops run native: child enforcement evidence refused: %s\n", evidenceReason)
	}
	receipt.ExitCode, receipt.Finished = code, time.Now().UTC()
	if err := writeOpsRunReceipt(*receiptPath, receipt); err != nil {
		fmt.Fprintln(stderr, "ops run native:", err)
		return 1
	}
	return code
}
