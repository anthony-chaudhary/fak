package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessmodelset"
	"github.com/anthony-chaudhary/fak/internal/modelinventory"
	"github.com/anthony-chaudhary/fak/internal/modelsetlock"
	"github.com/anthony-chaudhary/fak/internal/modelsetreceipt"
	"github.com/anthony-chaudhary/fak/internal/modelsetresolve"
)

const (
	harnessModelSetExpectationFile = "harness.model-set.expectation.json"
	harnessModelSetIntentFile      = "harness.model-set.json"
	harnessModelSetRuleBytes       = "{\"schema\":\"fak.harness-model-set-selection-policy/1\"}\n"
)

const harnessModelSetUsage = `usage: fak harness model-set <resolve|inspect|selfcheck> [flags]

  resolve   --intent PATH --inventory PATH --out PATH [--expectation PATH]
            [--as-of RFC3339] [--os NAME --arch NAME --accelerator NAME --runtime NAME] [--json]
  inspect   --lock PATH [--receipt PATH] [--json]
  selfcheck --lock PATH --inventory PATH --receipt PATH [--intent PATH]
            [--expectation PATH] [--as-of RFC3339] [--json]

resolve and selfcheck use only local files. selfcheck never downloads an artifact,
contacts a provider, or starts a model server.`

type harnessModelSetResolveResult struct {
	Schema          string                      `json:"schema"`
	Status          string                      `json:"status"`
	LockPath        string                      `json:"lock_path,omitempty"`
	ExpectationPath string                      `json:"expectation_path,omitempty"`
	ContentDigest   string                      `json:"content_digest,omitempty"`
	EvaluatedAt     string                      `json:"evaluated_at"`
	Roles           int                         `json:"roles"`
	Resolution      *modelsetresolve.Resolution `json:"resolution,omitempty"`
}

type harnessModelSetSelfcheckResult struct {
	Schema      string                    `json:"schema"`
	Status      modelsetreceipt.Status    `json:"status"`
	LockDigest  string                    `json:"lock_digest"`
	ReceiptPath string                    `json:"receipt_path"`
	EvaluatedAt string                    `json:"evaluated_at"`
	Roles       int                       `json:"roles"`
	Failures    []modelsetreceipt.Failure `json:"failures"`
}

type harnessModelSetInspection struct {
	Schema          string                         `json:"schema"`
	LockPath        string                         `json:"lock_path"`
	ContentDigest   string                         `json:"content_digest"`
	ResolverVersion string                         `json:"resolver_version"`
	EvaluatedAt     string                         `json:"evaluated_at"`
	Target          modelsetlock.Target            `json:"target"`
	Roles           []harnessModelSetInspectedRole `json:"roles"`
	Receipt         *harnessModelSetReceiptSummary `json:"receipt,omitempty"`
}

type harnessModelSetInspectedRole struct {
	ID             string                     `json:"id"`
	Required       bool                       `json:"required"`
	Status         modelsetresolve.RoleStatus `json:"status"`
	AlternativeID  string                     `json:"alternative_id,omitempty"`
	CandidateID    string                     `json:"candidate_id,omitempty"`
	Source         modelinventory.SourceKind  `json:"source,omitempty"`
	Artifact       string                     `json:"artifact,omitempty"`
	ArtifactDigest string                     `json:"artifact_digest,omitempty"`
	Rejections     int                        `json:"rejections"`
}

type harnessModelSetReceiptSummary struct {
	Path        string                 `json:"path"`
	Status      modelsetreceipt.Status `json:"status"`
	EvaluatedAt string                 `json:"evaluated_at"`
	Failures    int                    `json:"failures"`
}

func runHarnessModelSet(stdout, stderr io.Writer, argv []string) int {
	if stdout == nil || stderr == nil {
		panic("runHarnessModelSet requires writers")
	}
	if len(argv) == 0 {
		fmt.Fprintln(stderr, harnessModelSetUsage)
		return 2
	}
	switch argv[0] {
	case "resolve":
		return runHarnessModelSetResolve(stdout, stderr, argv[1:])
	case "inspect":
		return runHarnessModelSetInspect(stdout, stderr, argv[1:])
	case "selfcheck":
		return runHarnessModelSetSelfcheck(stdout, stderr, argv[1:])
	default:
		fmt.Fprintf(stderr, "fak harness model-set: unknown command %q\n%s\n", argv[0], harnessModelSetUsage)
		return 2
	}
}

func runHarnessModelSetResolve(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness model-set resolve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	intentPath := fs.String("intent", "", "role-indexed harness model-set intent")
	inventoryPath := fs.String("inventory", "", "normalized model inventory")
	outPath := fs.String("out", "", "canonical model-set lock output")
	expectationPath := fs.String("expectation", "", "startup expectation sidecar (default: next to lock)")
	asOfText := fs.String("as-of", "", "explicit RFC3339 evaluation time (default: inventory as_of)")
	osName := fs.String("os", runtime.GOOS, "target operating system")
	architecture := fs.String("arch", runtime.GOARCH, "target architecture")
	accelerator := fs.String("accelerator", "none", "target accelerator")
	runtimeName := fs.String("runtime", "mixed-runtime", "target serving runtime envelope")
	jsonOutput := fs.Bool("json", false, "emit machine-readable result")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *intentPath == "" || *inventoryPath == "" || *outPath == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak harness model-set resolve: --intent, --inventory, and --out are required")
		return 2
	}
	if *expectationPath == "" {
		*expectationPath = defaultModelSetExpectationPath(*outPath)
	}
	if sameCleanPath(*outPath, *expectationPath) {
		fmt.Fprintln(stderr, "fak harness model-set resolve: --out and --expectation must name different files")
		return 2
	}

	intent, err := readHarnessModelSetIntent(*intentPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness model-set resolve: intent: %v\n", err)
		return 1
	}
	inventory, authoredAt, err := readHarnessModelInventory(*inventoryPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness model-set resolve: inventory: %v\n", err)
		return 1
	}
	evaluatedAt := authoredAt
	if *asOfText != "" {
		evaluatedAt, err = parseModelSetTime(*asOfText)
		if err != nil {
			fmt.Fprintf(stderr, "fak harness model-set resolve: --as-of: %v\n", err)
			return 2
		}
	}

	resolution, resolveErr := modelsetresolve.Resolve(intent, inventory, evaluatedAt)
	if resolveErr != nil {
		var unresolved *modelsetresolve.RequiredRolesError
		if errors.As(resolveErr, &unresolved) {
			result := harnessModelSetResolveResult{
				Schema: "fak.harness-model-set-resolve/1", Status: "incompatible",
				EvaluatedAt: resolution.EvaluatedAt, Roles: len(resolution.Roles), Resolution: &resolution,
			}
			if *jsonOutput {
				if err := writeHarnessModelSetJSON(stdout, result); err != nil {
					fmt.Fprintf(stderr, "fak harness model-set resolve: %v\n", err)
					return 1
				}
			} else {
				fmt.Fprintf(stderr, "fak harness model-set resolve: %v\n", resolveErr)
				writeModelSetRejections(stderr, resolution.Rejections())
			}
			return 3
		}
		fmt.Fprintf(stderr, "fak harness model-set resolve: %v\n", resolveErr)
		return 1
	}

	inputs := modelsetlock.Inputs{
		Intent: intent, Inventory: inventory, RuleBytes: []byte(harnessModelSetRuleBytes),
		Target: modelsetlock.Target{
			OS: *osName, Architecture: *architecture, Accelerator: *accelerator, Runtime: *runtimeName,
		},
		ResolverVersion: modelsetresolve.Schema,
	}
	lock, err := modelsetlock.New(inputs, resolution)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness model-set resolve: lock: %v\n", err)
		return 1
	}
	expectation, err := modelsetreceipt.Bind(intent, resolution, inventory)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness model-set resolve: startup expectation: %v\n", err)
		return 1
	}
	expectationRaw, err := expectation.CanonicalJSON()
	if err != nil {
		fmt.Fprintf(stderr, "fak harness model-set resolve: startup expectation: %v\n", err)
		return 1
	}
	// The expectation lands first; the canonical lock is the commit point. A
	// failed resolution never reaches either write, and a failed lock publish
	// leaves the prior lock intact for startup to keep rejecting stale state.
	if err := writeHarnessModelSetAtomic(*expectationPath, expectationRaw); err != nil {
		fmt.Fprintf(stderr, "fak harness model-set resolve: expectation output: %v\n", err)
		return 1
	}
	if err := modelsetlock.WriteFile(*outPath, lock); err != nil {
		fmt.Fprintf(stderr, "fak harness model-set resolve: lock output: %v\n", err)
		return 1
	}

	result := harnessModelSetResolveResult{
		Schema: "fak.harness-model-set-resolve/1", Status: "resolved", LockPath: *outPath,
		ExpectationPath: *expectationPath, ContentDigest: lock.ContentDigest,
		EvaluatedAt: lock.EvaluatedAt, Roles: len(lock.Roles),
	}
	if *jsonOutput {
		if err := writeHarnessModelSetJSON(stdout, result); err != nil {
			fmt.Fprintf(stderr, "fak harness model-set resolve: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "resolved %d model-set roles; lock %s (%s); startup expectation %s\n", len(lock.Roles), *outPath, lock.ContentDigest, *expectationPath)
	}
	return 0
}

func runHarnessModelSetSelfcheck(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness model-set selfcheck", flag.ContinueOnError)
	fs.SetOutput(stderr)
	lockPath := fs.String("lock", "", "canonical model-set lock")
	intentPath := fs.String("intent", "", "model-set intent (default: harness.model-set.json next to lock)")
	inventoryPath := fs.String("inventory", "", "current normalized model inventory")
	expectationPath := fs.String("expectation", "", "startup expectation sidecar (default: next to lock)")
	receiptPath := fs.String("receipt", "", "canonical compatibility receipt output")
	asOfText := fs.String("as-of", "", "explicit RFC3339 startup time (default: current UTC time)")
	jsonOutput := fs.Bool("json", false, "emit machine-readable result")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *lockPath == "" || *inventoryPath == "" || *receiptPath == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak harness model-set selfcheck: --lock, --inventory, and --receipt are required")
		return 2
	}
	if *intentPath == "" {
		*intentPath = filepath.Join(filepath.Dir(*lockPath), harnessModelSetIntentFile)
	}
	if *expectationPath == "" {
		*expectationPath = defaultModelSetExpectationPath(*lockPath)
	}
	if sameCleanPath(*receiptPath, *lockPath) || sameCleanPath(*receiptPath, *expectationPath) {
		fmt.Fprintln(stderr, "fak harness model-set selfcheck: --receipt must not overwrite the lock or expectation")
		return 2
	}

	lock, err := modelsetlock.ReadFile(*lockPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness model-set selfcheck: lock: %v\n", err)
		return 1
	}
	intent, err := readHarnessModelSetIntent(*intentPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness model-set selfcheck: intent: %v\n", err)
		return 1
	}
	inventory, _, err := readHarnessModelInventory(*inventoryPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness model-set selfcheck: inventory: %v\n", err)
		return 1
	}
	expectationRaw, err := os.ReadFile(*expectationPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness model-set selfcheck: expectation: %v\n", err)
		return 1
	}
	expectation, err := modelsetreceipt.ParseExpectationJSON(expectationRaw)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness model-set selfcheck: expectation: %v\n", err)
		return 1
	}
	if err := validateModelSetExpectationLock(expectation, lock); err != nil {
		fmt.Fprintf(stderr, "fak harness model-set selfcheck: expectation: %v\n", err)
		return 1
	}

	evaluatedAt := time.Now().UTC().Truncate(time.Second)
	if *asOfText != "" {
		evaluatedAt, err = parseModelSetTime(*asOfText)
		if err != nil {
			fmt.Fprintf(stderr, "fak harness model-set selfcheck: --as-of: %v\n", err)
			return 2
		}
	}
	resolution := resolutionFromModelSetLock(lock)
	receipt, evaluateErr := modelsetreceipt.Evaluate(expectation, intent, resolution, inventory, evaluatedAt)
	receiptRaw, err := receipt.CanonicalJSON()
	if err != nil {
		fmt.Fprintf(stderr, "fak harness model-set selfcheck: receipt: %v\n", err)
		return 1
	}
	if err := writeHarnessModelSetAtomic(*receiptPath, receiptRaw); err != nil {
		fmt.Fprintf(stderr, "fak harness model-set selfcheck: receipt output: %v\n", err)
		return 1
	}
	result := harnessModelSetSelfcheckResult{
		Schema: "fak.harness-model-set-selfcheck/1", Status: receipt.Status,
		LockDigest: lock.ContentDigest, ReceiptPath: *receiptPath,
		EvaluatedAt: receipt.EvaluatedAt, Roles: len(receipt.Roles),
		Failures: append([]modelsetreceipt.Failure(nil), receipt.Failures...),
	}
	if *jsonOutput {
		if err := writeHarnessModelSetJSON(stdout, result); err != nil {
			fmt.Fprintf(stderr, "fak harness model-set selfcheck: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "model-set selfcheck %s: %d roles, %d failures; receipt %s\n", receipt.Status, len(receipt.Roles), len(receipt.Failures), *receiptPath)
	}
	var incompatible *modelsetreceipt.IncompatibleError
	if errors.As(evaluateErr, &incompatible) {
		writeModelSetFailures(stderr, receipt.Failures)
		return 3
	}
	if evaluateErr != nil {
		fmt.Fprintf(stderr, "fak harness model-set selfcheck: %v\n", evaluateErr)
		return 1
	}
	return 0
}

func runHarnessModelSetInspect(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness model-set inspect", flag.ContinueOnError)
	fs.SetOutput(stderr)
	lockPath := fs.String("lock", "", "canonical model-set lock")
	receiptPath := fs.String("receipt", "", "optional compatibility receipt")
	jsonOutput := fs.Bool("json", false, "emit machine-readable inspection")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *lockPath == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak harness model-set inspect: --lock is required")
		return 2
	}
	lock, err := modelsetlock.ReadFile(*lockPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness model-set inspect: lock: %v\n", err)
		return 1
	}
	report := inspectHarnessModelSetLock(*lockPath, lock)
	if *receiptPath != "" {
		raw, readErr := os.ReadFile(*receiptPath)
		if readErr != nil {
			fmt.Fprintf(stderr, "fak harness model-set inspect: receipt: %v\n", readErr)
			return 1
		}
		receipt, parseErr := modelsetreceipt.ParseJSON(raw)
		if parseErr != nil {
			fmt.Fprintf(stderr, "fak harness model-set inspect: receipt: %v\n", parseErr)
			return 1
		}
		report.Receipt = &harnessModelSetReceiptSummary{
			Path: *receiptPath, Status: receipt.Status, EvaluatedAt: receipt.EvaluatedAt, Failures: len(receipt.Failures),
		}
	}
	if *jsonOutput {
		if err := writeHarnessModelSetJSON(stdout, report); err != nil {
			fmt.Fprintf(stderr, "fak harness model-set inspect: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "model-set lock %s\ndigest: %s\nresolver: %s\nevaluated: %s\ntarget: %s/%s accelerator=%s runtime=%s\nroles: %d\n",
		report.LockPath, report.ContentDigest, report.ResolverVersion, report.EvaluatedAt,
		report.Target.OS, report.Target.Architecture, report.Target.Accelerator, report.Target.Runtime, len(report.Roles))
	for _, role := range report.Roles {
		if role.CandidateID == "" {
			fmt.Fprintf(stdout, "- %s: %s (required=%t, rejections=%d)\n", role.ID, role.Status, role.Required, role.Rejections)
			continue
		}
		fmt.Fprintf(stdout, "- %s: %s via %s/%s (%s %s, rejections=%d)\n", role.ID, role.CandidateID, role.AlternativeID, role.Source, role.Artifact, role.ArtifactDigest, role.Rejections)
	}
	if report.Receipt != nil {
		fmt.Fprintf(stdout, "receipt: %s at %s (%d failures)\n", report.Receipt.Status, report.Receipt.EvaluatedAt, report.Receipt.Failures)
	}
	return 0
}

func readHarnessModelSetIntent(path string) (harnessmodelset.Intent, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return harnessmodelset.Intent{}, err
	}
	return harnessmodelset.ParseJSON(raw)
}

func readHarnessModelInventory(path string) (modelinventory.Inventory, time.Time, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return modelinventory.Inventory{}, time.Time{}, err
	}
	var header struct {
		AsOf string `json:"as_of"`
	}
	if err := json.Unmarshal(raw, &header); err != nil {
		return modelinventory.Inventory{}, time.Time{}, fmt.Errorf("invalid JSON: %w", err)
	}
	authoredAt, err := parseModelSetTime(header.AsOf)
	if err != nil {
		return modelinventory.Inventory{}, time.Time{}, fmt.Errorf("as_of: %w", err)
	}
	inventory, diagnostics := modelinventory.ParseJSON(raw, authoredAt)
	if len(diagnostics) != 0 {
		return modelinventory.Inventory{}, time.Time{}, diagnostics
	}
	return inventory, authoredAt, nil
}

func parseModelSetTime(raw string) (time.Time, error) {
	parsed, err := time.Parse(time.RFC3339, strings.TrimSpace(raw))
	if err != nil {
		return time.Time{}, fmt.Errorf("must be RFC3339: %w", err)
	}
	return parsed.UTC().Truncate(time.Second), nil
}

func defaultModelSetExpectationPath(lockPath string) string {
	if filepath.Base(lockPath) == modelsetlock.DefaultFileName {
		return filepath.Join(filepath.Dir(lockPath), harnessModelSetExpectationFile)
	}
	return lockPath + ".expectation.json"
}

func sameCleanPath(left, right string) bool {
	leftAbs, leftErr := filepath.Abs(filepath.Clean(left))
	rightAbs, rightErr := filepath.Abs(filepath.Clean(right))
	if leftErr != nil || rightErr != nil {
		return filepath.Clean(left) == filepath.Clean(right)
	}
	return strings.EqualFold(leftAbs, rightAbs)
}

func validateModelSetExpectationLock(expectation modelsetreceipt.Expectation, lock modelsetlock.Lock) error {
	if expectation.Digests.Requirements != lock.Inputs.Intent {
		return fmt.Errorf("requirements digest does not match lock input")
	}
	if expectation.Digests.Inventory != lock.Inputs.Inventory {
		return fmt.Errorf("inventory digest does not match lock input")
	}
	selected := make(map[string]modelsetlock.Selected)
	for _, role := range lock.Roles {
		if role.Selected != nil {
			selected[role.ID] = *role.Selected
		}
	}
	if len(expectation.Roles) != len(selected) {
		return fmt.Errorf("role binding count does not match lock selections")
	}
	for _, binding := range expectation.Roles {
		locked, ok := selected[binding.RoleID]
		if !ok || binding.AlternativeID != locked.AlternativeID || binding.CandidateID != locked.CandidateID {
			return fmt.Errorf("role %s selection does not match lock", binding.RoleID)
		}
	}
	return nil
}

func resolutionFromModelSetLock(lock modelsetlock.Lock) modelsetresolve.Resolution {
	resolution := modelsetresolve.Resolution{
		Schema: lock.ResolverVersion, EvaluatedAt: lock.EvaluatedAt,
		Roles: make([]modelsetresolve.RoleResolution, 0, len(lock.Roles)),
	}
	for _, role := range lock.Roles {
		outcome := modelsetresolve.RoleResolution{
			RoleID: role.ID, Required: role.Required, Status: role.Status,
			Rejections: append([]modelsetresolve.Rejection(nil), role.Rejections...),
		}
		if role.Selected != nil {
			outcome.Selection = &modelsetresolve.Selection{
				AlternativeID: role.Selected.AlternativeID, CandidateID: role.Selected.CandidateID,
			}
		}
		resolution.Roles = append(resolution.Roles, outcome)
	}
	return resolution
}

func inspectHarnessModelSetLock(path string, lock modelsetlock.Lock) harnessModelSetInspection {
	report := harnessModelSetInspection{
		Schema: "fak.harness-model-set-inspect/1", LockPath: path,
		ContentDigest: lock.ContentDigest, ResolverVersion: lock.ResolverVersion,
		EvaluatedAt: lock.EvaluatedAt, Target: lock.Target,
		Roles: make([]harnessModelSetInspectedRole, 0, len(lock.Roles)),
	}
	for _, role := range lock.Roles {
		item := harnessModelSetInspectedRole{
			ID: role.ID, Required: role.Required, Status: role.Status, Rejections: len(role.Rejections),
		}
		if role.Selected != nil {
			item.AlternativeID = role.Selected.AlternativeID
			item.CandidateID = role.Selected.CandidateID
			item.Source = role.Selected.Identity.Source
			item.Artifact = role.Selected.Identity.Artifact
			item.ArtifactDigest = role.Selected.Identity.Digest
		}
		report.Roles = append(report.Roles, item)
	}
	return report
}

func writeHarnessModelSetAtomic(path string, raw []byte) error {
	if strings.TrimSpace(path) == "" {
		return errors.New("output path is empty")
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	keep := true
	defer func() {
		_ = tmp.Close()
		if keep {
			_ = os.Remove(tmpPath)
		}
	}()
	if err := tmp.Chmod(0o644); err != nil {
		return err
	}
	if _, err := tmp.Write(raw); err != nil {
		return err
	}
	if err := tmp.Sync(); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return err
	}
	keep = false
	return nil
}

func writeHarnessModelSetJSON(w io.Writer, value any) error {
	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "  ")
	return encoder.Encode(value)
}

func writeModelSetRejections(w io.Writer, rejections []modelsetresolve.Rejection) {
	for _, rejection := range rejections {
		fmt.Fprintf(w, "%s role=%s candidate=%s constraint=%s expected=%s actual=%s evidence=%s remediation=%s\n",
			rejection.Code, rejection.RoleID, rejection.CandidateID, rejection.Constraint,
			rejection.Expected, rejection.Actual, rejection.EvidenceSource, rejection.Remediation)
	}
}

func writeModelSetFailures(w io.Writer, failures []modelsetreceipt.Failure) {
	for _, failure := range failures {
		fmt.Fprintf(w, "%s source=%s role=%s candidate=%s field=%s expected=%s actual=%s evidence=%s remediation=%s\n",
			failure.Code, failure.SourceCode, failure.RoleID, failure.CandidateID, failure.Field,
			failure.Expected, failure.Actual, failure.EvidenceSource, failure.Remediation)
	}
}
