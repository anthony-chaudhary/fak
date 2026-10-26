package remotemaint

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"
)

// fak-test:runtime medium est=4s
func TestCommandAuthorityPhasesAndConfigMutation(t *testing.T) {
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	rawExecutable, err := os.ReadFile(executable)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(rawExecutable)
	now := time.Now().UTC()
	grant := validTestGrant(now)
	config := Config{Schema: ConfigSchema, Profiles: []Profile{{
		ProfileID: grant.ProfileID, Target: grant.Target, HostKey: grant.HostKey,
		Operation: grant.Operation, PayloadSHA256: grant.PayloadSHA256, Holder: grant.Holder,
		Paths: grant.Paths, PathPrefixes: grant.PathPrefixes, Units: grant.Units,
		LeaseNodes: []string{"node-1"}, ExpiresAt: now.Add(time.Hour),
		Authority: Command{Executable: executable, SHA256: hex.EncodeToString(sum[:]), Args: []string{"-test.run=^TestAdapterProcess$", "--", "remote-adapter"}},
	}}}
	configPath := filepath.Join(t.TempDir(), "remote-maintenance.json")
	writeTestConfig(t, configPath, config)
	t.Setenv("TMP", "remotemaint-selected-host-value")
	t.Setenv("PROVIDER_API_KEY", "must-not-reach-adapter")
	authority, err := LoadAuthority(configPath)
	if err != nil {
		t.Fatalf("LoadAuthority() error = %v", err)
	}
	prepared, err := Prepare(context.Background(), authority, Request{ProfileID: grant.ProfileID, OperationDigest: grant.OperationDigest})
	if err != nil {
		t.Fatalf("Prepare() error = %v", err)
	}
	config.Profiles[0].Target = "changed.example.invalid"
	writeTestConfig(t, configPath, config)
	if _, err := prepared.Execute(context.Background()); !errors.Is(err, ErrChanged) && !errors.Is(err, ErrScope) {
		t.Fatalf("Execute() error = %v, want ErrChanged or ErrScope", err)
	}

	config.Profiles[0].Target = grant.Target
	config.Profiles[0].Authority.Args = []string{"-test.run=^TestAdapterProcess$", "--", "remote-adapter", "remote-adapter-failure"}
	writeTestConfig(t, configPath, config)
	authority, err = LoadAuthority(configPath)
	if err != nil {
		t.Fatalf("LoadAuthority(failure) error = %v", err)
	}
	prepared, err = Prepare(context.Background(), authority, Request{ProfileID: grant.ProfileID, OperationDigest: grant.OperationDigest})
	if err != nil {
		t.Fatalf("Prepare(failure) error = %v", err)
	}
	receipt, err := prepared.Execute(context.Background())
	if !errors.Is(err, ErrUnknownEffect) || receipt.Outcome != OutcomeUnknownEffect || receipt.BackupDisposition != BackupRetained {
		t.Fatalf("Execute(failure) = (%+v, %v), want bound unknown effect", receipt, err)
	}

	config.Profiles[0].Authority.Args[3] = "remote-adapter-nonzero-failure"
	writeTestConfig(t, configPath, config)
	authority, err = LoadAuthority(configPath)
	if err != nil {
		t.Fatalf("LoadAuthority(nonzero failure) error = %v", err)
	}
	prepared, err = Prepare(context.Background(), authority, Request{ProfileID: grant.ProfileID, OperationDigest: grant.OperationDigest})
	if err != nil {
		t.Fatalf("Prepare(nonzero failure) error = %v", err)
	}
	receipt, err = prepared.Execute(context.Background())
	if !errors.Is(err, ErrRollbackFailed) || receipt.Outcome != OutcomeRollbackFailed || receipt.BackupDisposition != BackupRestored {
		t.Fatalf("Execute(nonzero failure) = (%+v, %v), want bound rollback failure", receipt, err)
	}

	config.Profiles[0].Authority.Args[3] = "remote-adapter-wrong-failure"
	writeTestConfig(t, configPath, config)
	authority, err = LoadAuthority(configPath)
	if err != nil {
		t.Fatalf("LoadAuthority(wrong failure) error = %v", err)
	}
	prepared, err = Prepare(context.Background(), authority, Request{ProfileID: grant.ProfileID, OperationDigest: grant.OperationDigest})
	if err != nil {
		t.Fatalf("Prepare(wrong failure) error = %v", err)
	}
	receipt, err = prepared.Execute(context.Background())
	if !errors.Is(err, ErrUnknownEffect) || receipt.Outcome != OutcomeUnknownEffect || receipt.BootID != grant.BootID || receipt.BackupDisposition != BackupUnknown {
		t.Fatalf("Execute(wrong failure) = (%+v, %v), want synthesized bound unknown effect", receipt, err)
	}

	config.Profiles[0].Authority.Args[3] = "remote-adapter-malformed"
	writeTestConfig(t, configPath, config)
	authority, err = LoadAuthority(configPath)
	if err != nil {
		t.Fatalf("LoadAuthority(malformed) error = %v", err)
	}
	prepared, err = Prepare(context.Background(), authority, Request{ProfileID: grant.ProfileID, OperationDigest: grant.OperationDigest})
	if err != nil {
		t.Fatalf("Prepare(malformed) error = %v", err)
	}
	preparedGrant := prepared.Grant()
	receipt, err = prepared.Execute(context.Background())
	if !errors.Is(err, ErrUnknownEffect) || receipt.Outcome != OutcomeUnknownEffect || receipt.GrantDigest != GrantDigest(preparedGrant) || receipt.BackupDisposition != BackupUnknown {
		t.Fatalf("Execute(malformed) = (%+v, %v), want synthesized bound unknown effect", receipt, err)
	}
}

// fak-test:runtime fast est=0.1s
func TestProtectedPathsIncludesMissingContext(t *testing.T) {
	want, _ := filepath.Abs(filepath.Join(t.TempDir(), "missing.json"))
	got, err := ProtectedPaths(want)
	if err != nil {
		t.Fatalf("ProtectedPaths() error = %v", err)
	}
	if !slices.Equal(got, []string{want}) {
		t.Fatalf("ProtectedPaths() = %v, want [%s]", got, want)
	}
}

// fak-test:runtime fast est=0.1s
func TestParseConfigRejectsInterpreterAndExtraField(t *testing.T) {
	if _, err := ParseConfig([]byte(`{"schema":"fak.remote-maintenance.config.v1","profiles":[],"environment":{}}`)); !errors.Is(err, ErrAuthority) {
		t.Fatalf("ParseConfig(extra field) error = %v, want ErrAuthority", err)
	}
	if err := validateCommand(Command{Executable: filepath.Join(string(filepath.Separator), "bin", "sh"), SHA256: repeated("a")}); !errors.Is(err, ErrAuthority) {
		t.Fatalf("validateCommand(shell) error = %v, want ErrAuthority", err)
	}
}

func TestAdapterProcess(t *testing.T) {
	if !slices.Contains(os.Args, "remote-adapter") {
		return
	}
	if os.Getenv("TMP") != "remotemaint-selected-host-value" || os.Getenv("PROVIDER_API_KEY") != "" {
		os.Exit(3)
	}
	raw, err := io.ReadAll(os.Stdin)
	if err != nil {
		os.Exit(2)
	}
	var request adapterRequest
	if err := json.Unmarshal(raw, &request); err != nil {
		os.Exit(2)
	}
	response := adapterResponse{Schema: ResponseEnvelopeSchema}
	nonzero := false
	switch request.Phase {
	case "resolve":
		grant := validTestGrant(time.Now().UTC())
		grant.ProfileID = request.ProfileID
		grant.OperationDigest = request.OperationDigest
		response.Grant = &grant
	case "validate":
		response.Grant = request.Grant
	case "execute":
		if slices.Contains(os.Args, "remote-adapter-malformed") {
			_, _ = io.WriteString(os.Stdout, `{"schema":`)
			os.Exit(7)
		}
		now := time.Now().UTC()
		receipt := receiptFor(*request.Grant, now)
		if slices.Contains(os.Args, "remote-adapter-failure") || slices.Contains(os.Args, "remote-adapter-nonzero-failure") || slices.Contains(os.Args, "remote-adapter-wrong-failure") {
			receipt.Outcome = OutcomeUnknownEffect
			receipt.Verified = false
			receipt.BackupDisposition = BackupRetained
			if slices.Contains(os.Args, "remote-adapter-nonzero-failure") {
				receipt.Outcome = OutcomeRollbackFailed
				receipt.BackupDisposition = BackupRestored
				nonzero = true
			}
			if slices.Contains(os.Args, "remote-adapter-wrong-failure") {
				receipt.BootID = "wrong-boot"
				nonzero = true
			}
			response.Error = &adapterError{Code: "effect_unknown", Message: "redacted"}
		}
		response.Receipt = &receipt
	default:
		response.Error = &adapterError{Code: "phase", Message: "denied"}
	}
	if err := json.NewEncoder(os.Stdout).Encode(response); err != nil {
		os.Exit(2)
	}
	if nonzero {
		os.Exit(7)
	}
	os.Exit(0)
}

func writeTestConfig(t *testing.T, path string, config Config) {
	t.Helper()
	raw, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
}
