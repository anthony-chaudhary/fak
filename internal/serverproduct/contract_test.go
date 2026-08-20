package serverproduct_test

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/serverproduct"
)

type goldenPair struct {
	name        string
	spec        serverproduct.ServerSpec
	receipt     serverproduct.ServerReceipt
	ready       serverproduct.ReadyReceipt
	specJSON    []byte
	receiptJSON []byte
}

func TestSchemaV1GoldenFixturesRoundTripByteStably(t *testing.T) {
	pairs := readGoldenPairs(t)
	if len(pairs) < 4 {
		t.Fatalf("valid_round_trips=%d, want >=4", len(pairs))
	}
	secretValues := 0
	for _, pair := range pairs {
		pair := pair
		t.Run(pair.name, func(t *testing.T) {
			specJSON, err := serverproduct.EncodeSpec(pair.spec)
			if err != nil {
				t.Fatalf("EncodeSpec: %v", err)
			}
			if !bytes.Equal(specJSON, pair.specJSON) {
				t.Fatalf("spec JSON is not canonical\n got: %s\nwant: %s", specJSON, pair.specJSON)
			}
			receiptJSON, err := serverproduct.EncodeReadyReceipt(pair.ready)
			if err != nil {
				t.Fatalf("EncodeReadyReceipt: %v", err)
			}
			if !bytes.Equal(receiptJSON, pair.receiptJSON) {
				t.Fatalf("receipt JSON is not canonical\n got: %s\nwant: %s", receiptJSON, pair.receiptJSON)
			}
			secretValues += countSecretValues(t, receiptJSON)
		})
	}
	if secretValues != 0 {
		t.Fatalf("secret_values_in_ready_receipt=%d, want 0", secretValues)
	}
}

func TestSchemaV1InvalidContractFixturesFailClosed(t *testing.T) {
	base := readGoldenPairs(t)[0]

	type invalidFixture struct {
		name string
		want string
		run  func() error
	}
	receiptFixture := func(mutate func(*serverproduct.ServerReceipt)) func() error {
		return func() error {
			receipt := base.receipt
			receipt.Protocol.Capabilities = append([]string(nil), receipt.Protocol.Capabilities...)
			mutate(&receipt)
			raw, err := json.Marshal(receipt)
			if err != nil {
				return err
			}
			_, err = serverproduct.DecodeReadyReceipt(base.spec, raw)
			return err
		}
	}
	specFixture := func(mutate func(*serverproduct.ServerSpec)) func() error {
		return func() error {
			spec := base.spec
			spec.Protocol.RequiredCapabilities = append([]string(nil), spec.Protocol.RequiredCapabilities...)
			mutate(&spec)
			raw, err := json.Marshal(spec)
			if err != nil {
				return err
			}
			_, err = serverproduct.DecodeSpec(raw)
			return err
		}
	}

	fixtures := []invalidFixture{
		{
			name: "bearer token field",
			want: "unknown field \"bearer_token\"",
			run: func() error {
				var raw map[string]any
				if err := json.Unmarshal(base.receiptJSON, &raw); err != nil {
					return err
				}
				auth := raw["auth"].(map[string]any)
				auth["bearer_token"] = "Bearer should-never-be-a-receipt-value"
				mutated, err := json.Marshal(raw)
				if err != nil {
					return err
				}
				_, err = serverproduct.DecodeReadyReceipt(base.spec, mutated)
				return err
			},
		},
		{name: "wildcard bind", want: "literal loopback IP", run: specFixture(func(spec *serverproduct.ServerSpec) {
			spec.Endpoint.BindHost = "0.0.0.0"
		})},
		{name: "missing probe evidence", want: "readiness probe is required", run: receiptFixture(func(receipt *serverproduct.ServerReceipt) {
			receipt.Readiness.Probe = ""
		})},
		{name: "mismatched spec digest", want: "does not match the authored spec", run: receiptFixture(func(receipt *serverproduct.ServerReceipt) {
			receipt.SpecDigest = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
		})},
		{name: "uppercase artifact digest", want: "lowercase sha256", run: specFixture(func(spec *serverproduct.ServerSpec) {
			spec.Artifact.Digest = "sha256:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
		})},
		{name: "duplicate required capability", want: "duplicate capability", run: specFixture(func(spec *serverproduct.ServerSpec) {
			spec.Protocol.RequiredCapabilities = append(spec.Protocol.RequiredCapabilities, spec.Protocol.RequiredCapabilities[0])
		})},
		{name: "duplicate probed capability", want: "duplicate capability", run: receiptFixture(func(receipt *serverproduct.ServerReceipt) {
			receipt.Protocol.Capabilities = append(receipt.Protocol.Capabilities, receipt.Protocol.Capabilities[0])
		})},
		{name: "authored observation", want: "kind must be \"observed\"", run: receiptFixture(func(receipt *serverproduct.ServerReceipt) {
			receipt.Provenance.Artifact.Kind = serverproduct.ProvenanceAuthored
		})},
		{name: "zero generation", want: "generation must be positive", run: receiptFixture(func(receipt *serverproduct.ServerReceipt) {
			receipt.Generation = 0
		})},
		{name: "remote receipt endpoint", want: "literal loopback IP", run: receiptFixture(func(receipt *serverproduct.ServerReceipt) {
			receipt.Endpoint.BaseURL = "http://0.0.0.0:8080"
		})},
		{name: "different loopback endpoint", want: "endpoint host does not match", run: receiptFixture(func(receipt *serverproduct.ServerReceipt) {
			receipt.Endpoint.BaseURL = "http://127.0.0.2:8125"
		})},
		{name: "missing required capability", want: "missing required capability", run: receiptFixture(func(receipt *serverproduct.ServerReceipt) {
			receipt.Protocol.Capabilities = []string{"models.list"}
		})},
		{name: "mismatched authored provenance", want: "authored provenance does not match", run: receiptFixture(func(receipt *serverproduct.ServerReceipt) {
			receipt.Provenance.Spec.Source = "server-spec:/different/spec.json"
		})},
		{name: "secret-shaped credential value", want: "uppercase symbolic name", run: receiptFixture(func(receipt *serverproduct.ServerReceipt) {
			receipt.Auth = serverproduct.AuthReference{Mode: serverproduct.AuthCredentialRef, CredentialRef: "sk-live-credential-value"}
		})},
		{name: "missing process start identity", want: "process_start_identity is required", run: receiptFixture(func(receipt *serverproduct.ServerReceipt) {
			receipt.Ownership.ProcessStartIdentity = ""
		})},
	}

	if len(fixtures) < 8 {
		t.Fatalf("invalid_contract_classes=%d, want >=8", len(fixtures))
	}
	for _, fixture := range fixtures {
		fixture := fixture
		t.Run(fixture.name, func(t *testing.T) {
			err := fixture.run()
			if err == nil {
				t.Fatal("invalid fixture was accepted")
			}
			if !strings.Contains(err.Error(), fixture.want) {
				t.Fatalf("error %q does not identify class %q", err, fixture.want)
			}
		})
	}
}

func TestWriteReadyReceiptIsAtomicAndReadyOnly(t *testing.T) {
	base := readGoldenPairs(t)[0]
	path := filepath.Join(t.TempDir(), "server-receipt.json")

	if err := serverproduct.WriteReadyReceipt(path, base.ready); err != nil {
		t.Fatalf("first WriteReadyReceipt: %v", err)
	}
	want, err := serverproduct.EncodeReadyReceipt(base.ready)
	if err != nil {
		t.Fatal(err)
	}
	assertFileBytes(t, path, want)

	next := base.ready.Receipt()
	next.Generation++
	next.CreatedAt = "2026-08-19T12:30:02Z"
	readyNext, err := serverproduct.NewReadyReceipt(base.spec, next)
	if err != nil {
		t.Fatalf("NewReadyReceipt next generation: %v", err)
	}
	if err := serverproduct.WriteReadyReceipt(path, readyNext); err != nil {
		t.Fatalf("replace WriteReadyReceipt: %v", err)
	}
	wantNext, err := serverproduct.EncodeReadyReceipt(readyNext)
	if err != nil {
		t.Fatal(err)
	}
	assertFileBytes(t, path, wantNext)

	if err := serverproduct.WriteReadyReceipt(path, serverproduct.ReadyReceipt{}); err == nil {
		t.Fatal("zero ReadyReceipt was written")
	}
	assertFileBytes(t, path, wantNext)
	if temps, err := filepath.Glob(filepath.Join(filepath.Dir(path), ".server-receipt-*.tmp")); err != nil || len(temps) != 0 {
		t.Fatalf("temporary files after atomic write: %v, err=%v", temps, err)
	}
	if runtime.GOOS != "windows" {
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		if got := info.Mode().Perm(); got != 0o600 {
			t.Fatalf("receipt permissions=%#o, want 0600", got)
		}
	}
}

func TestReadyReceiptDoesNotExposeMutableState(t *testing.T) {
	base := readGoldenPairs(t)[0]
	copy := base.ready.Receipt()
	copy.Protocol.Capabilities[0] = "tampered"
	got, err := serverproduct.EncodeReadyReceipt(base.ready)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(got, []byte("tampered")) {
		t.Fatal("Receipt returned mutable ReadyReceipt slice state")
	}
}

func readGoldenPairs(t *testing.T) []goldenPair {
	t.Helper()
	specPaths, err := filepath.Glob(filepath.Join("testdata", "valid", "*.spec.json"))
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(specPaths)
	pairs := make([]goldenPair, 0, len(specPaths))
	for _, specPath := range specPaths {
		name := strings.TrimSuffix(filepath.Base(specPath), ".spec.json")
		receiptPath := filepath.Join(filepath.Dir(specPath), name+".receipt.json")
		specJSON, err := os.ReadFile(specPath)
		if err != nil {
			t.Fatalf("read %s: %v", specPath, err)
		}
		receiptJSON, err := os.ReadFile(receiptPath)
		if err != nil {
			t.Fatalf("read %s: %v", receiptPath, err)
		}
		spec, err := serverproduct.DecodeSpec(specJSON)
		if err != nil {
			t.Fatalf("DecodeSpec %s: %v", name, err)
		}
		ready, err := serverproduct.DecodeReadyReceipt(spec, receiptJSON)
		if err != nil {
			t.Fatalf("DecodeReadyReceipt %s: %v", name, err)
		}
		pairs = append(pairs, goldenPair{
			name:        name,
			spec:        spec,
			receipt:     ready.Receipt(),
			ready:       ready,
			specJSON:    specJSON,
			receiptJSON: receiptJSON,
		})
	}
	return pairs
}

func countSecretValues(t *testing.T, data []byte) int {
	t.Helper()
	var receipt map[string]any
	if err := json.Unmarshal(data, &receipt); err != nil {
		t.Fatal(err)
	}
	count := 0
	var visit func(any)
	visit = func(value any) {
		switch value := value.(type) {
		case map[string]any:
			for key, nested := range value {
				switch strings.ToLower(key) {
				case "token", "bearer_token", "api_key", "secret", "password", "credential_value":
					count++
				}
				visit(nested)
			}
		case []any:
			for _, nested := range value {
				visit(nested)
			}
		case string:
			lower := strings.ToLower(value)
			if strings.HasPrefix(lower, "bearer ") || strings.HasPrefix(lower, "sk-") || strings.Contains(lower, "-----begin ") {
				count++
			}
		}
	}
	visit(receipt)
	return count
}

func assertFileBytes(t *testing.T, path string, want []byte) {
	t.Helper()
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("file bytes differ\n got: %s\nwant: %s", got, want)
	}
}
