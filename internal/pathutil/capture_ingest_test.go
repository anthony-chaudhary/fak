package pathutil

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestCaptureRefusalReasonUsesClosedABIVocabulary(t *testing.T) {
	code, ok := abi.ReasonByName(CaptureRefusalReason)
	if !ok || code != abi.ReasonSecretExfil {
		t.Fatalf("capture refusal reason %q = %v/%v, want existing ReasonSecretExfil", CaptureRefusalReason, code, ok)
	}
}

func TestConservativeSourceTable(t *testing.T) {
	tests := []struct {
		name  string
		value string
		class CaptureSourceClass
	}{
		{"claude credential home", `C:\\Users\\u\\.claude\\projects\\session.jsonl`, CaptureClassCredential},
		{"dotenv variant", `/srv/app/.env.production`, CaptureClassCredential},
		{"ssh private key", `cat ~/.ssh/id_ed25519`, CaptureClassCredential},
		{"private companion", `../fak-private/tools/bridge`, CaptureClassLabAccess},
		{"lab bridge", `C:\\lab\\tools\\dgxbridge\\README.md`, CaptureClassLabAccess},
		{"public lab access stub", `docs/private-comms-channel.md`, CaptureClassLabAccess},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := CheckCaptureSource(tc.value)
			if !got.Refused || got.Reason != CaptureRefusalReason || got.Class != tc.class || got.SourceDigest == "" {
				t.Fatalf("Decide(%q) = %+v", tc.value, got)
			}
		})
	}
}

func TestSourceTableDoesNotBecomeContentSubstringFilter(t *testing.T) {
	for _, allowed := range []string{
		`C:\\work\\fak`,
		`internal/secretgate/obfuscate.go`,
		`python -c "print(os.environ.get('CI'))"`,
		`docs/config.env.example`,
		`USD`,
	} {
		if got := CheckCaptureSource(allowed); got.Refused {
			t.Errorf("Decide(%q) = %+v, want allow", allowed, got)
		}
	}
}

func TestDecideJSONReadsOnlySourceSelectors(t *testing.T) {
	denied := CheckCaptureJSON([]byte(`{"path":"../fak-private/private.txt","payload":"not-a-source"}`), nil)
	if !denied.Refused || denied.Class != CaptureClassLabAccess {
		t.Fatalf("path decision = %+v, want lab refusal", denied)
	}
	allowed := CheckCaptureJSON([]byte(`{"source":"USD","data":"../fak-private/not-a-selector"}`), map[string]string{"query": "read ../fak-private"})
	if allowed.Refused {
		t.Fatalf("content/query text was mistaken for a source selector: %+v", allowed)
	}
}
