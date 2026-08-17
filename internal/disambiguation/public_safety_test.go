package disambiguation

import (
	"errors"
	"testing"
)

func TestPublicSafetyRejectsSensitiveCorpus(t *testing.T) {
	tests := []struct {
		name, text, code string
	}{
		{"windows local path", `operator file C:\\Users\\alice\\secrets.txt`, ErrPublicSafetyLocalPath},
		{"unix local path", "operator file /home/alice/secrets.txt", ErrPublicSafetyLocalPath},
		{"credential", "Authorization: Bearer abcdefghijklmnop", ErrPublicSafetyCredential},
		{"private repository", "copied from github.com/example/fak-private", ErrPublicSafetyPrivateRepo},
		{"private hostname", "run this on gpu-01.internal", ErrPublicSafetyPrivateHost},
		{"private address", "connect to 10.20.30.40", ErrPublicSafetyPrivateHost},
		{"control text", "copied terminal\x1b[31moutput", ErrPublicSafetyControlText},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			entry := fixtureEntry()
			entry.Definition = tt.text
			err := entry.Validate()
			if ValidationCode(err) != tt.code {
				t.Fatalf("code=%q want %q (%v)", ValidationCode(err), tt.code, err)
			}
			var validation *ValidationError
			if !errors.As(err, &validation) || validation.Field != "entry.definition" {
				t.Fatalf("error=%#v want entry.definition", err)
			}
		})
	}
}

func TestPublicSafetyAcceptsPublicRepositoryContent(t *testing.T) {
	entry := fixtureEntry()
	entry.Definition = "Public generator documented in docs/repro-packet.md and owned by the disambiguation lane."
	entry.Sources[0].Locator = "internal/disambiguation/public_safety.go#validatePublicSafety"
	if err := entry.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestPublicSafetyCoversNestedAndFutureStringFields(t *testing.T) {
	entry := fixtureEntry()
	entry.Contrasts[0].Explanation = "password=do-not-publish"
	if code := ValidationCode(entry.Validate()); code != ErrPublicSafetyCredential {
		t.Fatalf("nested field code=%q want %q", code, ErrPublicSafetyCredential)
	}
}
