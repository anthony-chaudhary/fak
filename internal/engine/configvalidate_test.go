package engine

import (
	"reflect"
	"testing"
)

// TestValidateVLLMConfigThreeVerdicts is the issue #3399 witness: one table
// drives the config validator through all three verdicts — hard-refuse illegal,
// auto-correct+warn safe drift, and accept — and pins the corrected config plus
// the named reasons on every row.
func TestValidateVLLMConfigThreeVerdicts(t *testing.T) {
	cases := []struct {
		name         string
		cfg          VLLMConfig
		wantVerdict  ConfigVerdict
		wantRefusals []ConfigRefusal
		wantWarnings []ConfigWarning
		wantBaseURL  string // corrected value, asserted only when admitted
		wantWorkerID string // corrected value, asserted only when admitted
	}{
		// --- accept: nothing illegal, nothing to correct -------------------
		{
			name:         "accept well-formed config",
			cfg:          VLLMConfig{BaseURL: "http://127.0.0.1:8000/v1", Model: "m", WorkerID: "w0"},
			wantVerdict:  ConfigVerdictAccept,
			wantBaseURL:  "http://127.0.0.1:8000/v1",
			wantWorkerID: "w0",
		},
		{
			name:         "accept https base URL",
			cfg:          VLLMConfig{BaseURL: "https://host:8443/v1", WorkerID: "w1"},
			wantVerdict:  ConfigVerdictAccept,
			wantBaseURL:  "https://host:8443/v1",
			wantWorkerID: "w1",
		},

		// --- auto-correct+warn: safe drift with one unambiguous fix --------
		{
			name:         "auto-correct trailing slash",
			cfg:          VLLMConfig{BaseURL: "http://127.0.0.1:8000/v1/", WorkerID: "w2"},
			wantVerdict:  ConfigVerdictAutoCorrect,
			wantWarnings: []ConfigWarning{ConfigWarningTrailingSlashTrimmed},
			wantBaseURL:  "http://127.0.0.1:8000/v1",
			wantWorkerID: "w2",
		},
		{
			name:         "auto-correct surrounding whitespace",
			cfg:          VLLMConfig{BaseURL: "  http://127.0.0.1:8000/v1 ", Model: " m ", WorkerID: "w3"},
			wantVerdict:  ConfigVerdictAutoCorrect,
			wantWarnings: []ConfigWarning{ConfigWarningWhitespaceTrimmed},
			wantBaseURL:  "http://127.0.0.1:8000/v1",
			wantWorkerID: "w3",
		},
		{
			name:         "auto-correct empty worker id to adapter default",
			cfg:          VLLMConfig{BaseURL: "http://127.0.0.1:8000/v1"},
			wantVerdict:  ConfigVerdictAutoCorrect,
			wantWarnings: []ConfigWarning{ConfigWarningWorkerIDDefaulted},
			wantBaseURL:  "http://127.0.0.1:8000/v1",
			wantWorkerID: "vllm",
		},
		{
			name:        "auto-correct accumulates every applied fix",
			cfg:         VLLMConfig{BaseURL: " http://127.0.0.1:8000/v1/ "},
			wantVerdict: ConfigVerdictAutoCorrect,
			wantWarnings: []ConfigWarning{
				ConfigWarningWhitespaceTrimmed,
				ConfigWarningTrailingSlashTrimmed,
				ConfigWarningWorkerIDDefaulted,
			},
			wantBaseURL:  "http://127.0.0.1:8000/v1",
			wantWorkerID: "vllm",
		},

		// --- refuse: illegal, no safe correction exists --------------------
		{
			name:         "refuse missing base URL",
			cfg:          VLLMConfig{WorkerID: "w4"},
			wantVerdict:  ConfigVerdictRefuse,
			wantRefusals: []ConfigRefusal{ConfigRefusalBaseURLMissing},
		},
		{
			name:         "refuse whitespace-only base URL",
			cfg:          VLLMConfig{BaseURL: "   ", WorkerID: "w5"},
			wantVerdict:  ConfigVerdictRefuse,
			wantRefusals: []ConfigRefusal{ConfigRefusalBaseURLMissing},
		},
		{
			name:         "refuse unparseable base URL",
			cfg:          VLLMConfig{BaseURL: "://bad", WorkerID: "w6"},
			wantVerdict:  ConfigVerdictRefuse,
			wantRefusals: []ConfigRefusal{ConfigRefusalBaseURLMalformed},
		},
		{
			name:         "refuse base URL without scheme or host",
			cfg:          VLLMConfig{BaseURL: "not-a-url", WorkerID: "w7"},
			wantVerdict:  ConfigVerdictRefuse,
			wantRefusals: []ConfigRefusal{ConfigRefusalBaseURLMalformed},
		},
		{
			name:         "refuse base URL with empty host",
			cfg:          VLLMConfig{BaseURL: "http://", WorkerID: "w8"},
			wantVerdict:  ConfigVerdictRefuse,
			wantRefusals: []ConfigRefusal{ConfigRefusalBaseURLMalformed},
		},
		{
			name:         "refuse non-HTTP scheme",
			cfg:          VLLMConfig{BaseURL: "ftp://host:21/v1", WorkerID: "w9"},
			wantVerdict:  ConfigVerdictRefuse,
			wantRefusals: []ConfigRefusal{ConfigRefusalBaseURLSchemeNotHTTP},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ValidateVLLMConfig(tc.cfg)
			if got.Verdict != tc.wantVerdict {
				t.Fatalf("verdict = %q, want %q (refusals=%v warnings=%v)",
					got.Verdict, tc.wantVerdict, got.Refusals, got.Warnings)
			}
			if !reflect.DeepEqual(got.Refusals, tc.wantRefusals) {
				t.Fatalf("refusals = %v, want %v", got.Refusals, tc.wantRefusals)
			}
			if !reflect.DeepEqual(got.Warnings, tc.wantWarnings) {
				t.Fatalf("warnings = %v, want %v", got.Warnings, tc.wantWarnings)
			}
			if tc.wantVerdict == ConfigVerdictRefuse {
				// Fail closed: a refused validation yields no usable config.
				if !reflect.DeepEqual(got.Config, VLLMConfig{}) {
					t.Fatalf("refused validation returned a non-zero config: %+v", got.Config)
				}
				return
			}
			if got.Config.BaseURL != tc.wantBaseURL {
				t.Fatalf("corrected BaseURL = %q, want %q", got.Config.BaseURL, tc.wantBaseURL)
			}
			if got.Config.WorkerID != tc.wantWorkerID {
				t.Fatalf("corrected WorkerID = %q, want %q", got.Config.WorkerID, tc.wantWorkerID)
			}
		})
	}
}

// TestConfigVerdictVocabularyIsClosedAndFailClosed pins the verdict enum: exactly
// three members, every member Valid, the zero value invalid and NOT admitted (an
// unset verdict never reads as an accept), and only auto-correct/accept admit.
func TestConfigVerdictVocabularyIsClosedAndFailClosed(t *testing.T) {
	want := []ConfigVerdict{ConfigVerdictRefuse, ConfigVerdictAutoCorrect, ConfigVerdictAccept}
	if got := ConfigVerdicts(); !reflect.DeepEqual(got, want) {
		t.Fatalf("ConfigVerdicts() = %v, want %v", got, want)
	}
	for _, v := range want {
		if !v.Valid() {
			t.Fatalf("verdict %q not Valid", v)
		}
	}
	var zero ConfigVerdict
	if zero.Valid() {
		t.Fatal("zero verdict must not be Valid")
	}
	if zero.Admitted() {
		t.Fatal("zero verdict must fail closed (not Admitted)")
	}
	if ConfigVerdictRefuse.Admitted() {
		t.Fatal("refuse must not be Admitted")
	}
	if !ConfigVerdictAutoCorrect.Admitted() || !ConfigVerdictAccept.Admitted() {
		t.Fatal("auto-correct and accept must both be Admitted")
	}
	if ConfigVerdict("out-of-vocabulary").Admitted() {
		t.Fatal("an out-of-vocabulary verdict must fail closed (not Admitted)")
	}
}
