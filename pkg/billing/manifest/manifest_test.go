package manifest

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// sampleManifestJSON is a complete v1 operator file with obviously fake
// values and generic names: one model bound to one local upstream, two caps
// set, two left uncapped.
const sampleManifestJSON = `{
  "version": "fak-billing-manifest/v1",
  "state_dir": "./state",
  "prices_path": "./prices.json",
  "models": [
    {"name": "model-a", "upstream": "local", "upstream_model": "model-a"}
  ],
  "upstreams": [
    {"name": "local", "base_url": "http://127.0.0.1:8000/v1"}
  ],
  "limits": {
    "requests_per_minute": 60,
    "tokens_per_minute": 100000,
    "daily_token_cap": 0,
    "session_token_cap": 0
  }
}`

func TestLoadRoundTrip(t *testing.T) {
	first, err := Load([]byte(sampleManifestJSON))
	if err != nil {
		t.Fatalf("first Load() = %v", err)
	}
	out, err := first.MarshalJSON()
	if err != nil {
		t.Fatalf("MarshalJSON() = %v", err)
	}
	second, err := Load(out)
	if err != nil {
		t.Fatalf("second Load() = %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("round trip not identical:\nfirst=%#v\nsecond=%#v", first, second)
	}
	if first.Limits.RequestsPerMinute != 60 || first.Limits.TokensPerMinute != 100000 {
		t.Fatalf("declared caps not preserved: %#v", first.Limits)
	}
	if first.Limits.DailyTokenCap != 0 || first.Limits.SessionTokenCap != 0 {
		t.Fatalf("zero caps must stay uncapped: %#v", first.Limits)
	}
}

func TestLoadRefusesForeignInput(t *testing.T) {
	cases := map[string]string{
		"top-level unknown field": `{"version":"fak-billing-manifest/v1","state_dir":"./s","prices_path":"./p.json","extra_field":123}`,
		"model unknown field":     `{"version":"fak-billing-manifest/v1","state_dir":"./s","prices_path":"./p.json","models":[{"name":"m","upstream":"u","upstream_model":"m","extra_field":true}]}`,
		"upstream unknown field":  `{"version":"fak-billing-manifest/v1","state_dir":"./s","prices_path":"./p.json","upstreams":[{"name":"u","base_url":"http://127.0.0.1:8000/v1","extra_field":true}]}`,
		"limits unknown field":    `{"version":"fak-billing-manifest/v1","state_dir":"./s","prices_path":"./p.json","limits":{"requests_per_minute":1,"extra_field":true}}`,
		"trailing data":           sampleManifestJSON + `{"version":"fak-billing-manifest/v1"}`,
		"truncated json":          `{"version":"fak-billing-manifest/v1",`,
		"empty input":             ``,
	}
	for name, in := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := Load([]byte(in))
			if !errors.Is(err, ErrMalformedPayload) {
				t.Fatalf("Load() = %v, want ErrMalformedPayload", err)
			}
		})
	}
}

func TestValidateShapes(t *testing.T) {
	base := func() *Manifest {
		m, err := Load([]byte(sampleManifestJSON))
		if err != nil {
			t.Fatalf("seed Load() = %v", err)
		}
		return m
	}
	cases := []struct {
		name   string
		mutate func(*Manifest)
		want   error
	}{
		{"wrong version", func(m *Manifest) { m.Version = "fak-billing-manifest/v2" }, ErrUnknownVersion},
		{"missing version", func(m *Manifest) { m.Version = "" }, ErrUnknownVersion},
		{"empty state dir", func(m *Manifest) { m.StateDir = "" }, ErrMissingStateDir},
		{"empty prices path", func(m *Manifest) { m.PricesPath = "" }, ErrMissingPricesPath},
		{"empty model name", func(m *Manifest) { m.Models[0].Name = "" }, ErrMissingModelName},
		{"duplicate model name", func(m *Manifest) { m.Models = append(m.Models, m.Models[0]) }, ErrDuplicateModelName},
		{"empty upstream model id", func(m *Manifest) { m.Models[0].UpstreamModel = "" }, ErrMissingUpstreamModel},
		{"undeclared upstream ref", func(m *Manifest) { m.Models[0].Upstream = "ghost" }, ErrUnknownUpstreamRef},
		{"empty upstream name", func(m *Manifest) { m.Upstreams[0].Name = "" }, ErrMissingUpstreamName},
		{"duplicate upstream name", func(m *Manifest) { m.Upstreams = append(m.Upstreams, m.Upstreams[0]) }, ErrDuplicateUpstreamName},
		{"relative base url", func(m *Manifest) { m.Upstreams[0].BaseURL = "localhost:8000/v1" }, ErrInvalidBaseURL},
		{"non-http scheme", func(m *Manifest) { m.Upstreams[0].BaseURL = "ftp://127.0.0.1:8000/v1" }, ErrInvalidBaseURL},
		{"scheme without host", func(m *Manifest) { m.Upstreams[0].BaseURL = "http://" }, ErrInvalidBaseURL},
		{"negative rpm", func(m *Manifest) { m.Limits.RequestsPerMinute = -1 }, ErrNegativeLimit},
		{"negative tpm", func(m *Manifest) { m.Limits.TokensPerMinute = -1 }, ErrNegativeLimit},
		{"negative daily cap", func(m *Manifest) { m.Limits.DailyTokenCap = -1 }, ErrNegativeLimit},
		{"negative session cap", func(m *Manifest) { m.Limits.SessionTokenCap = -1 }, ErrNegativeLimit},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := base()
			tc.mutate(m)
			err := m.Validate()
			if !errors.Is(err, tc.want) {
				t.Fatalf("Validate() = %v, want %v", err, tc.want)
			}
			if !errors.Is(err, ErrInvalidManifest) {
				t.Fatalf("Validate() = %v, want match ErrInvalidManifest", err)
			}
		})
	}
}

func TestLimitsValidate(t *testing.T) {
	cases := []struct {
		name string
		l    Limits
		want error
	}{
		{"all uncapped", Limits{}, nil},
		{"rpm set", Limits{RequestsPerMinute: 60}, nil},
		{"tpm set", Limits{TokensPerMinute: 1000000}, nil},
		{"daily set", Limits{DailyTokenCap: 10000000}, nil},
		{"session set", Limits{SessionTokenCap: 100000}, nil},
		{"all set", Limits{RequestsPerMinute: 1, TokensPerMinute: 2, DailyTokenCap: 3, SessionTokenCap: 4}, nil},
		{"negative rpm", Limits{RequestsPerMinute: -1}, ErrNegativeLimit},
		{"negative tpm", Limits{TokensPerMinute: -1}, ErrNegativeLimit},
		{"negative daily", Limits{DailyTokenCap: -1}, ErrNegativeLimit},
		{"negative session", Limits{SessionTokenCap: -1}, ErrNegativeLimit},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.l.Validate()
			if tc.want == nil && err != nil {
				t.Fatalf("Validate() = %v, want nil", err)
			}
			if tc.want != nil && !errors.Is(err, tc.want) {
				t.Fatalf("Validate() = %v, want %v", err, tc.want)
			}
		})
	}
}

func TestZeroValuesAreNeutral(t *testing.T) {
	// No caps declared: every zero stays uncapped and valid.
	m := &Manifest{
		Version:    ManifestVersionV1,
		StateDir:   "./state",
		PricesPath: "./prices.json",
	}
	if err := m.Validate(); err != nil {
		t.Fatalf("zero-limits manifest Validate() = %v, want nil", err)
	}
	// A file that omits the limits key entirely decodes to the same thing.
	minimal := `{"version":"fak-billing-manifest/v1","state_dir":"./state","prices_path":"./prices.json"}`
	got, err := Load([]byte(minimal))
	if err != nil {
		t.Fatalf("Load(minimal) = %v", err)
	}
	if got.Limits != (Limits{}) {
		t.Fatalf("omitted limits = %#v, want zero Limits", got.Limits)
	}
}

func TestLoadReportsValidationSentinels(t *testing.T) {
	in := `{"version":"fak-billing-manifest/v1","state_dir":"./state","prices_path":"./prices.json","limits":{"requests_per_minute":-1}}`
	_, err := Load([]byte(in))
	if !errors.Is(err, ErrNegativeLimit) {
		t.Fatalf("Load() = %v, want ErrNegativeLimit", err)
	}
}

func TestLoadFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "manifest.json")
	if err := os.WriteFile(path, []byte(sampleManifestJSON), 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	m, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile() = %v", err)
	}
	if len(m.Models) != 1 || len(m.Upstreams) != 1 {
		t.Fatalf("LoadFile() = %#v, want one model and one upstream", m)
	}
	if _, err := LoadFile(filepath.Join(dir, "missing.json")); err == nil {
		t.Fatal("LoadFile(missing): want error")
	}
}

func TestMarshalJSONRefusesInvalidManifest(t *testing.T) {
	m := &Manifest{Version: ManifestVersionV1, StateDir: "", PricesPath: "./prices.json"}
	if _, err := m.MarshalJSON(); !errors.Is(err, ErrMissingStateDir) {
		t.Fatalf("MarshalJSON() = %v, want ErrMissingStateDir", err)
	}
}
