package modelroute

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/borrowprovenance"
)

func TestDefaultProviderContractsAreCompleteAndDeterministic(t *testing.T) {
	contracts := DefaultProviderContracts()
	if err := ValidateProviderContracts(contracts); err != nil {
		t.Fatal(err)
	}
	if len(contracts) != 3 || contracts[0].Provider != "anthropic" || contracts[1].Provider != "openai" || contracts[2].Provider != "openrouter" {
		t.Fatalf("registry ordering/coverage = %+v", contracts)
	}
	for _, contract := range contracts {
		if contract.ContextTokens.State != KnowledgeUnknown || contract.MaxOutputTokens.State != KnowledgeUnknown {
			t.Errorf("%s guessed model-specific token limits: context=%s output=%s", contract.Provider, contract.ContextTokens.State, contract.MaxOutputTokens.State)
		}
		// PromptCaching must be KnowledgeKnown (explicitly declared), but Value can be true or false
		if contract.PromptCaching.State != KnowledgeKnown {
			t.Errorf("%s prompt_caching state must be known: %+v", contract.Provider, contract)
		}
		// UsageDetails must be explicitly declared (known or not_applicable), not unknown
		if contract.UsageDetails.State != KnowledgeKnown && contract.UsageDetails.State != KnowledgeNotApplicable {
			t.Errorf("%s usage_details state must be known or not_applicable: %+v", contract.Provider, contract)
		}
	}
	first, err := ProviderContractsJSON()
	if err != nil {
		t.Fatal(err)
	}
	second, err := ProviderContractsJSON()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatal("operator JSON is nondeterministic")
	}
	var round []ProviderContract
	if err := json.Unmarshal(first, &round); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(round, contracts) {
		t.Fatalf("registry JSON round trip changed data")
	}
}

func TestProviderProfilesProjectCanonicalContracts(t *testing.T) {
	registry, err := NewProviderRegistry(DefaultProviderProfiles())
	if err != nil {
		t.Fatal(err)
	}
	for _, provider := range []string{"openai", "anthropic", "openrouter"} {
		contract, ok := LookupProviderContract(provider)
		if !ok {
			t.Fatalf("missing %s contract", provider)
		}
		profile, ok := registry.Lookup(provider)
		if !ok {
			t.Fatalf("missing %s profile", provider)
		}
		if profile.Endpoint != contract.Endpoint.Value || profile.AuthEnv != contract.AuthEnv.Value || profile.SupportsPromptCaching != contract.PromptCaching.Value {
			t.Fatalf("%s profile drifted from contract: profile=%+v contract=%+v", provider, profile, contract)
		}
	}
}

func TestProviderContractValidationRejectsImplicitAndDuplicateFacts(t *testing.T) {
	base := DefaultProviderContracts()[0]
	tests := []struct {
		name      string
		contracts []ProviderContract
		want      string
	}{
		{"duplicate scope", []ProviderContract{base, base}, "duplicate provider contract scope"},
		{"implicit state", func() []ProviderContract { c := base; c.Endpoint.State = ""; return []ProviderContract{c} }(), "invalid knowledge state"},
		{"missing provenance", func() []ProviderContract { c := base; c.Endpoint.Source = FactSource{}; return []ProviderContract{c} }(), "fact provenance"},
		{"unknown with value", func() []ProviderContract {
			c := base
			c.ContextTokens = ContractFact[int]{State: KnowledgeUnknown, Value: 200000, Source: c.ContextTokens.Source}
			return []ProviderContract{c}
		}(), "optimistic value"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateProviderContracts(tt.contracts)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error=%v want substring %q", err, tt.want)
			}
		})
	}
}

func TestProviderRetryFactsMatchCopiedSDKBaselines(t *testing.T) {
	files := map[string]string{
		"openai":    "openai_should_retry.go.txt",
		"anthropic": "anthropic_should_retry.go.txt",
	}
	for provider, name := range files {
		contract, ok := LookupProviderContract(provider)
		if !ok {
			t.Fatalf("missing %s contract", provider)
		}
		raw, err := os.ReadFile(filepath.Join("testdata", "provider_contracts", "upstream", name))
		if err != nil {
			t.Fatal(err)
		}
		for _, baseline := range []string{"http.StatusRequestTimeout", "http.StatusConflict", "http.StatusTooManyRequests", "http.StatusInternalServerError", `x-should-retry`} {
			if !bytes.Contains(raw, []byte(baseline)) {
				t.Errorf("%s copied retry baseline missing %s", provider, baseline)
			}
		}
		if contract.RetryStatusCodes.Source.Ref == "" || contract.RetryStatusCodes.Source.Path != "internal/requestconfig/requestconfig.go" || contract.RetryStatusCodes.Source.Symbol != "shouldRetry" {
			t.Errorf("%s retry provenance does not identify copied source: %+v", provider, contract.RetryStatusCodes.Source)
		}
		source := contract.RetryStatusCodes.Source
		verification, err := borrowprovenance.Verify(borrowprovenance.Record{
			Schema: borrowprovenance.Schema, SourceURL: source.URL, SourceRef: source.Ref,
			SourcePath: source.Path, SourceSHA256: source.CopiedSHA256, License: source.License,
		}, raw)
		if err != nil {
			t.Fatalf("%s copied retry provenance invalid: %v", provider, err)
		}
		if source.CopiedPath != filepath.ToSlash(filepath.Join("testdata", "provider_contracts", "upstream", name)) || !verification.Match {
			t.Errorf("%s copied retry baseline drifted: path=%q verification=%+v", provider, source.CopiedPath, verification)
		}
		license, err := os.ReadFile(filepath.Join("testdata", "provider_contracts", "upstream", provider+".LICENSE"))
		if err != nil || len(bytes.TrimSpace(license)) == 0 {
			t.Fatalf("%s copied baseline license missing: %v", provider, err)
		}
	}
}
