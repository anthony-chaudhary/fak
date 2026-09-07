package modelroute

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func testPayloadFixture() (Target, Target, PayloadClassificationConfig) {
	localTarget := Target{
		Model:         "small",
		Account:       "local",
		Kind:          KindLocal,
		BaseURL:       "http://127.0.0.1:11434/v1",
		UpstreamModel: "llama3.2",
	}
	remoteTarget := Target{
		Model:         "large",
		Account:       "claude",
		Kind:          KindAnthropic,
		BaseURL:       "https://api.anthropic.com",
		CredEnv:       "ANTHROPIC_API_KEY",
		UpstreamModel: "claude-opus-4-6",
	}
	cfg := PayloadClassificationConfig{
		Version: PayloadClassificationVersion,
		Rules: []ClassificationRule{
			{Pattern: "secret/**", Class: ClassLocal, Description: "keys and credentials"},
			{Pattern: "internal/private/*.go", Class: ClassClassified, Description: "private source code"},
			{Pattern: "*.key", Class: ClassRestricted, Description: "private cryptographic keys"},
			{Pattern: "customer/pii/**", Class: ClassPII, Description: "personally identifiable information"},
			{Pattern: "public/**", Class: ClassPublic, Description: "publicly disclosed documentation"},
		},
	}
	return localTarget, remoteTarget, cfg
}

func TestClassifiedPathRemoteTargetDenied(t *testing.T) {
	_, remoteTarget, cfg := testPayloadFixture()

	cases := []struct {
		name        string
		path        string
		wantRulePat string
		wantClass   string
	}{
		{
			name:        "exact directory prefix glob",
			path:        "secret/credentials.env",
			wantRulePat: "secret/**",
			wantClass:   ClassLocal,
		},
		{
			name:        "internal private pattern",
			path:        "internal/private/core.go",
			wantRulePat: "internal/private/*.go",
			wantClass:   ClassClassified,
		},
		{
			name:        "key extension glob",
			path:        "certs/ca/server.key",
			wantRulePat: "*.key",
			wantClass:   ClassRestricted,
		},
		{
			name:        "pii class pattern",
			path:        "customer/pii/users.csv",
			wantRulePat: "customer/pii/**",
			wantClass:   ClassPII,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			witness := NewEgressWitness("test-run-deny")
			enforcer := NewPayloadResidencyEnforcer(cfg).WithWitness(witness)

			payload := NewPayload(tc.path)
			dec, err := enforcer.Check(remoteTarget, payload)

			if err == nil {
				t.Fatalf("expected refusal error for classified path %q to remote target, got nil", tc.path)
			}
			if dec.Action != ActionDeny {
				t.Fatalf("decision action = %q, want %q", dec.Action, ActionDeny)
			}
			if dec.Reason != ReasonPayloadClassifiedRemoteDenied {
				t.Fatalf("decision reason = %q, want %q", dec.Reason, ReasonPayloadClassifiedRemoteDenied)
			}

			var egressErr *EgressViolationError
			if !errors.As(err, &egressErr) {
				t.Fatalf("err type = %T, want *EgressViolationError", err)
			}
			if egressErr.Reason != ReasonPayloadClassifiedRemoteDenied {
				t.Errorf("egressErr.Reason = %q, want %q", egressErr.Reason, ReasonPayloadClassifiedRemoteDenied)
			}
			if len(egressErr.Matches) == 0 {
				t.Fatalf("egressErr.Matches is empty")
			}
			if egressErr.Matches[0].Pattern != tc.wantRulePat {
				t.Errorf("matched pattern = %q, want %q", egressErr.Matches[0].Pattern, tc.wantRulePat)
			}
			if egressErr.Matches[0].Class != tc.wantClass {
				t.Errorf("matched class = %q, want %q", egressErr.Matches[0].Class, tc.wantClass)
			}

			// Witness ledger must record the denial and 0 remote classified bytes.
			if len(witness.Denials) != 1 {
				t.Fatalf("witness denials count = %d, want 1", len(witness.Denials))
			}
			if witness.Denials[0].Reason != ReasonPayloadClassifiedRemoteDenied {
				t.Errorf("witness denial reason = %q, want %q", witness.Denials[0].Reason, ReasonPayloadClassifiedRemoteDenied)
			}
			if witness.ClassifiedBytesRemote != 0 {
				t.Fatalf("witness ClassifiedBytesRemote = %d, want 0", witness.ClassifiedBytesRemote)
			}
			if err := witness.Validate(); err != nil {
				t.Fatalf("witness validation failed: %v", err)
			}
		})
	}
}

func TestClassifiedPathRemoteTargetReroutedLocal(t *testing.T) {
	localTarget, remoteTarget, cfg := testPayloadFixture()

	// 1. Deterministic reroute via explicit LocalTarget
	cfgWithLocal := cfg
	cfgWithLocal.LocalTarget = &localTarget

	witness := NewEgressWitness("test-run-reroute")
	enforcer := NewPayloadResidencyEnforcer(cfgWithLocal).WithWitness(witness)

	payload := Payload{
		Items: []PayloadItem{
			{Path: "secret/token.json", SizeBytes: 512},
		},
	}
	dec, err := enforcer.Check(remoteTarget, payload)
	if err != nil {
		t.Fatalf("unexpected error on reroute: %v", err)
	}
	if dec.Action != ActionReroute {
		t.Fatalf("decision action = %q, want %q", dec.Action, ActionReroute)
	}
	if dec.Reason != ReasonPayloadClassifiedReroutedLocal {
		t.Fatalf("decision reason = %q, want %q", dec.Reason, ReasonPayloadClassifiedReroutedLocal)
	}
	if dec.Target.Account != localTarget.Account || !dec.Target.Local() {
		t.Fatalf("target was not rerouted to local: %+v", dec.Target)
	}
	if dec.OriginalTarget.Account != remoteTarget.Account {
		t.Fatalf("original target was not preserved: %+v", dec.OriginalTarget)
	}

	// Witness verification
	if len(witness.Reroutes) != 1 {
		t.Fatalf("witness reroutes count = %d, want 1", len(witness.Reroutes))
	}
	if witness.Reroutes[0].ToTarget.Account != "local" {
		t.Errorf("witness reroute destination = %q, want local", witness.Reroutes[0].ToTarget.Account)
	}
	if witness.ClassifiedBytesRemote != 0 {
		t.Fatalf("classified bytes remote = %d, want 0", witness.ClassifiedBytesRemote)
	}
	if witness.ClassifiedBytesLocal != 512 {
		t.Errorf("classified bytes local = %d, want 512", witness.ClassifiedBytesLocal)
	}
	if err := witness.Validate(); err != nil {
		t.Fatalf("witness validation failed: %v", err)
	}

	// 2. Deterministic reroute via LocalAccount + Roster
	r := rosterFixture()
	cfgWithAcct := cfg
	cfgWithAcct.LocalAccount = "local"

	witness2 := NewEgressWitness("test-run-reroute-roster")
	enforcer2 := NewPayloadResidencyEnforcer(cfgWithAcct).WithRoster(&r).WithWitness(witness2)

	dec2, err := enforcer2.Check(remoteTarget, payload)
	if err != nil {
		t.Fatalf("unexpected error on reroute via roster: %v", err)
	}
	if dec2.Action != ActionReroute {
		t.Fatalf("decision action = %q, want %q", dec2.Action, ActionReroute)
	}
	if !dec2.Target.Local() {
		t.Fatalf("target was not local: %+v", dec2.Target)
	}
}

func TestClassifiedPathLocalTargetAllowed(t *testing.T) {
	localTarget, _, cfg := testPayloadFixture()

	witness := NewEgressWitness("test-run-local-allow")
	enforcer := NewPayloadResidencyEnforcer(cfg).WithWitness(witness)

	payload := Payload{
		Items: []PayloadItem{
			{Path: "secret/master.key", SizeBytes: 256},
			{Path: "internal/private/secrets.go", SizeBytes: 1024},
		},
	}

	dec, err := enforcer.Check(localTarget, payload)
	if err != nil {
		t.Fatalf("classified payload to local target must be allowed: %v", err)
	}
	if dec.Action != ActionAllow {
		t.Fatalf("decision action = %q, want %q", dec.Action, ActionAllow)
	}
	if !dec.Classified {
		t.Fatalf("payload should be marked classified")
	}
	if dec.ClassifiedBytes != 1280 {
		t.Errorf("classified bytes = %d, want 1280", dec.ClassifiedBytes)
	}
	if !dec.Target.Local() {
		t.Fatalf("target must remain local")
	}

	// Witness verification: zero classified remote bytes
	if witness.ClassifiedBytesRemote != 0 {
		t.Fatalf("classified bytes remote = %d, want 0", witness.ClassifiedBytesRemote)
	}
	if witness.ClassifiedBytesLocal != 1280 {
		t.Errorf("classified bytes local = %d, want 1280", witness.ClassifiedBytesLocal)
	}
	if len(witness.Denials) != 0 || len(witness.Reroutes) != 0 {
		t.Fatalf("no denials or reroutes expected for local dispatch: denials=%d reroutes=%d", len(witness.Denials), len(witness.Reroutes))
	}
	if err := witness.Validate(); err != nil {
		t.Fatalf("witness validation failed: %v", err)
	}
}

func TestUnclassifiedPathAllowed(t *testing.T) {
	localTarget, remoteTarget, cfg := testPayloadFixture()

	unclassifiedPayload := Payload{
		Items: []PayloadItem{
			{Path: "cmd/main.go", SizeBytes: 1500},
			{Path: "pkg/sdk/client.go", SizeBytes: 2500},
			{Path: "public/readme.md", SizeBytes: 500}, // explicit ClassPublic rule
		},
	}

	// 1. Remote target allows unclassified payload
	witnessRemote := NewEgressWitness("test-run-unclass-remote")
	enforcerRemote := NewPayloadResidencyEnforcer(cfg).WithWitness(witnessRemote)

	dec, err := enforcerRemote.Check(remoteTarget, unclassifiedPayload)
	if err != nil {
		t.Fatalf("unclassified payload to remote target must be allowed: %v", err)
	}
	if dec.Action != ActionAllow {
		t.Fatalf("decision action = %q, want %q", dec.Action, ActionAllow)
	}
	if dec.Classified {
		t.Fatalf("payload should not be classified")
	}
	if dec.Target.Account != remoteTarget.Account {
		t.Fatalf("target changed unexpectedly: %+v", dec.Target)
	}
	if witnessRemote.ClassifiedBytesRemote != 0 {
		t.Fatalf("classified bytes remote = %d, want 0", witnessRemote.ClassifiedBytesRemote)
	}
	if witnessRemote.UnclassifiedBytesRemote != 4500 {
		t.Errorf("unclassified bytes remote = %d, want 4500", witnessRemote.UnclassifiedBytesRemote)
	}
	if err := witnessRemote.Validate(); err != nil {
		t.Fatalf("witness validation failed: %v", err)
	}

	// 2. Local target also allows unclassified payload
	witnessLocal := NewEgressWitness("test-run-unclass-local")
	enforcerLocal := NewPayloadResidencyEnforcer(cfg).WithWitness(witnessLocal)

	decLocal, err := enforcerLocal.Check(localTarget, unclassifiedPayload)
	if err != nil {
		t.Fatalf("unclassified payload to local target must be allowed: %v", err)
	}
	if decLocal.Action != ActionAllow {
		t.Fatalf("decision action = %q, want %q", decLocal.Action, ActionAllow)
	}
	if witnessLocal.ClassifiedBytesRemote != 0 {
		t.Fatalf("classified bytes remote = %d, want 0", witnessLocal.ClassifiedBytesRemote)
	}
	if witnessLocal.UnclassifiedBytesLocal != 4500 {
		t.Errorf("unclassified bytes local = %d, want 4500", witnessLocal.UnclassifiedBytesLocal)
	}
}

func TestInvalidClassificationFailsClosed(t *testing.T) {
	_, remoteTarget, _ := testPayloadFixture()

	// 1. Empty pattern
	badCfg1 := PayloadClassificationConfig{
		Rules: []ClassificationRule{
			{Pattern: "", Class: ClassLocal},
		},
	}
	if err := badCfg1.Validate(); err == nil {
		t.Fatalf("expected validation error for empty pattern")
	}

	// 2. Empty class
	badCfg2 := PayloadClassificationConfig{
		Rules: []ClassificationRule{
			{Pattern: "secret/**", Class: ""},
		},
	}
	if err := badCfg2.Validate(); err == nil {
		t.Fatalf("expected validation error for empty class")
	}

	// 3. LocalTarget that is remote
	remoteAsLocal := remoteTarget
	badCfg3 := PayloadClassificationConfig{
		Rules: []ClassificationRule{
			{Pattern: "secret/**", Class: ClassLocal},
		},
		LocalTarget: &remoteAsLocal,
	}
	if err := badCfg3.Validate(); err == nil {
		t.Fatalf("expected validation error when LocalTarget is remote")
	}

	// 4. Enforcer initialized with invalid config fails closed
	witness := NewEgressWitness("test-invalid-config")
	enforcer := NewPayloadResidencyEnforcer(badCfg1).WithWitness(witness)

	payload := NewPayload("normal/path.txt")
	dec, err := enforcer.Check(remoteTarget, payload)
	if err == nil {
		t.Fatalf("expected error from enforcer with invalid config")
	}
	if dec.Action != ActionDeny {
		t.Fatalf("action = %q, want %q", dec.Action, ActionDeny)
	}
	if dec.Reason != ReasonClassificationInvalid {
		t.Fatalf("reason = %q, want %q", dec.Reason, ReasonClassificationInvalid)
	}
	if witness.ClassifiedBytesRemote != 0 {
		t.Fatalf("classified bytes remote must be 0, got %d", witness.ClassifiedBytesRemote)
	}

	// 5. Unreadable/invalid JSON files fail closed
	tempDir := t.TempDir()
	badJSONPath := filepath.Join(tempDir, "broken.json")
	if err := os.WriteFile(badJSONPath, []byte(`{ broken json`), 0600); err != nil {
		t.Fatalf("failed to write broken JSON: %v", err)
	}
	if _, err := LoadPayloadClassification(badJSONPath); err == nil {
		t.Fatalf("LoadPayloadClassification must fail on broken JSON")
	}

	nonExistentPath := filepath.Join(tempDir, "does-not-exist.json")
	if _, err := LoadPayloadClassification(nonExistentPath); err == nil {
		t.Fatalf("LoadPayloadClassification must fail on non-existent file")
	}
}

func TestCentralOverridePrecedence(t *testing.T) {
	localTarget, remoteTarget, _ := testPayloadFixture()

	// Repo-level config: classifies only repo-secret/**
	repoCfg := PayloadClassificationConfig{
		Rules: []ClassificationRule{
			{Pattern: "repo-secret/**", Class: ClassLocal},
		},
	}

	// Central config: classifies enterprise-sensitive/** and defines a local fallback
	centralCfg := PayloadClassificationConfig{
		Rules: []ClassificationRule{
			{Pattern: "enterprise-sensitive/**", Class: ClassRestricted},
		},
		LocalTarget: &localTarget,
	}

	merged := repoCfg.Override(centralCfg)

	// Central rule must be first in merged rules
	if len(merged.Rules) != 2 {
		t.Fatalf("merged rules length = %d, want 2", len(merged.Rules))
	}
	if merged.Rules[0].Pattern != "enterprise-sensitive/**" {
		t.Errorf("first rule = %q, want enterprise-sensitive/**", merged.Rules[0].Pattern)
	}
	if merged.LocalTarget == nil || merged.LocalTarget.Account != "local" {
		t.Fatalf("central LocalTarget was not inherited")
	}

	witness := NewEgressWitness("test-run-override")
	enforcer := NewPayloadResidencyEnforcer(merged).WithWitness(witness)

	// Path classified by central config should be rerouted to central's LocalTarget
	payload := NewPayload("enterprise-sensitive/report.pdf")
	dec, err := enforcer.Check(remoteTarget, payload)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if dec.Action != ActionReroute {
		t.Fatalf("action = %q, want %q", dec.Action, ActionReroute)
	}
	if dec.Target.Account != "local" {
		t.Fatalf("rerouted target = %q, want local", dec.Target.Account)
	}

	// Path classified by repo config should also be protected
	payloadRepo := NewPayload("repo-secret/token.txt")
	decRepo, err := enforcer.Check(remoteTarget, payloadRepo)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if decRepo.Action != ActionReroute {
		t.Fatalf("action = %q, want %q", decRepo.Action, ActionReroute)
	}

	if witness.ClassifiedBytesRemote != 0 {
		t.Fatalf("classified bytes remote = %d, want 0", witness.ClassifiedBytesRemote)
	}
}

func TestLoadPayloadClassificationWithOverride(t *testing.T) {
	localTarget, _, _ := testPayloadFixture()
	tempDir := t.TempDir()

	repoCfg := PayloadClassificationConfig{
		Rules: []ClassificationRule{
			{Pattern: "repo/**", Class: ClassLocal},
		},
	}
	centralCfg := PayloadClassificationConfig{
		Rules: []ClassificationRule{
			{Pattern: "central/**", Class: ClassClassified},
		},
		LocalTarget: &localTarget,
	}

	repoPath := filepath.Join(tempDir, "repo.json")
	centralPath := filepath.Join(tempDir, "central.json")

	if err := os.WriteFile(repoPath, repoCfg.JSON(), 0600); err != nil {
		t.Fatalf("failed to write repo config: %v", err)
	}
	if err := os.WriteFile(centralPath, centralCfg.JSON(), 0600); err != nil {
		t.Fatalf("failed to write central config: %v", err)
	}

	merged, err := LoadPayloadClassificationWithOverride(repoPath, centralPath)
	if err != nil {
		t.Fatalf("failed to load with override: %v", err)
	}
	if len(merged.Rules) != 2 {
		t.Fatalf("merged rules length = %d, want 2", len(merged.Rules))
	}
	if merged.Rules[0].Pattern != "central/**" {
		t.Errorf("central rule did not take precedence")
	}
	if merged.LocalTarget == nil || merged.LocalTarget.Account != "local" {
		t.Errorf("central local target was not preserved")
	}

	// Fail closed if central file is unreadable
	if _, err := LoadPayloadClassificationWithOverride(repoPath, filepath.Join(tempDir, "missing.json")); err == nil {
		t.Fatalf("expected error when central file is missing")
	}
}

func TestEgressWitnessZeroClassifiedBytesRemote(t *testing.T) {
	localTarget, remoteTarget, cfg := testPayloadFixture()
	cfgWithLocal := cfg
	cfgWithLocal.LocalTarget = &localTarget

	witness := NewEgressWitness("run-comprehensive-witness")
	enforcerWithLocal := NewPayloadResidencyEnforcer(cfgWithLocal).WithWitness(witness)
	enforcerNoLocal := NewPayloadResidencyEnforcer(cfg).WithWitness(witness)

	// 1. Classified -> Local target (allowed)
	_, err := enforcerWithLocal.Check(localTarget, Payload{
		Items: []PayloadItem{{Path: "secret/a.key", SizeBytes: 100}},
	})
	if err != nil {
		t.Fatalf("local allow failed: %v", err)
	}

	// 2. Unclassified -> Remote target (allowed)
	_, err = enforcerWithLocal.Check(remoteTarget, Payload{
		Items: []PayloadItem{{Path: "docs/readme.txt", SizeBytes: 200}},
	})
	if err != nil {
		t.Fatalf("remote unclassified allow failed: %v", err)
	}

	// 3. Classified -> Remote target with Local fallback (rerouted)
	_, err = enforcerWithLocal.Check(remoteTarget, Payload{
		Items: []PayloadItem{{Path: "secret/b.key", SizeBytes: 300}},
	})
	if err != nil {
		t.Fatalf("reroute failed: %v", err)
	}

	// 4. Classified -> Remote target without Local fallback (denied)
	_, err = enforcerNoLocal.Check(remoteTarget, Payload{
		Items: []PayloadItem{{Path: "secret/c.key", SizeBytes: 400}},
	})
	if err == nil {
		t.Fatalf("expected denial, got nil")
	}

	// Verify all counters and slices in witness
	if witness.RulesEvaluated == 0 {
		t.Fatalf("RulesEvaluated should be > 0, got %d", witness.RulesEvaluated)
	}
	if witness.ClassifiedBytesRemote != 0 {
		t.Fatalf("RESIDENCY INVARIANT VIOLATION: ClassifiedBytesRemote = %d, MUST be 0", witness.ClassifiedBytesRemote)
	}
	if witness.ClassifiedBytesLocal != 400 { // 100 from call 1 + 300 from call 3
		t.Errorf("ClassifiedBytesLocal = %d, want 400", witness.ClassifiedBytesLocal)
	}
	if witness.UnclassifiedBytesRemote != 200 {
		t.Errorf("UnclassifiedBytesRemote = %d, want 200", witness.UnclassifiedBytesRemote)
	}
	if len(witness.Denials) != 1 {
		t.Errorf("Denials count = %d, want 1", len(witness.Denials))
	}
	if len(witness.Reroutes) != 1 {
		t.Errorf("Reroutes count = %d, want 1", len(witness.Reroutes))
	}

	// Invariant validation method
	if err := witness.Validate(); err != nil {
		t.Fatalf("witness validation failed: %v", err)
	}

	// Tamper test: simulate invariant breach
	tampered := *witness
	tampered.ClassifiedBytesRemote = 42
	if err := tampered.Validate(); err == nil {
		t.Fatalf("tampered witness with remote classified bytes must fail Validate()")
	}

	// JSON format check
	jsonBytes := witness.JSON()
	var decoded EgressWitness
	if err := json.Unmarshal(jsonBytes, &decoded); err != nil {
		t.Fatalf("witness JSON failed to decode: %v", err)
	}
	if decoded.ClassifiedBytesRemote != 0 {
		t.Fatalf("decoded ClassifiedBytesRemote = %d, want 0", decoded.ClassifiedBytesRemote)
	}
	if len(decoded.Denials) != 1 || len(decoded.Reroutes) != 1 {
		t.Fatalf("decoded witness missing events: %+v", decoded)
	}
}

func TestGlobMatchingPatterns(t *testing.T) {
	tests := []struct {
		pattern string
		target  string
		want    bool
	}{
		{"secret/**", "secret/sub/keys.env", true},
		{"secret/**", "secret/keys.env", true},
		{"secret/**", "secret", true},
		{"secret/**", "other/secret/keys.env", false},
		{"**/secret/**", "foo/secret/bar.txt", true},
		{"*.key", "certs/domain.key", true},
		{"*.key", "domain.key", true},
		{"*.key", "domain.key.bak", false},
		{"internal/private/*.go", "internal/private/core.go", true},
		{"internal/private/*.go", "internal/private/nested/deep.go", false},
		{"internal/private/**", "internal/private/nested/deep.go", true},
		{"**", "any/file/at/all.go", true},
		{"exact/file.txt", "exact/file.txt", true},
		{"exact/file.txt", "exact/other.txt", false},
		// Windows backslash normalization
		{"secret/**", "secret\\nested\\file.txt", true},
		{"*.pem", "certs\\prod.pem", true},
	}

	for _, tc := range tests {
		got := matchPattern(tc.pattern, tc.target)
		if got != tc.want {
			t.Errorf("matchPattern(%q, %q) = %v, want %v", tc.pattern, tc.target, got, tc.want)
		}
	}
}

func TestCustomSensitivityClasses(t *testing.T) {
	localClasses := []string{
		ClassLocal,
		ClassClassified,
		ClassRestricted,
		ClassConfidential,
		ClassSecret,
		ClassTenant,
		ClassPII,
		"TOP_SECRET",
		"internal-only",
	}

	for _, c := range localClasses {
		if !IsLocalOnlyClass(c) {
			t.Errorf("class %q should require local residency", c)
		}
	}

	publicClasses := []string{
		ClassPublic,
		ClassUnclassified,
		"",
		"public",
		"UNCLASSIFIED",
	}

	for _, c := range publicClasses {
		if IsLocalOnlyClass(c) {
			t.Errorf("class %q should NOT require local residency", c)
		}
	}
}

func TestAccountLevelResidencyInvariantUnchanged(t *testing.T) {
	// 1. Account-level invariant: local account with remote base URL fails Validate
	badLocal := Account{
		ID:      "bad-local",
		Kind:    KindLocal,
		BaseURL: "https://api.openai.com/v1", // Remote URL under local kind
	}
	rBad := Roster{Accounts: []Account{badLocal}}
	if err := rBad.Validate(); err == nil {
		t.Fatalf("local account with remote base_url must fail validation")
	}

	// 2. Valid local account
	goodLocal := Account{
		ID:      "good-local",
		Kind:    KindLocal,
		BaseURL: "http://127.0.0.1:11434/v1",
	}
	rGood := Roster{Accounts: []Account{goodLocal}}
	if err := rGood.Validate(); err != nil {
		t.Fatalf("valid local account failed Validate(): %v", err)
	}

	// 3. Remote account without credentials fails Validate
	badRemote := Account{
		ID:   "bad-remote",
		Kind: KindOpenAI,
	}
	rBadRemote := Roster{Accounts: []Account{badRemote}}
	if err := rBadRemote.Validate(); err == nil {
		t.Fatalf("remote account without cred_env must fail validation")
	}

	// 4. Locality derivation: single source of truth is KindLocal
	tLocal := Target{Kind: KindLocal}
	if !tLocal.Local() || tLocal.Remote() {
		t.Fatalf("Target.Local() must agree with KindLocal")
	}
	tRemote := Target{Kind: KindOpenAI}
	if tRemote.Local() || !tRemote.Remote() {
		t.Fatalf("Target.Remote() must agree with non-KindLocal")
	}
}

func TestJSONRoundTripAndManifests(t *testing.T) {
	localTarget, _, cfg := testPayloadFixture()
	cfg.LocalTarget = &localTarget

	b := cfg.JSON()
	parsed, err := ParsePayloadClassification(b)
	if err != nil {
		t.Fatalf("failed to parse JSON manifest: %v", err)
	}
	if len(parsed.Rules) != len(cfg.Rules) {
		t.Fatalf("parsed rules count = %d, want %d", len(parsed.Rules), len(cfg.Rules))
	}
	if parsed.LocalTarget == nil || parsed.LocalTarget.Account != "local" {
		t.Fatalf("parsed LocalTarget not restored")
	}

	// Unknown fields should be rejected (DisallowUnknownFields)
	tampered := []byte(`{"unknown_field": "val", "rules": []}`)
	if _, err := ParsePayloadClassification(tampered); err == nil {
		t.Fatalf("ParsePayloadClassification must reject unknown fields")
	}
}

func TestPayloadItemSizingAndNormalization(t *testing.T) {
	p := Payload{
		Paths: []string{"a.txt", "b.txt"},
		Items: []PayloadItem{
			{Path: "c.txt", SizeBytes: 100},
			{Path: "d.txt", Data: []byte("hello world")},
		},
	}

	items := p.AllItems()
	if len(items) != 4 {
		t.Fatalf("items count = %d, want 4", len(items))
	}

	total := p.TotalBytes()
	// c.txt=100 + d.txt=11 + a.txt=5 + b.txt=5 = 121
	if total != 121 {
		t.Errorf("TotalBytes = %d, want 121", total)
	}
}

func TestDeterministicRerouteMatchesRuleOrdering(t *testing.T) {
	localTarget, remoteTarget, _ := testPayloadFixture()

	cfg := PayloadClassificationConfig{
		Rules: []ClassificationRule{
			{Pattern: "secret/special/**", Class: ClassLocal, Description: "special secret"},
			{Pattern: "secret/**", Class: ClassRestricted, Description: "general secret"},
		},
		LocalTarget: &localTarget,
	}

	enforcer := NewPayloadResidencyEnforcer(cfg)
	payload := NewPayload("secret/special/file.txt")
	dec, err := enforcer.Check(remoteTarget, payload)
	if err != nil {
		t.Fatalf("check failed: %v", err)
	}
	if dec.Action != ActionReroute {
		t.Fatalf("action = %q, want %q", dec.Action, ActionReroute)
	}
	if len(dec.Matches) != 1 {
		t.Fatalf("matches count = %d, want 1", len(dec.Matches))
	}
	if dec.Matches[0].Description != "special secret" {
		t.Errorf("matched description = %q, want special secret", dec.Matches[0].Description)
	}
}

func TestMultiplePayloadFilesPartialClassification(t *testing.T) {
	_, remoteTarget, cfg := testPayloadFixture()

	// A payload with 2 unclassified files and 1 classified file
	payload := Payload{
		Items: []PayloadItem{
			{Path: "src/main.go", SizeBytes: 1000},
			{Path: "secret/token.key", SizeBytes: 300},
			{Path: "README.md", SizeBytes: 500},
		},
	}

	witness := NewEgressWitness("test-partial-classification")
	enforcer := NewPayloadResidencyEnforcer(cfg).WithWitness(witness)

	dec, err := enforcer.Check(remoteTarget, payload)
	if err == nil {
		t.Fatalf("payload with even ONE classified item must be denied remote dispatch")
	}
	if dec.Action != ActionDeny {
		t.Fatalf("action = %q, want %q", dec.Action, ActionDeny)
	}
	if !dec.Classified {
		t.Fatalf("payload should be marked classified")
	}
	if dec.ClassifiedBytes != 300 {
		t.Errorf("classified bytes = %d, want 300", dec.ClassifiedBytes)
	}
	if dec.TotalBytes != 1800 {
		t.Errorf("total bytes = %d, want 1800", dec.TotalBytes)
	}
	if witness.ClassifiedBytesRemote != 0 {
		t.Fatalf("classified bytes remote must be 0, got %d", witness.ClassifiedBytesRemote)
	}
	if len(witness.Denials) != 1 {
		t.Errorf("denials count = %d, want 1", len(witness.Denials))
	}
}
