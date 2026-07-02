package toolprocgate

import (
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestSpawnBrokerDeniesUnmanagedAttemptAndAuditsWithoutSecrets(t *testing.T) {
	broker := NewSpawnBroker()
	_, err := broker.Admit(SpawnAttempt{
		AgentRunID: "agent-1",
		ToolCallID: "tool-1",
		Argv:       []string{"agent", "--token=raw-secret"},
		Env:        []EnvVar{{Name: "API_TOKEN", Value: "raw-secret"}},
		CWD:        t.TempDir(),
		Backend:    "guard",
		Envelope:   CapabilityEnvelope{Capabilities: []abi.Capability{CapAgentRunSpawn}},
	})
	var denied SpawnDeniedError
	if !errors.As(err, &denied) {
		t.Fatalf("Admit returned %T/%v, want SpawnDeniedError", err, err)
	}
	if denied.Audit.Reason != "MISSING_POLICY_DIGEST" {
		t.Fatalf("deny reason = %q, want MISSING_POLICY_DIGEST", denied.Audit.Reason)
	}
	audits := broker.Audits()
	if len(audits) != 1 || audits[0].Verdict != SpawnVerdictDeny {
		t.Fatalf("audits = %+v, want one deny audit", audits)
	}
	rendered := strings.Join([]string{audits[0].ArgvDigest, audits[0].EnvDigest, strings.Join(audits[0].EnvNames, ",")}, "\n")
	if strings.Contains(rendered, "raw-secret") {
		t.Fatalf("spawn audit leaked a raw argv/env secret: %+v", audits[0])
	}
	if len(audits[0].EnvNames) != 1 || audits[0].EnvNames[0] != "API_TOKEN" {
		t.Fatalf("deny audit env names = %v, want only API_TOKEN", audits[0].EnvNames)
	}
}

func TestSpawnBrokerGrantSanitizesPropagatesMetadataAndAudits(t *testing.T) {
	broker := NewSpawnBroker()
	cwd := filepath.Join(t.TempDir(), ".")
	grant, err := broker.Admit(SpawnAttempt{
		AgentRunID:   " agent-1 ",
		ParentRunID:  " parent-1 ",
		ToolCallID:   " tool-1 ",
		PolicyDigest: " sha256:policy ",
		Argv:         []string{"  agent  ", "--mode", "ok"},
		Env: []EnvVar{
			{Name: "OPENAI_BASE_URL", Value: "http://127.0.0.1:9/v1"},
			{Name: EnvAgentRunID, Value: "stale"},
		},
		CWD:     cwd,
		Backend: " guard ",
		Envelope: CapabilityEnvelope{
			Capabilities:     []abi.Capability{"z.cap", CapAgentRunSpawn, "z.cap"},
			DeadlineMS:       120_000,
			HeartbeatEveryMS: 5_000,
		},
	})
	if err != nil {
		t.Fatalf("Admit: %v", err)
	}
	if grant.AgentRunID != "agent-1" || grant.ParentRunID != "parent-1" ||
		grant.ToolCallID != "tool-1" || grant.PolicyDigest != "sha256:policy" {
		t.Fatalf("grant metadata was not normalized: %+v", grant)
	}
	if got := strings.Join(grant.Argv, " "); got != "agent --mode ok" {
		t.Fatalf("grant argv = %q, want sanitized argv", got)
	}
	if grant.CWD != filepath.Clean(cwd) || grant.Backend != "guard" {
		t.Fatalf("grant cwd/backend = %q/%q, want %q/guard", grant.CWD, grant.Backend, filepath.Clean(cwd))
	}
	wantEnv := map[string]string{
		EnvAgentRunID:     "agent-1",
		EnvParentRunID:    "parent-1",
		EnvToolCallID:     "tool-1",
		EnvPolicyDigest:   "sha256:policy",
		EnvSpawnBackend:   "guard",
		EnvSpawnGrantID:   grant.GrantID,
		"OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
	}
	for _, kv := range grant.Env {
		if want, ok := wantEnv[kv.Name]; ok && want == kv.Value {
			delete(wantEnv, kv.Name)
		}
	}
	if len(wantEnv) != 0 {
		t.Fatalf("grant env missing metadata entries: %v from %+v", wantEnv, grant.Env)
	}
	if len(grant.Envelope.Capabilities) != 2 ||
		grant.Envelope.Capabilities[0] != CapAgentRunSpawn ||
		grant.Envelope.Capabilities[1] != "z.cap" {
		t.Fatalf("capability envelope = %+v, want sorted unique caps", grant.Envelope)
	}
	audits := broker.Audits()
	if len(audits) != 1 || audits[0].Verdict != SpawnVerdictAllow || audits[0].GrantID != grant.GrantID {
		t.Fatalf("audits = %+v, want one allow audit for grant", audits)
	}
	if strings.Contains(audits[0].ArgvDigest+audits[0].EnvDigest, "http://127.0.0.1") {
		t.Fatalf("audit must carry digests, not raw argv/env values: %+v", audits[0])
	}
}

func TestSpawnBrokerRejectsInvalidEnvelope(t *testing.T) {
	broker := NewSpawnBroker()
	_, err := broker.Admit(SpawnAttempt{
		AgentRunID:   "agent-1",
		ToolCallID:   "tool-1",
		PolicyDigest: "sha256:policy",
		Argv:         []string{"agent"},
		CWD:          t.TempDir(),
		Backend:      "guard",
		Envelope:     CapabilityEnvelope{DeadlineMS: -1},
	})
	if err == nil || !strings.Contains(err.Error(), "NEGATIVE_CAPABILITY_ENVELOPE") {
		t.Fatalf("Admit invalid envelope error = %v, want NEGATIVE_CAPABILITY_ENVELOPE", err)
	}
}
