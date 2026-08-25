package main

import (
	"bytes"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
)

func TestGuardGeminiProfileUsesNativeRepoint(t *testing.T) {
	t.Run("provider autodetection", func(t *testing.T) {
		provider, recognized := guardDetectProvider("gemini")
		if !recognized || provider != "gemini" {
			t.Fatalf("guardDetectProvider(gemini) = %q, %v; want gemini, true", provider, recognized)
		}
	})

	t.Run("base URL environment", func(t *testing.T) {
		if got := guardEnvVar("gemini", ""); got != "GOOGLE_GEMINI_BASE_URL" {
			t.Fatalf("guardEnvVar(gemini) = %q, want GOOGLE_GEMINI_BASE_URL", got)
		}
	})

	t.Run("native route root", func(t *testing.T) {
		const gatewayURL = "http://127.0.0.1:43123"
		if got := guardEnvValue("gemini", gatewayURL); got != gatewayURL {
			t.Fatalf("guardEnvValue(gemini) = %q, want %q", got, gatewayURL)
		}
	})
}

func TestGuardGeminiSessionStartCommandQuotesPlatforms(t *testing.T) {
	args := []string{`C:\Program Files\fak\fak.exe`, "guard-sessionstart", "--provider", "gemini"}
	if got := geminiCommandLine(args, "windows"); !strings.HasPrefix(got, `& "C:\Program Files\fak\fak.exe"`) {
		t.Fatalf("Windows command = %q", got)
	}
	posix := geminiCommandLine([]string{"/opt/fak's bin/fak", "guard-sessionstart"}, "linux")
	if !strings.Contains(posix, `'"'"'`) {
		t.Fatalf("POSIX command did not escape apostrophe: %q", posix)
	}
}

func TestResolveWindowsBatchCommandUsesGeminiEntrypoint(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows npm shim resolution")
	}
	got := resolveWindowsBatchCommand([]string{"gemini", "--version"})
	if len(got) != 3 || !strings.EqualFold(filepath.Base(got[0]), "node.exe") ||
		filepath.Base(got[1]) != "gemini.js" || got[2] != "--version" {
		t.Fatalf("extensionless Gemini did not resolve to its npm entrypoint: %v", got)
	}
}

func TestGuardGeminiClearAndNewCreateOneBoundedChildEach(t *testing.T) {
	oldSessions, oldDurability := serveSessions, serveSessionDurability
	serveSessions, serveSessionDurability = session.NewTable(), nil
	t.Cleanup(func() {
		serveSessions, serveSessionDurability = oldSessions, oldDurability
	})
	t.Setenv("FLEET_REG_DIR", t.TempDir())
	t.Setenv(sessionjournal.EnvPath, filepath.Join(t.TempDir(), "session-journal.jsonl"))
	t.Setenv(guardSessionJournalEnvMode, "")
	t.Setenv("GEMINI_API_KEY", "gemini-api-key-test-value")
	stageGuardSessionStartWitness(t, 1, nil)

	const oldTrace = "guard-gemini-old"
	budget := session.Budget{
		TurnsLeft: 5, TokensLeft: 800, ContextTokensLeft: 100, ContextTokensCap: 1000,
		SpendMicroCentsLeft: 40, SpendMicroCentsCap: 100,
	}
	if _, ok := serveSessions.SetBudget(oldTrace, budget); !ok {
		t.Fatal("stage old Gemini session budget")
	}
	recordGuardSessionStartIdentityFor(oldTrace, "gemini-session-old", "gemini", "startup")
	recordGuardSessionStartJournalFor(oldTrace, "gemini-session-old", "gemini", 0)

	srv, err := gateway.New(gateway.Config{ExposeProfile: "headless", DefaultTraceID: oldTrace})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	ipc, err := startGuardLifecycleServer(srv)
	if err != nil {
		t.Fatalf("start lifecycle IPC: %v", err)
	}
	t.Cleanup(ipc.Close)
	t.Setenv(guardLifecycleSocketEnv, ipc.path)
	t.Setenv(guardLifecycleTokenEnv, ipc.token)

	injected := append(guardInjectedEnv("gemini", "", "http://127.0.0.1:43123"), ipc.Env()...)
	_, childEnv := guardChildCommandEnv([]string{"gemini"}, injected, false)
	for _, want := range []string{
		guardLifecycleSocketEnv + "=" + ipc.path,
		guardLifecycleTokenEnv + "=" + ipc.token,
		"GEMINI_API_KEY=gemini-api-key-test-value",
	} {
		if !containsEnvEntry(childEnv, want) {
			t.Fatalf("Gemini child did not inherit authenticated lifecycle credential %q", want)
		}
	}

	aliases := []struct {
		name      string
		sessionID string
	}{
		{name: "/clear", sessionID: "gemini-session-clear"},
		{name: "/new", sessionID: "gemini-session-new"},
	}
	previousTrace := oldTrace
	for i, alias := range aliases {
		payload := bytes.NewBufferString(`{"hook_event_name":"SessionStart","source":"clear","session_id":"` + alias.sessionID + `"}`)
		var stdout, stderr bytes.Buffer
		if code := runGuardSessionStartHook(&stdout, &stderr, payload, []string{"--mode", "off", "--provider", "gemini"}); code != 0 {
			t.Fatalf("%s hook exit=%d stderr=%s", alias.name, code, stderr.String())
		}
		newTrace := providerBoundaryTrace("gemini", alias.sessionID)
		old := serveSessions.Get(previousTrace)
		child := serveSessions.Get(newTrace)
		if old.Run != session.Stopped || old.Reason != session.ReasonProviderSessionClear {
			t.Fatalf("%s old session = run %s reason %q", alias.name, old.Run, old.Reason)
		}
		if child.Run != session.Running || child.ProviderBoundary.PreviousTrace != previousTrace {
			t.Fatalf("%s child session = %+v", alias.name, child)
		}
		if child.Budget.ContextTokensLeft != 1000 || child.Budget.TurnsLeft != 5 || child.Budget.TokensLeft != 800 || child.Budget.SpendMicroCentsLeft != 40 {
			t.Fatalf("%s child budget = %+v", alias.name, child.Budget)
		}
		if got := srv.DefaultTraceID(); got != newTrace {
			t.Fatalf("%s gateway default trace=%q, want %q", alias.name, got, newTrace)
		}
		if got := serveSessions.Len(); got != i+2 {
			t.Fatalf("%s session count=%d, want %d (one child per alias)", alias.name, got, i+2)
		}
		previousTrace = newTrace
	}
}

func containsEnvEntry(env []string, want string) bool {
	for _, entry := range env {
		if entry == want {
			return true
		}
	}
	return false
}
