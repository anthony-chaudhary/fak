package armbench

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
)

func TestCavemanPinnedInputsAndSemanticGate(t *testing.T) {
	root := filepath.Join("..", "..", "docs", "_witnesses", "armbench-caveman-native", "inputs")
	hashes, err := verifyCavemanInputs(root)
	if err != nil {
		t.Fatal(err)
	}
	if hashes["prompts.json"] != cavemanPromptsSHA || hashes["SKILL.md"] != cavemanSkillSHA {
		t.Fatalf("wrong hashes: %#v", hashes)
	}
	good := map[string]string{
		"react-rerender":      "Object reference identity changes; use React.memo and useMemo.",
		"auth-middleware-fix": "exp is seconds, Date.now is milliseconds; compare Date.now()/1000.",
		"pr-security-review":  "SQL injection: use parameter placeholders and handle errors.",
		"error-boundary":      "Use getDerivedStateFromError, componentDidCatch to log, and a retry button.",
	}
	for id, text := range good {
		if ok, missing := semanticGate(id, text); !ok {
			t.Errorf("%s failed: %v", id, missing)
		}
	}
	if ok, _ := semanticGate("auth-middleware-fix", "Looks fine."); ok {
		t.Fatal("empty correctness answer passed")
	}
}
func TestCavemanHashMismatchFailsClosed(t *testing.T) {
	d := t.TempDir()
	for _, n := range []string{"prompts.json", "SKILL.md", "run.py"} {
		if err := os.WriteFile(filepath.Join(d, n), []byte("changed"), 0644); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := verifyCavemanInputs(d); err == nil {
		t.Fatal("expected hash mismatch")
	}
}

func TestLiveCavemanReplacement(t *testing.T) {
	if os.Getenv("FAK_CAVEMAN_LIVE") != "1" {
		t.Skip("set FAK_CAVEMAN_LIVE=1 for captured provider run")
	}
	root := filepath.Join("..", "..")
	_, err := RunCaveman(t.Context(), CavemanOptions{InputDir: filepath.Join(root, "docs", "_witnesses", "armbench-caveman-native", "inputs"), OutDir: filepath.Join(root, "docs", "_witnesses", "armbench-caveman-native", "replacement-gpt-5.6-sol"), BaseURL: os.Getenv("OPENAI_BASE_URL"), APIKey: os.Getenv("OPENAI_API_KEY"), Model: "gpt-5.6-sol", Label: "replacement-gpt-5.6-sol", Trials: 3})
	if err != nil {
		t.Fatal(err)
	}
}

func TestCavemanRetiredExactModelWitness(t *testing.T) {
	if os.Getenv("FAK_CAVEMAN_RETIREMENT_LIVE") != "1" {
		t.Skip("set FAK_CAVEMAN_RETIREMENT_LIVE=1")
	}
	_, err := RunCaveman(t.Context(), CavemanOptions{InputDir: filepath.Join("..", "..", "docs", "_witnesses", "armbench-caveman-native", "inputs"), OutDir: t.TempDir(), BaseURL: os.Getenv("OPENAI_BASE_URL"), APIKey: os.Getenv("OPENAI_API_KEY"), Model: CavemanModel, Label: "exact-model", Trials: 3})
	if err == nil {
		t.Fatal("expected unavailable exact model")
	}
	if werr := os.WriteFile(filepath.Join("..", "..", "docs", "_witnesses", "armbench-caveman-native", "exact-model-unavailable.txt"), []byte(err.Error()+"\n"), 0644); werr != nil {
		t.Fatal(werr)
	}
	t.Log(err)
}

func TestCavemanNativeMediumCanonicalArmAndRawOutput(t *testing.T) {
	canonical, err := syspromptmmu.ResolveStyle("caveman:native:medium")
	if err != nil {
		t.Fatal(err)
	}
	wantDigest := sha256.Sum256([]byte(canonical.Segment))
	root := filepath.Join("..", "..", "docs", "_witnesses", "armbench-caveman-native", "inputs")
	p, err := RunCaveman(context.Background(), CavemanOptions{InputDir: root, OutDir: t.TempDir(), Model: "fixture", Label: "replacement-fixture", Trials: 3, DryRun: true})
	if err != nil {
		t.Fatal(err)
	}
	profile := p.Profiles["native_medium"]
	if profile.Identity != "caveman:native:medium" || profile.SHA256 != hex.EncodeToString(wantDigest[:]) {
		t.Fatalf("profile witness = %#v", profile)
	}
	if len(p.Calls) != 90 || len(p.Summary) != 3 || p.Summary[2].Arm != "native_medium" { //boundarylint:ignore CHANGE_DETECTOR_TEST the benchmark fixture is a fixed 30-prompt by 3-arm matrix with one summary per arm
		t.Fatalf("arms not paired: calls=%d summary=%#v", len(p.Calls), p.Summary)
	}
	for _, call := range p.Calls {
		if len(call.Raw) == 0 || !bytes.Contains(call.Raw, []byte(`"dry_run":true`)) {
			t.Fatalf("raw output missing for %#v", call)
		}
		if !call.SemanticPass {
			t.Fatalf("dry-run semantic gate failed for %s/%s: %v", call.PromptID, call.Arm, call.Missing)
		}
	}
}

func TestCavemanNativeMediumProviderReceivesCanonicalFragment(t *testing.T) {
	canonical, err := syspromptmmu.ResolveStyle("caveman:native:medium")
	if err != nil {
		t.Fatal(err)
	}
	var nativeCalls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var request struct {
			Messages []struct{ Role, Content string } `json:"messages"`
		}
		if err := json.Unmarshal(body, &request); err != nil {
			t.Error(err)
		}
		if len(request.Messages) != 2 {
			t.Errorf("messages=%#v", request.Messages)
		} else if request.Messages[0].Content == canonical.Segment {
			nativeCalls++
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"model":"fixture","choices":[{"message":{"content":"Reference identity memo useMemo seconds Date.now 1000 pool timeout error history rebase merge async await not found measure profile complexity operational SQL injection parameter placeholder FROM AS builder npm ci CMD atomic transaction returning update componentDidCatch getDerivedStateFromError retry log"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`)
	}))
	defer srv.Close()
	root := filepath.Join("..", "..", "docs", "_witnesses", "armbench-caveman-native", "inputs")
	if _, err := RunCaveman(context.Background(), CavemanOptions{InputDir: root, OutDir: t.TempDir(), BaseURL: srv.URL, APIKey: "fixture", Model: "fixture", Label: "replacement-fixture", Trials: 3}); err != nil {
		t.Fatal(err)
	}
	if nativeCalls != 30 {
		t.Fatalf("canonical native calls=%d, want 30", nativeCalls)
	}
}
