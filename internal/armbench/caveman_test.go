package armbench

import (
	"os"
	"path/filepath"
	"testing"
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
