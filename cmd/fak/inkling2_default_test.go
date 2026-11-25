package main

import (
	"bytes"
	"errors"
	"flag"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/macfit"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelreg"
)

// errTurnkeyLadderOnly closes the serving gate for tests that pin the ladder.
var errTurnkeyLadderOnly = errors.New("test: Inkling-2 serving gate closed")

func withRunDefaultModel(t *testing.T, ref string, ok bool) {
	t.Helper()
	old := runDefaultModel
	runDefaultModel = func() (string, bool) { return ref, ok }
	t.Cleanup(func() { runDefaultModel = old })
}

func withTurnkeyInkling2Servable(t *testing.T, err error) {
	t.Helper()
	old := turnkeyInkling2Servable
	turnkeyInkling2Servable = func() error { return err }
	t.Cleanup(func() { turnkeyInkling2Servable = old })
}

func TestTurnkeyMacDefaultTier(t *testing.T) {
	refused := errors.New("serve refuses inkling2_loop on metal")
	cases := []struct {
		name         string
		goos, goarch string
		memory       uint64
		servable     error
		want         string
	}{
		{"fits", "darwin", "arm64", 36 * macfit.GiB, nil, "inkling2-mini"},
		{"does not fit", "darwin", "arm64", 16 * macfit.GiB, nil, ""},
		{"non-darwin", "linux", "amd64", 64 * macfit.GiB, nil, ""},
		{"windows", "windows", "amd64", 64 * macfit.GiB, nil, ""},
		{"serve unsupported", "darwin", "arm64", 36 * macfit.GiB, refused, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			withTurnkeyInkling2Servable(t, tc.servable)
			tier, ok := turnkeyMacDefaultTierFor(tc.goos, tc.goarch, tc.memory)
			if ok != (tc.want != "") || tier.ModelID != tc.want {
				t.Fatalf("turnkeyMacDefaultTierFor = %q, %v; want %q", tier.ModelID, ok, tc.want)
			}
		})
	}
}

// The gate must be the serving surface's own answer, so the default follows it
// without a second switch to flip.
func TestTurnkeyInkling2ServableIsTheServeBackendGate(t *testing.T) {
	want := fakmodel.RefuseInkling2LoopBackend("metal")
	got := turnkeyInkling2Servable()
	if (got == nil) != (want == nil) {
		t.Fatalf("turnkeyInkling2Servable = %v, serve gate = %v", got, want)
	}
}

func TestOpenTurnkeyInkling2RefusesWithServeGate(t *testing.T) {
	refused := errors.New("serve refuses inkling2_loop on metal")
	withTurnkeyInkling2Servable(t, refused)
	if _, _, err := openTurnkeyInkling2(filepath.Join(t.TempDir(), "absent.gguf"), "inkling2-mini"); !errors.Is(err, refused) {
		t.Fatalf("openTurnkeyInkling2 error = %v, want the serve refusal", err)
	}
}

func TestRunDefaultModelFor(t *testing.T) {
	// The bare-run default does not consult the serving gate.
	withTurnkeyInkling2Servable(t, errors.New("serve refuses"))
	cases := []struct {
		name         string
		goos, goarch string
		memory       uint64
		want         string
		wantOK       bool
	}{
		{"mac fits", "darwin", "arm64", 36 * macfit.GiB, modelreg.MacDefaultAlias, true},
		{"mac 24 GiB fits", "darwin", "arm64", 24 * macfit.GiB, modelreg.MacDefaultAlias, true},
		{"mac 16 GiB falls back to ladder", "darwin", "arm64", 16 * macfit.GiB, "qwen2.5:7b", true},
		{"mac 8 GiB falls back to ladder", "darwin", "arm64", 8 * macfit.GiB, "qwen2.5:7b", true},
		{"linux has none", "linux", "amd64", 64 * macfit.GiB, "", false},
		{"windows has none", "windows", "amd64", 64 * macfit.GiB, "", false},
		{"intel mac has none", "darwin", "amd64", 64 * macfit.GiB, "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := runDefaultModelFor(tc.goos, tc.goarch, tc.memory)
			if got != tc.want || ok != tc.wantOK {
				t.Fatalf("runDefaultModelFor = %q, %v; want %q, %v", got, ok, tc.want, tc.wantOK)
			}
		})
	}
}

func TestRunBareUsesPlatformDefault(t *testing.T) {
	withRunDefaultModel(t, "inkling2-mini", true)
	existing := filepath.Join(t.TempDir(), "my model.gguf")
	if err := os.WriteFile(existing, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		argv        []string
		wantModel   string
		wantPrompt  string
		wantDefault bool
	}{
		{argv: nil, wantModel: "inkling2-mini", wantDefault: true},
		{argv: []string{"--temp", "0"}, wantModel: "inkling2-mini", wantDefault: true},
		{argv: []string{"--effort", "none", "--temp", "0", "--max-tokens", "16", "Say hi"}, wantModel: "inkling2-mini", wantPrompt: "Say hi", wantDefault: true},
		{argv: []string{"Say hi"}, wantModel: "inkling2-mini", wantPrompt: "Say hi", wantDefault: true},
		{argv: []string{"Say hi", "--temp", "0", "to me"}, wantModel: "inkling2-mini", wantPrompt: "Say hi to me", wantDefault: true},
		// A named model always wins over the default.
		{argv: []string{"smollm2"}, wantModel: "smollm2"},
		{argv: []string{"smollm2", "Say hi"}, wantModel: "smollm2", wantPrompt: "Say hi"},
		{argv: []string{"--temp", "0", "smollm2", "Say hi"}, wantModel: "smollm2", wantPrompt: "Say hi"},
		{argv: []string{"--temp", "0", "hello"}, wantModel: "hello"},
		{argv: []string{existing, "hi"}, wantModel: existing, wantPrompt: "hi"},
		{argv: []string{"--temp", "0", existing}, wantModel: existing},
	}
	for _, c := range cases {
		fs, flags := newRunFlagSet("run", flag.ContinueOnError)
		cmd, err := parseRunArgs(fs, flags, c.argv)
		if err != nil {
			t.Errorf("argv=%v: %v", c.argv, err)
			continue
		}
		if cmd.action != runActionChat || cmd.modelRef != c.wantModel || cmd.chatConfig.modelRef != c.wantModel ||
			cmd.prompt != c.wantPrompt || cmd.chatConfig.prompt != c.wantPrompt || cmd.defaultModel != c.wantDefault {
			t.Errorf("argv=%v: action=%v model=%q prompt=%q default=%v; want chat model=%q prompt=%q default=%v",
				c.argv, cmd.action, cmd.modelRef, cmd.prompt, cmd.defaultModel, c.wantModel, c.wantPrompt, c.wantDefault)
		}
	}
}

func TestRunBareTraceStillWinsOverDefault(t *testing.T) {
	withRunDefaultModel(t, "inkling2-mini", true)
	fs, flags := newRunFlagSet("run", flag.ContinueOnError)
	cmd, err := parseRunArgs(fs, flags, []string{"--trace", "x.json"})
	if err != nil || cmd.action != runActionTrace || cmd.tracePath != "x.json" {
		t.Fatalf("trace replay = %+v, %v", cmd, err)
	}
}

func TestRunWithoutPlatformDefaultKeepsUsageError(t *testing.T) {
	withRunDefaultModel(t, "", false)
	for _, argv := range [][]string{nil, {"--temp", "0"}} {
		fs, flags := newRunFlagSet("run", flag.ContinueOnError)
		cmd, err := parseRunArgs(fs, flags, argv)
		if err != nil || cmd.action != runActionUsage {
			t.Fatalf("argv=%v: action=%v err=%v, want the usage error", argv, cmd.action, err)
		}
	}
	// Without a default, a spaced first argument is still read as the model.
	fs, flags := newRunFlagSet("run", flag.ContinueOnError)
	cmd, err := parseRunArgs(fs, flags, []string{"--temp", "0", "Say hi"})
	if err != nil || cmd.action != runActionChat || cmd.modelRef != "Say hi" || cmd.defaultModel {
		t.Fatalf("spaced argument without a default = %+v, %v", cmd, err)
	}
}

func TestTurnkeyInkling2TierResolvesToConsistentArtifact(t *testing.T) {
	t.Setenv("FAK_MODELS_DIR", t.TempDir())
	tier := macfit.Inkling2MiniTier
	ref := resolveTurnkeyModelRef(tier.ModelID)
	uri, expanded := modelreg.Resolve(ref)
	if !expanded || uri != modelreg.Catalog[modelreg.MacDefaultAlias] {
		t.Fatalf("tier model %q resolved to %q (expanded=%v)", tier.ModelID, uri, expanded)
	}
	art := describeTurnkeyArtifact(uri, turnkeyRefIsExplicit(tier.ModelID))
	if art.Quant != tier.QuantTier {
		t.Fatalf("artifact quant = %q, tier = %q", art.Quant, tier.QuantTier)
	}
	if err := validateTurnkeyArtifact(tier, art); err != nil {
		t.Fatal(err)
	}
	// The pinned file's real size sits inside the tier's resident budget band.
	art.SizeBytes = 13_631_779_296
	if err := validateTurnkeyArtifact(tier, art); err != nil {
		t.Fatal(err)
	}
	if got, ok := macfit.LookupModelTier("inkling2-mini"); !ok || got.ModelID != tier.ModelID {
		t.Fatalf("--model inkling2-mini does not name the tier: %+v, %v", got, ok)
	}
	if budget := gateway.DefaultAdmissionPolicy().TokenBudget; tier.FixedContextTokens != uint64(budget) {
		t.Fatalf("tier context %d differs from the launcher's admitted session budget %d", tier.FixedContextTokens, budget)
	}
}

func TestTurnkeyInkling2OverrideKeepsEngineContext(t *testing.T) {
	const memory = 36 * macfit.GiB
	plan, err := macfit.ConfigureTurnkeyWithOptions(memory, macfit.TurnkeyOptions{StaticOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := applyTurnkeyModelOverride(&plan, memory, "inkling2-mini"); err != nil {
		t.Fatal(err)
	}
	if plan.Tier.ModelID != "inkling2-mini" || plan.ContextBudgetTokens != macfit.Inkling2MiniMinContextTokens {
		t.Fatalf("override plan = %s at %d tokens", plan.Tier.ModelID, plan.ContextBudgetTokens)
	}
	if plan.HeadroomRatio < macfit.DefaultMinHeadroomRatio {
		t.Fatalf("headroom = %.3f", plan.HeadroomRatio)
	}
}

// The dry-run plan is the selection `fak up` serves: the Mac default only on
// Apple Silicon, only when it fits, and only while the serving gate is open.
func TestFakUpDryRunFollowsServingGate(t *testing.T) {
	t.Setenv("FAK_MODELS_DIR", t.TempDir())
	t.Setenv("FAK_MODEL_DIR", t.TempDir())
	t.Setenv("FAK_UP_STATIC_ONLY", "1")
	plan := func(t *testing.T, memoryGiB string) string {
		t.Helper()
		var stdout, stderr bytes.Buffer
		runTurnkeyUp(nil, &stdout, &stderr, []string{"--dry-run", "--memory-gib", memoryGiB})
		if stderr.Len() != 0 {
			t.Fatalf("fak up dry-run stderr = %q", stderr.String())
		}
		return stdout.String()
	}
	mac := runtime.GOOS == "darwin" && runtime.GOARCH == "arm64"

	withTurnkeyInkling2Servable(t, nil)
	wantOpen := "Model Tier     : 27B"
	if mac {
		wantOpen = "Model Tier     : Inkling-2-Mini (inkling2-mini, quant: Q3_K_XL)"
	}
	if out := plan(t, "36"); !strings.Contains(out, wantOpen) {
		t.Fatalf("gate open, 36 GiB: missing %q:\n%s", wantOpen, out)
	}
	if out := plan(t, "16"); !strings.Contains(out, "Model Tier     : 7B") {
		t.Fatalf("gate open, 16 GiB must stay on the ladder:\n%s", out)
	}

	withTurnkeyInkling2Servable(t, errTurnkeyLadderOnly)
	if out := plan(t, "36"); !strings.Contains(out, "Model Tier     : 27B") {
		t.Fatalf("gate closed, 36 GiB must stay on the ladder:\n%s", out)
	}
}
