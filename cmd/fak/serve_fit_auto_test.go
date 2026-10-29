// Tests for `fak serve --fit auto` (T04, ticket model-fit-auto-planner-20261023-04):
// the flag surface, typed startup refusals, pin precedence over the search, and the
// pre-load joint-search receipt. The CLI cases follow the --native-allow-bash-command
// precedent: cmdServe runs in a child test binary so flag parsing, validation and
// exit codes are witnessed the way an operator sees them.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fitplan"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// fak-test:runtime fast est=5ms lane=default
func TestServeFitAutoFlagDefaultsAndParse(t *testing.T) {
	fs, sf := newServeFlagSet()
	if err := fs.Parse(nil); err != nil {
		t.Fatal(err)
	}
	if *sf.fit != "off" || *sf.workloadPromptTokens != 0 || *sf.workloadOutputTokens != 0 ||
		*sf.workloadConcurrency != 0 || *sf.sloTPOTMS != 0 || *sf.sloTTFTMS != 0 {
		t.Fatalf("defaults = fit %q prompt %d output %d conc %d tpot %v ttft %v, want off/0/0/0/0/0",
			*sf.fit, *sf.workloadPromptTokens, *sf.workloadOutputTokens, *sf.workloadConcurrency, *sf.sloTPOTMS, *sf.sloTTFTMS)
	}

	fs, sf = newServeFlagSet()
	err := fs.Parse([]string{
		"--fit", "auto",
		"--workload-prompt-tokens", "4096",
		"--workload-output-tokens", "512",
		"--workload-concurrency", "2",
		"--slo-tpot-ms", "50",
		"--slo-ttft-ms", "2000",
		"--native-context-tokens", "8192",
		"--kv-precision", "q8_0",
	})
	if err != nil {
		t.Fatal(err)
	}
	if *sf.fit != "auto" || *sf.workloadPromptTokens != 4096 || *sf.workloadOutputTokens != 512 ||
		*sf.workloadConcurrency != 2 || *sf.sloTPOTMS != 50 || *sf.sloTTFTMS != 2000 {
		t.Fatalf("parsed = fit %q prompt %d output %d conc %d tpot %v ttft %v",
			*sf.fit, *sf.workloadPromptTokens, *sf.workloadOutputTokens, *sf.workloadConcurrency, *sf.sloTPOTMS, *sf.sloTTFTMS)
	}
	pins := serveFitAutoPins(sf)
	if pins.ContextTokens != 8192 || pins.KVPrecision != "q8" {
		t.Fatalf("pins %+v, want ctx 8192 kv q8 (explicit flags pin their axes)", pins)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestServeFitAutoPinsAreExplicitOnly(t *testing.T) {
	fs, sf := newServeFlagSet()
	if err := fs.Parse(nil); err != nil {
		t.Fatal(err)
	}
	if pins := serveFitAutoPins(sf); pins != (fitplan.SearchPins{}) {
		t.Fatalf("unset flags pinned %+v, want zero pins (the search owns unpinned axes)", pins)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestServeFitAutoRequestDefaults(t *testing.T) {
	req := serveFitAutoRequestFromFlags("auto", 0, 0, 0, 0, 0)
	if !req.Enabled || req.Workload.PromptTokens != fitplan.DefaultPromptTokens ||
		req.Workload.OutputTokens != fitplan.DefaultOutputTokens || req.Workload.Concurrency != 1 {
		t.Fatalf("defaults %+v, want enabled %d/%d x1", req, fitplan.DefaultPromptTokens, fitplan.DefaultOutputTokens)
	}
	req = serveFitAutoRequestFromFlags("off", 4096, 512, 2, 50, 2000)
	if req.Enabled {
		t.Fatalf("fit off must not enable the search")
	}
	req = serveFitAutoRequestFromFlags("auto", 4096, 512, 2, 50, 2000)
	if req.Workload.PromptTokens != 4096 || req.Workload.OutputTokens != 512 || req.Workload.Concurrency != 2 ||
		req.SLO.TPOTMS != 50 || req.SLO.TTFTMS != 2000 {
		t.Fatalf("stated workload not honored: %+v", req)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestServeFitAutoValidateRefusesBadTokens(t *testing.T) {
	for _, tc := range []struct {
		name string
		args [6]any
		want string
	}{
		{name: "unknown fit token", args: [6]any{"bogus", 0, 0, 0, 0.0, 0.0}, want: "--fit"},
		{name: "negative prompt", args: [6]any{"auto", -1, 0, 0, 0.0, 0.0}, want: "--workload-*"},
		{name: "negative concurrency", args: [6]any{"auto", 0, 0, -3, 0.0, 0.0}, want: "--workload-*"},
		{name: "negative tpot", args: [6]any{"auto", 0, 0, 0, -0.5, 0.0}, want: "--slo-tpot-ms"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := validateServeFitFlags(tc.args[0].(string), tc.args[1].(int), tc.args[2].(int), tc.args[3].(int), tc.args[4].(float64), tc.args[5].(float64))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("validate = %v, want refusal naming %s", err, tc.want)
			}
		})
	}
	for _, fit := range []string{"", "off", "auto"} {
		if err := validateServeFitFlags(fit, 0, 0, 0, 0, 0); err != nil {
			t.Fatalf("fit %q: %v, want valid", fit, err)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestServeFitAutoApplyHonorsPins(t *testing.T) {
	mk := func(ctx int, kv string) *serveFlags {
		return &serveFlags{nativeContextTokens: &ctx, kvPrecision: &kv}
	}
	chosen := &fitplan.SearchReport{Chosen: &fitplan.SearchRow{KVPrecision: "q8", ContextTokens: 4096, Slots: 4, BStar: 4}}

	sf := mk(0, "")
	applyServeFitAutoChoice(sf, chosen, fitplan.SearchPins{})
	if *sf.nativeContextTokens != 4096 || *sf.kvPrecision != "q8_0" {
		t.Fatalf("unpinned apply = ctx %d kv %q, want 4096/q8_0", *sf.nativeContextTokens, *sf.kvPrecision)
	}

	ctx, kv := 8192, "f32"
	sf = mk(ctx, kv)
	applyServeFitAutoChoice(sf, chosen, fitplan.SearchPins{ContextTokens: 8192, KVPrecision: "f32"})
	if *sf.nativeContextTokens != 8192 || *sf.kvPrecision != "f32" {
		t.Fatalf("pinned apply moved the operator's axes: ctx %d kv %q, want 8192/f32 untouched", *sf.nativeContextTokens, *sf.kvPrecision)
	}

	applyServeFitAutoChoice(mk(0, ""), &fitplan.SearchReport{}, fitplan.SearchPins{}) // no chosen row: nothing applies
}

// writeServeFitAutoFixture is the packed-embedding fixture with a stated context length:
// the T04 search needs MaxContext > 0 or every KV tier fails kvSizable and the whole
// search degenerates to context_unknown. Everything else mirrors
// writeServePackedEmbeddingFixture so the arm, quant family and model shape stay identical.
func writeServeFitAutoFixture(t *testing.T, base string) string {
	t.Helper()
	const (
		dim          = 256
		vocab        = 3
		q4BlockBytes = 144
		q6BlockBytes = 210
	)
	embedBytes := uint64(vocab * q4BlockBytes)
	outputBytes := uint64(vocab * q6BlockBytes)
	linearBytes := uint64(dim * q4BlockBytes)
	align32 := func(n uint64) uint64 { return (n + 31) &^ 31 }

	var b bytes.Buffer
	writeMinimalHeaderForTest(&b, 5, 11)
	writeKVStringForTest(&b, "general.architecture", "qwen35")
	writeKVUint32ForTest(&b, "general.alignment", 32)
	writeKVUint32ForTest(&b, "qwen35.context_length", 8192)
	writeKVUint32ForTest(&b, "qwen35.embedding_length", dim)
	writeKVUint32ForTest(&b, "qwen35.block_count", 2)
	writeKVUint32ForTest(&b, "qwen35.attention.head_count", 1)
	writeKVUint32ForTest(&b, "qwen35.attention.head_count_kv", 1)
	// Interval 1 keeps every block full-attention: a hybrid (linear-attention) shape
	// would zero the KV layer count and every tier would fail kvSizable.
	writeKVUint32ForTest(&b, "qwen35.full_attention_interval", 1)
	writeKVUint32ForTest(&b, "qwen35.attention.key_length", dim)
	writeKVUint32ForTest(&b, "qwen35.feed_forward_length", dim)
	writeKVFloat32ForTest(&b, "qwen35.attention.layer_norm_rms_epsilon", 1e-6)

	offset := uint64(0)
	writeTensorInfoForTest(&b, "token_embd.weight", []uint64{dim, vocab}, uint32(ggufload.TensorQ4_K), offset)
	offset = align32(offset + embedBytes)
	writeTensorInfoForTest(&b, "output.weight", []uint64{dim, vocab}, uint32(ggufload.TensorQ6_K), offset)
	offset = align32(offset + outputBytes)
	for _, name := range []string{"blk.0.attn_v.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight"} {
		writeTensorInfoForTest(&b, name, []uint64{dim, dim}, uint32(ggufload.TensorQ4_K), offset)
		offset = align32(offset + linearBytes)
	}
	padToAlignmentForTest(&b, 32)
	b.Write(make([]byte, int(offset)))

	path := filepath.Join(t.TempDir(), base)
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// fitAutoSeedStore writes a calibration store holding ONE decode receipt keyed to this
// host's epoch under every CPU kernel-path label the fixture's arm can take, so the
// child process prices rows at a known bandwidth instead of failing open on unknown
// peaks. The receipt is the evidence the fitter refits into its rows.
func fitAutoSeedStore(t *testing.T, fixture string) string {
	t.Helper()
	ws, err := ggufload.OpenWeights(fixture)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	embed, opts := serveFitPlanEmbed(ws, nil, resolveServeNativeContextLoadArm(ws, nil, false, false), false, fixture, serveFitBudget{})
	m, err := ws.FitPlanModel(embed, opts...)
	if err != nil {
		t.Fatal(err)
	}
	const bw = 30e9 // effective bandwidth the seeded step implies
	receipts := make([]fitplan.CalibrationReceipt, 0, len([]string{"cpu-lean-q8", "cpu-resident-q4k", "cpu-f32"}))
	for _, kp := range []string{"cpu-lean-q8", "cpu-resident-q4k", "cpu-f32"} {
		receipts = append(receipts, fitplan.CalibrationReceipt{
			Key: fitplan.CalibrationKey{
				Epoch:       fitPlanHardwareEpoch(nil).Digest(),
				Backend:     "fak-cpu",
				QuantFamily: m.QuantFamily,
				KernelPath:  kp,
				BatchBucket: fitplan.BatchBucket(1),
			},
			Kind:         fitplan.ReceiptDecode,
			Label:        "fit-auto-test-seed",
			BytesPerStep: 4e9,
			StepSeconds:  4e9 / bw,
			MeasuredAt:   time.Now().UTC().Format(time.RFC3339),
			Provenance:   fitplan.Provenance{Path: "fit-auto-test-seed.json", SHA256: strings.Repeat("ab", 32)},
		})
	}
	store := fitplan.NewCalibrationStore()
	store.Add(receipts...)
	raw, err := json.Marshal(store)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "calibration.json")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// runServeFitAutoCLI runs cmdServe in a child test binary with the given serve args and
// returns stdout, stderr and the exit code. The child gets a clean calibration store
// (seeded when non-empty) so pricing is deterministic regardless of the operator's own
// store, and FAK_Q4K cleared so the load arm is the lean-q8 default.
func runServeFitAutoCLI(t *testing.T, calibrationStore string, args ...string) (string, string, int) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	childArgs := []string{"-test.run=^TestServeFitAutoCLIHelper$", "--"}
	args = append([]string{"--mock"}, args...)
	childArgs = append(childArgs, args...)
	cmd := exec.CommandContext(ctx, os.Args[0], childArgs...)
	// The package init() pins FAK_FITPLAN_CALIBRATION to an empty per-binary store for the WHOLE
	// test binary, child included -- so the store path travels under its own name and the helper
	// re-points the real variable after init has run. FAK_Q4K is filtered so the operator's
	// machine cannot flip the load arm out from under the fixture's pricing.
	env := make([]string, 0, len(os.Environ())+3)
	for _, e := range os.Environ() {
		if strings.HasPrefix(e, "FAK_Q4K=") {
			continue
		}
		env = append(env, e)
	}
	env = append(env, "FAK_SERVE_FIT_AUTO_CLI_HELPER=1", "FAK_Q4K=")
	if calibrationStore != "" {
		env = append(env, "FAK_SERVE_FIT_AUTO_STORE="+calibrationStore)
	}
	var stdout, stderr bytes.Buffer
	cmd.Env = env
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	if ctx.Err() != nil {
		t.Fatalf("fit auto CLI reached startup instead of exiting: %v\n%s\n%s", ctx.Err(), stdout.String(), stderr.String())
	}
	code := 0
	if ee, ok := err.(*exec.ExitError); ok {
		code = ee.ExitCode()
	}
	return stdout.String(), stderr.String(), code
}

// fak-test:runtime slow est=10s lane=default
func TestServeFitAutoCLIValidatesBeforeStartup(t *testing.T) {
	out, errOut, code := runServeFitAutoCLI(t, "", "--fit", "bogus")
	if code == 0 {
		t.Fatalf("--fit bogus succeeded: %s", out)
	}
	if !strings.Contains(errOut, "--fit") {
		t.Fatalf("refusal omitted the offending flag: %s", errOut)
	}
	_, errOut, code = runServeFitAutoCLI(t, "", "--workload-prompt-tokens=-1")
	if code == 0 || !strings.Contains(errOut, "--workload-*") {
		t.Fatalf("negative workload: code %d stderr %q, want refusal naming the flag", code, errOut)
	}
}

// fak-test:runtime slow est=10s lane=default
func TestServeFitAutoCLILiveBootRefusesTyped(t *testing.T) {
	fixture := writeServeFitAutoFixture(t, "fit-auto-refuse.gguf")
	store := fitAutoSeedStore(t, fixture)
	// A million concurrent streams fits no pool: the search must refuse with the typed
	// concurrency reason and exit 2 BEFORE any load, whatever the host size.
	_, errOut, code := runServeFitAutoCLI(t, store,
		"--gguf", fixture, "--fit", "auto", "--workload-concurrency", "1000000")
	if code != 2 {
		t.Fatalf("exit %d, want 2 (a refused live boot never loads)\nstderr:\n%s", code, errOut)
	}
	if !strings.Contains(errOut, "concurrency_exceeds_pool") {
		t.Fatalf("refusal did not name the typed reason:\n%s", errOut)
	}
}

// fak-test:runtime slow est=10s lane=default
func TestServeFitAutoCLIPlanJSONCarriesSearch(t *testing.T) {
	fixture := writeServeFitAutoFixture(t, "fit-auto-plan.gguf")
	store := fitAutoSeedStore(t, fixture)
	stdout, errOut, code := runServeFitAutoCLI(t, store,
		"--gguf", fixture, "--plan-json", "--fit", "auto",
		"--workload-prompt-tokens", "1024", "--workload-output-tokens", "256",
		"--native-context-tokens", "4096", "--kv-precision", "q8_0")
	if code != 0 {
		t.Fatalf("exit %d, want 0\nstderr:\n%s", code, errOut)
	}
	var art struct {
		Version string                `json:"version"`
		FitAuto *fitplan.SearchReport `json:"fit_auto"`
	}
	if err := json.Unmarshal([]byte(stdout), &art); err != nil {
		t.Fatalf("artifact is not JSON: %v\nstdout:\n%s\nstderr:\n%s", err, stdout, errOut)
	}
	if art.Version != serveSizingVersion || art.FitAuto == nil {
		t.Fatalf("artifact version %q fit_auto %+v, want %q with the search report", art.Version, art.FitAuto, serveSizingVersion)
	}
	rep := art.FitAuto
	if rep.Workload.PromptTokens != 1024 || rep.Workload.OutputTokens != 256 || rep.Workload.Concurrency != 1 {
		t.Fatalf("workload %+v, want the stated 1024/256 x1", rep.Workload)
	}
	if rep.Pins.KVPrecision != "q8" || rep.Pins.ContextTokens != 4096 {
		t.Fatalf("pins %+v, want ctx 4096 + kv q8 (explicit flags pin their axes)", rep.Pins)
	}
	if rep.Status != fitplan.StatusFits || rep.Chosen == nil {
		t.Fatalf("status %s chosen %+v refusal %+v omitted %v, want a chosen row on a priceable host",
			rep.Status, rep.Chosen, rep.Refusal, rep.Omitted)
	}
	ch := rep.Chosen
	if ch.KVPrecision != "q8" {
		t.Fatalf("chosen kv %q broke the q8 pin", ch.KVPrecision)
	}
	if ch.ContextTokens != 4096 {
		t.Fatalf("chosen ctx %d moved the pinned 4096 axis", ch.ContextTokens)
	}
	if ch.ContextTokens < rep.Workload.PromptTokens || ch.BStar < 1 || ch.AggregateTokS <= 0 {
		t.Fatalf("chosen %+v, want ctx >= ISL, B* >= 1, positive aggregate", ch)
	}
	if !(ch.TTFTSeconds > 0) || ch.TTFTSeconds != ch.TTFTSeconds {
		t.Fatalf("ttft %v, want positive finite", ch.TTFTSeconds)
	}
	for _, a := range rep.Alternatives {
		if a.KVPrecision != "q8" {
			t.Fatalf("alternative %+v left the pinned kv axis", a)
		}
	}
}

// TestServeFitAutoCLIHelper is the child side of the CLI tests above: it runs cmdServe
// directly on the args after the "--" separator, exactly as the operator's shell would.
func TestServeFitAutoCLIHelper(t *testing.T) {
	if os.Getenv("FAK_SERVE_FIT_AUTO_CLI_HELPER") != "1" {
		return
	}
	if p := os.Getenv("FAK_SERVE_FIT_AUTO_STORE"); p != "" {
		_ = os.Setenv(calibrationStoreEnv, p)
	}
	for i, arg := range os.Args {
		if arg == "--" {
			cmdServe(os.Args[i+1:])
			// A dry-run verb (--plan-json) returns through cmdServe instead of exiting;
			// stop here so the test framework cannot print its PASS into the artifact.
			os.Exit(0)
		}
	}
	os.Exit(97)
}
