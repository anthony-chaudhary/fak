package bitnetruntime

// golden_test.go is the #6243 witness (epic #6221). It drives the whole public
// path — a FAKE BitNet runtime process whose probe output is a committed
// fixture — and compares the typed result against a hand-authored golden.
//
// The fixtures are built to make the four failure modes this leaf exists to
// prevent FAIL LOUDLY rather than to make the happy path pass:
//
//   - version discovery: kernels_undeclared carries a perfectly good version
//     banner and no kernel line, so an implementation that reads "a recent
//     version implies the usual kernels" admits a runtime it never probed;
//   - model validation: model_family_int4 is a well-formed request for a
//     four-level artifact, so an implementation that treats "low-bit" as one
//     bucket delegates a model bitnet.cpp cannot run;
//   - CPU dispatch: cpu_feature_missing and kernel_arch_mismatch differ from
//     the delegating fixtures ONLY in a host field, so an implementation that
//     dispatches on the runtime's kernel list alone admits both;
//   - explicit unsupported: every non-delegating fixture names its own reason
//     code, so a silent fallback to a "safe" kernel shows up as a diff.
//
// Every expectation is an independently readable file under testdata/ — the
// result is compared against a hand-authored golden, never regenerated from the
// code under test.

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

// fixtures is the explicit witness set. It is stated here rather than globbed so
// that a deleted fixture fails loudly instead of shrinking the witness silently;
// TestFixtureSetMatchesDisk binds it to what is actually on disk.
var fixtures = []string{
	"cpu_feature_missing",
	"delegate_darwin_arm64_tl1",
	"delegate_linux_amd64_tl2",
	"delegate_windows_amd64_i2s",
	"host_arch_undeclared",
	"host_os_unsupported",
	"kernel_arch_mismatch",
	"kernel_not_built",
	"kernels_undeclared",
	"model_family_int4",
	"model_family_undeclared",
	"packing_narrower_than_kernel",
	"probe_conflict",
	"probe_empty",
	"probe_failed",
	"runtime_too_old",
}

// fixtureInput is one fake-runtime scenario: the bytes the delegate's probe
// would print, the host it would print them on, and the model asked for. Probe
// bytes are held as a string so the fixture stays readable as a file.
type fixtureInput struct {
	Probe      string `json:"probe"`
	ProbeError string `json:"probe_error,omitempty"`
	Host       Host   `json:"host"`
	Model      Model  `json:"model"`
}

func readFixture(t *testing.T, name, suffix string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name+suffix))
	if err != nil {
		t.Fatalf("read fixture %s%s: %v", name, suffix, err)
	}
	return raw
}

// fakeRuntime turns a fixture into a Prober — the fake BitNet process. A
// declared probe_error stands in for a runtime that is absent or refused to
// start, which is a different answer from one that started and said nothing.
func fakeRuntime(in fixtureInput) Prober {
	return func(context.Context) ([]byte, error) {
		if in.ProbeError != "" {
			return nil, errors.New(in.ProbeError)
		}
		return []byte(in.Probe), nil
	}
}

func loadFixture(t *testing.T, name string) fixtureInput {
	t.Helper()
	var in fixtureInput
	if err := json.Unmarshal(readFixture(t, name, ".input.json"), &in); err != nil {
		t.Fatalf("decode fixture %s: %v", name, err)
	}
	return in
}

// admitFixture runs the full public path a caller would use: probe the fake
// runtime, then adjudicate the request against what it reported.
func admitFixture(t *testing.T, name string) Result {
	t.Helper()
	in := loadFixture(t, name)
	return DiscoverAndAdmit(context.Background(), fakeRuntime(in), in.Host, in.Model)
}

// TestGoldenResults compares each fixture's adjudication against its committed
// golden. This is the whole witness in one assertion per case: outcome, reason
// vocabulary, claim class, and every field of the delegation record.
func TestGoldenResults(t *testing.T) {
	for _, name := range fixtures {
		t.Run(name, func(t *testing.T) {
			gotJSON, err := json.Marshal(admitFixture(t, name))
			if err != nil {
				t.Fatalf("marshal result: %v", err)
			}
			var got, want any
			if err := json.Unmarshal(gotJSON, &got); err != nil {
				t.Fatalf("re-read result: %v", err)
			}
			if err := json.Unmarshal(readFixture(t, name, ".golden.json"), &want); err != nil {
				t.Fatalf("decode golden: %v", err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Errorf("result mismatch\n got: %s\nwant: %s", gotJSON, readFixture(t, name, ".golden.json"))
			}
		})
	}
}

// TestFixtureSetMatchesDisk binds the declared witness set to the files on disk,
// so neither a forgotten fixture nor a deleted one shrinks the witness quietly.
func TestFixtureSetMatchesDisk(t *testing.T) {
	entries, err := os.ReadDir("testdata")
	if err != nil {
		t.Fatalf("read testdata: %v", err)
	}
	var onDisk []string
	for _, e := range entries {
		name, ok := strings.CutSuffix(e.Name(), ".input.json")
		if !ok {
			continue
		}
		onDisk = append(onDisk, name)
		if _, err := os.Stat(filepath.Join("testdata", name+".golden.json")); err != nil {
			t.Errorf("fixture %s has no golden: %v", name, err)
		}
	}
	sort.Strings(onDisk)
	declared := append([]string(nil), fixtures...)
	sort.Strings(declared)
	if !reflect.DeepEqual(onDisk, declared) {
		t.Errorf("declared fixture set != testdata on disk\n disk: %v\ndeclared: %v", onDisk, declared)
	}
}

// TestEveryReasonIsPublished catches a reason code emitted by the contract but
// missing from the published vocabulary — the drift that makes a caller's
// switch on Reason silently fall through.
func TestEveryReasonIsPublished(t *testing.T) {
	for _, name := range fixtures {
		for _, r := range admitFixture(t, name).Reasons {
			if !r.Known() {
				t.Errorf("fixture %s emitted unpublished reason %q", name, r)
			}
		}
	}
}

// TestOutcomeIsTypedForEveryFixture is the "no silent fallback" half: every
// input lands on one of the four declared outcomes, and only a delegation
// carries a kernel and a runtime-delegated claim.
func TestOutcomeIsTypedForEveryFixture(t *testing.T) {
	for _, name := range fixtures {
		t.Run(name, func(t *testing.T) {
			got := admitFixture(t, name)
			switch got.Outcome {
			case OutcomeDelegate, OutcomeUnsupported, OutcomeAbstain, OutcomeRefuse:
			default:
				t.Fatalf("outcome %q is outside the declared vocabulary", got.Outcome)
			}
			if len(got.Reasons) == 0 {
				t.Fatal("result carries no reason; every outcome must name why")
			}
			if got.Outcome == OutcomeDelegate {
				if got.Claim != ClaimRuntimeDelegated {
					t.Errorf("delegation licensed claim %q, want %q", got.Claim, ClaimRuntimeDelegated)
				}
				if got.Delegation.Kernel == KernelUnknown || got.Delegation.Kernel == "" {
					t.Error("delegation names no kernel; the caller cannot dispatch on it")
				}
				return
			}
			if got.Claim != ClaimNone {
				t.Errorf("non-delegation licensed claim %q, want none", got.Claim)
			}
			if got.Delegation.Kernel != KernelUnknown {
				t.Errorf("non-delegation named kernel %q; only an admitted request selects one", got.Delegation.Kernel)
			}
		})
	}
}

// TestNeverClaimsHardwareOrArtifact is the guardrail from epic #6221 stated as a
// test: this leaf adjudicates a RUNTIME DELEGATION and nothing else. It may not
// license a statement about what the artifact is, how it was produced, or how
// fast it runs — those are bitnetmeta's and a measured benchmark's to make.
func TestNeverClaimsHardwareOrArtifact(t *testing.T) {
	forbidden := map[ClaimClass]string{
		ClaimArtifactDescribed: "artifact description belongs to internal/bitnetmeta",
		ClaimRecipeDescribed:   "production recipe belongs to internal/bitnetmeta",
		ClaimHardwareEnvelope:  "a capability probe cannot measure hardware",
	}
	for _, name := range fixtures {
		got := admitFixture(t, name).Claim
		if why, bad := forbidden[got]; bad {
			t.Errorf("fixture %s licensed claim %q: %s", name, got, why)
		}
	}
}

// TestDispatchTurnsOnTheHostAlone pins the CPU-dispatch axis directly rather
// than through a fixture pair: the same admitted runtime and model must stop
// being delegable the moment the host loses the feature the kernel needs.
func TestDispatchTurnsOnTheHostAlone(t *testing.T) {
	in := loadFixture(t, "delegate_linux_amd64_tl2")
	stripped := in.Host
	stripped.Features = nil

	if got := DiscoverAndAdmit(context.Background(), fakeRuntime(in), in.Host, in.Model); got.Outcome != OutcomeDelegate {
		t.Fatalf("baseline outcome = %q, want delegate", got.Outcome)
	}
	got := DiscoverAndAdmit(context.Background(), fakeRuntime(in), stripped, in.Model)
	if got.Outcome != OutcomeUnsupported || !got.HasReason(ReasonCPUFeatureMissing) {
		t.Fatalf("host without avx2: outcome %q reasons %v, want unsupported/%s",
			got.Outcome, got.Reasons, ReasonCPUFeatureMissing)
	}
}

// TestUnknownRuntimeKernelIsNotUsable proves a forward-compatible kernel token
// is neither an error nor a licence: a runtime advertising a kernel this
// contract has never heard of may still serve the kernels it does know, and may
// not serve the unknown one.
func TestUnknownRuntimeKernelIsNotUsable(t *testing.T) {
	in := loadFixture(t, "delegate_linux_amd64_tl2")
	in.Probe = strings.Replace(in.Probe, "kernels: i2_s,tl2", "kernels: i2_s,tl2,tl9", 1)
	if got := DiscoverAndAdmit(context.Background(), fakeRuntime(in), in.Host, in.Model); got.Outcome != OutcomeDelegate {
		t.Fatalf("an extra unknown kernel token broke a known-kernel delegation: %q %v", got.Outcome, got.Reasons)
	}

	in.Model.Kernel = "tl9"
	got := DiscoverAndAdmit(context.Background(), fakeRuntime(in), in.Host, in.Model)
	if got.Outcome != OutcomeAbstain || !got.HasReason(ReasonModelKernelUnknown) {
		t.Fatalf("model asking for an unknown kernel: outcome %q reasons %v, want abstain/%s",
			got.Outcome, got.Reasons, ReasonModelKernelUnknown)
	}
}

// TestWitnessedPlatformCoverage states the operating envelope this witness
// actually covers, so a claim about it is read off the fixtures rather than
// asserted in prose.
func TestWitnessedPlatformCoverage(t *testing.T) {
	oses := map[string]bool{}
	arches := map[string]bool{}
	for _, name := range fixtures {
		in := loadFixture(t, name)
		if admitFixture(t, name).Outcome != OutcomeDelegate {
			continue
		}
		oses[in.Host.OS] = true
		arches[in.Host.Arch] = true
	}
	if len(oses) < 3 {
		t.Errorf("delegating fixtures cover %d OS(es) %v, want >= 3", len(oses), oses)
	}
	if len(arches) < 2 {
		t.Errorf("delegating fixtures cover %d arch(es) %v, want >= 2", len(arches), arches)
	}
}
