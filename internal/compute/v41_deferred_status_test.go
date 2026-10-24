package compute

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/enumlint"
)

// The compatibility table below pins verdict behavior; this source-bound check
// keeps the wrapper's attribution switch exhaustive without a lint exemption.
// fak-test:justify why=contract when=changed:internal/compute/v41_deferred_status.go
// fak-test:runtime fast est=10ms lane=default
func TestV41DeferredVerdictEnumCoverage(t *testing.T) {
	t.Parallel()
	source, err := os.ReadFile("v41_deferred_status.go")
	if err != nil {
		t.Fatal(err)
	}
	scan := func(source []byte) enumlint.Report {
		t.Helper()
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "v41_deferred_status.go"), source, 0o600); err != nil {
			t.Fatal(err)
		}
		report, err := enumlint.Scan(dir, enumlint.Config{Rules: []string{enumlint.RuleSwitch}, Exempt: enumlint.NoExemptions})
		if err != nil {
			t.Fatal(err)
		}
		if len(report.Unparsed) != 0 || report.Packages != 1 || report.Enums != 1 || report.Members != 3 || report.Sites < 2 {
			t.Fatalf("incomplete deferred-status enum scan: %+v", report)
		}
		return report
	}
	if report := scan(source); len(report.Findings) != 0 {
		t.Fatalf("non-exhaustive deferred-status switch: %+v", report.Findings)
	}
	const routedCase = "\tcase V41DeferredRoutedSwiGLU:\n\t\tsite = \"V41RoutedSwiGLU\"\n"
	if strings.Count(string(source), routedCase) != 1 {
		t.Fatal("routed attribution arm must occur exactly once for the negative control")
	}
	mutated := strings.Replace(string(source), routedCase, "", 1)
	report := scan([]byte(mutated))
	if len(report.Findings) != 1 || report.Findings[0].Owner != "v41DeferredVerdict" || len(report.Findings[0].Missing) != 1 || report.Findings[0].Missing[0].Name != "V41DeferredRoutedSwiGLU" {
		t.Fatalf("negative control did not detect the missing routed attribution: %+v", report.Findings)
	}
}

func v41DeferredRecord(words ...uint32) []byte {
	out := make([]byte, 4*len(words))
	for i, w := range words {
		binary.LittleEndian.PutUint32(out[4*i:], w)
	}
	return out
}

// A deferred status record decodes to exactly the error the synchronous
// operation returns after its own readback (#13820).
// fak-test:runtime fast est=50ms lane=default
func TestV41DeferredVerdictMatchesSynchronousDecode(t *testing.T) {
	const none = math.MaxUint32
	swiglu := V41DeferredStatus{Kind: V41DeferredRoutedSwiGLU, Count: 8}
	publish := V41DeferredStatus{Kind: V41DeferredBF16Publish, Count: 8, Site: V41BF16PublishRoutedDown}
	for _, c := range []struct {
		name   string
		status V41DeferredStatus
		record []byte
		check  func(error) bool
	}{
		{"swiglu ok", swiglu, v41DeferredRecord(v41RoutedSwiGLUStatusMagic, 8, none, none), func(err error) bool { return err == nil }},
		{"swiglu activation fault", swiglu, v41DeferredRecord(v41RoutedSwiGLUStatusMagic, 8, 3, 1), func(err error) bool {
			var fault *V41RoutedSwiGLUFault
			return errors.As(err, &fault) && *fault == V41RoutedSwiGLUFault{Stage: V41RoutedSwiGLUStageActivation, Index: 3}
		}},
		{"swiglu down operand fault", swiglu, v41DeferredRecord(v41RoutedSwiGLUStatusMagic, 8, none, 6), func(err error) bool {
			var fault *V41RoutedSwiGLUFault
			return errors.As(err, &fault) && *fault == V41RoutedSwiGLUFault{Stage: V41RoutedSwiGLUStageDownOperand, Index: 6}
		}},
		{"swiglu stale magic", swiglu, v41DeferredRecord(none, none, none, none), func(err error) bool {
			var backend *BackendError
			var fault *V41RoutedSwiGLUFault
			return errors.As(err, &backend) && backend.Site == "V41RoutedSwiGLU" && errors.Is(err, ErrVulkanExecutionFailed) && !errors.As(err, &fault)
		}},
		{"publish ok", publish, v41DeferredRecord(v41BF16CheckedStatusMagic, V41BF16PublishRoutedDown, none, 8), func(err error) bool { return err == nil }},
		{"publish arithmetic", publish, v41DeferredRecord(v41BF16CheckedStatusMagic, V41BF16PublishRoutedDown, 5, 8), func(err error) bool {
			var numeric *V41BF16CheckedArithmeticError
			var backend *BackendError
			return errors.As(err, &backend) && backend.Site == "V41BF16Publish" && errors.As(err, &numeric) &&
				*numeric == V41BF16CheckedArithmeticError{Site: V41BF16PublishRoutedDown, Index: 5}
		}},
		{"publish wrong site", publish, v41DeferredRecord(v41BF16CheckedStatusMagic, V41BF16PublishRoutedGateUp, none, 8), func(err error) bool {
			var protocol *V41BF16CheckedProtocolError
			var numeric *V41BF16CheckedArithmeticError
			return errors.As(err, &protocol) && !errors.As(err, &numeric) && errors.Is(err, ErrVulkanExecutionFailed)
		}},
		{"short record", publish, v41DeferredRecord(v41BF16CheckedStatusMagic), func(err error) bool {
			var backend *BackendError
			return errors.As(err, &backend) && backend.Site == "V41ReadDeferred"
		}},
		{"unknown kind", V41DeferredStatus{Count: 8}, v41DeferredRecord(0, 0, 0, 0), func(err error) bool {
			var backend *BackendError
			return errors.As(err, &backend) && backend.Site == "V41ReadDeferred"
		}},
	} {
		if err := v41DeferredVerdict("vulkan", c.status, c.record); !c.check(err) {
			t.Errorf("%s: verdict %v", c.name, err)
		}
	}
}

// Neutral causes and legacy attribution have separate expectations: in
// particular, envelope failures must not gain a recovered cause in the wrapper.
// fak-test:runtime fast est=10ms lane=default
func TestV41DeferredNeutralDecodeAndVulkanCompatibility(t *testing.T) {
	const none = math.MaxUint32
	swiglu := V41DeferredStatus{Kind: V41DeferredRoutedSwiGLU, Count: 8}
	publish := V41DeferredStatus{Kind: V41DeferredBF16Publish, Count: 8, Site: V41BF16PublishRoutedDown}
	finite := V41DeferredStatus{Kind: V41DeferredF32Finite, Count: 8, Site: V41F32FiniteRoutedSum}
	finiteOne, finiteNegative, finiteExtra := finite, finite, finite
	finiteOne.Records, finiteNegative.Records, finiteExtra.Records = 1, -1, 2
	finiteOK := v41DeferredRecord(v41F32FiniteMagic, finite.Site, none, 8)
	for _, c := range []struct {
		name   string
		status V41DeferredStatus
		record []byte
		cause  error
		site   string // Empty for success or a concrete routed arithmetic fault.
	}{
		{"swiglu success", swiglu, v41DeferredRecord(v41RoutedSwiGLUStatusMagic, 8, none, none), nil, ""},
		{"swiglu activation precedes down", swiglu, v41DeferredRecord(v41RoutedSwiGLUStatusMagic, 8, 3, 1), &V41RoutedSwiGLUFault{Stage: V41RoutedSwiGLUStageActivation, Index: 3}, ""},
		{"swiglu down fault", swiglu, v41DeferredRecord(v41RoutedSwiGLUStatusMagic, 8, none, 6), &V41RoutedSwiGLUFault{Stage: V41RoutedSwiGLUStageDownOperand, Index: 6}, ""},
		{"swiglu uninitialized", swiglu, v41DeferredRecord(none, none, none, none), errors.New("compute: V4.1 routed SwiGLU status magic or count does not match the issued operation"), "V41RoutedSwiGLU"},
		{"swiglu invalid lane", swiglu, v41DeferredRecord(v41RoutedSwiGLUStatusMagic, 8, 8, none), errors.New("compute: V4.1 routed SwiGLU fault lane 8 exceeds the issued count 8"), "V41RoutedSwiGLU"},
		{"publish success", publish, v41DeferredRecord(v41BF16CheckedStatusMagic, publish.Site, none, 8), nil, ""},
		{"publish arithmetic", publish, v41DeferredRecord(v41BF16CheckedStatusMagic, publish.Site, 5, 8), &V41BF16CheckedArithmeticError{Site: publish.Site, Index: 5}, "V41BF16Publish"},
		{"publish wrong site", publish, v41DeferredRecord(v41BF16CheckedStatusMagic, V41BF16PublishRoutedGateUp, none, 8), &V41BF16CheckedProtocolError{Message: "magic, site or count does not match the issued operation"}, "V41BF16Publish"},
		{"publish invalid lane", publish, v41DeferredRecord(v41BF16CheckedStatusMagic, publish.Site, 8, 8), &V41BF16CheckedProtocolError{Message: "invalid element index exceeds issued count"}, "V41BF16Publish"},
		{"finite implicit single record", finite, finiteOK, nil, ""},
		{"finite explicit single record", finiteOne, finiteOK, nil, ""},
		{"finite arithmetic", finite, v41DeferredRecord(v41F32FiniteMagic, finite.Site, 5, 8), &V41F32FiniteArithmeticError{Site: finite.Site, Index: 5}, "V41F32Finite"},
		{"finite uninitialized", finite, v41DeferredRecord(none, none, none, none), &V41F32FiniteProtocolError{Message: "magic, site or count does not match the issued operation"}, "V41F32Finite"},
		{"finite wrong count", finite, v41DeferredRecord(v41F32FiniteMagic, finite.Site, none, 7), &V41F32FiniteProtocolError{Message: "magic, site or count does not match the issued operation"}, "V41F32Finite"},
		{"finite invalid lane", finite, v41DeferredRecord(v41F32FiniteMagic, finite.Site, 8, 8), &V41F32FiniteProtocolError{Message: "invalid element index exceeds issued count"}, "V41F32Finite"},
		{"finite negative records", finiteNegative, finiteOK, &v41DeferredStatusProtocolError{finite.Kind, "raw F32 status must have one record"}, "V41ReadDeferred"},
		{"finite extra records", finiteExtra, finiteOK, &v41DeferredStatusProtocolError{finite.Kind, "raw F32 status must have one record"}, "V41ReadDeferred"},
		{"missing record", publish, nil, &v41DeferredStatusProtocolError{publish.Kind, "status record is not 16 bytes"}, "V41ReadDeferred"},
		{"short record before record count", finiteExtra, finiteOK[:15], &v41DeferredStatusProtocolError{finite.Kind, "status record is not 16 bytes"}, "V41ReadDeferred"},
		{"long record", finite, append(append([]byte(nil), finiteOK...), 0), &v41DeferredStatusProtocolError{finite.Kind, "status record is not 16 bytes"}, "V41ReadDeferred"},
		{"unknown kind", V41DeferredStatus{Kind: 255, Count: 8}, v41DeferredRecord(0, 0, 0, 0), &v41DeferredStatusProtocolError{255, "unknown status kind"}, "V41ReadDeferred"},
	} {
		t.Run(c.name, func(t *testing.T) {
			before := string(c.record)
			t.Run("neutral", func(t *testing.T) {
				got := decodeV41DeferredStatus(c.status, c.record)
				if !reflect.DeepEqual(got, c.cause) {
					t.Fatalf("cause = %T %v, want %T %v", got, got, c.cause, c.cause)
				}
				if envelope, ok := c.cause.(*v41DeferredStatusProtocolError); ok {
					wantMessage := fmt.Sprintf("deferred V4.1 status kind %d: %s", c.status.Kind, envelope.reason)
					if got.Error() != wantMessage {
						t.Fatalf("envelope message = %q, want %q", got.Error(), wantMessage)
					}
				}
				var backend *BackendError
				if errors.As(got, &backend) || errors.Is(got, ErrVulkanExecutionFailed) ||
					(got != nil && strings.Contains(strings.ToLower(got.Error()), "vulkan")) {
					t.Fatalf("neutral cause acquired backend attribution: %v", got)
				}
			})
			for _, backend := range []string{"vulkan", "fixture"} {
				t.Run("compatibility/"+backend, func(t *testing.T) {
					got := v41DeferredVerdict(backend, c.status, c.record)
					if c.site == "" {
						if !reflect.DeepEqual(got, c.cause) || errors.Is(got, ErrVulkanExecutionFailed) {
							t.Fatalf("direct verdict = %T %v, want %T %v", got, got, c.cause, c.cause)
						}
						return
					}
					want := &BackendError{Backend: backend, Site: c.site, Class: VulkanClassExecutionFailed, Err: ErrVulkanExecutionFailed, Message: c.cause.Error()}
					if c.site != "V41ReadDeferred" {
						want.Recovered = c.cause
					}
					var attributed *BackendError
					if !errors.As(got, &attributed) || !reflect.DeepEqual(attributed, want) || !errors.Is(got, ErrVulkanExecutionFailed) {
						t.Fatalf("attribution = %#v, want %#v", got, want)
					}
					if wantMessage := "compute: " + backend + " [execution_failed] at " + c.site + ": " + c.cause.Error(); got.Error() != wantMessage {
						t.Fatalf("message = %q, want %q", got.Error(), wantMessage)
					}
					if c.site == "V41ReadDeferred" {
						if chain := attributed.Unwrap(); len(chain) != 1 || chain[0] != ErrVulkanExecutionFailed {
							t.Fatalf("envelope error chain changed: %v", chain)
						}
						return
					}
					// Preserve errors.Is for the exact synchronous cause. A typed
					// cause must also survive errors.As; an untyped string cause
					// shares its dynamic type with the ErrVulkanExecutionFailed
					// sentinel that precedes it in the chain, so errors.As cannot
					// name it and identity is the contract.
					if !errors.Is(got, attributed.Recovered.(error)) {
						t.Fatalf("operation cause lost from error chain: %v", got)
					}
					if reflect.TypeOf(c.cause) != reflect.TypeOf(ErrVulkanExecutionFailed) {
						target := reflect.New(reflect.TypeOf(c.cause))
						if !errors.As(got, target.Interface()) || !reflect.DeepEqual(target.Elem().Interface(), c.cause) {
							t.Fatalf("typed operation cause lost from error chain: %v", got)
						}
					}
				})
			}
			if string(c.record) != before {
				t.Fatal("decoding mutated the status record")
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41DeferredLayoutAlignsRegions(t *testing.T) {
	offsets, total := v41DeferredLayout([]int{16, 16, 10, 4, 1024})
	if want := []int{0, 16, 32, 48, 64}; !reflect.DeepEqual(offsets, want) || total != 1088 {
		t.Fatalf("offsets=%v total=%d", offsets, total)
	}
}
