//go:build linux

package main

import (
	"bytes"
	"encoding/json"
	"math"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=10ms lane=default
// All identities and collector observations below are synthetic values. This
// test opens no artifacts, creates no processes, and issues no native authority.
func TestV41EPArtifactReviewIdentityContract(t *testing.T) {
	const now = int64(10 * time.Second)
	fixture := func() (guardArtifactReviewReceipt, guardArtifactReviewApproval, guardArtifactReviewObserved) {
		hash := strings.Repeat("a", 64)
		file := guardArtifactReviewFile{SHA256: hash, SizeBytes: 1, Regular: true}
		environment := []string{"LANG=C", "LC_ALL=C", "TZ=UTC"}
		scope := guardArtifactReviewScope{RequestSHA256: hash, Rank: 0, ContractID: "synthetic-review-only",
			NotBeforeUnixNS: now - 1, ExpiresUnixNS: now + int64(3*time.Second),
			TerminateNS: int64(time.Second), KillGraceNS: int64(time.Second),
			MinimumMemoryBytes: guardMemoryFloor, MinimumDiskBytes: guardDiskFloor}
		identity := guardArtifactReviewIdentity{Helper: file, Owner: file, NativeArchive: file,
			SourceCommit: strings.Repeat("b", 40), SourceTree: strings.Repeat("c", 40), SourceArchiveSHA256: hash,
			VulkanReceiptSHA256: hash, OwnerBuildSupplementSHA256: hash,
			Backend: "vulkan", GOOS: "linux", GOARCH: "amd64", EnvironmentSHA256: guardEnvironmentSHA(environment),
			ShaderRegistry: guardArtifactReviewRegistry, SPIRVBundleSHA256: hash, SPIRVModuleCount: 83,
			LoadingInputs: []guardArtifactReviewLoadingInput{
				{Role: "icd_library", Path: "/synthetic/icd.so", SHA256: hash, SizeBytes: 1, Regular: true},
				{Role: "icd_manifest", Path: "/synthetic/icd.json", SHA256: hash, SizeBytes: 1, Regular: true},
				{Role: "vulkan_loader", Path: "/synthetic/loader.so", SHA256: hash, SizeBytes: 1, Regular: true},
			}}
		receipt := guardArtifactReviewReceipt{Schema: guardArtifactReviewSchema, Scope: scope, Identity: identity, QualificationEvidenceSHA256: hash}
		observedIdentity := identity
		observedIdentity.LoadingInputs = append([]guardArtifactReviewLoadingInput(nil), identity.LoadingInputs...)
		return receipt, guardArtifactReviewApproval{qualificationEvidenceSHA256: hash, scope: scope},
			guardArtifactReviewObserved{scope: scope, identity: observedIdentity, environment: environment}
	}
	encode := func(t *testing.T, receipt guardArtifactReviewReceipt) []byte {
		t.Helper()
		raw, err := json.Marshal(receipt)
		if err != nil {
			t.Fatal(err)
		}
		return raw
	}
	checkRefusal := func(t *testing.T, raw []byte, approval guardArtifactReviewApproval, observed guardArtifactReviewObserved) {
		t.Helper()
		result, err := guardCheckReviewedArtifacts(raw, approval, observed, now)
		if err == nil || result != (guardArtifactReviewCheck{}) {
			t.Fatalf("invalid input produced diagnostics: result=%+v error=%v", result, err)
		}
	}
	t.Run("diagnostics-only-exact-boundaries", func(t *testing.T) {
		r, a, o := fixture()
		r.Scope.ExpiresUnixNS = now + r.Scope.TerminateNS + r.Scope.KillGraceNS
		a.scope, o.scope = r.Scope, r.Scope
		raw := encode(t, r)
		a.receiptSHA256 = guardSHA(raw)
		result, err := guardCheckReviewedArtifacts(raw, a, o, now)
		if err != nil || result.Claim != "identity-comparison-only" || result.NativeAdmission != "unavailable" || result.ReceiptSHA256 != a.receiptSHA256 {
			t.Fatalf("diagnostic result=%+v error=%v", result, err)
		}
		if string(encode(t, r)) != string(raw) {
			t.Fatal("checker changed receipt input")
		}
	})
	t.Run("strict-wire", func(t *testing.T) {
		r, a, o := fixture()
		raw := encode(t, r)
		for name, bad := range map[string][]byte{
			"duplicate":         bytes.Replace(raw, []byte(`"rank":0`), []byte(`"rank":0,"rank":0`), 1),
			"null":              bytes.Replace(raw, []byte(`"rank":0`), []byte(`"rank":null`), 1),
			"omitted-rank":      bytes.Replace(raw, []byte(`"rank":0,`), nil, 1),
			"unknown":           append(append([]byte(nil), raw[:len(raw)-1]...), []byte(`,"approved":true}`)...),
			"trailing":          append(append([]byte(nil), raw...), []byte(`{}`)...),
			"overflow":          bytes.Replace(raw, []byte(`"size_bytes":1`), []byte(`"size_bytes":9223372036854775808`), 1),
			"unsigned-overflow": bytes.Replace(raw, []byte(`"minimum_disk_bytes":3221225472`), []byte(`"minimum_disk_bytes":18446744073709551616`), 1),
			"whitespace":        append([]byte(" "), raw...),
			"byte-cap":          bytes.Repeat([]byte("x"), guardArtifactReviewMaxBytes+1),
		} {
			t.Run(name, func(t *testing.T) {
				a.receiptSHA256 = guardSHA(bad)
				checkRefusal(t, bad, a, o)
			})
		}
	})
	t.Run("independent-inputs", func(t *testing.T) {
		for name, edit := range map[string]func(*guardArtifactReviewApproval, *guardArtifactReviewObserved){
			"request-hash-is-not-approval": func(a *guardArtifactReviewApproval, _ *guardArtifactReviewObserved) {
				a.receiptSHA256 = a.scope.RequestSHA256
			},
			"missing-approval": func(a *guardArtifactReviewApproval, _ *guardArtifactReviewObserved) {
				*a = guardArtifactReviewApproval{}
			},
			"qualification-reference": func(a *guardArtifactReviewApproval, _ *guardArtifactReviewObserved) {
				a.qualificationEvidenceSHA256 = strings.Repeat("d", 64)
			},
			"rank":   func(_ *guardArtifactReviewApproval, o *guardArtifactReviewObserved) { o.scope.Rank++ },
			"budget": func(a *guardArtifactReviewApproval, _ *guardArtifactReviewObserved) { a.scope.TerminateNS++ },
			"held-image": func(_ *guardArtifactReviewApproval, o *guardArtifactReviewObserved) {
				o.identity.Owner.SHA256 = strings.Repeat("d", 64)
			},
			"held-file-status": func(_ *guardArtifactReviewApproval, o *guardArtifactReviewObserved) {
				o.identity.Helper.Regular = false
			},
			"loading-input": func(_ *guardArtifactReviewApproval, o *guardArtifactReviewObserved) {
				o.identity.LoadingInputs[0].SizeBytes++
			},
			"environment-order": func(_ *guardArtifactReviewApproval, o *guardArtifactReviewObserved) {
				o.environment[0], o.environment[1] = o.environment[1], o.environment[0]
			},
		} {
			t.Run(name, func(t *testing.T) {
				r, a, o := fixture()
				raw := encode(t, r)
				a.receiptSHA256 = guardSHA(raw)
				edit(&a, &o)
				checkRefusal(t, raw, a, o)
			})
		}
	})
	t.Run("invalid-reviewed-values", func(t *testing.T) {
		for name, edit := range map[string]func(*guardArtifactReviewReceipt){
			"schema":        func(r *guardArtifactReviewReceipt) { r.Schema = "unreviewed" },
			"not-yet-valid": func(r *guardArtifactReviewReceipt) { r.Scope.NotBeforeUnixNS = now + 1 },
			"execution-exceeds-expiry": func(r *guardArtifactReviewReceipt) {
				r.Scope.ExpiresUnixNS = now + r.Scope.TerminateNS + r.Scope.KillGraceNS - 1
			},
			"negative-window": func(r *guardArtifactReviewReceipt) { r.Scope.NotBeforeUnixNS = math.MinInt64 },
			"duration-cap":    func(r *guardArtifactReviewReceipt) { r.Scope.TerminateNS = math.MaxInt64 },
			"kill-cap":        func(r *guardArtifactReviewReceipt) { r.Scope.KillGraceNS = int64(guardMaxKillGrace) + 1 },
			"resource-floor":  func(r *guardArtifactReviewReceipt) { r.Scope.MinimumMemoryBytes-- },
			"git-format":      func(r *guardArtifactReviewReceipt) { r.Identity.SourceCommit = strings.Repeat("a", 64) },
			"hash-case":       func(r *guardArtifactReviewReceipt) { r.Identity.Owner.SHA256 = strings.Repeat("A", 64) },
			"old-registry": func(r *guardArtifactReviewReceipt) {
				r.Identity.ShaderRegistry = "fak.vulkan-shader-registry.v18.current82"
			},
			"module-count": func(r *guardArtifactReviewReceipt) { r.Identity.SPIRVModuleCount = 82 },
			"platform":     func(r *guardArtifactReviewReceipt) { r.Identity.GOARCH = "arm64" },
			"not-regular":  func(r *guardArtifactReviewReceipt) { r.Identity.NativeArchive.Regular = false },
			"image-cap":    func(r *guardArtifactReviewReceipt) { r.Identity.Helper.SizeBytes = guardMaxImage + 1 },
			"file-cap":     func(r *guardArtifactReviewReceipt) { r.Identity.LoadingInputs[0].SizeBytes = math.MaxInt64 },
			"loading-total": func(r *guardArtifactReviewReceipt) {
				for i := range r.Identity.LoadingInputs {
					r.Identity.LoadingInputs[i].SizeBytes = guardArtifactReviewMaxFile
				}
			},
			"duplicate-role": func(r *guardArtifactReviewReceipt) {
				r.Identity.LoadingInputs[1].Role = r.Identity.LoadingInputs[0].Role
			},
			"duplicate-path": func(r *guardArtifactReviewReceipt) {
				r.Identity.LoadingInputs[1].Path = r.Identity.LoadingInputs[0].Path
			},
			"missing-required-role": func(r *guardArtifactReviewReceipt) { r.Identity.LoadingInputs[2].Role = "layer:test" },
			"relative-path":         func(r *guardArtifactReviewReceipt) { r.Identity.LoadingInputs[0].Path = "relative.so" },
			"path-cap": func(r *guardArtifactReviewReceipt) {
				r.Identity.LoadingInputs[0].Path = "/" + strings.Repeat("a", 4096)
			},
			"entry-cap": func(r *guardArtifactReviewReceipt) {
				for len(r.Identity.LoadingInputs) <= guardArtifactReviewMaxInputs {
					r.Identity.LoadingInputs = append(r.Identity.LoadingInputs, r.Identity.LoadingInputs[0])
				}
			},
		} {
			t.Run(name, func(t *testing.T) {
				r, a, o := fixture()
				edit(&r)
				// Even matching, independently supplied comparison values cannot
				// make an invalid wire identity or scope pass the bounded checker.
				a.scope, o.scope, o.identity = r.Scope, r.Scope, r.Identity
				raw := encode(t, r)
				a.receiptSHA256 = guardSHA(raw)
				checkRefusal(t, raw, a, o)
			})
		}
	})
	t.Run("checked-sum", func(t *testing.T) {
		total := guardArtifactReviewMaxAggregate - 1
		if !guardArtifactReviewAdd(&total, 1, guardArtifactReviewMaxAggregate) || total != guardArtifactReviewMaxAggregate || guardArtifactReviewAdd(&total, 1, guardArtifactReviewMaxAggregate) || guardArtifactReviewAdd(&total, math.MaxInt64, guardArtifactReviewMaxAggregate) {
			t.Fatal("aggregate addition did not retain exact cap without overflow")
		}
	})
}
