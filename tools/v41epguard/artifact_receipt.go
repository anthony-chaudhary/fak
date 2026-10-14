//go:build linux

package main

import (
	"encoding/hex"
	"errors"
	"reflect"
	"strings"
)

const (
	guardArtifactReviewSchema       = "fak.v41-ep-owner.artifact-review/v1"
	guardArtifactReviewRegistry     = "fak.vulkan-shader-registry.v19.current83"
	guardArtifactReviewMaxBytes     = 64 << 10
	guardArtifactReviewMaxInputs    = 16
	guardArtifactReviewMaxFile      = int64(1 << 30)
	guardArtifactReviewMaxLoading   = int64(2 << 30)
	guardArtifactReviewMaxAggregate = int64(4 << 30)
)

// These fields describe reviewed identities, not authority. In particular,
// Regular is an asserted collector observation: this pure checker opens no
// file, establishes no held descriptor, and verifies no hardware execution.
type guardArtifactReviewFile struct {
	SHA256    string `json:"sha256"`
	SizeBytes int64  `json:"size_bytes"`
	Regular   bool   `json:"regular"`
}

type guardArtifactReviewLoadingInput struct {
	Role      string `json:"role"`
	Path      string `json:"path"`
	SHA256    string `json:"sha256"`
	SizeBytes int64  `json:"size_bytes"`
	Regular   bool   `json:"regular"`
}

type guardArtifactReviewScope struct {
	RequestSHA256      string `json:"request_sha256"`
	Rank               int    `json:"rank"`
	ContractID         string `json:"contract_id"`
	NotBeforeUnixNS    int64  `json:"not_before_unix_ns"`
	ExpiresUnixNS      int64  `json:"expires_unix_ns"`
	TerminateNS        int64  `json:"terminate_ns"`
	KillGraceNS        int64  `json:"kill_grace_ns"`
	MinimumMemoryBytes uint64 `json:"minimum_memory_bytes"`
	MinimumDiskBytes   uint64 `json:"minimum_disk_bytes"`
}

type guardArtifactReviewIdentity struct {
	Helper                     guardArtifactReviewFile           `json:"helper"`
	Owner                      guardArtifactReviewFile           `json:"owner"`
	SourceCommit               string                            `json:"source_commit"`
	SourceTree                 string                            `json:"source_tree"`
	SourceArchiveSHA256        string                            `json:"source_archive_sha256"`
	VulkanReceiptSHA256        string                            `json:"vulkan_receipt_sha256"`
	OwnerBuildSupplementSHA256 string                            `json:"owner_build_supplement_sha256"`
	Backend                    string                            `json:"backend"`
	GOOS                       string                            `json:"goos"`
	GOARCH                     string                            `json:"goarch"`
	EnvironmentSHA256          string                            `json:"environment_sha256"`
	ShaderRegistry             string                            `json:"shader_registry"`
	SPIRVBundleSHA256          string                            `json:"spirv_bundle_sha256"`
	SPIRVModuleCount           int                               `json:"spirv_module_count"`
	NativeArchive              guardArtifactReviewFile           `json:"native_archive"`
	LoadingInputs              []guardArtifactReviewLoadingInput `json:"loading_inputs"`
}

type guardArtifactReviewReceipt struct {
	Schema                      string                      `json:"schema"`
	Scope                       guardArtifactReviewScope    `json:"scope"`
	Identity                    guardArtifactReviewIdentity `json:"identity"`
	QualificationEvidenceSHA256 string                      `json:"qualification_evidence_sha256"`
}

// A future independently reviewed orchestration boundary must supply this
// input separately from the owner request and launch plan. There is no JSON,
// flag, environment, FD reader, or production constructor for it in this leaf.
// An FD would only transport bytes; it cannot prove approval or freshness.
type guardArtifactReviewApproval struct {
	receiptSHA256, qualificationEvidenceSHA256 string
	scope                                      guardArtifactReviewScope
}

// Observations must eventually come from the trusted collector's current held
// artifacts and selected invocation. Values supplied here do not prove their
// origin. The entire identity and scope are compared, including file status.
type guardArtifactReviewObserved struct {
	scope       guardArtifactReviewScope
	identity    guardArtifactReviewIdentity
	environment []string
}

type guardArtifactReviewCheck struct {
	ReceiptSHA256               string
	QualificationEvidenceSHA256 string
	Claim                       string
	NativeAdmission             string
}

// guardCheckReviewedArtifacts performs bounded, pure identity comparisons.
// It does not parse the referenced build/qualification receipts or authenticate
// their claims. Neither this result nor the review wire format can construct a
// guardArtifactAdmission. Production main remains unchanged and fail-closed.
func guardCheckReviewedArtifacts(raw []byte, approval guardArtifactReviewApproval, observed guardArtifactReviewObserved, nowUnixNS int64) (guardArtifactReviewCheck, error) {
	refuse := func(message string) (guardArtifactReviewCheck, error) {
		return guardArtifactReviewCheck{}, errors.New("v41epguard artifact review: " + message)
	}
	if len(raw) == 0 || len(raw) > guardArtifactReviewMaxBytes {
		return refuse("receipt byte bound")
	}
	if !guardDigest(approval.receiptSHA256) || !guardDigest(approval.qualificationEvidenceSHA256) || guardSHA(raw) != approval.receiptSHA256 {
		return refuse("separate approval digest missing or mismatched")
	}
	var receipt guardArtifactReviewReceipt
	if err := guardStrict(raw, &receipt); err != nil {
		return refuse("strict canonical receipt required")
	}
	if receipt.Schema != guardArtifactReviewSchema || !guardDigest(receipt.QualificationEvidenceSHA256) || receipt.QualificationEvidenceSHA256 != approval.qualificationEvidenceSHA256 {
		return refuse("review schema or qualification reference mismatch")
	}
	if receipt.Scope != approval.scope || receipt.Scope != observed.scope {
		return refuse("separate approval and observed invocation scope mismatch")
	}
	if !guardArtifactReviewScopeValid(receipt.Scope, nowUnixNS) {
		return refuse("invalid or expired full-execution scope")
	}
	if !guardArtifactReviewIdentityValid(receipt.Identity) {
		return refuse("invalid bounded artifact identities")
	}
	if !reflect.DeepEqual(receipt.Identity, observed.identity) {
		return refuse("current collector identity mismatch")
	}
	// Use the helper's exact ordered JSON-slice recipe. This is not an
	// environment allow-list or permission to pass loading variables to main.
	if len(observed.environment) == 0 || len(observed.environment) > 32 {
		return refuse("observed environment bound")
	}
	seen := make(map[string]bool, len(observed.environment))
	for _, item := range observed.environment {
		if len(item) > 4096 {
			return refuse("observed environment entry bound")
		}
		key, value, ok := strings.Cut(item, "=")
		if !ok || key == "" || value == "" || strings.ContainsAny(item, "\x00\r\n") || seen[key] {
			return refuse("invalid observed environment entry")
		}
		seen[key] = true
	}
	if receipt.Identity.EnvironmentSHA256 != guardEnvironmentSHA(observed.environment) {
		return refuse("exact ordered environment mismatch")
	}
	return guardArtifactReviewCheck{
		ReceiptSHA256:               approval.receiptSHA256,
		QualificationEvidenceSHA256: approval.qualificationEvidenceSHA256,
		Claim:                       "identity-comparison-only", NativeAdmission: "unavailable",
	}, nil
}

func guardArtifactReviewScopeValid(s guardArtifactReviewScope, now int64) bool {
	if !guardDigest(s.RequestSHA256) || s.Rank < 0 || s.Rank > 2 ||
		s.ContractID == "" || len(s.ContractID) > 128 || strings.TrimSpace(s.ContractID) != s.ContractID || strings.ContainsAny(s.ContractID, "\x00\r\n") ||
		s.NotBeforeUnixNS <= 0 || now < s.NotBeforeUnixNS || s.ExpiresUnixNS <= now ||
		s.TerminateNS <= 0 || s.TerminateNS > int64(guardMaxTerminate) || s.KillGraceNS <= 0 || s.KillGraceNS > int64(guardMaxKillGrace) ||
		s.MinimumMemoryBytes < guardMemoryFloor || s.MinimumDiskBytes < guardDiskFloor {
		return false
	}
	// All operands are positive and ordered before subtraction. Expiry bounds
	// the complete TERM/KILL window, not admission alone. A future issuer must
	// recheck against actual launch time; this check does not reserve time.
	remaining := s.ExpiresUnixNS - now
	return s.TerminateNS <= remaining && s.KillGraceNS <= remaining-s.TerminateNS
}

func guardArtifactReviewIdentityValid(id guardArtifactReviewIdentity) bool {
	if id.Backend != "vulkan" || id.GOOS != "linux" || id.GOARCH != "amd64" ||
		id.ShaderRegistry != guardArtifactReviewRegistry || id.SPIRVModuleCount != 83 ||
		!guardArtifactReviewGit(id.SourceCommit) || !guardArtifactReviewGit(id.SourceTree) ||
		!guardDigest(id.SourceArchiveSHA256) || !guardDigest(id.VulkanReceiptSHA256) || !guardDigest(id.OwnerBuildSupplementSHA256) ||
		!guardDigest(id.EnvironmentSHA256) || !guardDigest(id.SPIRVBundleSHA256) ||
		!guardArtifactReviewFileValid(id.Helper, guardMaxImage) || !guardArtifactReviewFileValid(id.Owner, guardMaxImage) ||
		!guardArtifactReviewFileValid(id.NativeArchive, guardArtifactReviewMaxFile) ||
		len(id.LoadingInputs) < 3 || len(id.LoadingInputs) > guardArtifactReviewMaxInputs {
		return false
	}
	total := int64(0)
	for _, file := range []guardArtifactReviewFile{id.Helper, id.Owner, id.NativeArchive} {
		if !guardArtifactReviewAdd(&total, file.SizeBytes, guardArtifactReviewMaxAggregate) {
			return false
		}
	}
	roles, paths := make(map[string]bool), make(map[string]bool)
	loadingTotal := int64(0)
	previousRole := ""
	for _, file := range id.LoadingInputs {
		if file.Role <= previousRole || len(file.Role) > 128 || roles[file.Role] || paths[file.Path] ||
			!guardAbsolute(file.Path) || file.Path == "/" ||
			!guardArtifactReviewFileValid(guardArtifactReviewFile{file.SHA256, file.SizeBytes, file.Regular}, guardArtifactReviewMaxFile) {
			return false
		}
		switch file.Role {
		case "vulkan_loader", "icd_manifest", "icd_library":
		default:
			name, ok := strings.CutPrefix(file.Role, "layer:")
			if !ok || name == "" || strings.TrimSpace(name) != name || strings.ContainsAny(name, "\x00\r\n") {
				return false
			}
		}
		if !guardArtifactReviewAdd(&loadingTotal, file.SizeBytes, guardArtifactReviewMaxLoading) || !guardArtifactReviewAdd(&total, file.SizeBytes, guardArtifactReviewMaxAggregate) {
			return false
		}
		roles[file.Role], paths[file.Path], previousRole = true, true, file.Role
	}
	return roles["vulkan_loader"] && roles["icd_manifest"] && roles["icd_library"]
}

func guardArtifactReviewFileValid(file guardArtifactReviewFile, limit int64) bool {
	return file.Regular && guardDigest(file.SHA256) && file.SizeBytes > 0 && file.SizeBytes <= limit
}

func guardArtifactReviewAdd(total *int64, size, limit int64) bool {
	if *total < 0 || *total > limit || size <= 0 || size > limit-*total {
		return false
	}
	*total += size
	return true
}

func guardArtifactReviewGit(value string) bool {
	if len(value) != 40 || strings.ToLower(value) != value {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}
