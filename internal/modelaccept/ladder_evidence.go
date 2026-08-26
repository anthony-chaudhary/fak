package modelaccept

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// LadderEvidenceReasonCode is the closed vocabulary for checksum-admission
// failures. Callers can route on the code without parsing operator prose.
type LadderEvidenceReasonCode string

const (
	LadderEvidenceInvalidManifest    LadderEvidenceReasonCode = "LADDER_EVIDENCE_INVALID_MANIFEST"
	LadderEvidenceDuplicatePath      LadderEvidenceReasonCode = "LADDER_EVIDENCE_DUPLICATE_PATH"
	LadderEvidencePathTraversal      LadderEvidenceReasonCode = "LADDER_EVIDENCE_PATH_TRAVERSAL"
	LadderEvidenceMissingArtifact    LadderEvidenceReasonCode = "LADDER_EVIDENCE_MISSING_ARTIFACT"
	LadderEvidenceExtraArtifact      LadderEvidenceReasonCode = "LADDER_EVIDENCE_EXTRA_ARTIFACT"
	LadderEvidenceUnreadableArtifact LadderEvidenceReasonCode = "LADDER_EVIDENCE_UNREADABLE_ARTIFACT"
	LadderEvidenceChecksumMismatch   LadderEvidenceReasonCode = "LADDER_EVIDENCE_CHECKSUM_MISMATCH"
)

type LadderEvidenceOptions struct {
	Directory string
	Manifest  string
}

// LadderEvidenceReason is typed HOLD evidence. ExpectedSHA256 and
// ActualSHA256 are both populated for a checksum mismatch so the operator can
// distinguish stale evidence from a missing or unreadable artifact.
type LadderEvidenceReason struct {
	Code           LadderEvidenceReasonCode `json:"code"`
	Path           string                   `json:"path,omitempty"`
	ExpectedSHA256 string                   `json:"expected_sha256,omitempty"`
	ActualSHA256   string                   `json:"actual_sha256,omitempty"`
	Detail         string                   `json:"detail,omitempty"`
}

func (r LadderEvidenceReason) String() string {
	if r.Code == "" {
		return ""
	}
	parts := []string{string(r.Code)}
	if r.Path != "" {
		parts = append(parts, "path="+strconv.Quote(r.Path))
	}
	if r.ExpectedSHA256 != "" {
		parts = append(parts, "expected_sha256="+r.ExpectedSHA256)
	}
	if r.ActualSHA256 != "" {
		parts = append(parts, "actual_sha256="+r.ActualSHA256)
	}
	if r.Detail != "" {
		parts = append(parts, "detail="+strconv.Quote(r.Detail))
	}
	return strings.Join(parts, " ")
}

type LadderEvidenceAdmission struct {
	Verdict Verdict              `json:"verdict"`
	Reason  LadderEvidenceReason `json:"reason,omitempty"`
}

type ladderChecksumEntry struct {
	File   string `json:"file"`
	SHA256 string `json:"sha256"`
}

// BuildInventoryWithLadderEvidence is the fail-closed readiness seam. It runs
// the existing inventory evaluator only after every declared artifact has been
// admitted byte-for-byte. A HOLD is constructed from declarations alone, so
// sample counts, observed values, and witnessed tiers cannot leak from evidence
// whose identity failed admission.
func BuildInventoryWithLadderEvidence(in Input, inventoryOpts InventoryOptions, evidenceOpts LadderEvidenceOptions) (Inventory, LadderEvidenceAdmission) {
	admission := VerifyLadderEvidence(evidenceOpts)
	if admission.Verdict == Pass {
		return BuildInventory(in, inventoryOpts), admission
	}
	return holdInventoryForLadderEvidence(in, inventoryOpts, admission.Reason), admission
}

// VerifyLadderEvidence proves that the manifest names exactly the regular
// evidence files in Directory and that every named file has the declared
// SHA-256. The manifest itself is metadata and is excluded from the exact-set
// comparison when it resides inside Directory.
func VerifyLadderEvidence(opts LadderEvidenceOptions) LadderEvidenceAdmission {
	directory := strings.TrimSpace(opts.Directory)
	manifest := strings.TrimSpace(opts.Manifest)
	if directory == "" || manifest == "" {
		return ladderEvidenceHold(LadderEvidenceReason{
			Code:   LadderEvidenceInvalidManifest,
			Detail: "evidence directory and checksum manifest are required",
		})
	}

	root, err := filepath.Abs(directory)
	if err != nil {
		return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceUnreadableArtifact, Detail: err.Error()})
	}
	root, err = filepath.EvalSymlinks(root)
	if err != nil {
		return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceUnreadableArtifact, Detail: err.Error()})
	}
	info, err := os.Stat(root)
	if err != nil || !info.IsDir() {
		detail := "evidence directory is not a directory"
		if err != nil {
			detail = err.Error()
		}
		return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceUnreadableArtifact, Detail: detail})
	}

	manifestPath, err := filepath.Abs(manifest)
	if err != nil {
		return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceInvalidManifest, Detail: err.Error()})
	}
	entries, reason := decodeLadderChecksumManifest(manifestPath)
	if reason.Code != "" {
		return ladderEvidenceHold(reason)
	}
	// root is already symlink-canonical. Canonicalize the manifest identity to
	// the same coordinate system before WalkDir compares it with candidates.
	// This matters on macOS, where /var/... and /private/var/... name the same
	// file, and for an explicit symlink alias on every platform. A manifest that
	// really lives outside root stays outside and is not excluded from the walk.
	manifestIdentity, err := filepath.EvalSymlinks(manifestPath)
	if err != nil {
		return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceInvalidManifest, Detail: err.Error()})
	}
	manifestIdentity = filepath.Clean(manifestIdentity)

	declared := make(map[string]struct{}, len(entries))
	for i := range entries {
		rel, pathReason := canonicalLadderEvidencePath(entries[i].File)
		if pathReason.Code != "" {
			return ladderEvidenceHold(pathReason)
		}
		entries[i].File = rel
		if _, exists := declared[rel]; exists {
			return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceDuplicatePath, Path: rel})
		}
		declared[rel] = struct{}{}
		decoded, decodeErr := hex.DecodeString(entries[i].SHA256)
		if decodeErr != nil || len(decoded) != sha256.Size {
			return ladderEvidenceHold(LadderEvidenceReason{
				Code:   LadderEvidenceInvalidManifest,
				Path:   rel,
				Detail: "sha256 must be exactly 64 hexadecimal characters",
			})
		}
	}

	for _, entry := range entries {
		artifactPath := filepath.Join(root, filepath.FromSlash(entry.File))
		artifactInfo, statErr := os.Lstat(artifactPath)
		if statErr != nil {
			if errors.Is(statErr, os.ErrNotExist) {
				return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceMissingArtifact, Path: entry.File, ExpectedSHA256: entry.SHA256})
			}
			return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceUnreadableArtifact, Path: entry.File, ExpectedSHA256: entry.SHA256, Detail: statErr.Error()})
		}
		if artifactInfo.IsDir() {
			return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceUnreadableArtifact, Path: entry.File, ExpectedSHA256: entry.SHA256, Detail: "declared artifact is a directory"})
		}
		resolved, resolveErr := filepath.EvalSymlinks(artifactPath)
		if resolveErr != nil {
			return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceUnreadableArtifact, Path: entry.File, ExpectedSHA256: entry.SHA256, Detail: resolveErr.Error()})
		}
		if !pathWithin(root, resolved) {
			return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidencePathTraversal, Path: entry.File, ExpectedSHA256: entry.SHA256, Detail: "resolved artifact escapes evidence directory"})
		}
		resolvedInfo, statErr := os.Stat(resolved)
		if statErr != nil || !resolvedInfo.Mode().IsRegular() {
			detail := "declared artifact is not a regular file"
			if statErr != nil {
				detail = statErr.Error()
			}
			return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceUnreadableArtifact, Path: entry.File, ExpectedSHA256: entry.SHA256, Detail: detail})
		}
		content, readErr := os.ReadFile(resolved)
		if readErr != nil {
			return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceUnreadableArtifact, Path: entry.File, ExpectedSHA256: entry.SHA256, Detail: readErr.Error()})
		}
		actual := sha256.Sum256(content)
		actualText := hex.EncodeToString(actual[:])
		if !strings.EqualFold(entry.SHA256, actualText) {
			return ladderEvidenceHold(LadderEvidenceReason{
				Code:           LadderEvidenceChecksumMismatch,
				Path:           entry.File,
				ExpectedSHA256: entry.SHA256,
				ActualSHA256:   actualText,
			})
		}
	}

	walkErr := filepath.WalkDir(root, func(candidate string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if candidate == root || entry.IsDir() || filepath.Clean(candidate) == manifestIdentity {
			return nil
		}
		rel, relErr := filepath.Rel(root, candidate)
		if relErr != nil {
			return relErr
		}
		rel = filepath.ToSlash(rel)
		if _, exists := declared[rel]; !exists {
			reason = LadderEvidenceReason{Code: LadderEvidenceExtraArtifact, Path: rel}
			return io.EOF
		}
		return nil
	})
	if reason.Code != "" {
		return ladderEvidenceHold(reason)
	}
	if walkErr != nil {
		return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceUnreadableArtifact, Detail: walkErr.Error()})
	}
	return LadderEvidenceAdmission{Verdict: Pass}
}

func decodeLadderChecksumManifest(manifestPath string) ([]ladderChecksumEntry, LadderEvidenceReason) {
	f, err := os.Open(manifestPath)
	if err != nil {
		return nil, LadderEvidenceReason{Code: LadderEvidenceInvalidManifest, Detail: err.Error()}
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	var entries []ladderChecksumEntry
	if err := dec.Decode(&entries); err != nil {
		return nil, LadderEvidenceReason{Code: LadderEvidenceInvalidManifest, Detail: err.Error()}
	}
	if len(entries) == 0 {
		return nil, LadderEvidenceReason{Code: LadderEvidenceInvalidManifest, Detail: "checksum manifest contains no artifacts"}
	}
	var trailing any
	if err := dec.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			err = errors.New("multiple JSON values")
		}
		return nil, LadderEvidenceReason{Code: LadderEvidenceInvalidManifest, Detail: err.Error()}
	}
	return entries, LadderEvidenceReason{}
}

func canonicalLadderEvidencePath(raw string) (string, LadderEvidenceReason) {
	if raw == "" || strings.Contains(raw, `\`) || path.IsAbs(raw) || filepath.IsAbs(raw) {
		return "", LadderEvidenceReason{Code: LadderEvidencePathTraversal, Path: raw, Detail: "artifact path must be a non-empty relative slash path"}
	}
	clean := path.Clean(raw)
	if clean != raw || clean == "." || clean == ".." || strings.HasPrefix(clean, "../") {
		return "", LadderEvidenceReason{Code: LadderEvidencePathTraversal, Path: raw, Detail: "artifact path is not canonical within the evidence directory"}
	}
	return clean, LadderEvidenceReason{}
}

func pathWithin(root, candidate string) bool {
	rel, err := filepath.Rel(root, candidate)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

func ladderEvidenceHold(reason LadderEvidenceReason) LadderEvidenceAdmission {
	return LadderEvidenceAdmission{Verdict: Hold, Reason: reason}
}

func holdInventoryForLadderEvidence(in Input, opts InventoryOptions, reason LadderEvidenceReason) Inventory {
	reasonText := reason.String()
	out := Inventory{
		Schema:   InventorySchema,
		Verdict:  Hold,
		CorpusID: in.Corpus.ID,
		Rows:     []InventoryRow{},
	}
	models := append([]ModelRequest(nil), in.Models...)
	sort.Slice(models, func(i, j int) bool { return models[i].Model < models[j].Model })
	for _, req := range models {
		out.Rows = append(out.Rows, InventoryRow{
			Model:            req.Model,
			Family:           req.Family,
			Generation:       req.Generation,
			Lifecycle:        req.Lifecycle,
			CapabilityGate:   Hold,
			RequestedTier:    req.RequestedTier,
			CorpusID:         in.Corpus.ID,
			DeclaredAt:       in.Corpus.DeclaredAt,
			Artifact:         opts.Artifact,
			ArtifactRevision: opts.ArtifactRevision,
			Reasons:          []string{reasonText},
		})
	}
	if len(out.Rows) == 0 {
		out.Reasons = append(out.Reasons, reasonText)
	}
	return out
}
