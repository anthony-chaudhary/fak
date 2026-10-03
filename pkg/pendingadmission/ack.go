// Package pendingadmission defines the neutral, pure-data acknowledgement that
// binds an exact prepared candidate to durable pending obligations. It contains
// no repository, storage, qualification, or release policy.
package pendingadmission

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"path"
	"reflect"
	"regexp"
	"strings"
)

const (
	AckSchema   = "fak.pending-debt-ack/v1"
	AckContract = "durable-pending-obligations/v1"
	ackState    = "pending"
	maxJSONSize = 256 << 10
)

var obligationKindPattern = regexp.MustCompile(`^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$`)

// ProofContext binds the executable and workspace inputs used to prepare the
// candidate. It is evidence identity, not qualification evidence.
type ProofContext struct {
	CompanionCommit    string `json:"companion_commit"`
	CompanionTree      string `json:"companion_tree"`
	GoExecutableDigest string `json:"go_executable_digest"`
	ToolchainDigest    string `json:"toolchain_digest"`
	TestEnvDigest      string `json:"test_env_digest"`
	WorkspaceDigest    string `json:"workspace_digest"`
	VerifierDigest     string `json:"verifier_digest"`
}

// Validate checks the closed proof-context grammar.
func (p ProofContext) Validate() error {
	if err := validateObject("companion_commit", p.CompanionCommit); err != nil {
		return err
	}
	if err := validateObject("companion_tree", p.CompanionTree); err != nil {
		return err
	}
	for name, value := range map[string]string{
		"go_executable_digest": p.GoExecutableDigest,
		"toolchain_digest":     p.ToolchainDigest,
		"test_env_digest":      p.TestEnvDigest,
		"workspace_digest":     p.WorkspaceDigest,
		"verifier_digest":      p.VerifierDigest,
	} {
		if err := validateDigest(name, value); err != nil {
			return err
		}
	}
	return nil
}

// Binding identifies the exact target transition and prepared candidate.
type Binding struct {
	TargetRef       string       `json:"target_ref"`
	ParentCommit    string       `json:"parent_commit"`
	ParentTree      string       `json:"parent_tree"`
	CandidateCommit string       `json:"candidate_commit"`
	CandidateTree   string       `json:"candidate_tree"`
	RecoveryRef     string       `json:"recovery_ref"`
	Paths           []string     `json:"paths"`
	PathsDigest     string       `json:"paths_digest"`
	ProofContext    ProofContext `json:"proof_context"`
	ContextDigest   string       `json:"context_digest"`
}

// Validate checks canonical refs, objects, paths, and derived digests.
func (b Binding) Validate() error {
	if err := validateRef("target_ref", b.TargetRef); err != nil {
		return err
	}
	if err := validateRef("recovery_ref", b.RecoveryRef); err != nil {
		return err
	}
	for name, value := range map[string]string{
		"parent_commit":    b.ParentCommit,
		"parent_tree":      b.ParentTree,
		"candidate_commit": b.CandidateCommit,
		"candidate_tree":   b.CandidateTree,
	} {
		if err := validateObject(name, value); err != nil {
			return err
		}
	}
	if b.CandidateCommit == b.ParentCommit {
		return errors.New("candidate_commit must differ from parent_commit")
	}
	if err := validatePaths(b.Paths); err != nil {
		return err
	}
	pathsDigest, err := digestJSON(b.Paths)
	if err != nil {
		return fmt.Errorf("paths_digest: %w", err)
	}
	if b.PathsDigest != pathsDigest {
		return fmt.Errorf("paths_digest mismatch: got %q want %q", b.PathsDigest, pathsDigest)
	}
	if err := b.ProofContext.Validate(); err != nil {
		return fmt.Errorf("proof_context: %w", err)
	}
	contextDigest, err := digestJSON(b.ProofContext)
	if err != nil {
		return fmt.Errorf("context_digest: %w", err)
	}
	if b.ContextDigest != contextDigest {
		return fmt.Errorf("context_digest mismatch: got %q want %q", b.ContextDigest, contextDigest)
	}
	return nil
}

// Obligation identifies one durable pending obligation without defining its
// private policy or execution semantics.
type Obligation struct {
	Kind   string `json:"kind"`
	ID     string `json:"id"`
	Digest string `json:"digest"`
}

// DebtAck binds durable pending obligations to a prepared receipt and exact
// candidate binding.
type DebtAck struct {
	Schema            string       `json:"schema"`
	Contract          string       `json:"contract"`
	AckID             string       `json:"ack_id"`
	State             string       `json:"state"`
	PreparedReceiptID string       `json:"prepared_receipt_id"`
	Binding           Binding      `json:"binding"`
	RecipeDigest      string       `json:"recipe_digest"`
	Obligations       []Obligation `json:"obligations"`
}

// ID returns the canonical acknowledgement ID. The current AckID is ignored.
func (a DebtAck) ID() (string, error) {
	a.AckID = ""
	return digestJSON(a)
}

// Validate checks the closed envelope, exact binding, canonical obligations,
// and acknowledgement ID.
func (a DebtAck) Validate() error {
	if a.Schema != AckSchema {
		return fmt.Errorf("schema must be %q", AckSchema)
	}
	if a.Contract != AckContract {
		return fmt.Errorf("contract must be %q", AckContract)
	}
	if a.State != ackState {
		return fmt.Errorf("state must be %q", ackState)
	}
	if err := validateDigest("prepared_receipt_id", a.PreparedReceiptID); err != nil {
		return err
	}
	if err := a.Binding.Validate(); err != nil {
		return fmt.Errorf("binding: %w", err)
	}
	if err := validateDigest("recipe_digest", a.RecipeDigest); err != nil {
		return err
	}
	if err := validateObligations(a.Obligations); err != nil {
		return err
	}
	if err := validateDigest("ack_id", a.AckID); err != nil {
		return err
	}
	want, err := a.ID()
	if err != nil {
		return fmt.Errorf("ack_id: %w", err)
	}
	if a.AckID != want {
		return fmt.Errorf("ack_id mismatch: got %q want %q", a.AckID, want)
	}
	return nil
}

// Matches verifies that an acknowledgement authorizes exactly the supplied
// prepared receipt, candidate binding, and recipe digest.
func (a DebtAck) Matches(receiptID string, binding Binding, recipeDigest string) error {
	if err := a.Validate(); err != nil {
		return err
	}
	if err := validateDigest("receipt_id", receiptID); err != nil {
		return err
	}
	if err := binding.Validate(); err != nil {
		return fmt.Errorf("binding: %w", err)
	}
	if err := validateDigest("recipe_digest", recipeDigest); err != nil {
		return err
	}
	if a.PreparedReceiptID != receiptID {
		return errors.New("prepared_receipt_id does not match")
	}
	if !reflect.DeepEqual(a.Binding, binding) {
		return errors.New("binding does not match")
	}
	if a.RecipeDigest != recipeDigest {
		return errors.New("recipe_digest does not match")
	}
	return nil
}

// DecodeProofContext strictly decodes and validates a bounded proof context.
func DecodeProofContext(r io.Reader) (ProofContext, error) {
	var context ProofContext
	if err := decodeStrict(r, &context); err != nil {
		return ProofContext{}, err
	}
	if err := context.Validate(); err != nil {
		return ProofContext{}, err
	}
	return context, nil
}

// DecodeDebtAck strictly decodes and validates a bounded acknowledgement.
func DecodeDebtAck(r io.Reader) (DebtAck, error) {
	var ack DebtAck
	if err := decodeStrict(r, &ack); err != nil {
		return DebtAck{}, err
	}
	if err := ack.Validate(); err != nil {
		return DebtAck{}, err
	}
	return ack, nil
}

func decodeStrict(r io.Reader, dst any) error {
	if r == nil {
		return errors.New("nil JSON reader")
	}
	data, err := io.ReadAll(io.LimitReader(r, maxJSONSize+1))
	if err != nil {
		return fmt.Errorf("read JSON: %w", err)
	}
	if len(data) > maxJSONSize {
		return fmt.Errorf("JSON exceeds %d-byte limit", maxJSONSize)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(dst); err != nil {
		return fmt.Errorf("decode JSON: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return errors.New("trailing JSON value")
		}
		return fmt.Errorf("trailing JSON: %w", err)
	}
	return nil
}

func validateObject(name, value string) error {
	if (len(value) != 40 && len(value) != 64) || !isLowerHex(value) {
		return fmt.Errorf("%s must be a lowercase 40- or 64-hex object ID", name)
	}
	return nil
}

func validateDigest(name, value string) error {
	if len(value) != 64 || !isLowerHex(value) {
		return fmt.Errorf("%s must be a lowercase 64-hex digest", name)
	}
	return nil
}

func isLowerHex(value string) bool {
	if value == "" {
		return false
	}
	_, err := hex.DecodeString(value)
	if err != nil {
		return false
	}
	return value == strings.ToLower(value)
}

func validateRef(name, value string) error {
	if !strings.HasPrefix(value, "refs/") || value == "refs/" || strings.HasSuffix(value, "/") ||
		strings.Contains(value, "//") || strings.Contains(value, "..") || strings.Contains(value, "@{") ||
		strings.ContainsAny(value, " ~^:?*[\\") {
		return fmt.Errorf("%s must be an anchored canonical refs/... name", name)
	}
	for _, component := range strings.Split(value, "/") {
		if component == "" || strings.HasPrefix(component, ".") || strings.HasSuffix(component, ".") || strings.HasSuffix(component, ".lock") {
			return fmt.Errorf("%s must be an anchored canonical refs/... name", name)
		}
		for _, r := range component {
			if r < 0x20 || r == 0x7f {
				return fmt.Errorf("%s must be an anchored canonical refs/... name", name)
			}
		}
	}
	return nil
}

func validatePaths(paths []string) error {
	if len(paths) == 0 {
		return errors.New("paths must be nonempty")
	}
	previous := ""
	for i, value := range paths {
		if value == "" || value == "." || value == ".." || path.IsAbs(value) || isDriveQualified(value) || strings.ContainsAny(value, "\\\x00") || path.Clean(value) != value || strings.HasPrefix(value, "../") {
			return fmt.Errorf("paths[%d] must be canonical repo-relative", i)
		}
		if i > 0 && value <= previous {
			return errors.New("paths must be sorted and unique")
		}
		previous = value
	}
	return nil
}

// isDriveQualified rejects Windows volume syntax without depending on the host
// OS running validation. Backslash-qualified forms are rejected separately.
func isDriveQualified(value string) bool {
	if len(value) < 2 || value[1] != ':' {
		return false
	}
	letter := value[0]
	return (letter >= 'a' && letter <= 'z') || (letter >= 'A' && letter <= 'Z')
}

func validateObligations(obligations []Obligation) error {
	if len(obligations) == 0 {
		return errors.New("obligations must be nonempty")
	}
	previous := ""
	for i, obligation := range obligations {
		if !obligationKindPattern.MatchString(obligation.Kind) {
			return fmt.Errorf("obligations[%d].kind must be a canonical lower name", i)
		}
		if i > 0 && obligation.Kind <= previous {
			return errors.New("obligations must be sorted and unique by kind")
		}
		if err := validateDigest(fmt.Sprintf("obligations[%d].id", i), obligation.ID); err != nil {
			return err
		}
		if err := validateDigest(fmt.Sprintf("obligations[%d].digest", i), obligation.Digest); err != nil {
			return err
		}
		previous = obligation.Kind
	}
	return nil
}

func digestJSON(value any) (string, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:]), nil
}
