package workerworktree

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/pendingadmission"
)

const (
	PendingLandSchema       = "fak.pending-land/v1"
	PendingLandContract     = "workerworktree-exact-candidate-pending-debt/v1"
	LandResultPending       = "pending-prepared"
	LandResultLandedPending = "LANDED_PENDING"

	maxPendingLandReceiptBytes = 256 << 10
)

// PendingLandReceipt records an exact off-branch candidate awaiting durable
// obligations. State pending is deliberately distinct from a verified verdict.
type PendingLandReceipt struct {
	Schema                  string                     `json:"schema"`
	Contract                string                     `json:"contract"`
	ReceiptID               string                     `json:"receipt_id"`
	State                   string                     `json:"state"`
	Binding                 pendingadmission.Binding   `json:"binding"`
	CandidatePaths          []string                   `json:"candidate_paths"`
	WorktreeDigest          string                     `json:"worktree_digest"`
	CommonDirDigest         string                     `json:"common_dir_digest"`
	WorktreeBase            string                     `json:"worktree_base"`
	WorktreeDiffDigest      string                     `json:"worktree_diff_digest"`
	CoordinatorIntentDigest string                     `json:"coordinator_intent_digest"`
	MessageDigest           string                     `json:"message_digest"`
	Disambiguation          PreparedLandDisambiguation `json:"disambiguation"`
}

// PendingLandExpectation is the caller-controlled portion of pending accept.
type PendingLandExpectation struct {
	ReceiptID    string
	RecipeDigest string
	Ack          pendingadmission.DebtAck
}

type landCandidateCapture interface {
	persistCandidateLand(root, wtPath, targetRef, parentSHA, treeSHA, candidateSHA string, candidatePaths []string, recoveryRef, worktreeBase, worktreeDiff string, disambiguation *DisambiguationWitnesses, git GitRunner) (Result, error)
}

type pendingLandCapture struct {
	context                 pendingadmission.ProofContext
	coordinatorIntentDigest string
	receipt                 PendingLandReceipt
}

// This method preserves the verified-v1 capture while allowing land.go to
// persist a distinct candidate kind without inspecting either receipt schema.
func (c *preparedLandCapture) persistCandidateLand(root, wtPath, targetRef, parentSHA, treeSHA, candidateSHA string, candidatePaths []string, recoveryRef, _, _ string, disambiguation *DisambiguationWitnesses, git GitRunner) (Result, error) {
	receipt, err := persistPreparedLand(root, wtPath, targetRef, parentSHA, treeSHA, candidateSHA, candidatePaths, recoveryRef, disambiguation, c, git)
	if err != nil {
		return Result{}, err
	}
	c.receipt = receipt
	return Result{OK: true, Code: LandResultPrepared, Applied: false, Committed: false, Preserved: true,
		Reason: "verified prospective landing prepared at " + shortSHA(candidateSHA)}, nil
}

// PreparePendingLand constructs and anchors an exact candidate, then persists
// its pending identity. It does not run or represent a verification verdict.
func PreparePendingLand(root, wtPath, baseSHA, msgFile string, paths []string, context pendingadmission.ProofContext, git GitRunner, opts ...LandOption) (PendingLandReceipt, Result) {
	if err := context.Validate(); err != nil {
		return PendingLandReceipt{}, pendingLandMismatch("invalid pending proof context", err.Error())
	}
	normalized, err := normalizePreparedPaths(paths)
	if err != nil || len(normalized) == 0 {
		return PendingLandReceipt{}, pendingLandMismatch("explicit pending landing paths are required", errorDetail(err))
	}
	intent, err := LoadIntent(wtPath)
	if err != nil {
		return PendingLandReceipt{}, pendingLandMismatch("prepared worker intent is unavailable", err.Error())
	}
	capture := &pendingLandCapture{context: context, coordinatorIntentDigest: digestPendingJSON(intent)}
	res := landPrepared(root, wtPath, baseSHA, msgFile, normalized, nil, nil, capture, git, opts...)
	return capture.receipt, res
}

func (c *pendingLandCapture) persistCandidateLand(root, wtPath, targetRef, parentSHA, treeSHA, candidateSHA string, candidatePaths []string, recoveryRef, worktreeBase, worktreeDiff string, disambiguation *DisambiguationWitnesses, git GitRunner) (Result, error) {
	receipt, err := persistPendingLand(root, wtPath, targetRef, parentSHA, treeSHA, candidateSHA, candidatePaths, recoveryRef, worktreeBase, worktreeDiff, disambiguation, c.context, c.coordinatorIntentDigest, git)
	if err != nil {
		return Result{}, err
	}
	c.receipt = receipt
	return Result{OK: true, Code: LandResultPending, Applied: false, Committed: false, Preserved: true,
		Reason: "pending landing prepared at " + shortSHA(candidateSHA)}, nil
}

// AcceptPendingLand validates a durable pending-debt acknowledgement before
// revalidating the candidate and performing the same expected-old target CAS
// used by verified prepared landing.
func AcceptPendingLand(root, wtPath string, expected PendingLandExpectation, git GitRunner, opts ...LandOption) Result {
	cfg := newLandConfig(opts)
	expectedID := expected.ReceiptID
	if !validPendingDigest(expectedID) {
		return pendingLandMismatch("invalid pending landing receipt ID", "")
	}
	if !validPendingDigest(expected.RecipeDigest) {
		return pendingLandMismatch("invalid expected recipe digest", "")
	}
	path, err := PendingLandReceiptPath(root, expectedID, git)
	if err != nil {
		return pendingLandReprepare("pending landing receipt is unavailable", err.Error())
	}
	receipt, err := loadPendingLandReceipt(path)
	if err != nil {
		return pendingLandMismatch("pending landing receipt is malformed", err.Error())
	}
	if receipt.ReceiptID != expectedID || receipt.Schema != PendingLandSchema || receipt.Contract != PendingLandContract || receipt.State != "pending" {
		return pendingLandMismatch("pending landing receipt contract mismatch", "")
	}
	canonicalID, err := pendingLandReceiptID(receipt)
	if err != nil || canonicalID != expectedID {
		return pendingLandMismatch("pending landing receipt integrity mismatch", errorDetail(err))
	}
	if err := receipt.Binding.Validate(); err != nil {
		return pendingLandMismatch("pending landing binding mismatch", err.Error())
	}
	for _, digest := range []string{receipt.WorktreeDigest, receipt.CommonDirDigest, receipt.WorktreeDiffDigest, receipt.CoordinatorIntentDigest, receipt.MessageDigest} {
		if !validPendingDigest(digest) {
			return pendingLandMismatch("pending landing receipt digest is invalid", "")
		}
	}
	paths, err := normalizePreparedPaths(receipt.CandidatePaths)
	if err != nil || len(paths) == 0 || digestPendingJSON(paths) != receipt.Binding.PathsDigest || !pendingEqualStrings(paths, receipt.Binding.Paths) {
		return pendingLandMismatch("pending landing path binding mismatch", errorDetail(err))
	}
	receipt.CandidatePaths = paths
	if err := expected.Ack.Matches(expectedID, receipt.Binding, expected.RecipeDigest); err != nil {
		return pendingLandMismatch("pending debt acknowledgement mismatch", err.Error())
	}
	if got := pathIdentityDigest(wtPath); got != receipt.WorktreeDigest {
		return pendingLandReprepare("prepared worktree identity changed", "")
	}
	common, err := preparedGitCommonDir(wtPath, git)
	if err != nil || pathIdentityDigest(common) != receipt.CommonDirDigest {
		return pendingLandReprepare("prepared worktree repository identity changed", errorDetail(err))
	}
	intent, err := LoadIntent(wtPath)
	if err != nil || digestPendingJSON(intent) != receipt.CoordinatorIntentDigest {
		return pendingLandReprepare("prepared worker intent changed", errorDetail(err))
	}
	if !validPendingObject(receipt.WorktreeBase) {
		return pendingLandMismatch("pending worktree base is invalid", "")
	}
	diffArgs := append([]string{"diff", "--binary", receipt.WorktreeBase, "--"}, receipt.Binding.Paths...)
	if rc, currentDiff := run(git, wtPath, diffArgs); rc != 0 || digestPendingString(currentDiff) != receipt.WorktreeDiffDigest {
		return pendingLandReprepare("prepared worktree intent changed", tail(currentDiff, 200))
	}
	if rc, out := run(git, root, []string{"rev-parse", "--verify", receipt.Binding.CandidateCommit + "^{commit}"}); rc != 0 || strings.TrimSpace(out) != receipt.Binding.CandidateCommit {
		return pendingLandReprepare("pending candidate commit is unavailable", tail(out, 200))
	}
	if rc, out := run(git, root, []string{"rev-parse", "--verify", receipt.Binding.ParentCommit + "^{tree}"}); rc != 0 || strings.TrimSpace(out) != receipt.Binding.ParentTree {
		return pendingLandMismatch("pending parent tree mismatch", tail(out, 200))
	}
	if rc, out := run(git, root, []string{"rev-parse", "--verify", receipt.Binding.CandidateCommit + "^{tree}"}); rc != 0 || strings.TrimSpace(out) != receipt.Binding.CandidateTree {
		return pendingLandMismatch("pending candidate tree mismatch", tail(out, 200))
	}
	if rc, out := run(git, root, []string{"rev-parse", "--verify", receipt.Binding.RecoveryRef + "^{commit}"}); rc != 0 || strings.TrimSpace(out) != receipt.Binding.CandidateCommit {
		return pendingLandMismatch("pending recovery reference is unavailable", tail(out, 200))
	}
	if rc, message := run(git, root, []string{"show", "-s", "--format=%B", receipt.Binding.CandidateCommit}); rc != 0 || digestPendingString(message) != receipt.MessageDigest {
		return pendingLandMismatch("pending candidate message mismatch", tail(message, 200))
	}
	legacy := pendingAsPreparedReceipt(receipt)
	if refusal := validatePreparedLandDisambiguation(root, wtPath, legacy, git); !refusal.OK {
		return refusal
	}
	if refusal := revalidatePreparedLand(root, wtPath, legacy, cfg, git); !refusal.OK {
		return refusal
	}
	rc, current := run(git, root, []string{"rev-parse", "--verify", receipt.Binding.TargetRef + "^{commit}"})
	if rc != 0 || strings.TrimSpace(current) != receipt.Binding.ParentCommit {
		return pendingLandReprepare("pending target moved; prepare again", tail(current, 200))
	}
	if rc, out := run(git, root, []string{"update-ref", receipt.Binding.TargetRef, receipt.Binding.CandidateCommit, receipt.Binding.ParentCommit}); rc != 0 {
		return pendingLandReprepare("pending target changed during acceptance; prepare again", tail(out, 200))
	}
	sync := syncSharedCheckout(git, root, receipt.Binding.ParentCommit, receipt.Binding.CandidateCommit, receipt.CandidatePaths)
	return withSharedSync(Result{OK: true, Code: LandResultLandedPending, Applied: true, Committed: true, CommitSHA: receipt.Binding.CandidateCommit,
		Reason: "committed-pending " + shortSHA(receipt.Binding.CandidateCommit), Detail: "recovery-ref=" + receipt.Binding.RecoveryRef,
		RecoveryRef: receipt.Binding.RecoveryRef}, sync, receipt.Binding.CandidateCommit)
}

func persistPendingLand(root, wtPath, targetRef, parentSHA, treeSHA, candidateSHA string, candidatePaths []string, recoveryRef, worktreeBase, worktreeDiff string, disambiguation *DisambiguationWitnesses, context pendingadmission.ProofContext, coordinatorIntentDigest string, git GitRunner) (PendingLandReceipt, error) {
	paths, err := normalizePreparedPaths(candidatePaths)
	if err != nil {
		return PendingLandReceipt{}, fmt.Errorf("candidate paths: %w", err)
	}
	if len(paths) == 0 {
		return PendingLandReceipt{}, errors.New("candidate paths are empty")
	}
	parentTree, err := preparedLandTreeSHA(root, parentSHA, git)
	if err != nil {
		return PendingLandReceipt{}, fmt.Errorf("parent tree: %w", err)
	}
	disambiguationBinding, err := bindPreparedLandDisambiguation(root, wtPath, treeSHA, paths, disambiguation, git)
	if err != nil {
		return PendingLandReceipt{}, err
	}
	binding := pendingadmission.Binding{
		TargetRef: strings.TrimSpace(targetRef), ParentCommit: strings.TrimSpace(parentSHA), ParentTree: parentTree,
		CandidateCommit: strings.TrimSpace(candidateSHA), CandidateTree: strings.TrimSpace(treeSHA), RecoveryRef: strings.TrimSpace(recoveryRef),
		Paths: paths, PathsDigest: digestPendingJSON(paths), ProofContext: context, ContextDigest: digestPendingJSON(context),
	}
	if err := binding.Validate(); err != nil {
		return PendingLandReceipt{}, err
	}
	common, err := preparedGitCommonDir(wtPath, git)
	if err != nil {
		return PendingLandReceipt{}, err
	}
	rc, message := run(git, root, []string{"show", "-s", "--format=%B", candidateSHA})
	if rc != 0 {
		return PendingLandReceipt{}, errors.New("could not read pending candidate message")
	}
	receipt := PendingLandReceipt{
		Schema: PendingLandSchema, Contract: PendingLandContract, State: "pending", Binding: binding, CandidatePaths: paths,
		WorktreeDigest: pathIdentityDigest(wtPath), CommonDirDigest: pathIdentityDigest(common), WorktreeBase: worktreeBase,
		WorktreeDiffDigest: digestPendingString(worktreeDiff), CoordinatorIntentDigest: coordinatorIntentDigest,
		MessageDigest:  digestPendingString(message),
		Disambiguation: disambiguationBinding,
	}
	if !validPendingObject(receipt.WorktreeBase) {
		return PendingLandReceipt{}, errors.New("pending worktree base is invalid")
	}
	receipt.ReceiptID, err = pendingLandReceiptID(receipt)
	if err != nil {
		return PendingLandReceipt{}, err
	}
	path, err := PendingLandReceiptPath(root, receipt.ReceiptID, git)
	if err != nil {
		return PendingLandReceipt{}, err
	}
	if err := writePendingLandReceipt(path, receipt); err != nil {
		return PendingLandReceipt{}, err
	}
	readback, err := loadPendingLandReceipt(path)
	if err != nil {
		return PendingLandReceipt{}, fmt.Errorf("pending landing receipt readback mismatch: %w", err)
	}
	if readback.ReceiptID != receipt.ReceiptID {
		return PendingLandReceipt{}, errors.New("pending landing receipt readback mismatch")
	}
	return receipt, nil
}

func pendingAsPreparedReceipt(receipt PendingLandReceipt) PreparedLandReceipt {
	return PreparedLandReceipt{
		TargetRef: receipt.Binding.TargetRef, ParentSHA: receipt.Binding.ParentCommit, TreeSHA: receipt.Binding.CandidateTree,
		CandidateSHA: receipt.Binding.CandidateCommit, RecoveryRef: receipt.Binding.RecoveryRef,
		Paths: receipt.Binding.Paths, CandidatePaths: receipt.CandidatePaths, Disambiguation: receipt.Disambiguation,
	}
}

// PendingLandReceiptPath returns the repository-local path for a pending-only receipt.
func PendingLandReceiptPath(root, receiptID string, git GitRunner) (string, error) {
	id := strings.ToLower(strings.TrimSpace(receiptID))
	if !validReceiptID(id) {
		return "", errors.New("invalid receipt ID")
	}
	common, err := preparedGitCommonDir(root, git)
	if err != nil {
		return "", err
	}
	return filepath.Join(filepath.Clean(common), "fak-pending-land", id+".json"), nil
}

func pendingLandReceiptID(receipt PendingLandReceipt) (string, error) {
	receipt.ReceiptID = ""
	data, err := json.Marshal(receipt)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

func writePendingLandReceipt(path string, receipt PendingLandReceipt) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if existing, readErr := os.ReadFile(path); readErr == nil {
		if bytes.Equal(existing, data) {
			return nil
		}
		return errors.New("canonical pending landing receipt already exists with different content")
	} else if !os.IsNotExist(readErr) {
		return readErr
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".pending-land-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err = tmp.Write(data); err == nil {
		err = tmp.Sync()
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	// Windows has no portable directory metadata flush in Go. The synced temp,
	// same-directory rename and strict readback do not claim power-loss durability.
	if err := os.Rename(tmpName, path); err != nil {
		return err
	}
	if runtime.GOOS != "windows" {
		if dir, openErr := os.Open(filepath.Dir(path)); openErr == nil {
			err = dir.Sync()
			_ = dir.Close()
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func loadPendingLandReceipt(path string) (PendingLandReceipt, error) {
	f, err := os.Open(path)
	if err != nil {
		return PendingLandReceipt{}, err
	}
	defer f.Close()
	limited := io.LimitReader(f, maxPendingLandReceiptBytes+1)
	data, err := io.ReadAll(limited)
	if err != nil {
		return PendingLandReceipt{}, err
	}
	if len(data) > maxPendingLandReceiptBytes {
		return PendingLandReceipt{}, errors.New("pending landing receipt exceeds size limit")
	}
	var receipt PendingLandReceipt
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&receipt); err != nil {
		return PendingLandReceipt{}, err
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return PendingLandReceipt{}, errors.New("pending landing receipt contains trailing JSON")
	}
	return receipt, nil
}

func pathIdentityDigest(path string) string {
	abs, err := filepath.Abs(path)
	if err != nil {
		abs = path
	}
	identity := filepath.Clean(abs)
	if runtime.GOOS == "windows" {
		identity = strings.ToLower(identity)
	}
	return digestPendingString(identity)
}

func digestPendingJSON(value any) string {
	data, _ := json.Marshal(value)
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func digestPendingString(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}

func validPendingDigest(value string) bool {
	if len(value) != sha256.Size*2 || value != strings.ToLower(value) {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func validPendingObject(value string) bool {
	if (len(value) != 40 && len(value) != 64) || value != strings.ToLower(value) {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func pendingEqualStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func pendingLandMismatch(reason, detail string) Result {
	return Result{OK: false, Code: "pending-land-mismatch", Preserved: true, Reason: reason, Detail: detail}
}

func pendingLandReprepare(reason, detail string) Result {
	return Result{OK: false, Code: "pending-land-reprepare", Preserved: true, Reason: reason, Detail: detail}
}
