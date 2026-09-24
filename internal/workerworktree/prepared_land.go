package workerworktree

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	pathpkg "path"
	"path/filepath"
	"sort"
	"strings"
)

const (
	PreparedLandSchema       = "fak.prepared-land/v1"
	PreparedLandGateContract = "workerworktree-exact-candidate/v1"

	LandResultPrepared          = "prepared"
	LandResultPreparedMismatch  = "prepared-land-mismatch"
	LandResultPreparedReprepare = "prepared-land-reprepare"
)

type ProspectiveVerificationBinding struct {
	Command string   `json:"command"`
	Tags    []string `json:"tags,omitempty"`
}

type PreparedLandReceipt struct {
	Schema              string                         `json:"schema"`
	GateContract        string                         `json:"gate_contract"`
	ReceiptID           string                         `json:"receipt_id"`
	TargetRef           string                         `json:"target_ref"`
	ParentSHA           string                         `json:"parent_sha"`
	TreeSHA             string                         `json:"tree_sha"`
	CandidateSHA        string                         `json:"candidate_sha"`
	PathsDigest         string                         `json:"paths_digest"`
	VerifyCommandDigest string                         `json:"verify_command_digest"`
	TagsDigest          string                         `json:"tags_digest"`
	Verdict             string                         `json:"verdict"`
	RecoveryRef         string                         `json:"recovery_ref"`
	Paths               []string                       `json:"paths"`
	CandidatePaths      []string                       `json:"candidate_paths"`
	Verification        ProspectiveVerificationBinding `json:"verification"`
}

type PreparedLandExpectation struct {
	ReceiptID    string
	Paths        []string
	Verification ProspectiveVerificationBinding
}

type preparedLandCapture struct {
	binding ProspectiveVerificationBinding
	paths   []string
	receipt PreparedLandReceipt
}

func PrepareProspectiveLand(root, wtPath, baseSHA, msgFile string, paths []string, binding ProspectiveVerificationBinding, verify VerifyHook, prospectiveVerify ProspectiveVerifyHook, git GitRunner, opts ...LandOption) (PreparedLandReceipt, Result) {
	if prospectiveVerify == nil {
		return PreparedLandReceipt{}, Result{OK: false, Code: LandResultPreparedMismatch, Preserved: true, Reason: "prospective verifier is required"}
	}
	normalizedPaths, err := normalizePreparedPaths(paths)
	if err != nil || len(normalizedPaths) == 0 {
		return PreparedLandReceipt{}, Result{OK: false, Code: LandResultPreparedMismatch, Preserved: true, Reason: "explicit prepared landing paths are required", Detail: errorDetail(err)}
	}
	normalizedBinding, err := normalizeVerificationBinding(binding)
	if err != nil {
		return PreparedLandReceipt{}, Result{OK: false, Code: LandResultPreparedMismatch, Preserved: true, Reason: "invalid prospective verification binding", Detail: err.Error()}
	}
	capture := &preparedLandCapture{binding: normalizedBinding, paths: normalizedPaths}
	res := landPrepared(root, wtPath, baseSHA, msgFile, paths, verify, prospectiveVerify, capture, git, opts...)
	return capture.receipt, res
}

func AcceptPreparedLand(root, wtPath string, expected PreparedLandExpectation, git GitRunner, opts ...LandOption) Result {
	cfg := newLandConfig(opts)
	expectedID := strings.ToLower(strings.TrimSpace(expected.ReceiptID))
	if !validReceiptID(expectedID) {
		return preparedMismatch("invalid prepared landing receipt ID", "")
	}
	expectedPaths, err := normalizePreparedPaths(expected.Paths)
	if err != nil || len(expectedPaths) == 0 {
		return preparedMismatch("invalid expected prepared landing paths", errorDetail(err))
	}
	expectedBinding, err := normalizeVerificationBinding(expected.Verification)
	if err != nil {
		return preparedMismatch("invalid expected verification binding", err.Error())
	}
	path, err := PreparedLandReceiptPath(root, expectedID, git)
	if err != nil {
		return preparedReprepare("prepared landing receipt is unavailable", err.Error())
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return preparedReprepare("prepared landing receipt is unavailable", err.Error())
	}
	var receipt PreparedLandReceipt
	if err := json.Unmarshal(data, &receipt); err != nil {
		return preparedMismatch("prepared landing receipt is malformed", err.Error())
	}
	if receipt.ReceiptID != expectedID || receipt.Schema != PreparedLandSchema ||
		receipt.GateContract != PreparedLandGateContract || receipt.Verdict != "verified" {
		return preparedMismatch("prepared landing receipt contract mismatch", "")
	}
	canonicalID, err := preparedReceiptID(receipt)
	if err != nil || canonicalID != expectedID {
		return preparedMismatch("prepared landing receipt integrity mismatch", errorDetail(err))
	}
	if receipt.PathsDigest != digestStrings(expectedPaths) ||
		receipt.VerifyCommandDigest != digestStrings([]string{expectedBinding.Command}) ||
		receipt.TagsDigest != digestStrings(expectedBinding.Tags) {
		return preparedMismatch("prepared landing expectations do not match receipt", "")
	}
	receiptPaths, err := normalizePreparedPaths(receipt.Paths)
	if err != nil || len(receiptPaths) == 0 || digestStrings(receiptPaths) != receipt.PathsDigest {
		return preparedMismatch("prepared landing receipt path binding mismatch", errorDetail(err))
	}
	storedBinding, err := normalizeVerificationBinding(receipt.Verification)
	if err != nil || digestStrings([]string{storedBinding.Command}) != receipt.VerifyCommandDigest || digestStrings(storedBinding.Tags) != receipt.TagsDigest {
		return preparedMismatch("prepared landing receipt verification binding mismatch", errorDetail(err))
	}
	if got, err := normalizePreparedPaths(receipt.CandidatePaths); err != nil || len(got) == 0 {
		return preparedMismatch("prepared landing receipt has invalid candidate paths", errorDetail(err))
	} else {
		receipt.CandidatePaths = got
	}
	if rc, out := run(git, root, []string{"rev-parse", "--verify", receipt.CandidateSHA + "^{commit}"}); rc != 0 || strings.TrimSpace(out) != receipt.CandidateSHA {
		return preparedReprepare("prepared candidate commit is unavailable", tail(out, 200))
	}
	if rc, out := run(git, root, []string{"rev-parse", "--verify", receipt.CandidateSHA + "^{tree}"}); rc != 0 || strings.TrimSpace(out) != receipt.TreeSHA {
		return preparedMismatch("prepared candidate tree mismatch", tail(out, 200))
	}
	if rc, out := run(git, root, []string{"rev-parse", "--verify", receipt.RecoveryRef + "^{commit}"}); rc != 0 || strings.TrimSpace(out) != receipt.CandidateSHA {
		return preparedMismatch("prepared recovery reference is unavailable", tail(out, 200))
	}
	if refusal := revalidatePreparedLand(root, wtPath, receipt, cfg, git); !refusal.OK {
		return refusal
	}
	rc, current := run(git, root, []string{"rev-parse", "--verify", receipt.TargetRef + "^{commit}"})
	if rc != 0 || strings.TrimSpace(current) != receipt.ParentSHA {
		return preparedReprepare("prepared target moved; prepare again", tail(current, 200))
	}
	if rc, out := run(git, root, []string{"update-ref", receipt.TargetRef, receipt.CandidateSHA, receipt.ParentSHA}); rc != 0 {
		return preparedReprepare("prepared target changed during acceptance; prepare again", tail(out, 200))
	}
	detail := "recovery-ref=" + receipt.RecoveryRef
	coArgs := append([]string{"checkout", receipt.CandidateSHA, "--"}, receipt.CandidatePaths...)
	if rc, out := run(git, root, coArgs); rc != 0 {
		detail += "; landed " + shortSHA(receipt.CandidateSHA) + " but working-tree sync failed: " + tail(out, 200)
	}
	return Result{OK: true, Code: LandResultSuccess, Applied: true, Committed: true, CommitSHA: receipt.CandidateSHA,
		Reason: "accepted verified prospective landing " + shortSHA(receipt.CandidateSHA),
		Detail: detail, RecoveryRef: receipt.RecoveryRef}
}

func revalidatePreparedLand(root, wtPath string, receipt PreparedLandReceipt, cfg landConfig, git GitRunner) Result {
	if info, err := os.Stat(wtPath); err != nil || !info.IsDir() {
		return preparedReprepare("prepared worktree is unavailable", errorDetail(err))
	}
	if rc, out := run(git, wtPath, []string{"rev-parse", "--is-inside-work-tree"}); rc != 0 || strings.TrimSpace(out) != "true" {
		return preparedReprepare("prepared worktree metadata is unavailable", tail(out, 200))
	}
	rootCommon, rootErr := preparedGitCommonDir(root, git)
	worktreeCommon, worktreeErr := preparedGitCommonDir(wtPath, git)
	rootInfo, rootStatErr := os.Stat(rootCommon)
	worktreeInfo, worktreeStatErr := os.Stat(worktreeCommon)
	if rootErr != nil || worktreeErr != nil || rootStatErr != nil || worktreeStatErr != nil || !os.SameFile(rootInfo, worktreeInfo) {
		return preparedReprepare("prepared worktree belongs to a different repository", "")
	}
	// The checked-out-branch equality is only meaningful for the default land,
	// which targets whatever the root checkout has checked out. With an explicit
	// branch ref the land targets that ref regardless of the root's HEAD, so the
	// equality is not required (and would wrongly refuse a main-based land while
	// the root sits on a peer branch).
	if cfg.branchRef == "" {
		rc, branch := run(git, root, []string{"symbolic-ref", "--quiet", "HEAD"})
		if rc != 0 || strings.TrimSpace(branch) != receipt.TargetRef {
			return preparedReprepare("checked-out target branch changed; prepare again", tail(branch, 200))
		}
	}
	rc, parent := run(git, root, []string{"rev-parse", "--verify", receipt.CandidateSHA + "^1"})
	if rc != 0 || strings.TrimSpace(parent) != receipt.ParentSHA {
		return preparedMismatch("prepared candidate parent mismatch", tail(parent, 200))
	}
	rc, names := run(git, root, []string{"diff", "--name-only", "-z", receipt.ParentSHA, receipt.CandidateSHA})
	if rc != 0 {
		return preparedMismatch("could not read prepared candidate paths", tail(names, 200))
	}
	nameParts := strings.Split(names, "\x00")
	actualPaths := make([]string, 0, len(nameParts))
	for _, name := range nameParts {
		if name != "" {
			actualPaths = append(actualPaths, strings.ReplaceAll(name, "\\", "/"))
		}
	}
	actualPaths = normalizeStrings(actualPaths)
	if digestStrings(actualPaths) != digestStrings(receipt.CandidatePaths) {
		return preparedMismatch("prepared candidate changed-path set mismatch", "")
	}
	if len(cfg.LeasedGlobs) > 0 {
		if err := ValidateWorkerTreeDisjointness(actualPaths, cfg.LeasedGlobs); err != nil {
			return preparedMismatch("prepared candidate is outside the accepted lease", err.Error())
		}
	}
	statusArgs := append([]string{"status", "--porcelain", "--"}, receipt.CandidatePaths...)
	if rc, out := run(git, root, statusArgs); rc != 0 || strings.TrimSpace(out) != "" {
		return preparedReprepare("shared checkout has overlapping local changes", tail(out, 200))
	}
	if disambiguationRelevant(receipt.CandidatePaths) {
		witnesses, valid := verifyApplicableDisambiguation(root, wtPath, receipt.TreeSHA)
		if !valid {
			return preparedMismatch("prepared candidate no longer satisfies disambiguation", witnesses.compactDetail())
		}
	}
	rc, diff := run(git, root, []string{"diff", "--binary", receipt.ParentSHA, receipt.CandidateSHA})
	if rc != 0 {
		return preparedMismatch("could not read prepared candidate diff", tail(diff, 200))
	}
	rc, message := run(git, root, []string{"show", "-s", "--format=%B", receipt.CandidateSHA})
	if rc != 0 {
		return preparedMismatch("could not read prepared candidate message", tail(message, 200))
	}
	msg, err := os.CreateTemp("", "fak-prepared-msg-*.txt")
	if err != nil {
		return preparedReprepare("could not stage prepared message for policy admission", err.Error())
	}
	msgPath := msg.Name()
	defer os.Remove(msgPath)
	if _, err := msg.WriteString(message); err != nil {
		_ = msg.Close()
		return preparedReprepare("could not stage prepared message for policy admission", err.Error())
	}
	if err := msg.Close(); err != nil {
		return preparedReprepare("could not stage prepared message for policy admission", err.Error())
	}
	if refusal, fired := coreLockLandGate(root, strings.Join(actualPaths, "\n"), diff, msgPath, cfg, git); fired {
		refusal.Preserved = true
		return refusal
	}
	return Result{OK: true}
}

func persistPreparedLand(root, targetRef, parentSHA, treeSHA, candidateSHA string, candidatePaths []string, recoveryRef string, capture *preparedLandCapture, git GitRunner) (PreparedLandReceipt, error) {
	paths, err := normalizePreparedPaths(candidatePaths)
	if err != nil {
		return PreparedLandReceipt{}, fmt.Errorf("candidate paths: %w", err)
	}
	if len(paths) == 0 {
		return PreparedLandReceipt{}, errors.New("candidate paths are empty")
	}
	binding, err := normalizeVerificationBinding(capture.binding)
	if err != nil {
		return PreparedLandReceipt{}, err
	}
	receipt := PreparedLandReceipt{
		Schema: PreparedLandSchema, GateContract: PreparedLandGateContract,
		TargetRef: strings.TrimSpace(targetRef), ParentSHA: strings.TrimSpace(parentSHA),
		TreeSHA: strings.TrimSpace(treeSHA), CandidateSHA: strings.TrimSpace(candidateSHA),
		PathsDigest:         digestStrings(capture.paths),
		VerifyCommandDigest: digestStrings([]string{binding.Command}), TagsDigest: digestStrings(binding.Tags),
		Verdict: "verified", RecoveryRef: strings.TrimSpace(recoveryRef),
		Paths: capture.paths, CandidatePaths: paths, Verification: binding,
	}
	receipt.ReceiptID, err = preparedReceiptID(receipt)
	if err != nil {
		return PreparedLandReceipt{}, err
	}
	path, err := PreparedLandReceiptPath(root, receipt.ReceiptID, git)
	if err != nil {
		return PreparedLandReceipt{}, err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return PreparedLandReceipt{}, err
	}
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return PreparedLandReceipt{}, err
	}
	data = append(data, '\n')
	if existing, readErr := os.ReadFile(path); readErr == nil {
		if string(existing) == string(data) {
			return receipt, nil
		}
		return PreparedLandReceipt{}, errors.New("canonical prepared receipt already exists with different content")
	} else if !os.IsNotExist(readErr) {
		return PreparedLandReceipt{}, readErr
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".prepared-land-*.tmp")
	if err != nil {
		return PreparedLandReceipt{}, err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err = tmp.Write(data); err == nil {
		err = tmp.Close()
	} else {
		_ = tmp.Close()
	}
	if err != nil {
		return PreparedLandReceipt{}, err
	}
	if err := os.Rename(tmpName, path); err != nil {
		return PreparedLandReceipt{}, err
	}
	return receipt, nil
}

// PreparedLandReceiptPath returns the canonical repository-local receipt path.
func PreparedLandReceiptPath(root, receiptID string, git GitRunner) (string, error) {
	id := strings.ToLower(strings.TrimSpace(receiptID))
	if !validReceiptID(id) {
		return "", errors.New("invalid receipt ID")
	}
	common, err := preparedGitCommonDir(root, git)
	if err != nil {
		return "", err
	}
	return filepath.Join(filepath.Clean(common), "fak-prepared-land", id+".json"), nil
}

func preparedGitCommonDir(root string, git GitRunner) (string, error) {
	rc, out := run(git, root, []string{"rev-parse", "--git-common-dir"})
	if rc != 0 || strings.TrimSpace(out) == "" {
		return "", errors.New("could not resolve git common directory")
	}
	common := strings.TrimSpace(out)
	if !filepath.IsAbs(common) {
		common = filepath.Join(root, common)
	}
	abs, err := filepath.Abs(common)
	if err != nil {
		return "", err
	}
	return filepath.Clean(abs), nil
}

func preparedReceiptID(receipt PreparedLandReceipt) (string, error) {
	receipt.ReceiptID = ""
	data, err := json.Marshal(receipt)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

func normalizeVerificationBinding(binding ProspectiveVerificationBinding) (ProspectiveVerificationBinding, error) {
	binding.Command = strings.TrimSpace(binding.Command)
	if binding.Command == "" {
		return ProspectiveVerificationBinding{}, errors.New("verification command is required")
	}
	binding.Tags = normalizeStrings(binding.Tags)
	return binding, nil
}

func normalizePreparedPaths(paths []string) ([]string, error) {
	out := make([]string, 0, len(paths))
	for _, raw := range paths {
		raw = strings.TrimSpace(raw)
		if strings.IndexByte(raw, 0) >= 0 || filepath.IsAbs(raw) || filepath.VolumeName(raw) != "" {
			return nil, fmt.Errorf("unsafe path %q", raw)
		}
		p := pathpkg.Clean(strings.ReplaceAll(raw, "\\", "/"))
		if p == "" || p == "." || p == ".." || strings.HasPrefix(p, "../") {
			return nil, fmt.Errorf("unsafe path %q", raw)
		}
		out = append(out, p)
	}
	return normalizeStrings(out), nil
}

func normalizeStrings(values []string) []string {
	seen := make(map[string]bool, len(values))
	out := make([]string, 0, len(values))
	for _, raw := range values {
		value := strings.TrimSpace(raw)
		if value == "" || seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func digestStrings(values []string) string {
	sum := sha256.Sum256([]byte(strings.Join(values, "\x00")))
	return hex.EncodeToString(sum[:])
}

func validReceiptID(id string) bool {
	if len(id) != sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(id)
	return err == nil
}

func preparedMismatch(reason, detail string) Result {
	return Result{OK: false, Code: LandResultPreparedMismatch, Preserved: true, Reason: reason, Detail: detail}
}

func preparedReprepare(reason, detail string) Result {
	return Result{OK: false, Code: LandResultPreparedReprepare, Preserved: true, Reason: reason, Detail: detail}
}

func errorDetail(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}
