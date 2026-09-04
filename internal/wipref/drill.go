package wipref

import (
	"archive/tar"
	"bytes"
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// RecoveryDrillSchema is the schema tag for RecoveryDrillReport.
const RecoveryDrillSchema = "fak-wip-recovery-drill/1"

// DrillOptions configures recovery drill execution.
type DrillOptions struct {
	Limit   int    `json:"limit,omitempty"`
	Session string `json:"session,omitempty"`
}

// DrillResult reports the recovery outcome for one checkpoint ref.
type DrillResult struct {
	Ref               string `json:"ref"`
	Session           string `json:"session"`
	CommitSHA         string `json:"commit_sha"`
	TreeSHA           string `json:"tree_sha"`
	RestoredPathCount int    `json:"restored_path_count"`
	ByteHashesMatch   bool   `json:"byte_hashes_match"`
	AttributionMatch  bool   `json:"attribution_match"`
	DurationMs        int64  `json:"duration_ms"`
	Status            string `json:"status"`
	Detail            string `json:"detail"`
}

// RecoveryDrillReport summarizes drill execution across all drilled checkpoints.
type RecoveryDrillReport struct {
	Schema            string        `json:"schema"`
	DrillTimestamp    time.Time     `json:"drill_timestamp"`
	Repo              string        `json:"repo"`
	TotalDrilled      int           `json:"total_drilled"`
	SuccessCount      int           `json:"success_count"`
	FailureCount      int           `json:"failure_count"`
	Results           []DrillResult `json:"results"`
	MainTreePreserved bool          `json:"main_tree_preserved"`
}

// JSON returns indented JSON encoding of the report.
func (r *RecoveryDrillReport) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

type treeStateSnapshot struct {
	head   string
	status string
	diff   string
}

func captureTreeState(ctx context.Context, dir string) treeStateSnapshot {
	head, _, _, _ := gitCmd(ctx, dir, nil, "rev-parse", "--verify", "HEAD")
	status, _, _, _ := gitCmd(ctx, dir, nil, "status", "--porcelain=v1", "-uall")
	diff, _, _, _ := gitCmd(ctx, dir, nil, "diff", "HEAD")
	return treeStateSnapshot{
		head:   strings.TrimSpace(head),
		status: status,
		diff:   diff,
	}
}

// RunRecoveryDrill exercises checkpoint recovery in detached isolation.
// It verifies that checkpointed trees restore byte-for-byte in a throwaway scratch directory
// without mutating the main checkout, index, or HEAD.
func RunRecoveryDrill(ctx context.Context, repo string, opts DrillOptions) (*RecoveryDrillReport, error) {
	if repo == "" {
		repo = "."
	}
	absRepo, err := filepath.Abs(repo)
	if err != nil {
		return nil, fmt.Errorf("resolve repo path: %w", err)
	}

	before := captureTreeState(ctx, absRepo)

	report := &RecoveryDrillReport{
		Schema:         RecoveryDrillSchema,
		DrillTimestamp: time.Now().UTC(),
		Repo:           absRepo,
		Results:        make([]DrillResult, 0),
	}

	type refItem struct {
		ref string
		sha string
	}

	var candidates []refItem

	if opts.Session != "" {
		targetRef := SessionRef(opts.Session)
		sha, _, code, _ := gitCmd(ctx, absRepo, nil, "rev-parse", "--verify", targetRef)
		sha = strings.TrimSpace(sha)
		if code != 0 || sha == "" {
			// Specific session requested but ref does not exist
			start := time.Now()
			report.Results = append(report.Results, DrillResult{
				Ref:        targetRef,
				Session:    opts.Session,
				DurationMs: time.Since(start).Milliseconds(),
				Status:     "MISSING_OBJECT",
				Detail:     fmt.Sprintf("checkpoint ref %q does not exist", targetRef),
			})
			report.TotalDrilled = 1
			report.FailureCount = 1
			after := captureTreeState(ctx, absRepo)
			report.MainTreePreserved = (before == after)
			return report, nil
		}
		candidates = append(candidates, refItem{ref: targetRef, sha: sha})
	} else {
		out, _, code, err := gitCmd(ctx, absRepo, nil, "for-each-ref", "--format=%(refname) %(objectname)", RefNamespace)
		if err != nil || code != 0 {
			return nil, fmt.Errorf("list checkpoint refs: %v", err)
		}
		lines := strings.Split(out, "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			parts := strings.Fields(line)
			if len(parts) >= 2 {
				candidates = append(candidates, refItem{ref: parts[0], sha: parts[1]})
			}
		}
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].ref < candidates[j].ref
		})
	}

	if opts.Limit > 0 && len(candidates) > opts.Limit {
		candidates = candidates[:opts.Limit]
	}

	for _, cand := range candidates {
		res := drillOneRef(ctx, absRepo, cand.ref, cand.sha)
		report.Results = append(report.Results, res)
		report.TotalDrilled++
		if res.Status == "PASS" {
			report.SuccessCount++
		} else {
			report.FailureCount++
		}
	}

	after := captureTreeState(ctx, absRepo)
	report.MainTreePreserved = (before == after)

	return report, nil
}

func drillOneRef(ctx context.Context, repo, ref, commitSHA string) DrillResult {
	start := time.Now()
	session := SessionFromRef(ref)
	res := DrillResult{
		Ref:       ref,
		Session:   session,
		CommitSHA: commitSHA,
	}

	// 1. Verify commit object exists and is valid
	_, errStr, code, err := gitCmd(ctx, repo, nil, "cat-file", "-e", commitSHA)
	if err != nil || code != 0 {
		status := classifyGitObjectError(errStr)
		if status == "" {
			if looseObjectExists(repo, commitSHA) {
				status = "CORRUPT_OBJECT"
			} else {
				status = "MISSING_OBJECT"
			}
		}
		res.Status = status
		res.Detail = fmt.Sprintf("commit object check failed: %s", strings.TrimSpace(errStr))
		res.DurationMs = time.Since(start).Milliseconds()
		return res
	}

	// 2. Read commit object to verify attribution
	commitData, errStr, code, err := gitCmd(ctx, repo, nil, "cat-file", "-p", commitSHA)
	if err != nil || code != 0 {
		status := classifyGitObjectError(errStr)
		if status == "" {
			if looseObjectExists(repo, commitSHA) {
				status = "CORRUPT_OBJECT"
			} else {
				status = "MISSING_OBJECT"
			}
		}
		res.Status = status
		res.Detail = fmt.Sprintf("read commit object failed: %s", strings.TrimSpace(errStr))
		res.DurationMs = time.Since(start).Milliseconds()
		return res
	}

	stamp, hasStamp := DecodeStamp(commitData)
	res.AttributionMatch = hasStamp && (stamp.SessionID == session || (stamp.SessionID == "" && session == ""))

	// 3. Resolve tree SHA
	treeSHA, errStr, code, err := gitCmd(ctx, repo, nil, "rev-parse", "--verify", commitSHA+"^{tree}")
	treeSHA = strings.TrimSpace(treeSHA)
	res.TreeSHA = treeSHA
	if err != nil || code != 0 || treeSHA == "" {
		status := classifyGitObjectError(errStr)
		if status == "" {
			status = "MISSING_OBJECT"
		}
		res.Status = status
		res.Detail = fmt.Sprintf("resolve tree SHA: %s", strings.TrimSpace(errStr))
		res.DurationMs = time.Since(start).Milliseconds()
		return res
	}

	// 4. Check for empty tree
	if treeSHA == "4b825dc642cb6eb9a060e54bf8d69288fbee4904" || treeSHA == "6ef19b41225c5369f1c104d45d8d85efa9b057b53b14b4b9b939dd74decc5321" {
		res.Status = "EMPTY_TREE"
		res.Detail = "checkpoint tree is empty"
		res.DurationMs = time.Since(start).Milliseconds()
		return res
	}

	// 5. Inspect tree entries via ls-tree
	lsOut, errStr, code, err := gitCmd(ctx, repo, nil, "ls-tree", "-r", "-z", treeSHA)
	if err != nil || code != 0 {
		status := classifyGitObjectError(errStr)
		if status == "" {
			if looseObjectExists(repo, treeSHA) {
				status = "CORRUPT_OBJECT"
			} else {
				status = "MISSING_OBJECT"
			}
		}
		res.Status = status
		res.Detail = fmt.Sprintf("ls-tree failed: %s", strings.TrimSpace(errStr))
		res.DurationMs = time.Since(start).Milliseconds()
		return res
	}

	treeBlobs := make(map[string]string)
	for _, item := range strings.Split(lsOut, "\x00") {
		if item == "" {
			continue
		}
		tabIdx := strings.IndexByte(item, '\t')
		if tabIdx == -1 {
			continue
		}
		meta := item[:tabIdx]
		filePath := item[tabIdx+1:]
		fields := strings.Fields(meta)
		if len(fields) >= 3 && fields[1] == "blob" {
			treeBlobs[filePath] = fields[2]
		}
	}

	if len(treeBlobs) == 0 {
		res.Status = "EMPTY_TREE"
		res.Detail = "tree contains no file entries"
		res.DurationMs = time.Since(start).Milliseconds()
		return res
	}

	// Determine target paths: if checkpoint has a parent, drill the checkpointed delta;
	// otherwise drill all files in tree.
	var targetPaths []string
	parentSHA, _, pcode, _ := gitCmd(ctx, repo, nil, "rev-parse", "--verify", commitSHA+"^1")
	parentSHA = strings.TrimSpace(parentSHA)
	if pcode == 0 && parentSHA != "" {
		diffOut, _, dcode, _ := gitCmd(ctx, repo, nil, "diff", "--name-status", parentSHA, commitSHA)
		if dcode == 0 {
			for _, line := range strings.Split(diffOut, "\n") {
				line = strings.TrimSpace(line)
				if line == "" {
					continue
				}
				parts := strings.Fields(line)
				if len(parts) >= 2 {
					action := parts[0]
					filePath := parts[1]
					if !strings.HasPrefix(action, "D") {
						if _, exists := treeBlobs[filePath]; exists {
							targetPaths = append(targetPaths, filePath)
						}
					}
				}
			}
		}
		if len(targetPaths) == 0 {
			res.Status = "EMPTY_TREE"
			res.Detail = "checkpoint delta contains no modified or added files"
			res.DurationMs = time.Since(start).Milliseconds()
			return res
		}
	} else {
		for p := range treeBlobs {
			targetPaths = append(targetPaths, p)
		}
		sort.Strings(targetPaths)
	}

	// 6. Allocate isolated scratch directory
	scratch, err := os.MkdirTemp("", "fak-wip-drill-")
	if err != nil {
		res.Status = "CORRUPT_OBJECT"
		res.Detail = fmt.Sprintf("create scratch dir: %v", err)
		res.DurationMs = time.Since(start).Milliseconds()
		return res
	}
	defer os.RemoveAll(scratch)

	// 7. Extract archive into isolation
	archiveArgs := []string{"archive", "--format=tar", treeSHA}
	if len(targetPaths) > 0 && len(targetPaths) <= 500 {
		archiveArgs = append(archiveArgs, targetPaths...)
	}
	archiveBytes, archiveErrStr, archiveCode, archiveErr := gitCmdBytes(ctx, repo, nil, archiveArgs...)
	if archiveErr != nil || archiveCode != 0 {
		status := classifyGitObjectError(archiveErrStr)
		if status == "" {
			status = "CORRUPT_OBJECT"
		}
		res.Status = status
		res.Detail = fmt.Sprintf("git archive failed: %s", strings.TrimSpace(archiveErrStr))
		res.DurationMs = time.Since(start).Milliseconds()
		return res
	}

	tr := tar.NewReader(bytes.NewReader(archiveBytes))
	restoredCount := 0
	readArchiveFail := false
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			res.Status = "CORRUPT_OBJECT"
			res.Detail = fmt.Sprintf("read archive stream: %v", err)
			readArchiveFail = true
			break
		}
		target := filepath.Join(scratch, filepath.FromSlash(hdr.Name))
		cleanTarget := filepath.Clean(target)
		cleanScratch := filepath.Clean(scratch)
		if !strings.HasPrefix(cleanTarget, cleanScratch+string(filepath.Separator)) {
			continue
		}
		switch hdr.Typeflag {
		case tar.TypeDir:
			_ = os.MkdirAll(cleanTarget, 0755)
		case tar.TypeReg, tar.TypeRegA:
			_ = os.MkdirAll(filepath.Dir(cleanTarget), 0755)
			f, err := os.OpenFile(cleanTarget, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
			if err != nil {
				res.Status = "HASH_MISMATCH"
				res.Detail = fmt.Sprintf("open restored file %s: %v", hdr.Name, err)
				readArchiveFail = true
				break
			}
			_, err = io.Copy(f, tr)
			f.Close()
			if err != nil {
				res.Status = "HASH_MISMATCH"
				res.Detail = fmt.Sprintf("write restored file %s: %v", hdr.Name, err)
				readArchiveFail = true
				break
			}
			restoredCount++
		}
		if readArchiveFail {
			break
		}
	}
	if readArchiveFail {
		res.DurationMs = time.Since(start).Milliseconds()
		return res
	}
	res.RestoredPathCount = restoredCount

	// 8. Verify byte hashes of restored files
	isSHA256 := len(treeSHA) == 64
	hashesMatch := true
	var hashDetail string
	for _, targetPath := range targetPaths {
		expectedSHA := treeBlobs[targetPath]
		target := filepath.Join(scratch, filepath.FromSlash(targetPath))
		data, err := os.ReadFile(target)
		if err != nil {
			hashesMatch = false
			hashDetail = fmt.Sprintf("restored file missing: %s", targetPath)
			break
		}
		actualSHA := gitBlobSHA(data, isSHA256 || len(expectedSHA) == 64)
		if actualSHA != expectedSHA {
			hashesMatch = false
			hashDetail = fmt.Sprintf("hash mismatch for %s: expected %s, got %s", targetPath, expectedSHA, actualSHA)
			break
		}
	}

	res.ByteHashesMatch = hashesMatch
	if !hashesMatch {
		res.Status = "HASH_MISMATCH"
		res.Detail = hashDetail
	} else if !res.AttributionMatch {
		res.Status = "HASH_MISMATCH"
		res.Detail = fmt.Sprintf("attribution mismatch: ref session %q != stamp session %q", session, stamp.SessionID)
	} else if restoredCount == 0 {
		res.Status = "EMPTY_TREE"
		res.Detail = "restored 0 files from tree"
	} else {
		res.Status = "PASS"
		res.Detail = fmt.Sprintf("restored %d paths, byte hashes & attribution verified in isolation", restoredCount)
	}

	res.DurationMs = time.Since(start).Milliseconds()
	return res
}

func gitBlobSHA(data []byte, isSHA256 bool) string {
	var h hash.Hash
	if isSHA256 {
		h = sha256.New()
	} else {
		h = sha1.New()
	}
	prefix := fmt.Sprintf("blob %d\x00", len(data))
	h.Write([]byte(prefix))
	h.Write(data)
	return hex.EncodeToString(h.Sum(nil))
}

func looseObjectExists(repo, sha string) bool {
	if len(sha) < 4 {
		return false
	}
	p := filepath.Join(repo, ".git", "objects", sha[:2], sha[2:])
	fi, err := os.Stat(p)
	return err == nil && !fi.IsDir()
}

func classifyGitObjectError(errStr string) string {
	errLower := strings.ToLower(errStr)
	if strings.Contains(errLower, "corrupt") ||
		strings.Contains(errLower, "checksum") ||
		strings.Contains(errLower, "inflate") ||
		strings.Contains(errLower, "data stream error") ||
		strings.Contains(errLower, "sha1 mismatch") ||
		strings.Contains(errLower, "sha256 mismatch") ||
		strings.Contains(errLower, "zlib") ||
		strings.Contains(errLower, "bad object") ||
		strings.Contains(errLower, "unable to unpack") {
		return "CORRUPT_OBJECT"
	}
	if strings.Contains(errLower, "missing") ||
		strings.Contains(errLower, "not a valid object") ||
		strings.Contains(errLower, "unable to read") ||
		strings.Contains(errLower, "unable to find") ||
		strings.Contains(errLower, "could not read") ||
		strings.Contains(errLower, "cannot read") ||
		strings.Contains(errLower, "does not exist") ||
		strings.Contains(errLower, "not a tree object") ||
		strings.Contains(errLower, "invalid object") ||
		strings.Contains(errLower, "unknown revision") {
		return "MISSING_OBJECT"
	}
	return ""
}

func gitCmd(ctx context.Context, dir string, env []string, args ...string) (stdout, stderr string, code int, err error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	if dir != "" {
		cmd.Dir = dir
	}
	if len(env) > 0 {
		cmd.Env = append(os.Environ(), env...)
	}
	configureDispatchHelperCommand(cmd)
	var o, e bytes.Buffer
	cmd.Stdout = &o
	cmd.Stderr = &e
	runErr := cmd.Run()
	if runErr == nil {
		return o.String(), e.String(), 0, nil
	}
	var ee *exec.ExitError
	if errors.As(runErr, &ee) {
		return o.String(), e.String(), ee.ExitCode(), nil
	}
	return "", e.String(), -1, runErr
}

func gitCmdBytes(ctx context.Context, dir string, env []string, args ...string) (stdout []byte, stderr string, code int, err error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	if dir != "" {
		cmd.Dir = dir
	}
	if len(env) > 0 {
		cmd.Env = append(os.Environ(), env...)
	}
	configureDispatchHelperCommand(cmd)
	var o, e bytes.Buffer
	cmd.Stdout = &o
	cmd.Stderr = &e
	runErr := cmd.Run()
	if runErr == nil {
		return o.Bytes(), e.String(), 0, nil
	}
	var ee *exec.ExitError
	if errors.As(runErr, &ee) {
		return o.Bytes(), e.String(), ee.ExitCode(), nil
	}
	return nil, e.String(), -1, runErr
}
