package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/dojocal"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
	"github.com/anthony-chaudhary/fak/internal/witness"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

// cmdWorktreeVerb fronts `fak worktree <sub>`. Today it hosts the `worker`
// subcommand (the CLI face of internal/workerworktree, #3182). The sibling
// `witness` subcommand is authored separately (cmd/fak/worktree.go) and folds in
// here once it lands on trunk — until then this dispatcher owns the `worktree`
// verb so `fak worktree worker` is reachable. Kept a distinct symbol from that
// in-flight file's cmdWorktree so the shared, peer-dirty working tree still
// builds while both land.
func cmdWorktreeVerb(argv []string) {
	if len(argv) == 0 {
		worktreeWorkerUsage()
		os.Exit(2)
	}
	switch argv[0] {
	case "worker":
		cmdWorktreeWorker(argv[1:])
	case "-h", "--help", "help":
		worktreeWorkerUsage()
	default:
		fmt.Fprintf(os.Stderr, "fak worktree: unknown subcommand %q\n", argv[0])
		worktreeWorkerUsage()
		os.Exit(2)
	}
}

func worktreeWorkerUsage() {
	fmt.Fprintln(os.Stderr, strings.TrimSpace(`
fak worktree <subcommand>

  worker <op>   Per-worker git worktree isolation (#3182). Ops:
      prepare --lane <l> --key <k> [--base-sha S] [--wt-root D]
              [--lease-id ID] [--owner-pid PID] [--capacity-reason WHY]
              [--sandbox-compatible]
                   Create ONE worker's DETACHED worktree pinned at trunk HEAD
                   (or --base-sha), stamped with owner PID, lease, and timestamp.
                   Above the advisory setpoint of 50, --capacity-reason records
                   why growth is needed; omission warns but never blocks prepare.
                   --sandbox-compatible translates the .git pointer to a localized
                   bridge for strict host sandboxes.
                   Prints {ok, path, base_sha, reused, env, ...}.
      land --worktree D [--base-sha S] [--msg-file F] [--paths p ...] [--verify go-build]
           [--core-lock-maintenance-witness CLAIM] [--recovery-remote R]
           [--require-remote-recovery] [--disambiguation-timeout-ms N]
           [--require-test-witness]
                   Apply the worktree's diff-since-base onto the trunk as one
                   signed-off commit. Prints {ok, applied, committed, ...}.
                   The optional disambiguation deadline is 1..900000 ms and uses
                   the oracle's existing resolver; omitted means 120000 ms.
                   Managed issue lands re-verify the Top-5 comment. Roll back with
                   FAK_THOUGHT_CHECK_MODE=observe|off (default: enforce).
                   A diff touching a hard-self core-locked path is REFUSED with
                   CORE_SELF_MODIFY unless the witness claim (flag, or a
                   Core-lock-maintenance-witness: trailer in the commit message)
                   resolves CONFIRMED — the same lock fak commit enforces.
                   --require-test-witness requires an independently verified test
                   witness receipt in the worktree, refusing unverified claims with
                   UNWITNESSED_WORKER_CLAIM.
      reap --worktree D [--superseded-by SHA] [--max-wait D]
                   Release ONE clean worker worktree within a shared deadline. A dirty
                   worktree is preserved by default. --superseded-by authorizes force
                   removal only when SHA is on trunk and exactly matches its bytes.
                   Prints a typed {ok, code, removed|preserved, ...} receipt.
      reap --all-cold [--apply] [--age-floor-min N] [--even-if-unlanded]
                   Bulk cold sweep: enumerate every worker worktree and reap only the
                   COLD ones — lane lease dead, past the age floor, AND working tree
                   clean. One still holding uncommitted work is KEPT and reported as
                   held_by_work: land or abandon that diff to reclaim its disk.
                   DRY-RUN by default — reports the would-reap set and deletes nothing;
                   pass --apply (or FAK_WORKTREE_COLD_COLLECT=apply) to actually collect.
                   --even-if-unlanded also collects the held ones, DESTROYING that work.
      gc [--max-age D] [--dry-run|--apply]
                   Owner-stamped leak GC. Selects only old, clean worktrees whose
                   owner PID is dead AND stamped lease is released. DRY-RUN by default;
                   --apply force-removes selected worktrees and prunes git admin entries.
      list [--json] [--capacity-reason WHY] [--remote R] [--fetch]
           [--worker NAME|PATH] [--session SESS] [--timeout D]
                   List the live per-worker worktrees. The default preserves the
                   existing {count, paths, inventory} output; --json emits the
                   typed association/liveness/cleanliness/lifecycle inventory.
      defaults [--json] [--root <repo>]
                   Expose portable managed-worker worktree defaults, root location,
                   and supported environment overrides (guaranteed read-only).
      publish --remote R [--dry-run|--apply]
                   Explicitly publish one bounded path-scrubbed per-host snapshot.
      recover [--remote R] [--fetch] [--cleanup REF] [--force]
              [--cleanup-remote REF] [--apply] [--allow-peer] [--worktree-name NAME]
                   List durable off-branch land candidates and their LANDED or
                   RECOVERABLE state. Cleanup refuses unlanded refs unless forced.

See docs/managed-worker-worktrees.md for the complete operator guide.
`))
}

// cmdWorktreeWorker routes `fak worktree worker <op>` to the matching
// internal/workerworktree primitive and prints exactly one JSON object, mirroring
// the tools/worker_worktree.py CLI contract. Production drives the real git via the
// package's default runner (nil GitRunner). Every op fails open: a git error is a
// JSON result with ok=false, never a crash.
func cmdWorktreeWorker(argv []string) {
	if len(argv) == 0 {
		worktreeWorkerUsage()
		os.Exit(2)
	}
	switch argv[0] {
	case "prepare":
		worktreeWorkerPrepare(argv[1:])
	case "land":
		worktreeWorkerLand(argv[1:])
	case "reap":
		worktreeWorkerReap(argv[1:])
	case "gc":
		worktreeWorkerGC(argv[1:])
	case "list":
		worktreeWorkerList(argv[1:])
	case "publish":
		worktreeWorkerPublish(argv[1:])
	case "recover":
		worktreeWorkerRecover(argv[1:])
	case "defaults":
		worktreeWorkerDefaults(argv[1:])
	case "-h", "--help", "help":
		worktreeWorkerUsage()
	default:
		fmt.Fprintf(os.Stderr, "fak worktree worker: unknown op %q\n", argv[0])
		worktreeWorkerUsage()
		os.Exit(2)
	}
}

// worktreeWorkerRoot resolves the repo root a worker op runs against: the explicit
// --root, else discovered from cwd. An empty result is a usage error (mirrors the
// Python CLI, which requires a resolvable repo).
func worktreeWorkerRoot(flagVal string) string {
	root := strings.TrimSpace(flagVal)
	if root == "" {
		root = discoverRepoRoot()
	}
	if root == "" {
		fmt.Fprintln(os.Stderr, "fak worktree worker: could not resolve a git repo root (pass --root)")
		os.Exit(2)
	}
	return root
}

// worktreeWorkerEmit prints one JSON object (compact, one line) — the single
// machine-readable result the caller parses. Uses SetEscapeHTML(false) so paths
// with `&` etc. render verbatim.
func worktreeWorkerEmit(v any) {
	enc := json.NewEncoder(os.Stdout)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}

func worktreeWorkerProgressEmitter(w io.Writer) func(workerworktree.LandProgressEvent) {
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	return func(event workerworktree.LandProgressEvent) { _ = enc.Encode(event) }
}

// worktreePrepareOut is the prepare JSON: the primitive's Result plus the child
// env the caller needs to spawn the worker in the isolated worktree (the Python
// CLI adds `env` on a successful prepare the same way). Embedding flattens Result's
// fields to the top level.
type worktreePrepareOut struct {
	workerworktree.Result
	Env               map[string]string               `json:"env,omitempty"`
	Capacity          workerworktree.CapacityAdvisory `json:"capacity"`
	SandboxCompatible bool                            `json:"sandbox_compatible,omitempty"`
}

func worktreeWorkerPrepare(argv []string) {
	fs := flag.NewFlagSet("worktree worker prepare", flag.ExitOnError)
	lane := fs.String("lane", "", "worker's lane (e.g. cmd, gateway) — a segment of the worktree dir name")
	key := fs.String("key", "", "worker's unique key (issue number, wave id, pid) — hashed into the dir name")
	baseSHA := fs.String("base-sha", "", "commit to pin the detached worktree at (default: trunk HEAD)")
	leaseID := fs.String("lease-id", "", "lease identity to retain in the owner stamp (default: FAK_LEASE_ID or resolve-<lane>)")
	ownerPID := fs.Int("owner-pid", os.Getpid(), "owner process PID to retain in the owner stamp")
	capacityReason := fs.String("capacity-reason", "", "why worker-worktree growth above the advisory setpoint is needed (advisory; never blocks)")
	message := fs.String("message", "", "intended signed commit message retained for lifecycle recovery")
	sandboxCompatible := fs.Bool("sandbox-compatible", false, "localize worktree gitdir pointer for strict host sandboxes")
	var paths repeatedString
	fs.Var(&paths, "path", "explicit intended land path (repeatable; required with --message for LAND_READY inventory)")
	wtRoot := fs.String("wt-root", "", "parent dir for the worktree (default: FLEET_WORKER_WORKTREE_ROOT or per-OS scratch)")
	root := fs.String("root", "", "repo root (default: discover from cwd)")
	fs.Parse(argv)

	repoRoot := worktreeWorkerRoot(*root)
	capacityCensus := workerworktree.CapacityCensusFor(repoRoot, nil)
	owner := workerworktree.OwnerStamp{PID: *ownerPID, LeaseID: strings.TrimSpace(*leaseID), CreatedAt: time.Now().UTC()}
	if owner.LeaseID == "" {
		owner.LeaseID = strings.TrimSpace(os.Getenv("FAK_LEASE_ID"))
	}
	if owner.LeaseID == "" && strings.TrimSpace(*lane) != "" {
		owner.LeaseID = "resolve-" + strings.TrimSpace(*lane)
	}
	res := workerworktree.PrepareOwnedBounded(repoRoot, *lane, *key, strings.TrimSpace(*baseSHA), strings.TrimSpace(*wtRoot), owner, 2*time.Minute)
	prospectiveCount := len(capacityCensus.Paths)
	if res.OK && !res.Reused {
		prospectiveCount++
	}
	capacity := worktreeWorkerCapacityAdvisory(repoRoot, capacityCensus, prospectiveCount, *capacityReason, nil)
	out := worktreePrepareOut{Result: res, Capacity: capacity, SandboxCompatible: *sandboxCompatible}
	if res.OK && res.Path != "" {
		hasInclude := false
		if fi, err := os.Stat(filepath.Join(repoRoot, ".worktreeinclude")); err == nil && !fi.IsDir() {
			hasInclude = true
		} else if fi, err := os.Stat(filepath.Join(res.Path, ".worktreeinclude")); err == nil && !fi.IsDir() {
			hasInclude = true
		}
		if hasInclude {
			if err := dojocal.ApplyWorktreeInclude(res.Path, nil); err != nil {
				res.OK, res.Reason = false, "apply worktree include: "+err.Error()
				out.Result = res
			}
		}
		if res.OK && *sandboxCompatible {
			if err := localizeWorktreeGitdir(repoRoot, res.Path); err != nil {
				res.OK, res.Reason = false, "sandbox gitdir localization: "+err.Error()
				out.Result = res
			}
		}
		if res.OK {
			out.Env = workerworktree.WorktreeEnv(nil, res.Path)
			if strings.TrimSpace(*message) != "" || len(paths) > 0 {
				if strings.TrimSpace(*message) == "" || len(paths) == 0 {
					res.OK, res.Reason = false, "--message and at least one --path must be supplied together"
					out.Result = res
					out.Env = nil
				} else if err := workerworktree.SaveIntent(res.Path, res.BaseSHA, *message, paths); err != nil {
					res.OK, res.Reason = false, "save worker land intent: "+err.Error()
					out.Result = res
					out.Env = nil
				}
			}
		}
	}
	worktreeWorkerWriteCapacityHuman(os.Stderr, capacity)
	worktreeWorkerEmit(out)
	if !res.OK {
		os.Exit(1)
	}
}

func localizeWorktreeGitdir(repoRoot, wtPath string) error {
	dotGit := filepath.Join(wtPath, ".git")
	b, err := os.ReadFile(dotGit)
	if err != nil {
		return fmt.Errorf("read .git pointer: %w", err)
	}
	line := strings.TrimSpace(string(b))
	rest, ok := strings.CutPrefix(line, "gitdir:")
	if !ok {
		return fmt.Errorf(".git is not a gitdir pointer: %q", line)
	}
	targetDir := strings.TrimSpace(rest)
	if targetDir == "" {
		return fmt.Errorf(".git contains empty gitdir pointer")
	}
	if !filepath.IsAbs(targetDir) {
		targetDir = filepath.Join(wtPath, targetDir)
	}
	targetDir = filepath.Clean(targetDir)

	// If targetDir is already inside wtPath, nothing to localize.
	rel, err := filepath.Rel(wtPath, targetDir)
	if err == nil && !strings.HasPrefix(rel, "..") && rel != "." {
		return nil
	}

	var commonGitDir string
	if commonBytes, err := os.ReadFile(filepath.Join(targetDir, "commondir")); err == nil {
		c := strings.TrimSpace(string(commonBytes))
		if filepath.IsAbs(c) {
			commonGitDir = filepath.Clean(c)
		} else {
			commonGitDir = filepath.Clean(filepath.Join(targetDir, c))
		}
	}
	if commonGitDir == "" {
		commonGitDir = filepath.Join(repoRoot, ".git")
	}
	if fi, err := os.Stat(commonGitDir); err == nil && !fi.IsDir() {
		if data, err := os.ReadFile(commonGitDir); err == nil {
			if r, ok := strings.CutPrefix(strings.TrimSpace(string(data)), "gitdir:"); ok {
				ptr := strings.TrimSpace(r)
				if !filepath.IsAbs(ptr) {
					ptr = filepath.Join(filepath.Dir(commonGitDir), ptr)
				}
				commonGitDir = filepath.Clean(ptr)
			}
		}
	}

	localGitDir := filepath.Join(wtPath, ".gitdir")
	if err := os.MkdirAll(localGitDir, 0o755); err != nil {
		return fmt.Errorf("create localized gitdir: %w", err)
	}

	entries, err := os.ReadDir(targetDir)
	if err != nil {
		return fmt.Errorf("read target gitdir %s: %w", targetDir, err)
	}
	for _, entry := range entries {
		name := entry.Name()
		if name == "commondir" || name == "gitdir" {
			continue
		}
		srcPath := filepath.Join(targetDir, name)
		dstPath := filepath.Join(localGitDir, name)
		if entry.IsDir() {
			if err := copyWorktreeDir(srcPath, dstPath, false); err != nil {
				return fmt.Errorf("copy worktree dir %s: %w", name, err)
			}
		} else {
			if err := copyWorktreeFile(srcPath, dstPath); err != nil {
				return fmt.Errorf("copy worktree file %s: %w", name, err)
			}
		}
	}

	_ = os.WriteFile(filepath.Join(localGitDir, "original_gitdir"), []byte(targetDir+"\n"), 0o644)

	srcObjects := filepath.Join(commonGitDir, "objects")
	dstObjects := filepath.Join(localGitDir, "objects")
	if fi, err := os.Stat(srcObjects); err == nil && fi.IsDir() {
		if err := copyWorktreeDir(srcObjects, dstObjects, true); err != nil {
			return fmt.Errorf("copy git objects: %w", err)
		}
	}

	srcRefs := filepath.Join(commonGitDir, "refs")
	dstRefs := filepath.Join(localGitDir, "refs")
	if fi, err := os.Stat(srcRefs); err == nil && fi.IsDir() {
		_ = copyWorktreeDir(srcRefs, dstRefs, false)
	}

	srcPackedRefs := filepath.Join(commonGitDir, "packed-refs")
	dstPackedRefs := filepath.Join(localGitDir, "packed-refs")
	if _, err := os.Stat(srcPackedRefs); err == nil {
		_ = copyWorktreeFile(srcPackedRefs, dstPackedRefs)
	}

	srcConfig := filepath.Join(commonGitDir, "config")
	dstConfig := filepath.Join(localGitDir, "config")
	if _, err := os.Stat(srcConfig); err == nil {
		_ = copyWorktreeFile(srcConfig, dstConfig)
	}

	infoDir := filepath.Join(localGitDir, "info")
	if err := os.MkdirAll(infoDir, 0o755); err != nil {
		return fmt.Errorf("create info dir: %w", err)
	}
	srcExclude := filepath.Join(commonGitDir, "info", "exclude")
	dstExclude := filepath.Join(infoDir, "exclude")
	var excludeContent []byte
	if b, err := os.ReadFile(srcExclude); err == nil {
		excludeContent = b
	}
	if !strings.Contains(string(excludeContent), ".gitdir") {
		if len(excludeContent) > 0 && !strings.HasSuffix(string(excludeContent), "\n") {
			excludeContent = append(excludeContent, '\n')
		}
		excludeContent = append(excludeContent, []byte(".gitdir\n.gitdir/\n")...)
	}
	if err := os.WriteFile(dstExclude, excludeContent, 0o644); err != nil {
		return fmt.Errorf("write exclude file: %w", err)
	}

	if err := os.WriteFile(dotGit, []byte("gitdir: .gitdir\n"), 0o644); err != nil {
		return fmt.Errorf("update .git pointer: %w", err)
	}

	return nil
}

func copyWorktreeFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	fi, err := in.Stat()
	if err != nil {
		return err
	}

	out, err := os.OpenFile(dst, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, fi.Mode().Perm())
	if err != nil {
		return err
	}
	defer out.Close()

	if _, err := io.Copy(out, in); err != nil {
		return err
	}
	return out.Close()
}

func linkOrCopyWorktreeFile(src, dst string) error {
	_ = os.Remove(dst)
	if err := os.Link(src, dst); err == nil {
		return nil
	}
	return copyWorktreeFile(src, dst)
}

func copyWorktreeDir(src, dst string, linkFiles bool) error {
	entries, err := os.ReadDir(src)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dst, 0o755); err != nil {
		return err
	}
	for _, entry := range entries {
		srcPath := filepath.Join(src, entry.Name())
		dstPath := filepath.Join(dst, entry.Name())
		if entry.IsDir() {
			if err := copyWorktreeDir(srcPath, dstPath, linkFiles); err != nil {
				return err
			}
		} else {
			if linkFiles {
				if err := linkOrCopyWorktreeFile(srcPath, dstPath); err != nil {
					return err
				}
			} else {
				if err := copyWorktreeFile(srcPath, dstPath); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func worktreeCommitSubject(wtPath, baseSHA, msgFile string) string {
	if strings.TrimSpace(msgFile) != "" {
		if b, err := os.ReadFile(msgFile); err == nil {
			for _, line := range strings.Split(string(b), "\n") {
				line = strings.TrimSpace(line)
				if line != "" {
					return line
				}
			}
		}
	}
	base := strings.TrimSpace(baseSHA)
	if base == "" {
		if in, err := workerworktree.LoadIntent(wtPath); err == nil {
			base = strings.TrimSpace(in.BaseSHA)
		}
	}
	var head string
	headCmd := windowgate.Command("git", "-C", wtPath, "rev-parse", "HEAD")
	windowgate.ConfigureBackgroundCommand(headCmd)
	if headOut, err := headCmd.Output(); err == nil {
		head = strings.TrimSpace(string(headOut))
	}
	// A worktree commit at tip is present when HEAD differs from base.
	// When HEAD == base, the worker made no tip commit; do not inherit the base commit subject (#8813).
	hasWorkerCommit := head != "" && (base == "" || head != base)
	if hasWorkerCommit || head == "" {
		cmd := windowgate.Command("git", "-C", wtPath, "log", "-1", "--format=%s")
		windowgate.ConfigureBackgroundCommand(cmd)
		if out, err := cmd.Output(); err == nil {
			subj := strings.TrimSpace(string(out))
			if subj != "" {
				return subj
			}
		}
	}
	if in, err := workerworktree.LoadIntent(wtPath); err == nil && strings.TrimSpace(in.Message) != "" {
		for _, line := range strings.Split(in.Message, "\n") {
			line = strings.TrimSpace(line)
			if line != "" {
				return line
			}
		}
	}
	return ""
}

func isFixSubject(s string) bool {
	lower := strings.ToLower(strings.TrimSpace(s))
	return strings.HasPrefix(lower, "fix(") || strings.HasPrefix(lower, "fix:")
}

func workerLandSymptomUnwitnessed(detail string) workerworktree.Result {
	return workerworktree.Result{
		OK:     false,
		Code:   "SYMPTOM_UNWITNESSED",
		Reason: "SYMPTOM_UNWITNESSED: fix commit must include a test that fails on parent and passes on fix (red-then-green); bypass only with --unsafe-skip-symptom-witness",
		Detail: detail,
	}
}

func verifyWorkerLandSymptom(wtPath, ref string) workerworktree.Result {
	resolver := witness.NewWithRunner(nil, wtPath)
	outcome := resolver.ResolveSymptom(context.Background(), ref, true)
	switch outcome {
	case abi.WitnessConfirmed:
		return workerworktree.Result{OK: true}
	case abi.WitnessRefuted:
		return workerLandSymptomUnwitnessed("symptom witness was refuted")
	default:
		return workerLandSymptomUnwitnessed("symptom witness abstained")
	}
}

func verifyWorkerLandTestWitness(wtPath string) workerworktree.Result {
	candidates := []string{
		filepath.Join(wtPath, ".fak", "test-witness.json"),
		filepath.Join(wtPath, "test-witness.json"),
		filepath.Join(wtPath, ".test-witness.json"),
		filepath.Join(wtPath, ".fak", "test_witness.json"),
		filepath.Join(wtPath, ".fak", "test-witness.receipt"),
		filepath.Join(wtPath, ".fak", "test-witness-receipt.json"),
		filepath.Join(filepath.Dir(wtPath), ".fak-worker-intents", filepath.Base(wtPath)+"-test-witness.json"),
		filepath.Join(filepath.Dir(wtPath), ".fak-worker-receipts", filepath.Base(wtPath)+"-test-witness.json"),
	}

	var foundCandidate bool
	var lastReason string

	for _, cand := range candidates {
		data, err := os.ReadFile(cand)
		if err != nil {
			continue
		}
		foundCandidate = true
		valid, reason := validateTestWitnessBytes(data)
		if valid {
			return workerworktree.Result{OK: true}
		}
		if lastReason == "" {
			lastReason = reason
		}
	}

	// Also check if test-witness receipt is committed in git HEAD if not found on disk
	if !foundCandidate {
		for _, gitPath := range []string{".fak/test-witness.json", "test-witness.json"} {
			cmd := windowgate.Command("git", "-C", wtPath, "show", "HEAD:"+gitPath)
			windowgate.ConfigureBackgroundCommand(cmd)
			if out, err := cmd.Output(); err == nil && len(out) > 0 {
				foundCandidate = true
				valid, reason := validateTestWitnessBytes(out)
				if valid {
					return workerworktree.Result{OK: true}
				}
				if lastReason == "" {
					lastReason = reason
				}
				break
			}
		}
	}

	if foundCandidate {
		return workerworktree.Result{
			OK:     false,
			Code:   "UNWITNESSED_WORKER_CLAIM",
			Reason: "UNWITNESSED_WORKER_CLAIM: " + lastReason,
		}
	}

	return workerworktree.Result{
		OK:     false,
		Code:   "UNWITNESSED_WORKER_CLAIM",
		Reason: "UNWITNESSED_WORKER_CLAIM: landing requires an independently verified test witness receipt; none found in worker worktree",
	}
}

func isWitnessTruthy(v any) bool {
	switch val := v.(type) {
	case bool:
		return val
	case string:
		s := strings.ToLower(strings.TrimSpace(val))
		return s == "true" || s == "1" || s == "yes"
	}
	return false
}

func isWitnessExplicitFalse(v any) bool {
	switch val := v.(type) {
	case bool:
		return !val
	case string:
		s := strings.ToLower(strings.TrimSpace(val))
		return s == "false" || s == "0" || s == "no"
	}
	return false
}

func validateTestWitnessBytes(data []byte) (bool, string) {
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return false, "malformed test witness receipt: invalid JSON"
	}
	if len(raw) == 0 {
		return false, "test witness receipt contains no evidence (empty object)"
	}

	// 1. Guard against self-report claims
	if isWitnessTruthy(raw["self_report"]) || isWitnessTruthy(raw["self-report"]) || isWitnessTruthy(raw["selfReport"]) {
		return false, "test witness is an unverified self-report claim"
	}
	for _, key := range []string{"witness", "source", "claim", "type", "kind"} {
		if val, ok := raw[key].(string); ok {
			s := strings.ToLower(strings.TrimSpace(val))
			if s == "self-report" || s == "self_report" || s == "subject-only" || s == "self" {
				return false, "test witness is an unverified self-report claim"
			}
		}
	}
	if rung, ok := raw["rung"].(string); ok {
		if strings.EqualFold(strings.TrimSpace(rung), "w0") {
			return false, "test witness is an unverified self-report claim (W0 rung)"
		}
	}

	// 2. Guard against explicit false verification flags
	if isWitnessExplicitFalse(raw["verified"]) {
		return false, "test witness receipt is explicitly unverified"
	}
	if isWitnessExplicitFalse(raw["witnesses"]) {
		return false, "test witness receipt does not witness claim"
	}
	if isWitnessExplicitFalse(raw["passed"]) {
		return false, "test witness receipt indicates test did not pass"
	}
	if isWitnessExplicitFalse(raw["ok"]) {
		return false, "test witness receipt indicates failure"
	}

	// 3. Guard against non-zero exit code
	if ec, ok := raw["exit_code"]; ok {
		switch val := ec.(type) {
		case float64:
			if val != 0 {
				return false, fmt.Sprintf("test witness exited with non-zero code %d", int(val))
			}
		case int:
			if val != 0 {
				return false, fmt.Sprintf("test witness exited with non-zero code %d", val)
			}
		}
	}

	// 4. Guard against negative status/verdict/outcome strings
	unwitnessedVerdicts := map[string]bool{
		"FAIL":              true,
		"FAILED":            true,
		"CLAIM_UNWITNESSED": true,
		"UNWITNESSED":       true,
		"UNVERIFIED":        true,
		"ABSTAIN":           true,
		"VACUOUS":           true,
		"REJECTED":          true,
		"ERROR":             true,
		"RED":               true,
	}
	if verdict, ok := raw["verdict"].(string); ok {
		v := strings.ToUpper(strings.TrimSpace(verdict))
		if unwitnessedVerdicts[v] {
			return false, fmt.Sprintf("test witness receipt has negative verdict %q", verdict)
		}
	}
	if status, ok := raw["status"].(string); ok {
		s := strings.ToUpper(strings.TrimSpace(status))
		if unwitnessedVerdicts[s] {
			return false, fmt.Sprintf("test witness receipt has negative status %q", status)
		}
	}
	if outcome, ok := raw["outcome"].(string); ok {
		o := strings.ToUpper(strings.TrimSpace(outcome))
		if unwitnessedVerdicts[o] || o == "NOTYET" || o == "NOT_YET" {
			return false, fmt.Sprintf("test witness receipt has negative outcome %q", outcome)
		}
	}

	// 5. Require positive verification / witness evidence
	positiveVerdicts := map[string]bool{
		"PASS":           true,
		"EXEC_PASS":      true,
		"RESOLVED_MATCH": true,
		"OK":             true,
		"DISCRIMINATES":  true,
		"PASSED":         true,
		"SUCCESS":        true,
		"CONFIRMED":      true,
		"GREEN":          true,
	}
	hasPositive := false
	if isWitnessTruthy(raw["verified"]) || isWitnessTruthy(raw["witnesses"]) || isWitnessTruthy(raw["passed"]) || isWitnessTruthy(raw["ok"]) {
		hasPositive = true
	}
	if verdict, ok := raw["verdict"].(string); ok && positiveVerdicts[strings.ToUpper(strings.TrimSpace(verdict))] {
		hasPositive = true
	}
	if status, ok := raw["status"].(string); ok && positiveVerdicts[strings.ToUpper(strings.TrimSpace(status))] {
		hasPositive = true
	}
	if outcome, ok := raw["outcome"].(string); ok && positiveVerdicts[strings.ToUpper(strings.TrimSpace(outcome))] {
		hasPositive = true
	}
	if witness, ok := raw["witness"].(string); ok {
		w := strings.ToLower(strings.TrimSpace(witness))
		if w == "test-witnessed" || w == "diff-witnessed" || w == "verified" || w == "test" || w == "independent" {
			hasPositive = true
		}
	}
	if rung, ok := raw["rung"].(string); ok {
		r := strings.ToUpper(strings.TrimSpace(rung))
		if r == "W3" || r == "W2" || r == "OS_RECORDED" {
			hasPositive = true
		}
	}
	if schema, ok := raw["schema"].(string); ok && strings.Contains(schema, "test-witness") {
		hasPositive = true
	}

	if !hasPositive {
		return false, "test witness receipt contains no positive verification or passed test evidence"
	}
	return true, ""
}

func runWorktreeWorkerLand(stdout, stderr io.Writer, argv []string) (workerworktree.Result, int) {
	fs := flag.NewFlagSet("worktree worker land", flag.ContinueOnError)
	fs.SetOutput(stderr)
	worktree := fs.String("worktree", "", "the worker's worktree dir to land from (required)")
	baseSHA := fs.String("base-sha", "", "the sha the worktree was pinned at — the diff ref (default: HEAD)")
	msgFile := fs.String("msg-file", "", "commit message file for `git commit -s -F` (default: derive from the worktree tip)")
	verify := fs.String("verify", "go-build", "pre-land witness run IN the worktree: off | go-build (default: go-build)")
	root := fs.String("root", "", "repo root the change lands on (default: discover from cwd)")
	disambiguationTimeoutMS := fs.String("disambiguation-timeout-ms", "", "one shared whole-tree disambiguation deadline in milliseconds (1..900000; default 120000; no retries)")
	coreLockWitness := fs.String("core-lock-maintenance-witness", "",
		"independent witness claim that clears a hard-self core-lock land (same claim vocabulary as fak commit)")
	recoveryRemote := fs.String("recovery-remote", "", "publish/read-back candidate on this git remote before trunk CAS")
	requireRemote := fs.Bool("require-remote-recovery", false, "refuse trunk CAS unless remote recovery read-back succeeds")
	unsafeSkipSymptomWitness := fs.Bool("unsafe-skip-symptom-witness", false,
		"bypass mandatory fail-to-pass symptom witness for fix(*) commits")
	requireTestWitness := fs.Bool("require-test-witness", false,
		"require verified test witness receipt before landing worker diff")
	var paths repeatedString
	fs.Var(&paths, "paths", "path to scope the commit to (repeatable); omit to commit the whole applied diff")
	if err := fs.Parse(argv); err != nil {
		return workerworktree.Result{OK: false, Reason: err.Error()}, 2
	}

	worktreeDir := strings.TrimSpace(*worktree)
	if worktreeDir == "" {
		fmt.Fprintln(stderr, "fak worktree worker land: --worktree is required")
		return workerworktree.Result{OK: false, Reason: "--worktree is required"}, 2
	}
	repoRoot := worktreeWorkerRoot(*root)

	// Mandatory test witness receipt check (#11532)
	if *requireTestWitness {
		testWitnessRes := verifyWorkerLandTestWitness(worktreeDir)
		if !testWitnessRes.OK {
			return testWitnessRes, 1
		}
	}

	// Mandatory fail-to-pass symptom witness for fix(*) commits (#10926)
	subj := worktreeCommitSubject(worktreeDir, strings.TrimSpace(*baseSHA), strings.TrimSpace(*msgFile))
	requireSymptomWitness := isFixSubject(subj) && !*unsafeSkipSymptomWitness

	var hook workerworktree.VerifyHook
	switch strings.ToLower(strings.TrimSpace(*verify)) {
	case "off", "none":
		hook = nil
	case "", "go-build", "gobuild", "build":
		hook = worktreeWorkerGoBuildVerify
	default:
		fmt.Fprintf(stderr, "fak worktree worker land: unknown --verify %q (want off|go-build)\n", *verify)
		return workerworktree.Result{OK: false, Reason: fmt.Sprintf("unknown --verify %q", *verify)}, 2
	}

	opts := []workerworktree.LandOption{
		workerworktree.WithCoreLockWitness(*coreLockWitness),
		workerworktree.WithLandProgress(worktreeWorkerProgressEmitter(stderr)),
	}
	if strings.TrimSpace(*recoveryRemote) != "" || *requireRemote {
		remote := strings.TrimSpace(*recoveryRemote)
		if remote == "" {
			remote = "origin"
		}
		opts = append(opts, workerworktree.WithRecoveryRemote(remote, *requireRemote))
	}
	timeoutSet := flagWasSet(fs, "disambiguation-timeout-ms")
	res, err := withWorkerLandDisambiguationTimeout(*disambiguationTimeoutMS, timeoutSet, func() workerworktree.Result {
		if requireSymptomWitness {
			prospectiveVerify := func(dir string, materializationErr error) workerworktree.Result {
				if materializationErr != nil {
					return workerLandSymptomUnwitnessed(materializationErr.Error())
				}
				return verifyWorkerLandSymptom(dir, "HEAD")
			}
			return workerworktree.LandProspectiveVerified(
				repoRoot, worktreeDir, strings.TrimSpace(*baseSHA), strings.TrimSpace(*msgFile),
				[]string(paths), hook, prospectiveVerify, nil, opts...,
			)
		}
		return workerworktree.Land(repoRoot, worktreeDir, strings.TrimSpace(*baseSHA), strings.TrimSpace(*msgFile), []string(paths), hook, nil, opts...)
	})
	if err != nil {
		res = workerworktree.Result{
			OK: false, Code: workerworktree.DisambiguationTimeoutCode,
			Reason: "configure worker land disambiguation timeout: " + err.Error(),
		}
	}
	if !res.OK && res.Code == "SYMPTOM_UNWITNESSED" {
		return res, 1
	}
	return res, 0
}

func worktreeWorkerLand(argv []string) {
	res, code := runWorktreeWorkerLand(os.Stdout, os.Stderr, argv)
	worktreeWorkerEmit(res)
	if !res.OK {
		if code != 0 {
			os.Exit(code)
		}
		os.Exit(1)
	}
}

// withWorkerLandDisambiguationTimeout bridges the explicit CLI spelling onto the
// workerworktree resolver's bootstrap input for exactly one Land call. The CLI
// deliberately does not parse or clamp the value: the existing resolver remains
// the single authority for the inclusive 1..900000 ms bound and writes the same
// requested/effective/default fields into the disambiguation receipt. An omitted
// flag leaves the environment untouched, preserving the 120000 ms default (or an
// explicitly supplied bootstrap environment for older callers).
func withWorkerLandDisambiguationTimeout(raw string, explicit bool, land func() workerworktree.Result) (workerworktree.Result, error) {
	if !explicit {
		return land(), nil
	}
	previous, hadPrevious := os.LookupEnv(workerworktree.DisambiguationTimeoutEnv)
	if err := os.Setenv(workerworktree.DisambiguationTimeoutEnv, raw); err != nil {
		return workerworktree.Result{}, err
	}
	defer func() {
		if hadPrevious {
			_ = os.Setenv(workerworktree.DisambiguationTimeoutEnv, previous)
		} else {
			_ = os.Unsetenv(workerworktree.DisambiguationTimeoutEnv)
		}
	}()
	return land(), nil
}

// worktreeWorkerListOut is the list JSON: a count and the sorted live-worktree
// paths (never null — an empty slice renders `[]`), mirroring the Python CLI.
type worktreeWorkerListOut struct {
	Count     int                             `json:"count"`
	Paths     []string                        `json:"paths"`
	Inventory []workerworktree.InventoryRow   `json:"inventory"`
	Capacity  workerworktree.CapacityAdvisory `json:"capacity"`
	Partial   bool                            `json:"partial,omitempty"`
	Timeout   bool                            `json:"timeout,omitempty"`
}

type worktreeWorkerRecoverOut struct {
	OK            bool                                `json:"ok"`
	Count         int                                 `json:"count"`
	Candidates    []workerworktree.RecoveryEntry      `json:"candidates"`
	Cleaned       string                              `json:"cleaned,omitempty"`
	Reason        string                              `json:"reason,omitempty"`
	RemoteCleanup *workerworktree.RemoteCleanupReport `json:"remote_cleanup,omitempty"`
}

// worktreeWorkerRecover is the crash-resume inventory for isolated lands. With
// no mutation flags it is read-only. Cleanup is guarded by HEAD reachability;
// --force is deliberately required to discard an unlanded candidate.
func worktreeWorkerRecover(argv []string) {
	fs := flag.NewFlagSet("worktree worker recover", flag.ExitOnError)
	root := fs.String("root", "", "repo root (default: discover from cwd)")
	cleanup := fs.String("cleanup", "", "delete one landed recovery ref")
	force := fs.Bool("force", false, "allow cleanup of an unlanded recovery ref")
	remote := fs.String("remote", "origin", "remote whose worker-land mirror is inspected")
	fetch := fs.Bool("fetch", false, "refresh the read-only remote mirror before listing")
	cleanupRemote := fs.String("cleanup-remote", "", "report/delete one remote recovery ref after default-branch ancestry proof")
	apply := fs.Bool("apply", false, "apply remote cleanup; otherwise report-only")
	allowPeer := fs.Bool("allow-peer", false, "permit cleanup of a peer-named recovery ref")
	worktreeName := fs.String("worktree-name", "", "local worktree identity used for remote cleanup ownership guard")
	fs.Parse(argv)

	repoRoot := worktreeWorkerRoot(*root)
	finishLifecycle := beginAutomaticWIPLifecycle(repoRoot, "crash-recovery", os.Stderr)
	defer finishLifecycle()
	out := worktreeWorkerRecoverOut{Candidates: []workerworktree.RecoveryEntry{}}
	if *fetch {
		if err := workerworktree.FetchRecoveryMirror(repoRoot, *remote, nil); err != nil {
			out.Reason = err.Error()
			worktreeWorkerEmit(out)
			finishLifecycle()
			os.Exit(1)
		}
	}
	if *cleanupRemote != "" {
		plan := workerworktree.CleanupRemoteRecoveryRef(repoRoot, *remote, *cleanupRemote, *worktreeName, *allowPeer, *apply, nil)
		out.RemoteCleanup = &plan
		if *apply && !plan.Applied {
			out.Reason = plan.Reason
			worktreeWorkerEmit(out)
			finishLifecycle()
			os.Exit(1)
		}
	}
	if *cleanup != "" {
		if err := workerworktree.DeleteRecoveryRef(repoRoot, *cleanup, *force, nil); err != nil {
			out.Reason = err.Error()
			worktreeWorkerEmit(out)
			finishLifecycle()
			os.Exit(1)
		}
		out.Cleaned = *cleanup
	}
	items, err := workerworktree.RecoveryEntries(repoRoot, nil)
	if err != nil {
		out.Reason = err.Error()
		worktreeWorkerEmit(out)
		finishLifecycle()
		os.Exit(1)
	}
	out.OK = true
	out.Count = len(items)
	out.Candidates = items
	worktreeWorkerEmit(out)
}
