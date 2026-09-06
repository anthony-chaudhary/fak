package devcmd

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	codexPluginSyncSchema    = "fak/codex-plugin-sync/v1"
	codexPluginAttestTimeout = 20 * time.Second
)

var codexPluginProfileInspect = inspectCodexHookProfileContext

type codexPluginArtifact struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

type codexPluginSyncReceipt struct {
	Schema              string                `json:"schema"`
	Status              string                `json:"status"`
	SourceRevision      string                `json:"source_revision,omitempty"`
	MarketplaceRevision string                `json:"marketplace_revision,omitempty"`
	PluginVersion       string                `json:"plugin_version"`
	PackageVersion      string                `json:"package_version,omitempty"`
	CodexHome           string                `json:"codex_home"`
	Source              string                `json:"source"`
	Destination         string                `json:"destination"`
	StagedDestination   string                `json:"staged_destination,omitempty"`
	Artifacts           []codexPluginArtifact `json:"artifacts"`
	ExistingState       string                `json:"existing_state,omitempty"`
	DriftedArtifacts    []string              `json:"drifted_artifacts,omitempty"`
	LoadedHooks         []string              `json:"loaded_hooks,omitempty"`
	LoadedInventoryHash string                `json:"loaded_inventory_hash,omitempty"`
	ProfileVerdict      string                `json:"profile_verdict,omitempty"`
	FailureStage        string                `json:"failure_stage,omitempty"`
	Detail              string                `json:"detail,omitempty"`
	CreatedAt           string                `json:"created_at"`
}

type codexPluginSyncOps struct {
	rename func(string, string) error
	remove func(string) error
}

func RunCodexPluginSync(stdout, stderr io.Writer, args []string) int {
	if len(args) > 0 {
		switch args[0] {
		case "marketplace-maintenance":
			return runCodexMarketplaceMaintenance(stdout, stderr, args[1:])
		case "marketplace-upgrade":
			return runCodexMarketplaceUpgrade(stdout, stderr, args[1:])
		}
	}
	fs := flag.NewFlagSet("codex-plugin-sync", flag.ContinueOnError)
	fs.SetOutput(stderr)
	home := fs.String("codex-home", os.Getenv("CODEX_HOME"), "active Codex home")
	source := fs.String("source", "", "plugin source directory (default: active dos marketplace claude-plugin)")
	destination := fs.String("destination", "", "installed plugin directory (default: active dos-kernel cache version)")
	workspace := fs.String("workspace", ".", "workspace used for hooks/list attestation")
	jsonOut := fs.Bool("json", false, "emit JSON receipt")
	if err := fs.Parse(args); err != nil || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak-dev codex-plugin-sync [--codex-home DIR] [--source DIR] [--destination DIR] [--workspace DIR] [--json]")
		return 2
	}
	if *home == "" {
		if d, err := os.UserHomeDir(); err == nil {
			*home = filepath.Join(d, ".codex")
		}
	}
	var err error
	*home, err = filepath.Abs(*home)
	if err != nil {
		fmt.Fprintf(stderr, "codex-plugin-sync: resolve CODEX_HOME: %v\n", err)
		return 1
	}
	if *source == "" {
		*source = filepath.Join(*home, ".tmp", "marketplaces", "dos", "claude-plugin")
	}
	if *destination == "" {
		version, e := pluginManifestVersion(*source)
		if e != nil {
			fmt.Fprintf(stderr, "codex-plugin-sync: %v\n", e)
			return 1
		}
		*destination = filepath.Join(*home, "plugins", "cache", "dos", "dos-kernel", version)
	}
	receipt, syncErr := syncCodexPlugin(*home, *source, *destination, *workspace, codexPluginSyncOps{rename: os.Rename, remove: os.RemoveAll})
	if *jsonOut {
		_ = json.NewEncoder(stdout).Encode(receipt)
	} else {
		writeCodexPluginSyncReceipt(stdout, receipt)
	}
	if syncErr != nil {
		if !*jsonOut {
			fmt.Fprintf(stderr, "codex-plugin-sync: %v\n", syncErr)
		}
		return 1
	}
	return 0
}

func renamePluginPath(ops codexPluginSyncOps, old, new string) error {
	var err error
	for attempt := 0; attempt < 8; attempt++ {
		err = ops.rename(old, new)
		if err == nil {
			return nil
		}
		time.Sleep(10 * time.Millisecond)
	}
	return err
}

func syncCodexPlugin(home, source, destination, workspace string, ops codexPluginSyncOps) (codexPluginSyncReceipt, error) {
	now := time.Now().UTC()
	r := codexPluginSyncReceipt{Schema: codexPluginSyncSchema, Status: "FAILED", CodexHome: filepath.Clean(home), Source: filepath.Clean(source), Destination: filepath.Clean(destination), CreatedAt: now.Format(time.RFC3339)}
	version, err := pluginManifestVersion(source)
	if err != nil {
		r.FailureStage, r.Detail = "source", err.Error()
		return r, err
	}
	r.PluginVersion = version
	r.PackageVersion = packageVersion(filepath.Dir(source))
	if r.PackageVersion == "" || r.PackageVersion != r.PluginVersion {
		err := fmt.Errorf("plugin version %q does not match package version %q", r.PluginVersion, r.PackageVersion)
		r.FailureStage, r.Detail = "source_version", err.Error()
		return r, err
	}
	repoRoot := filepath.Dir(source)
	r.MarketplaceRevision = marketplaceRevision(repoRoot)
	headRevision := ""
	if gitRoot(repoRoot) == filepath.Clean(repoRoot) {
		headRevision = gitRevision(repoRoot)
	}
	if r.MarketplaceRevision == "" || (headRevision != "" && headRevision != r.MarketplaceRevision) {
		err := fmt.Errorf("source HEAD %q does not match marketplace revision %q", headRevision, r.MarketplaceRevision)
		r.FailureStage, r.Detail = "source_revision", err.Error()
		return r, err
	}
	r.SourceRevision = r.MarketplaceRevision
	sourceArtifacts, err := pluginArtifacts(source)
	if err != nil {
		r.FailureStage, r.Detail = "source_hash", err.Error()
		return r, err
	}
	r.Artifacts = sourceArtifacts
	if pathExists(destination) {
		if installed, installedErr := pluginArtifacts(destination); installedErr != nil {
			r.ExistingState = "MIXED_OR_INCOMPLETE"
			r.DriftedArtifacts = []string{"artifact_inventory"}
		} else {
			r.DriftedArtifacts = artifactDrift(sourceArtifacts, installed)
			if len(r.DriftedArtifacts) == 0 {
				r.ExistingState = "COHERENT"
			} else {
				r.ExistingState = "SELECTIVELY_STALE"
			}
		}
	} else {
		r.ExistingState = "ABSENT"
	}
	parent := filepath.Dir(destination)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		r.FailureStage, r.Detail = "stage", err.Error()
		return r, err
	}
	stage, err := prepareCodexPluginStage(destination, ops)
	if err != nil {
		r.FailureStage, r.Detail = "stage", err.Error()
		return r, err
	}
	r.StagedDestination = stage
	keepStage := false
	defer func() {
		if !keepStage && stage != "" {
			_ = ops.remove(stage)
		}
	}()
	if err := copyTree(source, stage); err != nil {
		r.FailureStage, r.Detail = "stage", err.Error()
		return r, err
	}
	err = verifyPluginArtifacts(sourceArtifacts, stage, "staged")
	if err != nil {
		r.FailureStage, r.Detail = "verify_stage", err.Error()
		return r, err
	}
	backup := codexPluginBackupPath(destination, now)
	destinationExists := pathExists(destination)
	if destinationExists {
		if err := ops.rename(destination, backup); err != nil {
			keepStage = true
			r.Status, r.FailureStage = "RESTART_REQUIRED", "cutover_locked"
			r.Detail = fmt.Sprintf("destination is in use; close Codex processes and rerun (staged coherent plugin retained): %v", err)
			return r, errors.New(r.Detail)
		}
	}
	rollback := func(stageName string, cause error) error {
		if !destinationExists {
			_ = ops.remove(destination)
			return cause
		}
		_ = ops.remove(destination)
		if rollbackErr := renamePluginPath(ops, backup, destination); rollbackErr != nil {
			r.Status = "ROLLBACK_FAILED"
			r.Detail = fmt.Sprintf("%s: %v; rollback: %v", stageName, cause, rollbackErr)
			return errors.New(r.Detail)
		}
		return cause
	}
	if err := ops.rename(stage, destination); err != nil {
		if destinationExists {
			if rollbackErr := renamePluginPath(ops, backup, destination); rollbackErr != nil {
				r.Status, r.FailureStage = "ROLLBACK_FAILED", "cutover"
				r.Detail = fmt.Sprintf("cutover: %v; rollback: %v", err, rollbackErr)
				return r, errors.New(r.Detail)
			}
		}
		r.FailureStage, r.Detail = "cutover", err.Error()
		return r, err
	}
	stage = ""
	err = verifyPluginArtifacts(sourceArtifacts, destination, "installed")
	if err != nil {
		r.FailureStage, r.Detail = "verify_install", err.Error()
		return r, rollback("verify_install", err)
	}
	r.StagedDestination = ""
	r.Status = "INSTALLED"
	ctx, cancel := context.WithTimeout(context.Background(), codexPluginAttestTimeout)
	profile, profileErr := codexPluginProfileInspect(ctx, home, workspace, "")
	cancel()
	if profileErr != nil {
		if errors.Is(profileErr, context.DeadlineExceeded) || errors.Is(ctx.Err(), context.DeadlineExceeded) {
			profileErr = fmt.Errorf("attestation timed out after %s: %w", codexPluginAttestTimeout, context.DeadlineExceeded)
		}
		r.Status = "INSTALLED_UNATTESTED"
		r.FailureStage, r.Detail = "attest_profile", profileErr.Error()
		return r, rollback("attest_profile", profileErr)
	}
	r.ProfileVerdict = profile.Verdict
	for _, h := range profile.Hooks {
		if strings.EqualFold(filepath.Clean(h.SourcePath), filepath.Clean(filepath.Join(destination, "hooks", "hooks.json"))) {
			r.LoadedHooks = append(r.LoadedHooks, h.EventName+":"+h.State+":"+h.CurrentHash)
		}
	}
	sort.Strings(r.LoadedHooks)
	r.LoadedInventoryHash = hashStrings(r.LoadedHooks)
	if err := attestInstalledHookInventory(profile, destination); err != nil {
		r.Status = "INSTALLED_PROFILE_UNHEALTHY"
		r.FailureStage = "attest_profile"
		r.Detail = err.Error()
		return r, rollback("attest_profile", err)
	}
	receiptPath := filepath.Join(destination, ".fak-install-receipt.json")
	b, _ := json.MarshalIndent(r, "", "  ")
	if err := os.WriteFile(receiptPath, append(b, '\n'), 0o644); err != nil {
		r.Status, r.FailureStage, r.Detail = "INSTALLED_UNATTESTED", "write_receipt", err.Error()
		return r, rollback("write_receipt", err)
	}
	if destinationExists {
		_ = ops.remove(backup)
	}
	return r, nil
}

func prepareCodexPluginStage(destination string, ops codexPluginSyncOps) (string, error) {
	parent := filepath.Dir(destination)
	stage := filepath.Join(parent, ".fak-plugin-stage-current")
	entries, err := os.ReadDir(parent)
	if err != nil {
		return "", fmt.Errorf("list retained plugin stages: %w", err)
	}
	for _, entry := range entries {
		if !strings.HasPrefix(entry.Name(), ".fak-plugin-stage-") {
			continue
		}
		if err := ops.remove(filepath.Join(parent, entry.Name())); err != nil {
			return "", fmt.Errorf("clear retained plugin stage %s: %w", entry.Name(), err)
		}
	}
	if err := os.Mkdir(stage, 0o755); err != nil {
		return "", fmt.Errorf("create plugin stage: %w", err)
	}
	return stage, nil
}
func codexPluginBackupPath(destination string, now time.Time) string {
	parent := filepath.Dir(destination)
	name := filepath.Base(destination)
	return filepath.Join(parent, fmt.Sprintf(".fak-plugin-backup-%s-%d", name, now.UnixNano()))
}
func attestInstalledHookInventory(profile hookProfileReport, destination string) error {
	manifest := filepath.Clean(filepath.Join(destination, "hooks", "hooks.json"))
	seen := map[string]int{}
	for _, h := range profile.Hooks {
		if !strings.EqualFold(filepath.Clean(h.SourcePath), manifest) {
			continue
		}
		seen[normalizeCodexHookEvent(h.EventName)]++
		if h.State != "effective" {
			return fmt.Errorf("loaded %s handler is %s, not effective", h.EventName, h.State)
		}
	}
	for _, event := range []string{"pretooluse", "posttooluse", "stop", "subagentstop"} {
		if seen[event] == 0 {
			return fmt.Errorf("Codex hooks/list did not load installed %s inventory from %s", event, manifest)
		}
	}
	return nil
}

func normalizeCodexHookEvent(event string) string {
	return strings.NewReplacer("_", "", "-", "").Replace(strings.ToLower(event))
}

func hashStrings(values []string) string {
	h := sha256.New()
	for _, value := range values {
		_, _ = io.WriteString(h, value+"\n")
	}
	return hex.EncodeToString(h.Sum(nil))
}

func pluginManifestVersion(root string) (string, error) {
	b, err := os.ReadFile(filepath.Join(root, ".claude-plugin", "plugin.json"))
	if err != nil {
		return "", fmt.Errorf("read plugin manifest: %w", err)
	}
	var v struct {
		Version string `json:"version"`
	}
	if err := json.Unmarshal(b, &v); err != nil || strings.TrimSpace(v.Version) == "" {
		return "", errors.New("plugin manifest has no valid version")
	}
	return v.Version, nil
}

func packageVersion(repo string) string {
	b, err := os.ReadFile(filepath.Join(repo, "pyproject.toml"))
	if err != nil {
		return ""
	}
	inProject := false
	for _, raw := range strings.Split(string(b), "\n") {
		line := strings.TrimSpace(strings.SplitN(raw, "#", 2)[0])
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			inProject = line == "[project]"
			continue
		}
		if !inProject {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if ok && strings.TrimSpace(key) == "version" {
			return strings.Trim(strings.TrimSpace(value), "\"")
		}
	}
	return ""
}

func marketplaceRevision(repo string) string {
	b, err := os.ReadFile(filepath.Join(repo, ".codex-marketplace-install.json"))
	if err != nil {
		return ""
	}
	var receipt struct {
		Revision string `json:"revision"`
	}
	if json.Unmarshal(b, &receipt) != nil {
		return ""
	}
	return strings.TrimSpace(receipt.Revision)
}

func gitRoot(path string) string {
	cmd := exec.Command("git", "-C", path, "rev-parse", "--show-toplevel")
	configureDispatchHelperCommand(cmd)
	b, err := cmd.Output()
	if err != nil {
		return ""
	}
	root, err := filepath.Abs(strings.TrimSpace(string(b)))
	if err != nil {
		return ""
	}
	return filepath.Clean(root)
}

func gitRevision(repo string) string {
	cmd := exec.Command("git", "-C", repo, "rev-parse", "HEAD")
	configureDispatchHelperCommand(cmd)
	b, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

func pluginArtifacts(root string) ([]codexPluginArtifact, error) {
	var paths []string
	for _, rel := range []string{".claude-plugin/plugin.json", "hooks/hooks.json", "bin/dos-hook-codex.ps1"} {
		if pathExists(filepath.Join(root, rel)) {
			paths = append(paths, rel)
		}
	}
	bins, _ := filepath.Glob(filepath.Join(root, "bin", "dos-hook-*"))
	for _, p := range bins {
		if strings.EqualFold(filepath.Base(p), "dos-hook-codex.ps1") {
			continue
		}
		if info, err := os.Stat(p); err == nil && !info.IsDir() {
			rel, _ := filepath.Rel(root, p)
			paths = append(paths, filepath.ToSlash(rel))
		}
	}
	sort.Strings(paths)
	if len(paths) < 3 {
		return nil, errors.New("plugin source is missing manifest, hooks, adapter, or native binaries")
	}
	out := make([]codexPluginArtifact, 0, len(paths))
	for _, rel := range paths {
		b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			return nil, err
		}
		h := sha256.Sum256(b)
		out = append(out, codexPluginArtifact{Path: rel, SHA256: hex.EncodeToString(h[:])})
	}
	return out, nil
}

func artifactDrift(want, got []codexPluginArtifact) []string {
	wm := make(map[string]string, len(want))
	gm := make(map[string]string, len(got))
	for _, a := range want {
		wm[a.Path] = a.SHA256
	}
	for _, a := range got {
		gm[a.Path] = a.SHA256
	}
	keys := map[string]struct{}{}
	for p := range wm {
		keys[p] = struct{}{}
	}
	for p := range gm {
		keys[p] = struct{}{}
	}
	var drift []string
	for p := range keys {
		if wm[p] != gm[p] {
			drift = append(drift, p)
		}
	}
	sort.Strings(drift)
	return drift
}

func verifyPluginArtifacts(source []codexPluginArtifact, destination, label string) error {
	artifacts, err := pluginArtifacts(destination)
	if err != nil {
		return err
	}
	if !sameArtifacts(source, artifacts) {
		return fmt.Errorf("%s artifact hashes differ from source", label)
	}
	return nil
}

func sameArtifacts(a, b []codexPluginArtifact) bool {
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

func copyTree(src, dst string) error {
	return filepath.WalkDir(src, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil || rel == "." {
			return err
		}
		target := filepath.Join(dst, rel)
		if d.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("plugin source contains unsupported symlink: %s", path)
		}
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		in, err := os.Open(path)
		if err != nil {
			return err
		}
		defer in.Close()
		out, err := os.OpenFile(target, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, info.Mode().Perm())
		if err != nil {
			return err
		}
		_, copyErr := io.Copy(out, in)
		closeErr := out.Close()
		if copyErr != nil {
			return copyErr
		}
		return closeErr
	})
}

func pathExists(path string) bool { _, err := os.Stat(path); return err == nil }

func writeCodexPluginSyncReceipt(w io.Writer, r codexPluginSyncReceipt) {
	fmt.Fprintf(w, "Codex plugin sync: %s\nsource: %s @ %s\ndestination: %s\nversion: plugin=%s package=%s\n", r.Status, r.Source, r.SourceRevision, r.Destination, r.PluginVersion, r.PackageVersion)
	for _, a := range r.Artifacts {
		fmt.Fprintf(w, "  %s %s\n", a.SHA256, a.Path)
	}
	if r.ExistingState != "" {
		fmt.Fprintf(w, "existing: %s", r.ExistingState)
		if len(r.DriftedArtifacts) > 0 {
			fmt.Fprintf(w, " (%s)", strings.Join(r.DriftedArtifacts, ", "))
		}
		fmt.Fprintln(w)
	}
	if r.StagedDestination != "" {
		fmt.Fprintf(w, "staged: %s\n", r.StagedDestination)
	}
	if r.ProfileVerdict != "" {
		fmt.Fprintf(w, "profile: %s (%s)\n", r.ProfileVerdict, strings.Join(r.LoadedHooks, ", "))
	}
	if r.Detail != "" {
		fmt.Fprintf(w, "%s: %s\n", r.FailureStage, r.Detail)
	}
}
