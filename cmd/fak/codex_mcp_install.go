package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
)

// fak codex mcp installs an immutable, versioned fak binary and points Codex's
// stdio MCP entry at it (#4657). Fixing server code does not help until Codex
// actually launches a fixed, non-worktree artifact: the in-tree fak.exe can be
// held open by a running Codex on Windows (blocking replacement), its revision
// drifts from source, and a hand-edited config has no rollback or handshake
// verification. This command owns that install/verify/rollback lifecycle.
//
// The load-bearing choice is a *versioned filename* per install
// (fak-<version>-<sha8>.exe): an upgrade writes a NEW file and only re-points the
// config, so the old binary can stay held open by a live Codex while new sessions
// select the new path — no in-place replacement, no OFF_TRUNK worktree binary.

const (
	codexMCPManifestSchema = "fak.codex-mcp-install.v1"
	codexMCPStatusSchema   = "fak.codex-mcp-status.v1"
	codexMCPDirName        = "fak-mcp"
	codexMCPManifestName   = "install.json"
)

// codexMCPManifest records exactly what an install owns so status can verify it
// and uninstall can remove only owned artifacts. It also carries the prior
// known-good config section so a rollback restores it byte-for-byte.
type codexMCPManifest struct {
	Schema           string   `json:"schema"`
	Server           string   `json:"server"`
	InstalledCommand string   `json:"installed_command"`
	Args             []string `json:"args"`
	PolicyPath       string   `json:"policy_path"`
	SourcePath       string   `json:"source_path"`
	Version          string   `json:"version"`
	ModuleRevision   string   `json:"module_revision,omitempty"`
	SHA256           string   `json:"sha256"`
	Arch             string   `json:"arch"`
	OS               string   `json:"os"`
	InstalledAt      string   `json:"installed_at"`
	ConfigPath       string   `json:"config_path"`
	// PriorConfigSection is the raw `[mcp_servers.<server>]` block that this
	// install replaced (empty when the entry was newly created). Rollback writes
	// it back verbatim; PriorCommand is the artifact rollback keeps alive.
	PriorConfigSection string `json:"prior_config_section,omitempty"`
	PriorCommand       string `json:"prior_command,omitempty"`
}

// codexMCPInitializeProbe runs a bounded MCP `initialize` handshake against an
// exact command+args and reports whether it answered cleanly. It is a package
// var so hermetic tests can stub the process boundary; production wires it to the
// same diagnoseMCP probe `fak doctor mcp` uses.
var codexMCPInitializeProbe = func(command string, args []string, timeout time.Duration) (ok bool, cause string) {
	rep := diagnoseMCP("fak", "", command, args, timeout)
	if rep.OK {
		return true, ""
	}
	for _, s := range rep.Stages {
		if s.Status == "fail" {
			return false, s.Cause
		}
	}
	return false, "PROBE_FAILED"
}

var codexMCPExit = os.Exit

func cmdCodexMCP(argv []string) {
	codexMCPExit(runCodexMCP(os.Stdout, os.Stderr, argv))
}

func runCodexMCP(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, codexMCPUsage())
		return 2
	}
	sub := argv[0]
	rest := argv[1:]
	switch sub {
	case "install":
		return runCodexMCPInstall(stdout, stderr, rest)
	case "status":
		return runCodexMCPStatus(stdout, stderr, rest)
	case "uninstall":
		return runCodexMCPUninstall(stdout, stderr, rest)
	case "-h", "--help", "help":
		fmt.Fprintln(stdout, codexMCPUsage())
		return 0
	default:
		fmt.Fprintf(stderr, "fak codex mcp: unknown subcommand %q\n%s\n", sub, codexMCPUsage())
		return 2
	}
}

func codexMCPUsage() string {
	return strings.TrimSpace(`
usage: fak codex mcp <install|status|uninstall> [flags]
  install    copy an immutable versioned fak into the Codex install dir and point mcp_servers.<server> at it
  status     verify the installed binary (present, checksum, arch, freshness) with an initialize handshake
  uninstall  remove the owned config entry and installed artifacts (idempotent)
common flags:
  --server NAME   Codex mcp_servers entry name (default: fak)
  --config PATH   Codex config.toml (default: $CODEX_HOME/config.toml or ~/.codex/config.toml)
  --json          emit a stable machine report`)
}

// codexMCPInstallDir is the non-worktree home for installed binaries, policy
// copies, and the manifest: <codex config dir>/fak-mcp. It sits beside the Codex
// config so it travels with the same operator state, never inside the fak clone.
func codexMCPInstallDir(configPath string) string {
	return filepath.Join(filepath.Dir(configPath), codexMCPDirName)
}

func codexMCPManifestPath(configPath string) string {
	return filepath.Join(codexMCPInstallDir(configPath), codexMCPManifestName)
}

// ---- install ----------------------------------------------------------------

func runCodexMCPInstall(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("codex mcp install", flag.ContinueOnError)
	fs.SetOutput(stderr)
	server := fs.String("server", "fak", "Codex mcp_servers entry name")
	config := fs.String("config", defaultCodexConfigPath(), "Codex config.toml path")
	source := fs.String("source", "", "source fak binary to install (default: this executable)")
	policy := fs.String("policy", "", "capability-floor policy JSON (default: <repo>/examples/dev-agent-policy.json)")
	timeout := fs.Duration("timeout", 20*time.Second, "initialize verification timeout")
	rollback := fs.Bool("rollback", false, "restore the prior known-good config entry recorded by the last install")
	skipVerify := fs.Bool("skip-verify", false, "skip the post-install initialize handshake (install only)")
	asJSON := fs.Bool("json", false, "emit a stable JSON report")
	if !parseFlags(fs, argv) {
		return 2
	}
	if !validCodexServerName(*server) {
		return codexMCPFail(stderr, asJSON, "install", "SERVER_NAME", fmt.Sprintf("invalid --server %q: use letters, digits, '-' or '_' (it is written into a TOML table header)", *server))
	}
	// Absolutize the config path so the derived install dir and the recorded
	// launch command never depend on the caller's cwd (a relative --config would
	// otherwise plant artifacts under whatever directory Codex happens to start in).
	if abs, err := filepath.Abs(*config); err == nil {
		*config = abs
	}
	if *rollback {
		return runCodexMCPRollback(stdout, stderr, *config, *server, *asJSON)
	}

	srcPath := strings.TrimSpace(*source)
	if srcPath == "" {
		exe, err := os.Executable()
		if err != nil {
			return codexMCPFail(stderr, asJSON, "install", "SOURCE_UNRESOLVED", fmt.Sprintf("resolve this executable: %v", err))
		}
		srcPath = exe
	}
	if abs, err := filepath.Abs(srcPath); err == nil {
		srcPath = abs
	}
	if _, err := os.Stat(srcPath); err != nil {
		return codexMCPFail(stderr, asJSON, "install", "SOURCE_MISSING", fmt.Sprintf("source binary unreadable: %v", err))
	}

	policyPath := strings.TrimSpace(*policy)
	if policyPath == "" {
		policyPath = filepath.Join(repoRoot(), "examples", "dev-agent-policy.json")
	}
	if abs, err := filepath.Abs(policyPath); err == nil {
		policyPath = abs
	}
	policyBytes, err := os.ReadFile(policyPath)
	if err != nil {
		return codexMCPFail(stderr, asJSON, "install", "POLICY_UNREADABLE", fmt.Sprintf("policy unreadable: %v", err))
	}
	if !json.Valid(policyBytes) {
		return codexMCPFail(stderr, asJSON, "install", "POLICY_MALFORMED", "policy is not valid JSON: "+policyPath)
	}

	sum, err := sha256File(srcPath)
	if err != nil {
		return codexMCPFail(stderr, asJSON, "install", "SOURCE_UNREADABLE", err.Error())
	}
	version := appversion.Current()
	moduleRevision := guardShortBuildID()
	if moduleRevision == "" {
		moduleRevision = "sha256:" + sum[:12]
	}

	dir := codexMCPInstallDir(*config)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return codexMCPFail(stderr, asJSON, "install", "INSTALL_DIR", err.Error())
	}

	// Versioned, immutable filename: version + short content hash. An upgrade
	// writes a *new* file, so a Codex holding the prior binary open never blocks
	// the install.
	exeName := fmt.Sprintf("fak-%s-%s%s", sanitizeVersion(version), sum[:8], exeSuffix())
	installedCmd := filepath.Join(dir, exeName)
	if err := copyImmutable(srcPath, installedCmd); err != nil {
		return codexMCPFail(stderr, asJSON, "install", "BINARY_COPY", err.Error())
	}
	// Copy the policy alongside so the install is self-contained and does not
	// depend on the worktree remaining present.
	installedPolicy := filepath.Join(dir, fmt.Sprintf("policy-%s.json", sum[:8]))
	if err := writeFileAtomic(installedPolicy, policyBytes, 0o644); err != nil {
		return codexMCPFail(stderr, asJSON, "install", "POLICY_COPY", err.Error())
	}

	args := []string{"serve", "--stdio", "--policy", installedPolicy}

	priorSection, priorCmd, err := readPriorCodexSection(*config, *server)
	if err != nil {
		return codexMCPFail(stderr, asJSON, "install", "CONFIG_READ", err.Error())
	}
	// On a re-install or upgrade the entry we are about to replace is our own
	// fak-managed entry, not the operator's original. Capturing it as "prior"
	// would make rollback restore a fak entry (or nothing) instead of the
	// pre-fak config. Carry the original forward from the existing manifest so
	// rollback still restores what the operator had before fak ever touched it.
	if existing, mErr := readCodexMCPManifest(*config); mErr == nil {
		if priorCmd == existing.InstalledCommand || (priorCmd != "" && withinDir(dir, priorCmd)) {
			priorSection = existing.PriorConfigSection
			priorCmd = existing.PriorCommand
		}
	}

	man := codexMCPManifest{
		Schema:             codexMCPManifestSchema,
		Server:             *server,
		InstalledCommand:   installedCmd,
		Args:               args,
		PolicyPath:         installedPolicy,
		SourcePath:         srcPath,
		Version:            version,
		ModuleRevision:     moduleRevision,
		SHA256:             sum,
		Arch:               runtime.GOARCH,
		OS:                 runtime.GOOS,
		InstalledAt:        time.Now().UTC().Format(time.RFC3339),
		ConfigPath:         *config,
		PriorConfigSection: priorSection,
		PriorCommand:       priorCmd,
	}
	// Persist the backup manifest BEFORE the destructive config rewrite. If the
	// patch fails (or the process dies) after this point, the prior entry is
	// still both in the live config and recorded here, so rollback recovers it;
	// writing the manifest afterwards would leave a crash window with the prior
	// entry overwritten and no record of it.
	if err := writeCodexMCPManifest(*config, man); err != nil {
		return codexMCPFail(stderr, asJSON, "install", "MANIFEST_WRITE", err.Error())
	}
	if err := patchCodexMCPConfig(*config, *server, installedCmd, args); err != nil {
		return codexMCPFail(stderr, asJSON, "install", "CONFIG_PATCH", err.Error())
	}

	verified := false
	verifyCause := ""
	if !*skipVerify {
		verified, verifyCause = codexMCPInitializeProbe(installedCmd, args, *timeout)
		if !verified {
			// The artifact and config are in place; report the probe failure but do
			// not roll back — status/diagnose can re-probe, and rollback is explicit.
			report := map[string]any{
				"schema":    codexMCPManifestSchema,
				"action":    "install",
				"ok":        false,
				"cause":     "INITIALIZE_FAILED:" + verifyCause,
				"command":   installedCmd,
				"config":    *config,
				"server":    *server,
				"verified":  false,
				"installed": true,
			}
			if *asJSON {
				return encodeJSONOrFail(stdout, stderr, report, "fak codex mcp install")
			}
			fmt.Fprintf(stderr, "fak codex mcp install: installed %s but initialize probe failed (%s)\n", installedCmd, verifyCause)
			fmt.Fprintf(stderr, "  run `fak codex mcp status --server %s` to diagnose; `--rollback` restores the prior entry\n", *server)
			return 1
		}
	}

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"schema":    codexMCPManifestSchema,
			"action":    "install",
			"ok":        true,
			"command":   installedCmd,
			"policy":    installedPolicy,
			"sha256":    sum,
			"version":   version,
			"revision":  man.ModuleRevision,
			"config":    *config,
			"server":    *server,
			"verified":  verified,
			"installed": true,
		}, "fak codex mcp install")
	}
	fmt.Fprintf(stdout, "installed fak codex MCP: %s\n", installedCmd)
	fmt.Fprintf(stdout, "  policy:   %s\n", installedPolicy)
	fmt.Fprintf(stdout, "  version:  %s (%s)\n", version, shortOr(man.ModuleRevision, "no-vcs"))
	fmt.Fprintf(stdout, "  sha256:   %s\n", sum)
	fmt.Fprintf(stdout, "  config:   %s [mcp_servers.%s]\n", *config, *server)
	if verified {
		fmt.Fprintln(stdout, "  verify:   initialize handshake OK")
	} else {
		fmt.Fprintln(stdout, "  verify:   skipped")
	}
	return 0
}

func runCodexMCPRollback(stdout, stderr io.Writer, config, server string, asJSON bool) int {
	man, err := readCodexMCPManifest(config)
	if err != nil {
		return codexMCPFail(stderr, &asJSON, "rollback", "NO_MANIFEST", fmt.Sprintf("no install manifest to roll back from: %v", err))
	}
	if man.PriorConfigSection == "" {
		// Nothing preceded this install; rollback means removing the entry we own.
		if err := removeCodexMCPSection(config, server); err != nil {
			return codexMCPFail(stderr, &asJSON, "rollback", "CONFIG_PATCH", err.Error())
		}
	} else {
		if err := restoreCodexMCPSection(config, server, man.PriorConfigSection); err != nil {
			return codexMCPFail(stderr, &asJSON, "rollback", "CONFIG_PATCH", err.Error())
		}
	}
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"schema":  codexMCPManifestSchema,
			"action":  "rollback",
			"ok":      true,
			"config":  config,
			"server":  server,
			"command": man.PriorCommand,
		}, "fak codex mcp rollback")
	}
	if man.PriorCommand == "" {
		fmt.Fprintf(stdout, "rolled back: removed mcp_servers.%s (no prior entry)\n", server)
	} else {
		fmt.Fprintf(stdout, "rolled back mcp_servers.%s to prior command: %s\n", server, man.PriorCommand)
	}
	return 0
}

// ---- status -----------------------------------------------------------------

type codexMCPStatusReport struct {
	Schema    string `json:"schema"`
	Server    string `json:"server"`
	Config    string `json:"config_path"`
	State     string `json:"state"`
	Command   string `json:"command,omitempty"`
	Detail    string `json:"detail,omitempty"`
	Managed   bool   `json:"managed"`
	Probed    bool   `json:"probed"`
	Probe     string `json:"probe,omitempty"`
	Installed string `json:"installed_version,omitempty"`
	Current   string `json:"current_version,omitempty"`
	OK        bool   `json:"ok"`
}

func runCodexMCPStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("codex mcp status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	server := fs.String("server", "fak", "Codex mcp_servers entry name")
	config := fs.String("config", defaultCodexConfigPath(), "Codex config.toml path")
	timeout := fs.Duration("timeout", 20*time.Second, "initialize probe timeout")
	noProbe := fs.Bool("no-probe", false, "skip the initialize handshake (static checks only)")
	asJSON := fs.Bool("json", false, "emit a stable JSON report")
	if !parseFlags(fs, argv) {
		return 2
	}
	if abs, err := filepath.Abs(*config); err == nil {
		*config = abs
	}
	rep := classifyCodexMCPStatus(*config, *server, *timeout, !*noProbe)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rep, "fak codex mcp status")
	}
	writeCodexMCPStatusHuman(stdout, rep)
	if !rep.OK {
		return 1
	}
	return 0
}

// classifyCodexMCPStatus is the pure decision core (given the config + install
// state), factored out so tests can exercise every typed state without the CLI.
func classifyCodexMCPStatus(config, server string, timeout time.Duration, probe bool) codexMCPStatusReport {
	rep := codexMCPStatusReport{Schema: codexMCPStatusSchema, Server: server, Config: config, Current: appversion.Current()}

	command, _, cfgErr := readCodexMCPEntry(config, server)
	if cfgErr != nil {
		rep.State, rep.Detail = "absent", "no mcp_servers."+server+" entry in "+config
		return rep
	}
	rep.Command = command

	man, manErr := readCodexMCPManifest(config)
	if manErr != nil {
		rep.State = "unmanaged"
		rep.Detail = "config entry exists but no fak install manifest owns it"
		return rep
	}
	rep.Managed = true
	rep.Installed = man.Version

	if command != man.InstalledCommand {
		rep.State = "hand_edited"
		rep.Detail = "config command does not match the installed artifact: " + command
		return rep
	}

	info, statErr := os.Stat(command)
	if statErr != nil {
		rep.State = "missing_binary"
		rep.Detail = "installed command is gone: " + command
		return rep
	}
	if info.IsDir() {
		rep.State = "missing_binary"
		rep.Detail = "installed command path is a directory: " + command
		return rep
	}

	sum, sumErr := sha256File(command)
	if sumErr != nil {
		if isSharingViolation(sumErr) {
			rep.State = "locked"
			rep.Detail = "installed binary is locked/unreadable: " + sumErr.Error()
			return rep
		}
		rep.State = "unreadable"
		rep.Detail = sumErr.Error()
		return rep
	}
	if sum != man.SHA256 {
		rep.State = "checksum_mismatch"
		rep.Detail = "installed binary bytes differ from the manifest checksum (tampered)"
		return rep
	}

	if arch := binaryArch(command); arch != "" && arch != runtime.GOARCH {
		rep.State = "wrong_arch"
		rep.Detail = fmt.Sprintf("installed binary is %s but this host is %s", arch, runtime.GOARCH)
		return rep
	}

	if probe {
		ok, cause := codexMCPInitializeProbe(command, man.Args, timeout)
		rep.Probed = true
		if !ok {
			rep.State = "probe_failed"
			rep.Probe = cause
			rep.Detail = "initialize handshake failed: " + cause
			return rep
		}
		rep.Probe = "ok"
	}

	// Healthy artifact — is it the current fak, or an older one still serving?
	if currentSum, err := currentExecutableSum(); err == nil && currentSum != "" && currentSum != man.SHA256 {
		rep.State = "stale"
		rep.Detail = "installed binary is healthy but older than the current fak; re-run install to upgrade"
		rep.OK = true // stale is advisory: the entry still serves.
		return rep
	}

	rep.State = "healthy"
	rep.OK = true
	return rep
}

func writeCodexMCPStatusHuman(w io.Writer, rep codexMCPStatusReport) {
	fmt.Fprintf(w, "== fak codex mcp status: %s ==\n", rep.Server)
	fmt.Fprintf(w, "state:   %s\n", rep.State)
	if rep.Command != "" {
		fmt.Fprintf(w, "command: %s\n", rep.Command)
	}
	if rep.Installed != "" {
		fmt.Fprintf(w, "version: installed=%s current=%s\n", rep.Installed, rep.Current)
	}
	if rep.Probed {
		fmt.Fprintf(w, "probe:   %s\n", shortOr(rep.Probe, "n/a"))
	}
	if rep.Detail != "" {
		fmt.Fprintf(w, "detail:  %s\n", rep.Detail)
	}
}

// ---- uninstall --------------------------------------------------------------

// removeCodexOwnedArtifact deletes one installed artifact, distinguishing the
// three outcomes uninstall must treat differently: removed, absent (idempotent
// no-op), or held open by a live Codex (locked → retry, not a failure). It clears
// the installed read-only bit first so the OS permits the delete, but never
// follows a symlink when doing so — a link planted under the install dir must not
// redirect the chmod onto its target.
func removeCodexOwnedArtifact(p string) (removed, locked bool, err error) {
	if p == "" {
		return false, false, nil
	}
	if fi, lerr := os.Lstat(p); lerr == nil && fi.Mode()&os.ModeSymlink == 0 {
		_ = os.Chmod(p, 0o644)
	}
	switch rmErr := os.Remove(p); {
	case rmErr == nil:
		return true, false, nil
	case errors.Is(rmErr, os.ErrNotExist):
		return false, false, nil
	case isSharingViolation(rmErr):
		return false, true, nil
	default:
		return false, false, rmErr
	}
}

// codexMCPUninstallLocked reports the atomic-abort outcome: the current binary is
// held open, so uninstall changed nothing (config included) and the operator
// should stop Codex and re-run. Kept a distinct exit-1 path so scripts see the
// uninstall did not complete, while a retry after Codex exits converges cleanly.
func codexMCPUninstallLocked(stdout, stderr io.Writer, asJSON *bool, config, server, command string) int {
	if asJSON != nil && *asJSON {
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"schema":          codexMCPManifestSchema,
			"action":          "uninstall",
			"ok":              false,
			"cause":           "BINARY_LOCKED",
			"config":          config,
			"server":          server,
			"command":         command,
			"removed_section": false,
			"removed_files":   []string{},
		}, "fak codex mcp uninstall")
	}
	fmt.Fprintf(stderr, "fak codex mcp uninstall: installed binary is held open (Codex still running?): %s\n", command)
	fmt.Fprintln(stderr, "  nothing was changed; stop Codex and re-run `fak codex mcp uninstall` (it is idempotent)")
	return 1
}

func runCodexMCPUninstall(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("codex mcp uninstall", flag.ContinueOnError)
	fs.SetOutput(stderr)
	server := fs.String("server", "fak", "Codex mcp_servers entry name")
	config := fs.String("config", defaultCodexConfigPath(), "Codex config.toml path")
	keepBinaries := fs.Bool("keep-binaries", false, "remove only the config entry, leave installed artifacts")
	asJSON := fs.Bool("json", false, "emit a stable JSON report")
	if !parseFlags(fs, argv) {
		return 2
	}
	if abs, err := filepath.Abs(*config); err == nil {
		*config = abs
	}

	dir := codexMCPInstallDir(*config)
	removedFiles := []string{}
	lockedFiles := []string{}

	// Artifacts first, config second: the current binary is the one a live Codex
	// most likely holds open, so we gate on it. If it is locked we change nothing
	// (config included) and tell the operator to retry — a half-removed state that
	// stripped the config while leaving a live binary must never happen.
	if !*keepBinaries {
		if man, err := readCodexMCPManifest(*config); err == nil {
			removed, locked, rmErr := removeCodexOwnedArtifact(man.InstalledCommand)
			if rmErr != nil {
				return codexMCPFail(stderr, asJSON, "uninstall", "ARTIFACT_REMOVE", rmErr.Error())
			}
			if locked {
				return codexMCPUninstallLocked(stdout, stderr, asJSON, *config, *server, man.InstalledCommand)
			}
			if removed {
				removedFiles = append(removedFiles, man.InstalledCommand)
			}
			// Reap every other artifact the install dir owns: the current policy
			// plus any binaries/policies orphaned by prior upgrades (versioned
			// filenames mean an upgrade leaves the superseded pair behind). Older
			// binaries may be held open by other live sessions; skip those
			// best-effort rather than block the whole uninstall on them.
			owned, _ := filepath.Glob(filepath.Join(dir, "fak-*"))
			if pol, _ := filepath.Glob(filepath.Join(dir, "policy-*.json")); len(pol) > 0 {
				owned = append(owned, pol...)
			}
			for _, p := range owned {
				if p == man.InstalledCommand {
					continue // gated above
				}
				removed, locked, rmErr := removeCodexOwnedArtifact(p)
				if rmErr != nil {
					return codexMCPFail(stderr, asJSON, "uninstall", "ARTIFACT_REMOVE", rmErr.Error())
				}
				switch {
				case removed:
					removedFiles = append(removedFiles, p)
				case locked:
					lockedFiles = append(lockedFiles, p)
				}
			}
		}
	}

	removedSection := false
	if _, _, err := readCodexMCPEntry(*config, *server); err == nil {
		if err := removeCodexMCPSection(*config, *server); err != nil {
			return codexMCPFail(stderr, asJSON, "uninstall", "CONFIG_PATCH", err.Error())
		}
		removedSection = true
	}

	if !*keepBinaries && len(lockedFiles) == 0 {
		// Everything owned is gone: drop the manifest and the dir (if empty).
		// When artifacts remain locked we keep the manifest so a later uninstall
		// can still find and reap them.
		manPath := codexMCPManifestPath(*config)
		if err := os.Remove(manPath); err == nil {
			removedFiles = append(removedFiles, manPath)
		}
		_ = os.Remove(dir) // drops the dir only if now empty
	}

	sort.Strings(removedFiles)
	sort.Strings(lockedFiles)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"schema":          codexMCPManifestSchema,
			"action":          "uninstall",
			"ok":              true,
			"config":          *config,
			"server":          *server,
			"removed_section": removedSection,
			"removed_files":   removedFiles,
			"locked_files":    lockedFiles,
		}, "fak codex mcp uninstall")
	}
	if !removedSection && len(removedFiles) == 0 && len(lockedFiles) == 0 {
		fmt.Fprintf(stdout, "uninstall: nothing owned for mcp_servers.%s (already clean)\n", *server)
		return 0
	}
	fmt.Fprintf(stdout, "uninstalled mcp_servers.%s from %s\n", *server, *config)
	for _, p := range removedFiles {
		fmt.Fprintf(stdout, "  removed %s\n", p)
	}
	for _, p := range lockedFiles {
		fmt.Fprintf(stdout, "  deferred (held open, retry after Codex exits): %s\n", p)
	}
	return 0
}

// ---- config patching --------------------------------------------------------

// patchCodexMCPConfig rewrites the `[mcp_servers.<server>]` block to point at
// command/args, preserving every other byte of the config (comments, unrelated
// tables). Written atomically via a temp+rename so a reader sees the whole old
// or whole new document, never a torn one.
func patchCodexMCPConfig(configPath, server, command string, args []string) error {
	existing, err := readConfigOrEmpty(configPath)
	if err != nil {
		return err
	}
	section := "mcp_servers." + server
	// Refuse to append a plain [section] next to an existing [[section]]
	// array-of-tables: that yields a config TOML itself rejects as a duplicate
	// key, and we would not own either entry cleanly.
	lines := strings.Split(existing, "\n")
	if start, _ := findTOMLSection(lines, section); start < 0 && tomlArrayTableExists(lines, section) {
		return fmt.Errorf("config declares [[%s]] as an array-of-tables; remove that form before installing (fak manages a plain [%s] table)", section, section)
	}
	block := renderCodexMCPSection(server, command, args)
	out, _, _ := replaceTOMLSection(existing, section, block)
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		return err
	}
	return writeFileAtomic(configPath, []byte(out), 0o600)
}

func restoreCodexMCPSection(configPath, server, priorBlock string) error {
	existing, err := readConfigOrEmpty(configPath)
	if err != nil {
		return err
	}
	out, _, _ := replaceTOMLSection(existing, "mcp_servers."+server, strings.TrimRight(priorBlock, "\n"))
	return writeFileAtomic(configPath, []byte(out), 0o600)
}

func removeCodexMCPSection(configPath, server string) error {
	existing, err := readConfigOrEmpty(configPath)
	if err != nil {
		return err
	}
	out := removeTOMLSection(existing, "mcp_servers."+server)
	if out == existing {
		return nil
	}
	return writeFileAtomic(configPath, []byte(out), 0o600)
}

func readPriorCodexSection(configPath, server string) (section, command string, err error) {
	existing, err := readConfigOrEmpty(configPath)
	if err != nil {
		return "", "", err
	}
	section = extractTOMLSection(existing, "mcp_servers."+server)
	if section != "" {
		command, _, _ = readCodexMCPEntry(configPath, server)
	}
	return section, command, nil
}

func readConfigOrEmpty(configPath string) (string, error) {
	b, err := os.ReadFile(configPath)
	if errors.Is(err, os.ErrNotExist) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// renderCodexMCPSection emits a canonical block. Windows paths carry backslashes,
// so it prefers TOML literal (single-quoted) strings — which the doctor/health
// parser round-trips — falling back to escaped basic strings when a value holds a
// quote or newline.
func renderCodexMCPSection(server, command string, args []string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "[mcp_servers.%s]\n", server)
	fmt.Fprintf(&b, "command = %s\n", tomlLiteralOrBasic(command))
	b.WriteString("args = [")
	for i, a := range args {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(tomlLiteralOrBasic(a))
	}
	b.WriteString("]\n")
	return b.String()
}

func tomlLiteralOrBasic(s string) string {
	if !strings.ContainsAny(s, "'\n\r") {
		return "'" + s + "'"
	}
	// Basic string with JSON-compatible escaping (round-trips via parseTOMLString).
	esc, _ := json.Marshal(s)
	return string(esc)
}

// replaceTOMLSection returns text with the block for `section` replaced by
// newBlock (which must include its own header line). The block spans the header
// line through the line before the next table header (including child sub-tables
// `section.*`) or EOF. When the section is absent it is appended. Returns the
// prior block text and whether it existed.
func replaceTOMLSection(text, section, newBlock string) (out, prior string, existed bool) {
	lines := strings.Split(text, "\n")
	start, end := findTOMLSection(lines, section)
	newLines := strings.Split(strings.TrimRight(newBlock, "\n"), "\n")
	if start < 0 {
		// Append, ensuring a blank separator line before the new block.
		trimmed := strings.TrimRight(text, "\n")
		var b strings.Builder
		if trimmed != "" {
			b.WriteString(trimmed)
			b.WriteString("\n\n")
		}
		b.WriteString(strings.Join(newLines, "\n"))
		b.WriteString("\n")
		return b.String(), "", false
	}
	prior = strings.Join(lines[start:end], "\n")
	merged := append([]string{}, lines[:start]...)
	merged = append(merged, newLines...)
	merged = append(merged, lines[end:]...)
	return strings.Join(merged, "\n"), prior, true
}

func removeTOMLSection(text, section string) string {
	lines := strings.Split(text, "\n")
	start, end := findTOMLSection(lines, section)
	if start < 0 {
		return text
	}
	// Also drop a single trailing blank separator left behind, if any.
	if end < len(lines) && strings.TrimSpace(lines[end]) == "" {
		end++
	}
	merged := append([]string{}, lines[:start]...)
	merged = append(merged, lines[end:]...)
	res := strings.Join(merged, "\n")
	// Collapse a leading blank run the removal may have exposed.
	return strings.TrimLeft(res, "\n")
}

func extractTOMLSection(text, section string) string {
	lines := strings.Split(text, "\n")
	start, end := findTOMLSection(lines, section)
	if start < 0 {
		return ""
	}
	return strings.Join(lines[start:end], "\n")
}

// findTOMLSection locates [start,end) line indices for a plain table header and
// its plain child sub-tables (`section.*`). Two invariants keep an install or
// uninstall from ever deleting unrelated config:
//
//   - An array-of-tables header (`[[x]]`) is ALWAYS a boundary, never absorbed.
//     Without this the span would run from the fak table straight through any
//     `[[foo]]` block that happened to follow it (until the next plain table),
//     silently deleting it on the next rewrite.
//   - Trailing blank/comment lines immediately before the boundary are excluded
//     from the span: they document the following table, so replacing or removing
//     the fak block must leave them attached to what comes after.
//
// Returns start=-1 when absent.
func findTOMLSection(lines []string, section string) (start, end int) {
	start = -1
	for i, raw := range lines {
		name, ok, array := tomlHeaderName(raw)
		if !ok {
			continue
		}
		if start < 0 {
			// Only a plain table defines the section we manage; a stray
			// `[[section]]` array-of-tables is a foreign construct, not our start.
			if name == section && !array {
				start = i
			}
			continue
		}
		// Inside the target section: a plain child sub-table stays in the span;
		// any other header (including every array-of-tables) ends it.
		if !array && (name == section || strings.HasPrefix(name, section+".")) {
			continue
		}
		return start, trimTrailingBlankComment(lines, start, i)
	}
	if start < 0 {
		return -1, -1
	}
	return start, len(lines)
}

// trimTrailingBlankComment pulls end back past a run of blank or comment lines
// directly preceding it, so those lines are not counted as part of the section.
// It never trims past start+1, so the header line always survives.
func trimTrailingBlankComment(lines []string, start, end int) int {
	for end > start+1 {
		s := strings.TrimSpace(lines[end-1])
		if s != "" && !strings.HasPrefix(s, "#") {
			break
		}
		end--
	}
	return end
}

// tomlArrayTableExists reports whether the document declares `section` as an
// array-of-tables (`[[section]]`) — a form fak does not manage and must refuse to
// collide with, rather than appending a conflicting plain `[section]` table.
func tomlArrayTableExists(lines []string, section string) bool {
	for _, raw := range lines {
		if name, ok, array := tomlHeaderName(raw); ok && array && name == section {
			return true
		}
	}
	return false
}

// tomlHeaderName extracts the table name from a `[a.b.c]` or `[[a.b.c]]` header
// line, ignoring a trailing inline comment. Returns ok=false for non-header
// lines and array=true for an array-of-tables header (`[[x]]`).
func tomlHeaderName(raw string) (name string, ok, array bool) {
	line := strings.TrimSpace(raw)
	if !strings.HasPrefix(line, "[") {
		return "", false, false
	}
	if strings.HasPrefix(line, "[[") {
		end := strings.Index(line, "]]")
		if end < 0 {
			return "", false, false
		}
		return strings.TrimSpace(line[2:end]), true, true
	}
	end := strings.IndexByte(line, ']')
	if end < 0 {
		return "", false, false
	}
	return strings.TrimSpace(line[1:end]), true, false
}

// ---- manifest IO ------------------------------------------------------------

func writeCodexMCPManifest(configPath string, man codexMCPManifest) error {
	dir := codexMCPInstallDir(configPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(man, "", "  ")
	if err != nil {
		return err
	}
	// 0o600: the manifest embeds the operator's prior config section, which may
	// hold API keys or other secrets carried in the entry we replaced.
	return writeFileAtomic(codexMCPManifestPath(configPath), b, 0o600)
}

func readCodexMCPManifest(configPath string) (codexMCPManifest, error) {
	var man codexMCPManifest
	b, err := os.ReadFile(codexMCPManifestPath(configPath))
	if err != nil {
		return man, err
	}
	if err := json.Unmarshal(b, &man); err != nil {
		return man, fmt.Errorf("decode install manifest: %w", err)
	}
	return man, nil
}

// ---- binary helpers ---------------------------------------------------------

func sha256File(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func currentExecutableSum() (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	return sha256File(exe)
}

// copyImmutable copies src to dst atomically (temp+rename) and marks it
// read-only, so a fat-fingered edit of the installed artifact is refused and the
// manifest checksum stays the source of truth. It is idempotent: because the
// destination filename encodes the content hash, an identical dst is left as-is
// (a re-install never has to overwrite a read-only file that may be held open).
func copyImmutable(src, dst string) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	if existing, err := os.ReadFile(dst); err == nil && bytes.Equal(existing, data) {
		return nil // already installed with identical bytes; immutable
	}
	// Clear a stale/read-only dst so the atomic rename can publish (Windows
	// refuses to replace a read-only target).
	_ = os.Chmod(dst, 0o644)
	_ = os.Remove(dst)
	if err := writeFileAtomic(dst, data, 0o755); err != nil {
		return err
	}
	// Best-effort read-only; ignore on platforms/filesystems that reject it.
	_ = os.Chmod(dst, 0o555)
	return nil
}

// binaryArch reads the executable header magic and returns a GOARCH string
// ("amd64","arm64","386") or "" when unknown. Deterministic and offline, so
// wrong-arch installs are caught without executing the artifact.
func binaryArch(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	hdr := make([]byte, 64)
	n, _ := io.ReadFull(f, hdr)
	if n < 20 {
		return ""
	}
	switch {
	case hdr[0] == 'M' && hdr[1] == 'Z': // PE (Windows)
		peOff := int64(binary.LittleEndian.Uint32(hdr[0x3C:0x40]))
		sig := make([]byte, 6)
		if _, err := f.ReadAt(sig, peOff); err != nil {
			return ""
		}
		if sig[0] != 'P' || sig[1] != 'E' || sig[2] != 0 || sig[3] != 0 {
			return ""
		}
		switch binary.LittleEndian.Uint16(sig[4:6]) {
		case 0x8664:
			return "amd64"
		case 0xAA64:
			return "arm64"
		case 0x14c:
			return "386"
		}
	case hdr[0] == 0x7F && hdr[1] == 'E' && hdr[2] == 'L' && hdr[3] == 'F': // ELF
		switch binary.LittleEndian.Uint16(hdr[18:20]) {
		case 0x3E:
			return "amd64"
		case 0xB7:
			return "arm64"
		case 0x03:
			return "386"
		}
	case binary.LittleEndian.Uint32(hdr[0:4]) == 0xFEEDFACF: // Mach-O 64
		switch binary.LittleEndian.Uint32(hdr[4:8]) {
		case 0x01000007:
			return "amd64"
		case 0x0100000C:
			return "arm64"
		}
	}
	return ""
}

// ---- small utilities --------------------------------------------------------

func exeSuffix() string {
	if runtime.GOOS == "windows" {
		return ".exe"
	}
	return ""
}

// sanitizeVersion maps an arbitrary version string to a filesystem-safe token for
// the versioned artifact name. It keeps only an allowlist of characters safe on
// every target filesystem — anything else (path separators, the Windows-reserved
// `:*?"<>|`, control bytes) becomes '-' — then trims trailing dots/spaces, which
// Windows silently strips from filenames. The <sha8> suffix on the final name
// keeps installs distinct even if two versions collapse to the same token.
func sanitizeVersion(v string) string {
	var b strings.Builder
	for _, r := range strings.TrimSpace(v) {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9',
			r == '.', r == '-', r == '_', r == '+':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}
	out := strings.Trim(b.String(), ". ")
	// Fall back to "dev" when nothing alphanumeric survived (empty input, or a
	// version that was all separators/reserved chars) — a token of only '-'/'_'/
	// '+'/'.' is a useless filename fragment; the <sha8> suffix still keeps
	// installs distinct.
	if strings.Trim(out, "-_+.") == "" {
		return "dev"
	}
	return out
}

// validCodexServerName gates the --server value before it is written into a TOML
// table header (`[mcp_servers.<name>]`). Restricting it to an unambiguous
// identifier keeps a crafted name from injecting extra tables or breaking the
// exact-string match uninstall relies on to own its entry.
func validCodexServerName(name string) bool {
	if name == "" {
		return false
	}
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9',
			r == '-', r == '_':
		default:
			return false
		}
	}
	return true
}

func shortOr(s, fallback string) string {
	if strings.TrimSpace(s) == "" {
		return fallback
	}
	return s
}

// isSharingViolation reports whether err is a Windows sharing violation or POSIX
// busy/permission error — the signature of a binary held open by a running Codex.
// Kept string/errno-tolerant so it needs no per-OS build tag.
func isSharingViolation(err error) bool {
	if err == nil {
		return false
	}
	if os.IsPermission(err) {
		return true
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "sharing violation") ||
		strings.Contains(msg, "used by another process") ||
		strings.Contains(msg, "text file busy")
}

func withinDir(dir, path string) bool {
	rel, err := filepath.Rel(dir, path)
	if err != nil {
		return false
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

// codexMCPFail writes a typed failure and returns exit 1. asJSON may be nil.
func codexMCPFail(stderr io.Writer, asJSON *bool, action, cause, detail string) int {
	if asJSON != nil && *asJSON {
		b, _ := json.MarshalIndent(map[string]any{
			"schema": codexMCPManifestSchema,
			"action": action,
			"ok":     false,
			"cause":  cause,
			"detail": detail,
		}, "", "  ")
		fmt.Fprintln(stderr, string(b))
		return 1
	}
	fmt.Fprintf(stderr, "fak codex mcp %s: %s: %s\n", action, cause, detail)
	return 1
}
