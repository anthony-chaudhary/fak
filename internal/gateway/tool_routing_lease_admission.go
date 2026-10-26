package gateway

import (
	"context"
	"encoding/json"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/vdso"
	"github.com/anthony-chaudhary/fak/internal/wipref"
)

// adjudicateLeaseAdmission is the file-ownership half of the pre-execution
// boundary. The kernel answers whether a call is permitted in principle; this
// check answers whether this guarded session may write the named workspace tree
// now. It reads the host-injected canonical lease/presence view, whose workspace
// is selected by the host rather than by untrusted tool arguments.
//
// refused is false when the call proves no workspace write footprint or when
// the authority admits its write set. Once a write footprint exists, an absent,
// unreadable, or malformed authority fails closed instead of turning an
// observational outage into permission to overwrite a peer. The wire's
// read_only field is only a cache hint, so it cannot exempt a structurally
// mutating call from ownership admission.
func (s *Server) adjudicateLeaseAdmission(ctx context.Context, tool, rawArgs string, readOnly bool) (WireVerdict, bool) {
	args, rawTargets, mutating, known := adjudicateRawWriteTargets(tool, rawArgs, readOnly)
	if !mutating {
		return WireVerdict{}, false
	}
	if !known || len(rawTargets) == 0 {
		return leaseAdmissionRefusal(abi.ReasonDefaultDeny, "write footprint is unknown"), true
	}
	admissionProvider := workspaceLeaseAdmissionProvider
	if admissionProvider == nil {
		return leaseAdmissionRefusal(abi.ReasonDefaultDeny, "workspace lease authority is unavailable"), true
	}
	admission, err := admissionProvider(ctx)
	if err != nil {
		return leaseAdmissionRefusal(abi.ReasonDefaultDeny, "workspace lease authority read failed"), true
	}
	root, err := filepath.Abs(strings.TrimSpace(admission.WorkspaceRoot))
	if err != nil || strings.TrimSpace(admission.WorkspaceRoot) == "" {
		return leaseAdmissionRefusal(abi.ReasonDefaultDeny, "workspace lease authority returned an invalid root"), true
	}
	root, err = canonicalExistingWorkspacePath(root)
	if err != nil {
		return leaseAdmissionRefusal(abi.ReasonDefaultDeny, "workspace lease authority root cannot be canonicalized"), true
	}
	targets, known := adjudicateNormalizeWriteTargets(args, rawTargets, root)
	if !known {
		return leaseAdmissionRefusal(abi.ReasonDefaultDeny, "write footprint is outside the trusted workspace"), true
	}
	provider := leasePlanePresence
	if provider == nil {
		return leaseAdmissionRefusal(abi.ReasonDefaultDeny, "lease authority is unavailable"), true
	}
	view, err := provider(ctx)
	if err != nil {
		return leaseAdmissionRefusal(abi.ReasonDefaultDeny, "lease authority read failed"), true
	}
	var leases []adjudicateClassifiedLease
	if err := json.Unmarshal(view.ClassifiedLeases, &leases); err != nil {
		return leaseAdmissionRefusal(abi.ReasonDefaultDeny, "lease authority returned an invalid view"), true
	}
	for _, lease := range admission.Leases {
		if len(lease.TreeGlobs) == 0 {
			return leaseAdmissionRefusal(abi.ReasonDefaultDeny, "workspace lease authority returned an invalid view"), true
		}
		leases = append(leases, adjudicateClassifiedLease(lease))
	}
	for _, lease := range leases {
		if len(lease.TreeGlobs) == 0 {
			return leaseAdmissionRefusal(abi.ReasonDefaultDeny, "lease authority returned an invalid view"), true
		}
		canonicalGlobs := make([]string, 0, len(lease.TreeGlobs))
		for _, treeGlob := range lease.TreeGlobs {
			canonicalGlob, ok := adjudicateCanonicalTreeGlob(root, treeGlob)
			if !ok {
				return leaseAdmissionRefusal(abi.ReasonDefaultDeny, "lease authority tree cannot be canonicalized"), true
			}
			canonicalGlobs = append(canonicalGlobs, canonicalGlob)
		}
		if dispatchorder.TreesOverlap(targets, canonicalGlobs) {
			claim := "peer lease holds " + strings.Join(lease.TreeGlobs, ",")
			return leaseAdmissionRefusal(abi.ReasonLeaseHeld, claim), true
		}
	}
	return WireVerdict{}, false
}

// WorkspaceAdmissionLease is one host-observed lease row used by the gateway's
// execution boundary. SessionID is an authority-supplied holder identity, never
// a request trace.
type WorkspaceAdmissionLease struct {
	TreeGlobs []string `json:"tree_globs"`
	SessionID string   `json:"session_id"`
}

// WorkspaceLeaseAdmissionView binds lease rows to the same trusted workspace
// root used to canonicalize proposed write paths. OwnSession is retained for wire
// compatibility only: no current production acquisition path authenticates a DOS
// row to a guard launch, so admission grants no ownership exemption from this text.
type WorkspaceLeaseAdmissionView struct {
	WorkspaceRoot string
	OwnSession    string
	Leases        []WorkspaceAdmissionLease
}

var workspaceLeaseAdmissionProvider func(context.Context) (WorkspaceLeaseAdmissionView, error)

// SetWorkspaceLeaseAdmissionProvider installs the host adapter for the
// canonical workspace lease authority. A nil provider makes writes fail closed.
func SetWorkspaceLeaseAdmissionProvider(provider func(context.Context) (WorkspaceLeaseAdmissionView, error)) {
	workspaceLeaseAdmissionProvider = provider
}

type adjudicateClassifiedLease struct {
	TreeGlobs []string `json:"tree_globs"`
	SessionID string   `json:"session_id"`
}

func leaseAdmissionRefusal(reason abi.ReasonCode, claim string) WireVerdict {
	return renderVerdict(abi.Verdict{
		Kind:    abi.VerdictDeny,
		Reason:  reason,
		By:      "lease-admission",
		Payload: abi.WitnessPayload{Claim: claim},
	}, nil)
}

// adjudicateRawWriteTargets extracts an untrusted write footprint without path
// normalization. Keeping absolute paths intact is required until the host has
// supplied the trusted workspace root.
func adjudicateRawWriteTargets(tool, rawArgs string, readOnly bool) (args map[string]any, targets []string, mutating, known bool) {
	if err := json.Unmarshal([]byte(rawArgs), &args); err != nil {
		return nil, nil, adjudicateMutatingTool(tool), false
	}
	base := adjudicateToolBase(tool)
	mutating = adjudicateMutatingTool(base)
	if adjudicateCodexPatchTool(base) {
		patchTargets, patchKnown := adjudicateCodexPatchFootprint(rawArgs)
		return args, patchTargets, true, patchKnown
	}
	targets = adjudicateDirectWriteTargets(args)
	// An allowed but unrecognized tool does not earn read status merely because
	// its name is absent from wipref's small write/edit vocabulary. Path-shaped
	// or content-shaped arguments make its write footprint conservative unless
	// the tool is a known read operation.
	if !mutating && !adjudicateKnownReadTool(base) && (len(targets) > 0 || adjudicateHasWriteLikeArgs(args)) {
		return args, nil, true, false
	}
	if command, ok := adjudicateCommandArgument(args); ok {
		commandTargets := adjudicator.ExtractCommandWriteTargets(command)
		if len(commandTargets) > 0 {
			mutating = true
			targets = append(targets, commandTargets...)
		} else if adjudicateShellCommandTool(base) {
			// A shell can hide writes behind scripts, aliases, environment stripping,
			// or syntax the target extractor does not recognize. The caller's
			// read_only hint requests classification; only the existing positive
			// gh/git grammar or in-process shell-read grammar can prove a read.
			_, promotableRead := vdso.ParseShellRead(command)
			if readOnly && (adjudicator.PositiveCLIReadOnly(command) || promotableRead) {
				return args, nil, false, true
			}
			return args, nil, true, false
		}
	}
	if !mutating {
		return args, nil, false, true
	}
	return args, targets, true, len(targets) > 0
}

func adjudicateShellCommandTool(tool string) bool {
	tool = strings.ToLower(tool)
	return tool == "exec_command" || tool == "shell_command" || tool == "bash" ||
		tool == "sh" || tool == "shell" || tool == "powershell" || tool == "terminal" || tool == "zsh" ||
		strings.HasSuffix(tool, "_exec_command")
}

func adjudicateNormalizeWriteTargets(args map[string]any, targets []string, root string) ([]string, bool) {
	workdir, validWorkdir := adjudicateWorkdir(args, root)
	if !validWorkdir {
		return nil, false
	}
	for i := range targets {
		var ok bool
		targets[i], ok = adjudicateWorkspacePath(root, workdir, targets[i])
		if !ok {
			return nil, false
		}
	}
	return targets, len(targets) > 0
}

// adjudicateDirectWriteTargets mirrors the mutating tool argument vocabulary
// without normalizing first. Absolute paths must remain absolute until
// adjudicateWorkspacePath proves they are inside the trusted workspace root.
func adjudicateDirectWriteTargets(args map[string]any) []string {
	var targets []string
	for _, key := range []string{
		"file_path", "filePath", "path", "filepath", "file",
		"target", "filename", "dest", "destination", "target_path", "targetPath",
		"source", "src", "from", "to", "old_path", "oldPath", "new_path", "newPath",
	} {
		if value, ok := args[key].(string); ok && strings.TrimSpace(value) != "" {
			targets = append(targets, value)
		}
	}
	for _, key := range []string{"file_paths", "filePaths", "paths", "files", "targets"} {
		switch values := args[key].(type) {
		case []string:
			for _, value := range values {
				if strings.TrimSpace(value) != "" {
					targets = append(targets, value)
				}
			}
		case []any:
			for _, item := range values {
				if value, ok := item.(string); ok && strings.TrimSpace(value) != "" {
					targets = append(targets, value)
				}
			}
		}
	}
	return targets
}

func adjudicateToolBase(tool string) string {
	tool = strings.TrimSpace(tool)
	if i := strings.LastIndex(tool, "__"); i >= 0 {
		tool = tool[i+2:]
	}
	if i := strings.LastIndexByte(tool, '.'); i >= 0 {
		tool = tool[i+1:]
	}
	return tool
}

func adjudicateMutatingTool(tool string) bool {
	base := strings.ToLower(adjudicateToolBase(tool))
	if wipref.IsMutatingTool(base) {
		return true
	}
	for _, token := range adjudicateToolTokens(base) {
		switch token {
		case "write", "edit", "create", "overwrite", "append", "replace", "save",
			"patch", "apply", "delete", "remove", "rm", "unlink", "move", "mv",
			"rename", "copy", "cp", "mkdir", "makedir", "touch", "truncate":
			return true
		}
	}
	return false
}

func adjudicateKnownReadTool(tool string) bool {
	for _, token := range adjudicateToolTokens(strings.ToLower(adjudicateToolBase(tool))) {
		switch token {
		case "read", "get", "list", "search", "find", "glob", "grep", "stat",
			"inspect", "view", "open", "cat", "head", "tail", "diff":
			return true
		}
	}
	return false
}

func adjudicateToolTokens(tool string) []string {
	return strings.FieldsFunc(tool, func(r rune) bool {
		return r == '_' || r == '-' || r == '.' || r == '/' || r == ':'
	})
}

func adjudicateHasWriteLikeArgs(args map[string]any) bool {
	for _, key := range []string{
		"content", "contents", "data", "patch", "replacement", "new_content", "newContent",
		"append", "text", "bytes", "mode", "permissions",
	} {
		if value, ok := args[key]; ok && value != nil {
			return true
		}
	}
	return false
}

func adjudicateCommandArgument(args map[string]any) (string, bool) {
	for _, key := range []string{"command", "cmd"} {
		if value, ok := args[key].(string); ok && strings.TrimSpace(value) != "" {
			return value, true
		}
	}
	return "", false
}

func adjudicateWorkdir(args map[string]any, root string) (string, bool) {
	for _, key := range []string{"workdir", "cwd"} {
		if value, ok := args[key].(string); ok && strings.TrimSpace(value) != "" {
			path, valid := adjudicateWorkspacePath(root, "", value)
			if !valid {
				return "", false
			}
			info, err := os.Stat(filepath.Join(root, filepath.FromSlash(path)))
			return path, err == nil && info.IsDir()
		}
	}
	return "", true
}

func adjudicateWorkspacePath(root, workdir, target string) (string, bool) {
	// Normalize both sides of Rel: host aliases such as macOS /var and
	// /private/var must describe the same workspace boundary.
	canonicalRoot, rootErr := canonicalExistingWorkspacePath(root)
	if rootErr != nil {
		return "", false
	}
	root = canonicalRoot
	target = strings.TrimSpace(strings.Trim(target, `"'`))
	if target == "" {
		return "", false
	}
	osTarget := filepath.FromSlash(strings.ReplaceAll(target, `\`, "/"))
	if !filepath.IsAbs(osTarget) {
		osTarget = filepath.Join(root, filepath.FromSlash(workdir), osTarget)
	}
	osTarget = filepath.Clean(osTarget)
	canonicalTarget, err := canonicalExistingWorkspacePath(osTarget)
	if err != nil {
		if !os.IsNotExist(err) {
			return "", false
		}
		// A new leaf is safe only when its complete parent already exists and
		// canonicalizes. Missing intermediate parents are ambiguous because a
		// later symlink could redirect the proposed write after admission.
		canonicalParent, parentErr := canonicalExistingWorkspacePath(filepath.Dir(osTarget))
		if parentErr != nil {
			return "", false
		}
		canonicalTarget = filepath.Join(canonicalParent, filepath.Base(osTarget))
	}
	rel, err := filepath.Rel(root, filepath.Clean(canonicalTarget))
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", false
	}
	return filepath.ToSlash(rel), true
}

// adjudicateCanonicalTreeGlob resolves the non-pattern prefix of a lease tree
// through the same canonical workspace used for proposed targets. This makes a
// lease declared through a directory symlink overlap writes through either the
// alias or its real spelling.
func adjudicateCanonicalTreeGlob(root, treeGlob string) (string, bool) {
	treeGlob = filepath.ToSlash(strings.ReplaceAll(strings.TrimSpace(treeGlob), `\`, "/"))
	if treeGlob == "" || strings.IndexByte(treeGlob, 0) >= 0 {
		return "", false
	}
	for _, component := range strings.Split(treeGlob, "/") {
		if component == ".." {
			return "", false
		}
	}
	globPath := filepath.FromSlash(treeGlob)
	if filepath.VolumeName(globPath) != "" && !filepath.IsAbs(globPath) {
		return "", false
	}
	if _, err := path.Match(treeGlob, ""); err != nil {
		return "", false
	}
	meta := strings.IndexAny(treeGlob, "*?[")
	if meta < 0 {
		return adjudicateCanonicalLeaseStaticPrefix(root, treeGlob)
	}
	boundary := strings.LastIndex(treeGlob[:meta], "/") + 1
	staticDir := strings.TrimSuffix(treeGlob[:boundary], "/")
	suffix := treeGlob[boundary:]
	canonicalDir := ""
	if staticDir != "" {
		var ok bool
		canonicalDir, ok = adjudicateCanonicalLeaseStaticPrefix(root, staticDir)
		if !ok {
			return "", false
		}
	}
	if canonicalDir == "" {
		return suffix, suffix != ""
	}
	return canonicalDir + "/" + suffix, true
}

// adjudicateCanonicalLeaseStaticPrefix projects a possibly not-yet-created
// lease prefix through its nearest existing ancestor. Proposed write targets
// deliberately remain strict: their complete parent must already canonicalize.
// Lstat precedes every ascent so dangling links are indirection, not absence.
func adjudicateCanonicalLeaseStaticPrefix(root, staticPrefix string) (string, bool) {
	canonicalRoot, err := canonicalExistingWorkspacePath(root)
	if err != nil {
		return "", false
	}
	staticPrefix = strings.TrimSpace(staticPrefix)
	if staticPrefix == "" {
		return "", true
	}
	prefixPath := filepath.FromSlash(staticPrefix)
	target := prefixPath
	if !filepath.IsAbs(prefixPath) && filepath.VolumeName(prefixPath) == "" {
		target = filepath.Join(canonicalRoot, prefixPath)
	}
	target = filepath.Clean(target)
	candidate := target
	var missingTail []string
	for {
		_, lstatErr := os.Lstat(candidate)
		if lstatErr == nil {
			canonicalAncestor, canonicalErr := canonicalExistingWorkspacePath(candidate)
			if canonicalErr != nil || !pathWithinWorkspace(canonicalRoot, canonicalAncestor) {
				return "", false
			}
			if len(missingTail) > 0 {
				canonicalInfo, statErr := os.Stat(canonicalAncestor)
				if statErr != nil || !canonicalInfo.IsDir() {
					return "", false
				}
			}
			projected := canonicalAncestor
			for i := len(missingTail) - 1; i >= 0; i-- {
				projected = filepath.Join(projected, missingTail[i])
			}
			if !pathWithinWorkspace(canonicalRoot, projected) {
				return "", false
			}
			rel, relErr := filepath.Rel(canonicalRoot, projected)
			if relErr != nil {
				return "", false
			}
			return filepath.ToSlash(rel), true
		}
		if !os.IsNotExist(lstatErr) {
			return "", false
		}
		parent := filepath.Dir(candidate)
		if parent == candidate {
			return "", false
		}
		missingTail = append(missingTail, filepath.Base(candidate))
		candidate = parent
	}
}

func pathWithinWorkspace(root, candidate string) bool {
	rel, err := filepath.Rel(filepath.Clean(root), filepath.Clean(candidate))
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}
