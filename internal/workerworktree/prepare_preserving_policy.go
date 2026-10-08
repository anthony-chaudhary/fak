package workerworktree

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// Keep success diagnostics visible to admission. The shared bounded runner
// intentionally returns stdout alone on success, which cannot prove no warnings.
func boundedPreservingGitRunner(ctx context.Context) GitRunner {
	return func(root string, args []string) (int, string) {
		if err := ctx.Err(); err != nil {
			return ReapTimeoutExitCode, err.Error()
		}
		cmd := exec.CommandContext(ctx, "git", args...)
		cmd.Dir = root
		cmd.WaitDelay = 100 * time.Millisecond
		windowgate.ConfigureBackgroundCommand(cmd)
		var stdout, stderr bytes.Buffer
		cmd.Stdout, cmd.Stderr = &stdout, &stderr
		err := cmd.Run()
		if ctx.Err() != nil {
			return ReapTimeoutExitCode, ctx.Err().Error()
		}
		code := 0
		if err != nil {
			code = 127
			if exit, ok := err.(*exec.ExitError); ok {
				code = exit.ExitCode()
			}
			if stderr.Len() == 0 {
				stderr.WriteString(err.Error())
			}
		}
		return preservingGitOutput(args, code, stdout.String(), stderr.String())
	}
}

func preservingGitOutput(args []string, code int, stdout, stderr string) (int, string) {
	if code != 0 {
		return code, stdout + stderr
	}
	// Add's ordinary progress is written to stderr. Read queries may not hide it.
	add := len(args) == 10 && args[0] == "-c" && args[1] == "gc.worktreePruneExpire=never" && args[2] == "-c" && args[3] == "core.longpaths=true" && args[4] == "worktree" && args[5] == "add" && args[6] == "--quiet" && args[7] == "--detach"
	progress := false
	if add && strings.HasPrefix(stderr, "Preparing worktree (detached HEAD ") && strings.HasSuffix(stderr, ")\n") && strings.Count(stderr, "\n") == 1 {
		identity := strings.TrimSuffix(strings.TrimPrefix(stderr, "Preparing worktree (detached HEAD "), ")\n")
		if len(identity) >= 4 && len(identity) <= 40 {
			_, err := hex.DecodeString(identity + strings.Repeat("0", len(identity)%2))
			progress = err == nil
		}
	}
	if stderr != "" && !progress {
		return 128, stdout + stderr
	}
	return 0, stdout
}

// The qualification is deliberately limited to the reviewed Git release. It
// reads policy without overriding it or running a filter, hook or status helper.
// The digest must remain equal before add and in the new checkout before status.
type preservingCheckoutProof struct {
	digest                string
	root                  string
	worktreeConfigPath    string
	worktreeConfigData    string
	worktreeConfigPresent bool
}

func preservingCheckoutPolicy(root, base string, git GitRunner) (string, error) {
	proof, err := preservingCheckoutPolicyProof(root, root, base, git, nil)
	return proof.digest, err
}

// admitted permits only Git's native copy of the directly identified source
// config.worktree into the actual child context. All other origins stay bound.
func preservingCheckoutPolicyProof(root, target, base string, git GitRunner, admitted *preservingCheckoutProof) (preservingCheckoutProof, error) {
	proof := preservingCheckoutProof{root: root}
	hash := sha256.New()
	bind := func(data string) { fmt.Fprintf(hash, "%d:", len(data)); hash.Write([]byte(data)) }
	query := func(args ...string) (string, error) {
		rc, data := run(git, root, args)
		if rc != 0 || !utf8.ValidString(data) {
			return "", fmt.Errorf("checkout policy query unavailable or malformed: %s", args[0])
		}
		return data, nil
	}
	read := func(args ...string) (string, error) {
		data, err := query(args...)
		if err == nil {
			bind(data)
		}
		return data, err
	}
	version, err := read("--version")
	if err != nil || version != "git version 2.45.0\n" {
		return preservingCheckoutProof{}, fmt.Errorf("checkout policy requires qualified Git 2.45.0")
	}
	config, err := query("config", "--null", "--show-origin", "--list")
	if err != nil {
		return preservingCheckoutProof{}, err
	}
	records, err := preservingNULRecords(config)
	if err != nil || len(records)%2 != 0 {
		return preservingCheckoutProof{}, fmt.Errorf("checkout configuration malformed or contains diagnostics")
	}
	worktreeConfig := false
	for i := 1; i < len(records); i += 2 {
		key, value, valued := strings.Cut(records[i], "\n")
		if strings.EqualFold(key, "extensions.worktreeconfig") {
			if !valued || !preservingGitFalse(value) && !preservingGitTrue(value) {
				return preservingCheckoutProof{}, fmt.Errorf("malformed worktree-config boolean")
			}
			// Git uses the last value. Do not grant relocation for a disabled
			// extension merely because an earlier origin enabled it.
			worktreeConfig = preservingGitTrue(value)
		}
	}
	var worktreeEntries []string
	if worktreeConfig {
		location, err := query("rev-parse", "--git-path", "config.worktree")
		if err != nil {
			return preservingCheckoutProof{}, err
		}
		proof.worktreeConfigPath, err = preservingPolicyLocation(root, location)
		if err != nil {
			return preservingCheckoutProof{}, fmt.Errorf("worktree configuration location unavailable: %w", err)
		}
		if _, err := os.Lstat(proof.worktreeConfigPath); !os.IsNotExist(err) {
			proof.worktreeConfigData, err = preservingReadDirectConfig(proof.worktreeConfigPath)
			if err != nil {
				return preservingCheckoutProof{}, err
			}
			proof.worktreeConfigPresent = true
			data, err := query("config", "--null", "--no-includes", "--file", proof.worktreeConfigPath, "--list")
			if err != nil {
				return preservingCheckoutProof{}, err
			}
			worktreeEntries, err = preservingWorktreeConfigEntries(data)
			if err != nil {
				return preservingCheckoutProof{}, err
			}
		}
	}
	if admitted != nil {
		if root == admitted.root || admitted.digest == "" || proof.worktreeConfigPresent != admitted.worktreeConfigPresent {
			return preservingCheckoutProof{}, fmt.Errorf("new-worktree configuration copy provenance unavailable")
		}
		if proof.worktreeConfigPresent && (proof.worktreeConfigPath == admitted.worktreeConfigPath || proof.worktreeConfigData != admitted.worktreeConfigData) {
			return preservingCheckoutProof{}, fmt.Errorf("new-worktree configuration is not an identical native copy")
		}
		if proof.worktreeConfigPresent {
			original, err := preservingReadDirectConfig(admitted.worktreeConfigPath)
			sourceInfo, sourceErr := os.Lstat(admitted.worktreeConfigPath)
			copyInfo, copyErr := os.Lstat(proof.worktreeConfigPath)
			if err != nil || original != admitted.worktreeConfigData || sourceErr != nil || copyErr != nil || os.SameFile(sourceInfo, copyInfo) {
				return preservingCheckoutProof{}, fmt.Errorf("worktree configuration source drifted or copy aliases source")
			}
		}
	}
	filters, submodules, hooksPathSet := false, false, false
	var effectiveWorktreeEntries []string
	origins := map[string]bool{}
	for i := 0; i < len(records); i += 2 {
		origin, entry := records[i], records[i+1]
		if !strings.HasPrefix(origin, "file:") {
			return preservingCheckoutProof{}, fmt.Errorf("unqualified checkout configuration origin")
		}
		path := strings.TrimPrefix(origin, "file:")
		if !filepath.IsAbs(path) {
			path = filepath.Join(root, path)
		}
		path = filepath.Clean(path)
		boundPath := path
		if proof.worktreeConfigPresent && path == proof.worktreeConfigPath {
			effectiveWorktreeEntries = append(effectiveWorktreeEntries, entry)
			if admitted != nil {
				boundPath = admitted.worktreeConfigPath
			}
		}
		bind(boundPath)
		bind(entry)
		if !origins[path] {
			info, statErr := os.Lstat(path)
			if statErr != nil || !info.Mode().IsRegular() {
				return preservingCheckoutProof{}, fmt.Errorf("configuration source unavailable or indirect")
			}
			data, readErr := os.ReadFile(path)
			if readErr != nil {
				return preservingCheckoutProof{}, fmt.Errorf("configuration source unreadable")
			}
			if path == proof.worktreeConfigPath && string(data) != proof.worktreeConfigData {
				return preservingCheckoutProof{}, fmt.Errorf("worktree configuration changed during qualification")
			}
			bind(boundPath)
			bind(string(data))
			origins[path] = true
		}
		key, value, valued := strings.Cut(entry, "\n")
		lower := strings.ToLower(key)
		if key == "" || strings.ContainsAny(key, "\r\n\t ") {
			return preservingCheckoutProof{}, fmt.Errorf("malformed checkout configuration key")
		}
		switch {
		case strings.HasPrefix(lower, "includeif."):
			// The worktree administrative path is allocated by Git add. Do not
			// guess how dormant conditional includes would resolve there.
			return preservingCheckoutProof{}, fmt.Errorf("conditional new-worktree configuration unqualified")
		case lower == "include.path":
			if !valued || value == "" || strings.Contains(value, "%(prefix)") {
				return preservingCheckoutProof{}, fmt.Errorf("include identity unqualified")
			}
			included := value
			if strings.HasPrefix(included, "~/") {
				included = filepath.Join(os.Getenv("HOME"), included[2:])
			} else if !filepath.IsAbs(included) {
				included = filepath.Join(filepath.Dir(path), included)
			}
			included = filepath.Clean(included)
			if !filepath.IsAbs(included) {
				return preservingCheckoutProof{}, fmt.Errorf("include identity unavailable")
			}
			bind(included)
			info, statErr := os.Lstat(included)
			if os.IsNotExist(statErr) {
				bind("absent include")
			} else if statErr != nil || !info.Mode().IsRegular() {
				return preservingCheckoutProof{}, fmt.Errorf("include source unavailable or indirect")
			} else {
				data, readErr := os.ReadFile(included)
				if readErr != nil {
					return preservingCheckoutProof{}, fmt.Errorf("include source unreadable")
				}
				bind(string(data))
			}
		case lower == "extensions.worktreeconfig":
			if !valued || !preservingGitFalse(value) && !preservingGitTrue(value) {
				return preservingCheckoutProof{}, fmt.Errorf("malformed worktree-config boolean")
			}
		case lower == "core.fsmonitor":
			if !valued || !preservingGitFalse(value) {
				return preservingCheckoutProof{}, fmt.Errorf("active or malformed fsmonitor configuration")
			}
		case lower == "core.hookspath":
			if !valued || value == "" || strings.ContainsAny(value, "\x00\r\n") || !preservingASCIIPath(value) {
				return preservingCheckoutProof{}, fmt.Errorf("hooks path unavailable, malformed or non-ASCII")
			}
			hooksPathSet = true
		case strings.HasPrefix(lower, "core.sparsecheckout") || lower == "core.attributesfile":
			return preservingCheckoutProof{}, fmt.Errorf("unqualified checkout configuration: %s", key)
		case strings.HasPrefix(lower, "submodule."):
			if lower != "submodule.active" || !valued {
				return preservingCheckoutProof{}, fmt.Errorf("unqualified submodule configuration")
			}
			submodules = true
		case strings.HasPrefix(lower, "filter."):
			parts := strings.Split(key, ".")
			if len(parts) != 3 || parts[1] == "" || !valued {
				return preservingCheckoutProof{}, fmt.Errorf("malformed filter definition")
			}
			switch strings.ToLower(parts[2]) {
			case "clean", "smudge", "process":
			case "required":
				if !preservingGitFalse(value) && !preservingGitTrue(value) {
					return preservingCheckoutProof{}, fmt.Errorf("malformed required-filter boolean")
				}
			default:
				return preservingCheckoutProof{}, fmt.Errorf("unqualified filter configuration")
			}
			filters = true
		}
	}
	if strings.Join(effectiveWorktreeEntries, "\x00") != strings.Join(worktreeEntries, "\x00") {
		return preservingCheckoutProof{}, fmt.Errorf("worktree configuration origin or effective entries differ from direct source")
	}
	tree, err := read("ls-tree", "-r", "-z", "--full-tree", base)
	if err != nil {
		return preservingCheckoutProof{}, err
	}
	paths, entries, err := preservingPolicyTree(tree)
	if err != nil {
		return preservingCheckoutProof{}, err
	}
	if filters || submodules {
		for _, path := range paths {
			if entries[path] == "160000" || path == ".gitmodules" {
				return preservingCheckoutProof{}, fmt.Errorf("dormant-policy proof requires no gitlinks or .gitmodules")
			}
		}
	}
	// Add initializes refs in the parent context, then reset updates the child
	// refs/index. Its final post-checkout lookup still uses parent policy.
	// Each context must prove every reachable helper absent, without disabling
	// any hook or relying on the post-add check to stop pre-status execution.
	for _, name := range []string{"reference-transaction", "post-index-change", "post-checkout"} {
		location, err := query("rev-parse", "--git-path", "hooks/"+name)
		if err != nil {
			return preservingCheckoutProof{}, err
		}
		hook, err := preservingPolicyLocation(root, location)
		if err != nil {
			return preservingCheckoutProof{}, fmt.Errorf("%s hook location unavailable: %w", name, err)
		}
		// Case folding below does not prove Unicode-normalization aliases on
		// every qualified host filesystem. Keep configured path identities in
		// the ASCII envelope rather than guessing their future equivalence.
		nativeHook := strings.TrimSuffix(location, "\n")
		if hooksPathSet && (!preservingASCIIPath(nativeHook) || filepath.IsAbs(nativeHook) && !preservingASCIIPath(target)) {
			return preservingCheckoutProof{}, fmt.Errorf("non-ASCII creation hook or checkout identity unqualified")
		}
		if hooksPathSet && !filepath.IsAbs(nativeHook) {
			rel := filepath.Clean(nativeHook)
			if err := preservingPinnedHookAbsent(rel, entries); err != nil {
				return preservingCheckoutProof{}, err
			}
			// The copied relative policy also applies from the new checkout.
			if err := preservingHookAbsent(filepath.Join(target, rel)); err != nil {
				return preservingCheckoutProof{}, err
			}
		} else if rel, inside := preservingPathWithin(target, hook); hooksPathSet && inside {
			// Absolute hooksPath may name a directory in the as-yet absent
			// target. Its hook or ancestor must not appear during checkout.
			if err := preservingPinnedHookAbsent(rel, entries); err != nil {
				return preservingCheckoutProof{}, err
			}
		}
		if err := preservingHookAbsent(hook); err != nil {
			return preservingCheckoutProof{}, err
		}
	}
	if filters {
		infoPath, err := query("rev-parse", "--git-path", "info/attributes")
		if err != nil || strings.Count(infoPath, "\n") != 1 || !strings.HasSuffix(infoPath, "\n") {
			return preservingCheckoutProof{}, fmt.Errorf("info attributes location malformed or unavailable")
		}
		infoPath = strings.TrimSuffix(infoPath, "\n")
		if infoPath == "" {
			return preservingCheckoutProof{}, fmt.Errorf("empty info attributes location")
		}
		if !filepath.IsAbs(infoPath) {
			infoPath = filepath.Join(root, infoPath)
		}
		if _, err := os.Lstat(infoPath); !os.IsNotExist(err) {
			return preservingCheckoutProof{}, fmt.Errorf("info attributes exist or cannot be inspected")
		}
		global := os.Getenv("XDG_CONFIG_HOME")
		if global == "" {
			global = filepath.Join(os.Getenv("HOME"), ".config")
		}
		if !filepath.IsAbs(global) {
			return preservingCheckoutProof{}, fmt.Errorf("global attributes identity unavailable")
		}
		if _, err := os.Lstat(filepath.Join(global, "git", "attributes")); !os.IsNotExist(err) {
			return preservingCheckoutProof{}, fmt.Errorf("global attributes unqualified")
		}
		// Every pinned path gets exactly one filter triple. Batches avoid argv
		// limits; no sampling, working-tree attributes or truncated results.
		for start := 0; start < len(paths); {
			end, bytes := start, 0
			for end < len(paths) && end-start < 128 && bytes+len(paths[end])+1 < 16384 {
				bytes += len(paths[end]) + 1
				end++
			}
			if end == start {
				return preservingCheckoutProof{}, fmt.Errorf("attribute path exceeds qualified argument bound")
			}
			args := append([]string{"check-attr", "--source=" + base, "-z", "filter", "--"}, paths[start:end]...)
			data, err := read(args...)
			if err != nil {
				return preservingCheckoutProof{}, err
			}
			triples, err := preservingNULRecords(data)
			if err != nil || len(triples) != 3*(end-start) {
				return preservingCheckoutProof{}, fmt.Errorf("incomplete or malformed pinned attribute proof")
			}
			for i, path := range paths[start:end] {
				if triples[3*i] != path || triples[3*i+1] != "filter" || triples[3*i+2] != "unspecified" {
					return preservingCheckoutProof{}, fmt.Errorf("active, reordered or unknown pinned filter attribute")
				}
			}
			start = end
		}
	}
	if proof.worktreeConfigPath != "" {
		if proof.worktreeConfigPresent {
			data, err := preservingReadDirectConfig(proof.worktreeConfigPath)
			if err != nil || data != proof.worktreeConfigData {
				return preservingCheckoutProof{}, fmt.Errorf("worktree configuration changed during qualification")
			}
		} else if _, err := os.Lstat(proof.worktreeConfigPath); !os.IsNotExist(err) {
			return preservingCheckoutProof{}, fmt.Errorf("worktree configuration appeared during qualification")
		}
	}
	proof.digest = hex.EncodeToString(hash.Sum(nil))
	return proof, nil
}

func preservingPolicyLocation(root, data string) (string, error) {
	if strings.Count(data, "\n") != 1 || !strings.HasSuffix(data, "\n") || strings.ContainsAny(data, "\x00\r") {
		return "", fmt.Errorf("malformed Git path")
	}
	path := strings.TrimSuffix(data, "\n")
	if path == "" {
		return "", fmt.Errorf("empty Git path")
	}
	if !filepath.IsAbs(path) {
		path = filepath.Join(root, path)
	}
	if !filepath.IsAbs(path) {
		return "", fmt.Errorf("absolute Git path unavailable")
	}
	return filepath.Clean(path), nil
}

func preservingReadDirectConfig(path string) (string, error) {
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() {
		return "", fmt.Errorf("worktree configuration source unavailable or indirect")
	}
	physical, err := filepath.EvalSymlinks(path)
	if err != nil || physical != path {
		return "", fmt.Errorf("worktree configuration identity indirect")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("worktree configuration source unreadable")
	}
	after, err := os.Lstat(path)
	if err != nil || !after.Mode().IsRegular() || !os.SameFile(info, after) {
		return "", fmt.Errorf("worktree configuration source replaced during read")
	}
	return string(data), nil
}

func preservingWorktreeConfigEntries(data string) ([]string, error) {
	entries, err := preservingNULRecords(data)
	if err != nil || len(entries) != 2 {
		return nil, fmt.Errorf("worktree configuration requires exactly core.bare=false and core.hooksPath")
	}
	seen := map[string]bool{}
	for _, entry := range entries {
		key, value, valued := strings.Cut(entry, "\n")
		key = strings.ToLower(key)
		if !valued || seen[key] {
			return nil, fmt.Errorf("duplicate or malformed worktree configuration")
		}
		seen[key] = true
		switch key {
		case "core.bare":
			if value != "false" {
				return nil, fmt.Errorf("worktree configuration requires literal core.bare=false")
			}
		case "core.hookspath":
			if value == "" || strings.ContainsAny(value, "\x00\r\n") {
				return nil, fmt.Errorf("worktree hooks path unavailable or malformed")
			}
		default:
			return nil, fmt.Errorf("unqualified worktree configuration: %s", key)
		}
	}
	return entries, nil
}

func preservingASCIIPath(path string) bool {
	for i := 0; i < len(path); i++ {
		if path[i] >= utf8.RuneSelf {
			return false
		}
	}
	return true
}

func preservingPinnedHookAbsent(rel string, entries map[string]string) error {
	rel = filepath.ToSlash(filepath.Clean(rel))
	if rel == "." || rel == ".." || strings.HasPrefix(rel, "../") || filepath.IsAbs(rel) {
		return fmt.Errorf("relative hook path escapes the future checkout context")
	}
	for p := rel; p != "."; p = filepath.ToSlash(filepath.Dir(p)) {
		for entry := range entries {
			// Conservative across case-sensitive and case-insensitive hosts.
			if strings.EqualFold(entry, p) {
				return fmt.Errorf("pinned source can materialize a creation hook or ancestor")
			}
		}
	}
	return nil
}

func preservingPathWithin(root, path string) (string, bool) {
	root, path = filepath.ToSlash(filepath.Clean(root)), filepath.ToSlash(filepath.Clean(path))
	if strings.EqualFold(root, path) {
		return ".", true
	}
	prefix := strings.TrimSuffix(root, "/") + "/"
	if len(path) > len(prefix) && strings.EqualFold(path[:len(prefix)], prefix) {
		return filepath.FromSlash(path[len(prefix):]), true
	}
	return "", false
}

func preservingHookAbsent(path string) error {
	// Lstat of an absent leaf alone follows symlink ancestors. Inspect the
	// entire path so a dangling or redirected hooks directory fails closed.
	for p := filepath.Clean(path); ; p = filepath.Dir(p) {
		info, err := os.Lstat(p)
		if !os.IsNotExist(err) {
			if err != nil || p == filepath.Clean(path) || !info.IsDir() {
				return fmt.Errorf("creation hook exists, is indirect, or cannot be inspected")
			}
		}
		if filepath.Dir(p) == p {
			break
		}
	}
	return nil
}

func preservingGitFalse(value string) bool {
	switch strings.ToLower(value) {
	case "false", "no", "off", "0":
		return true
	}
	return false
}

func preservingGitTrue(value string) bool {
	switch strings.ToLower(value) {
	case "true", "yes", "on", "1":
		return true
	}
	return false
}

func preservingNULRecords(data string) ([]string, error) {
	if data == "" {
		return nil, nil
	}
	if !strings.HasSuffix(data, "\x00") {
		return nil, fmt.Errorf("unterminated Git records")
	}
	return strings.Split(strings.TrimSuffix(data, "\x00"), "\x00"), nil
}

func preservingPolicyTree(data string) ([]string, map[string]string, error) {
	records, err := preservingNULRecords(data)
	if err != nil || len(records) == 0 {
		return nil, nil, fmt.Errorf("pinned tree unavailable or malformed")
	}
	var paths []string
	entries := map[string]string{}
	for _, record := range records {
		meta, path, found := strings.Cut(record, "\t")
		fields := strings.Fields(meta)
		if !found || len(fields) != 3 || path == "" || entries[path] != "" || filepath.ToSlash(filepath.Clean(path)) != path || filepath.IsAbs(path) || strings.HasPrefix(path, "../") {
			return nil, nil, fmt.Errorf("malformed or duplicate pinned tree entry")
		}
		if fields[0] != "100644" && fields[0] != "100755" && fields[0] != "120000" && fields[0] != "160000" || fields[1] != "blob" && fields[1] != "commit" || (fields[0] == "160000") != (fields[1] == "commit") {
			return nil, nil, fmt.Errorf("unknown pinned tree object")
		}
		if len(fields[2]) != 40 {
			return nil, nil, fmt.Errorf("unknown pinned object identity")
		}
		if _, err := hex.DecodeString(fields[2]); err != nil {
			return nil, nil, fmt.Errorf("malformed pinned object identity")
		}
		paths = append(paths, path)
		entries[path] = fields[0]
	}
	return paths, entries, nil
}
