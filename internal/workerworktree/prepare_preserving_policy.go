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
	add := len(args) == 9 && args[0] == "-c" && args[1] == "gc.worktreePruneExpire=never" && args[2] == "-c" && args[3] == "core.longpaths=true" && args[4] == "worktree" && args[5] == "add" && args[6] == "--detach"
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
func preservingCheckoutPolicy(root, base string, git GitRunner) (string, error) {
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
		return "", fmt.Errorf("checkout policy requires qualified Git 2.45.0")
	}
	config, err := query("config", "--null", "--show-origin", "--list")
	if err != nil {
		return "", err
	}
	records, err := preservingNULRecords(config)
	if err != nil || len(records)%2 != 0 {
		return "", fmt.Errorf("checkout configuration malformed or contains diagnostics")
	}
	filters, submodules, worktreeConfig := false, false, false
	origins := map[string]bool{}
	for i := 0; i < len(records); i += 2 {
		origin, entry := records[i], records[i+1]
		if !strings.HasPrefix(origin, "file:") {
			return "", fmt.Errorf("unqualified checkout configuration origin")
		}
		path := strings.TrimPrefix(origin, "file:")
		if !filepath.IsAbs(path) {
			path = filepath.Join(root, path)
		}
		path = filepath.Clean(path)
		bind(path)
		bind(entry)
		if !origins[path] {
			info, statErr := os.Lstat(path)
			if statErr != nil || !info.Mode().IsRegular() {
				return "", fmt.Errorf("configuration source unavailable or indirect")
			}
			data, readErr := os.ReadFile(path)
			if readErr != nil {
				return "", fmt.Errorf("configuration source unreadable")
			}
			bind(path)
			bind(string(data))
			origins[path] = true
		}
		key, value, valued := strings.Cut(entry, "\n")
		lower := strings.ToLower(key)
		if key == "" || strings.ContainsAny(key, "\r\n\t ") {
			return "", fmt.Errorf("malformed checkout configuration key")
		}
		switch {
		case strings.HasPrefix(lower, "includeif."):
			// The worktree administrative path is allocated by Git add. Do not
			// guess how dormant conditional includes would resolve there.
			return "", fmt.Errorf("conditional new-worktree configuration unqualified")
		case lower == "include.path":
			if !valued || value == "" || strings.Contains(value, "%(prefix)") {
				return "", fmt.Errorf("include identity unqualified")
			}
			included := value
			if strings.HasPrefix(included, "~/") {
				included = filepath.Join(os.Getenv("HOME"), included[2:])
			} else if !filepath.IsAbs(included) {
				included = filepath.Join(filepath.Dir(path), included)
			}
			included = filepath.Clean(included)
			if !filepath.IsAbs(included) {
				return "", fmt.Errorf("include identity unavailable")
			}
			bind(included)
			info, statErr := os.Lstat(included)
			if os.IsNotExist(statErr) {
				bind("absent include")
			} else if statErr != nil || !info.Mode().IsRegular() {
				return "", fmt.Errorf("include source unavailable or indirect")
			} else {
				data, readErr := os.ReadFile(included)
				if readErr != nil {
					return "", fmt.Errorf("include source unreadable")
				}
				bind(string(data))
			}
		case lower == "extensions.worktreeconfig":
			if !valued || !preservingGitFalse(value) && !preservingGitTrue(value) {
				return "", fmt.Errorf("malformed worktree-config boolean")
			}
			worktreeConfig = worktreeConfig || preservingGitTrue(value)
		case lower == "core.fsmonitor":
			if !valued || !preservingGitFalse(value) {
				return "", fmt.Errorf("active or malformed fsmonitor configuration")
			}
		case strings.HasPrefix(lower, "core.sparsecheckout") || lower == "core.attributesfile":
			return "", fmt.Errorf("unqualified checkout configuration: %s", key)
		case strings.HasPrefix(lower, "submodule."):
			if lower != "submodule.active" || !valued {
				return "", fmt.Errorf("unqualified submodule configuration")
			}
			submodules = true
		case strings.HasPrefix(lower, "filter."):
			parts := strings.Split(key, ".")
			if len(parts) != 3 || parts[1] == "" || !valued {
				return "", fmt.Errorf("malformed filter definition")
			}
			switch strings.ToLower(parts[2]) {
			case "clean", "smudge", "process":
			case "required":
				if !preservingGitFalse(value) && !preservingGitTrue(value) {
					return "", fmt.Errorf("malformed required-filter boolean")
				}
			default:
				return "", fmt.Errorf("unqualified filter configuration")
			}
			filters = true
		}
	}
	if worktreeConfig {
		// Existing peers may retain their own safeguards. The context being
		// qualified must have no per-worktree config, and Git 2.45.0 creates a
		// new administrative directory rather than copying peer configuration.
		location, err := query("rev-parse", "--git-path", "config.worktree")
		if err != nil || strings.Count(location, "\n") != 1 || !strings.HasSuffix(location, "\n") || location == "\n" {
			return "", fmt.Errorf("worktree configuration location unavailable")
		}
		location = strings.TrimSuffix(location, "\n")
		if !filepath.IsAbs(location) {
			location = filepath.Join(root, location)
		}
		if _, err := os.Lstat(location); !os.IsNotExist(err) {
			return "", fmt.Errorf("context-specific worktree configuration unqualified")
		}
	}
	tree, err := read("ls-tree", "-r", "-z", "--full-tree", base)
	if err != nil {
		return "", err
	}
	paths, entries, err := preservingPolicyTree(tree)
	if err != nil {
		return "", err
	}
	if filters || submodules {
		for _, path := range paths {
			if entries[path] == "160000" || path == ".gitmodules" {
				return "", fmt.Errorf("dormant-policy proof requires no gitlinks or .gitmodules")
			}
		}
	}
	hook, err := query("rev-parse", "--git-path", "hooks/post-checkout")
	if err != nil || strings.TrimSpace(hook) == "" || strings.Count(hook, "\n") != 1 || !strings.HasSuffix(hook, "\n") {
		return "", fmt.Errorf("checkout hook location malformed or unavailable")
	}
	// Locations may differ between contexts. Configuration origin identities
	// and source bytes are bound; the resolved hook must be absent in each.
	hook = strings.TrimSuffix(hook, "\n")
	if !filepath.IsAbs(hook) {
		rel := filepath.ToSlash(filepath.Clean(hook))
		if rel == ".." || strings.HasPrefix(rel, "../") {
			return "", fmt.Errorf("relative hook path escapes the future checkout context")
		}
		for p := rel; p != "."; p = filepath.ToSlash(filepath.Dir(p)) {
			if _, exists := entries[p]; exists {
				return "", fmt.Errorf("pinned source can materialize a checkout hook or ancestor")
			}
			if p == ".." || filepath.IsAbs(p) {
				break
			}
		}
		hook = filepath.Join(root, hook)
	}
	if _, err := os.Lstat(hook); !os.IsNotExist(err) {
		return "", fmt.Errorf("post-checkout hook exists or cannot be inspected")
	}
	if filters {
		infoPath, err := query("rev-parse", "--git-path", "info/attributes")
		if err != nil || strings.Count(infoPath, "\n") != 1 || !strings.HasSuffix(infoPath, "\n") {
			return "", fmt.Errorf("info attributes location malformed or unavailable")
		}
		infoPath = strings.TrimSuffix(infoPath, "\n")
		if infoPath == "" {
			return "", fmt.Errorf("empty info attributes location")
		}
		if !filepath.IsAbs(infoPath) {
			infoPath = filepath.Join(root, infoPath)
		}
		if _, err := os.Lstat(infoPath); !os.IsNotExist(err) {
			return "", fmt.Errorf("info attributes exist or cannot be inspected")
		}
		global := os.Getenv("XDG_CONFIG_HOME")
		if global == "" {
			global = filepath.Join(os.Getenv("HOME"), ".config")
		}
		if !filepath.IsAbs(global) {
			return "", fmt.Errorf("global attributes identity unavailable")
		}
		if _, err := os.Lstat(filepath.Join(global, "git", "attributes")); !os.IsNotExist(err) {
			return "", fmt.Errorf("global attributes unqualified")
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
				return "", fmt.Errorf("attribute path exceeds qualified argument bound")
			}
			args := append([]string{"check-attr", "--source=" + base, "-z", "filter", "--"}, paths[start:end]...)
			data, err := read(args...)
			if err != nil {
				return "", err
			}
			triples, err := preservingNULRecords(data)
			if err != nil || len(triples) != 3*(end-start) {
				return "", fmt.Errorf("incomplete or malformed pinned attribute proof")
			}
			for i, path := range paths[start:end] {
				if triples[3*i] != path || triples[3*i+1] != "filter" || triples[3*i+2] != "unspecified" {
					return "", fmt.Errorf("active, reordered or unknown pinned filter attribute")
				}
			}
			start = end
		}
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
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
