package adjudicator

import (
	"crypto/sha1"
	"crypto/subtle"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// selfAuthoredUntrackedRemoval recognizes the one deletion for which a confirm
// round-trip adds no witness: one plain rm of a file successfully written by this
// trace after the run began and absent from Git's index. Every uncertainty fails
// closed back to the normal REQUIRE_WITNESS path (#4999).
func (a *Adjudicator) selfAuthoredUntrackedRemoval(c *abi.ToolCall, args map[string]any) bool {
	if c == nil || c.TraceID == "" || a.receiptRoot == "" {
		return false
	}
	switch strings.ToLower(c.Tool) {
	case "bash", "shell", "shell_command":
	default:
		return false
	}
	command, _ := argString(args, "command")
	if command == "" {
		command, _ = argString(args, "cmd")
	}
	target, ok := plainSingleRMTarget(command)
	if !ok {
		return false
	}
	base := a.receiptRoot
	if workdir, _ := argString(args, "workdir"); workdir != "" {
		if filepath.IsAbs(workdir) {
			base = workdir
		} else {
			base = filepath.Join(a.receiptRoot, workdir)
		}
	}
	if !filepath.IsAbs(target) {
		target = filepath.Join(base, target)
	}
	canonical, ok := canonicalLocalReceiptPath(a.receiptRoot, target)
	if !ok {
		return false
	}
	info, err := os.Lstat(canonical)
	if err != nil || !info.Mode().IsRegular() || !a.hasPriorWriteReceipt(c.TraceID, canonical, c.SeqNo) {
		return false
	}
	ignored, err := repositoryIgnores(a.receiptRoot, canonical)
	if err != nil || ignored {
		return false
	}
	tracked, err := gitIndexTracks(a.receiptRoot, canonical)
	return err == nil && !tracked
}

func plainSingleRMTarget(command string) (string, bool) {
	// Nested command sources, pipelines, lists, substitutions, wrappers, and flags
	// are deliberately outside the carve-out. It is exactly `rm [--] literal`.
	if len(rceShellSources(command)) != 1 {
		return "", false
	}
	segments := rceShellSegments(command)
	if len(segments) != 1 || segments[0].sep != 0 {
		return "", false
	}
	argv := segments[0].argv
	i := rceCommandWord(argv)
	if i < 0 || i != 0 || argv[i] != "rm" {
		return "", false
	}
	args := argv[i+1:]
	if len(args) == 2 && args[0] == "--" {
		args = args[1:]
	}
	if len(args) != 1 || strings.HasPrefix(args[0], "-") || strings.ContainsAny(args[0], "*?[$`{") {
		return "", false
	}
	return args[0], true
}

// gitIndexTracks reads a SHA-1 v2/v3 Git index directly. The adjudication hot
// path never launches git; unsupported/corrupt index forms return an error and
// preserve REQUIRE_WITNESS. Staged files therefore remain protected.
func gitIndexTracks(root, target string) (bool, error) {
	indexPath, err := repositoryIndexPath(root)
	if err != nil {
		return false, err
	}
	data, err := os.ReadFile(indexPath)
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	if err != nil || len(data) < 12+sha1.Size || string(data[:4]) != "DIRC" {
		return false, errors.New("unreadable git index")
	}
	payload, checksum := data[:len(data)-sha1.Size], data[len(data)-sha1.Size:]
	sum := sha1.Sum(payload)
	if subtle.ConstantTimeCompare(sum[:], checksum) != 1 {
		return false, errors.New("invalid git index checksum")
	}
	version := binary.BigEndian.Uint32(payload[4:8])
	if version != 2 && version != 3 {
		return false, errors.New("unsupported git index version")
	}
	count := int(binary.BigEndian.Uint32(payload[8:12]))
	rel, err := filepath.Rel(root, target)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
		return false, errors.New("target outside repository")
	}
	want := filepath.ToSlash(rel)
	off := 12
	tracked := false
	for n := 0; n < count; n++ {
		entryStart := off
		if off+62 > len(payload) {
			return false, errors.New("truncated git index")
		}
		flags := binary.BigEndian.Uint16(payload[off+60 : off+62])
		off += 62
		if flags&0x4000 != 0 {
			if version != 3 || off+2 > len(payload) {
				return false, errors.New("invalid extended index entry")
			}
			off += 2
		}
		end := off
		for end < len(payload) && payload[end] != 0 {
			end++
		}
		if end == len(payload) {
			return false, errors.New("unterminated git index path")
		}
		name := string(payload[off:end])
		if name == want || filepath.Separator == '\\' && strings.EqualFold(name, want) {
			tracked = true
		}
		off = entryStart + ((end-entryStart+1+7)/8)*8
		if off > len(payload) {
			return false, errors.New("truncated git index padding")
		}
	}
	// Parse extensions so split/sparse indexes cannot make a base-index entry
	// look untracked. Unknown mandatory (lowercase-leading) extensions fail closed.
	for off < len(payload) {
		if off+8 > len(payload) {
			return false, errors.New("truncated git index extension")
		}
		sig := string(payload[off : off+4])
		sz := int(binary.BigEndian.Uint32(payload[off+4 : off+8]))
		if sz < 0 || off+8+sz > len(payload) {
			return false, errors.New("invalid git index extension")
		}
		if sig == "link" || sig == "sdir" || sig[0] >= 'a' && sig[0] <= 'z' {
			return false, errors.New("unsupported git index extension")
		}
		off += 8 + sz
	}
	return tracked, nil
}

// repositoryIgnores recognizes repository-local excludes without launching Git.
// Unsupported patterns/configuration fail closed to REQUIRE_WITNESS.
func repositoryIgnores(root, target string) (bool, error) {
	rel, err := filepath.Rel(root, target)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
		return false, errors.New("target outside repository")
	}
	unsupported, err := configuredExternalExcludes(root)
	if err != nil {
		return false, err
	}
	if unsupported {
		return false, errors.New("external excludes configuration unsupported")
	}
	type source struct{ path, base string }
	sources := []source{{filepath.Join(root, ".git", "info", "exclude"), ""}, {filepath.Join(root, ".gitignore"), ""}}
	dir := filepath.Dir(rel)
	if dir != "." {
		parts := strings.Split(filepath.ToSlash(dir), "/")
		for i := range parts {
			base := filepath.Join(parts[:i+1]...)
			sources = append(sources, source{filepath.Join(root, base, ".gitignore"), filepath.ToSlash(base)})
		}
	}
	ignored := false
	for _, src := range sources {
		data, readErr := os.ReadFile(src.path)
		if errors.Is(readErr, os.ErrNotExist) {
			continue
		}
		if readErr != nil {
			return false, readErr
		}
		for _, line := range strings.Split(strings.ReplaceAll(string(data), "\r\n", "\n"), "\n") {
			match, negate, parseErr := gitIgnorePatternMatch(line, src.base, filepath.ToSlash(rel))
			if parseErr != nil {
				return false, parseErr
			}
			if match {
				ignored = !negate
			}
		}
	}
	return ignored, nil
}

func gitIgnorePatternMatch(raw, base, rel string) (bool, bool, error) {
	if raw == "" || strings.HasPrefix(raw, "#") {
		return false, false, nil
	}
	if strings.Contains(raw, "\\") || strings.HasSuffix(raw, " ") || strings.Contains(raw, "**") {
		return false, false, errors.New("unsupported gitignore pattern")
	}
	negate := strings.HasPrefix(raw, "!")
	if negate {
		raw = strings.TrimPrefix(raw, "!")
	}
	dirOnly := strings.HasSuffix(raw, "/")
	raw = strings.TrimSuffix(strings.TrimPrefix(raw, "/"), "/")
	if raw == "" {
		return false, false, errors.New("invalid gitignore pattern")
	}
	candidate := rel
	if base != "" {
		prefix := strings.TrimSuffix(base, "/") + "/"
		if !strings.HasPrefix(candidate, prefix) {
			return false, negate, nil
		}
		candidate = strings.TrimPrefix(candidate, prefix)
	}
	if !strings.Contains(raw, "/") {
		for _, part := range strings.Split(candidate, "/") {
			matched, matchErr := filepath.Match(raw, part)
			if matchErr != nil {
				return false, false, matchErr
			}
			if matched && (!dirOnly || part != filepath.Base(candidate)) {
				return true, negate, nil
			}
		}
		return false, negate, nil
	}
	matched, matchErr := filepath.Match(raw, candidate)
	if matchErr != nil {
		return false, false, matchErr
	}
	if dirOnly && strings.HasPrefix(candidate, raw+"/") {
		matched = true
	}
	return matched, negate, nil
}

func configuredExternalExcludes(root string) (bool, error) {
	type configSource struct {
		path     string
		required bool
	}
	sources := []configSource{{path: filepath.Join(root, ".git", "config")}}
	if home, err := os.UserHomeDir(); err == nil {
		sources = append(sources, configSource{path: filepath.Join(home, ".gitconfig")})
	}
	if xdg := os.Getenv("XDG_CONFIG_HOME"); xdg != "" {
		sources = append(sources, configSource{path: filepath.Join(xdg, "git", "config")})
	}
	seen := make(map[string]bool)
	for len(sources) > 0 {
		source := sources[0]
		sources = sources[1:]
		path := filepath.Clean(source.path)
		if seen[path] {
			continue
		}
		seen[path] = true
		data, err := os.ReadFile(path)
		if errors.Is(err, os.ErrNotExist) && !source.required {
			continue
		}
		if err != nil {
			return false, err
		}
		unsupported, includes, err := inspectGitConfig(path, data)
		if err != nil || unsupported {
			return unsupported, err
		}
		for _, include := range includes {
			sources = append(sources, configSource{path: include, required: true})
		}
	}
	return false, nil
}

func inspectGitConfig(path string, data []byte) (bool, []string, error) {
	section := ""
	var includes []string
	for _, raw := range strings.Split(string(data), "\n") {
		line := strings.TrimSpace(strings.TrimSuffix(raw, "\r"))
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, ";") {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			section = strings.ToLower(strings.TrimSpace(line[1 : len(line)-1]))
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.ToLower(strings.TrimSpace(key))
		value = strings.Trim(strings.TrimSpace(value), `"`)
		if key == "excludesfile" {
			return true, nil, nil
		}
		if section == "include" && key == "path" {
			include := expandGitConfigPath(filepath.Dir(path), value)
			if include == "" {
				return false, nil, errors.New("invalid git config include path")
			}
			includes = append(includes, include)
		}
		if strings.HasPrefix(section, "includeif ") {
			return false, nil, errors.New("conditional git config includes unsupported")
		}
	}
	return false, includes, nil
}

func expandGitConfigPath(base, value string) string {
	if value == "" {
		return ""
	}
	if value == "~" || strings.HasPrefix(value, "~/") || strings.HasPrefix(value, `~\`) {
		home, err := os.UserHomeDir()
		if err != nil {
			return ""
		}
		value = filepath.Join(home, strings.TrimLeft(value[1:], `/\`))
	} else if !filepath.IsAbs(value) {
		value = filepath.Join(base, value)
	}
	return filepath.Clean(value)
}

func repositoryIndexPath(root string) (string, error) {
	gitPath := filepath.Join(root, ".git")
	info, err := os.Stat(gitPath)
	if err != nil {
		return "", err
	}
	if info.IsDir() {
		return filepath.Join(gitPath, "index"), nil
	}
	data, err := os.ReadFile(gitPath)
	if err != nil {
		return "", err
	}
	line := strings.TrimSpace(string(data))
	if !strings.HasPrefix(line, "gitdir:") {
		return "", errors.New("invalid .git indirection")
	}
	dir := strings.TrimSpace(strings.TrimPrefix(line, "gitdir:"))
	if !filepath.IsAbs(dir) {
		dir = filepath.Join(root, dir)
	}
	return filepath.Join(filepath.Clean(dir), "index"), nil
}
