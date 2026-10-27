// Package mcpinstall installs a source-selected stdio MCP executable into Codex.
// It does not execute the executable, reload a host, authenticate an MCP peer, or
// change approval policy. Callers must review the complete Spec before Install.
package mcpinstall

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
	"strconv"
	"strings"
	"unicode/utf8"
)

// Spec is a reviewed launch specification, supplied by trusted Go source, not
// model-selected command fields. Paths must be absolute and free of symlinks.
// ConfigPath must name an existing config; ExpectedConfigSHA256 binds all its
// bytes, including other servers, policy and approval settings.
type Spec struct {
	Server, SourcePath, SourceSHA256, InstallDir string
	ConfigPath, ExpectedConfigSHA256, WorkingDir string
	Args                                         []string
}

// Receipt describes exact launch bytes and the durable rollback record. It says
// nothing about server execution, MCP initialization, or peer authentication.
// Retain the whole receipt, especially ReceiptSHA256, outside the install folder.
type Receipt struct {
	Server            string   `json:"server"`
	Command           string   `json:"command"`
	Args              []string `json:"args"`
	WorkingDir        string   `json:"working_dir"`
	ConfigPath        string   `json:"config_path"`
	PriorConfigSHA256 string   `json:"prior_config_sha256"`
	ConfigSHA256      string   `json:"config_sha256"`
	ReceiptPath       string   `json:"receipt_path"`
	ReceiptSHA256     string   `json:"receipt_sha256"`
	ConfigIdentity    string   `json:"config_identity"`
	Guard             string   `json:"guard"`
	UpdateMethod      string   `json:"update_method"`
	Durability        string   `json:"durability"`
}

// Plan captures validated source and config bytes. Prepare and Preview are
// read-only; changing the caller's Spec or a returned Preview cannot alter Plan.
type Plan struct {
	spec          Spec
	source, prior []byte
	next          []byte
	receipt       Receipt
}

type rollbackRecord struct {
	Schema       string  `json:"schema"`
	Receipt      Receipt `json:"receipt"`
	PrefixLength int64   `json:"prefix_length"`
	OwnedSuffix  []byte  `json:"owned_suffix"`
}

// Prepare validates a closed server namespace, both expected hashes and the
// exact cwd/argv. Existing namespaces are never replaced. Unsupported ambiguous
// MCP TOML declarations fail closed; unrelated config bytes are never rewritten.
func Prepare(spec Spec) (*Plan, error) {
	if !identifier(spec.Server) || !digest(spec.SourceSHA256) || !digest(spec.ExpectedConfigSHA256) {
		return nil, errors.New("mcpinstall: invalid server or expected SHA256")
	}
	for _, path := range []string{spec.SourcePath, spec.InstallDir, spec.ConfigPath, spec.WorkingDir} {
		if err := plainPath(path); err != nil {
			return nil, err
		}
	}
	if err := workingDirectory(spec.WorkingDir); err != nil {
		return nil, err
	}
	if info, err := os.Stat(spec.InstallDir); err != nil || !info.IsDir() {
		return nil, errors.New("mcpinstall: install directory must already exist")
	}
	if len(spec.Args) > 64 {
		return nil, errors.New("mcpinstall: too many arguments")
	}
	for _, arg := range spec.Args {
		if len(arg) > 8192 || !safeText(arg) {
			return nil, errors.New("mcpinstall: invalid argument")
		}
	}
	source, _, err := readRegular(spec.SourcePath, 256<<20)
	if err != nil {
		return nil, err
	}
	if len(source) == 0 || hash(source) != spec.SourceSHA256 {
		return nil, errors.New("mcpinstall: source hash mismatch")
	}
	prior, _, err := readRegular(spec.ConfigPath, 4<<20)
	if err != nil {
		return nil, err
	}
	if hash(prior) != spec.ExpectedConfigSHA256 {
		return nil, errors.New("mcpinstall: config changed")
	}
	if err := absentNamespace(prior, spec.Server); err != nil {
		return nil, err
	}
	f, err := os.Open(spec.ConfigPath)
	if err != nil {
		return nil, err
	}
	identity, err := configIdentity(f)
	_ = f.Close()
	if err != nil {
		return nil, err
	}
	spec.Args = append([]string{}, spec.Args...)
	suffix := ""
	if strings.EqualFold(filepath.Ext(spec.SourcePath), ".exe") {
		suffix = ".exe"
	}
	command := filepath.Join(spec.InstallDir, spec.Server+"-"+spec.SourceSHA256+suffix)
	if command == spec.SourcePath || command == spec.ConfigPath {
		return nil, errors.New("mcpinstall: overlapping install paths")
	}
	next := appendSection(prior, spec.Server, command, spec.Args, spec.WorkingDir)
	r := Receipt{Server: spec.Server, Command: command, Args: spec.Args, WorkingDir: spec.WorkingDir,
		ConfigPath: spec.ConfigPath, PriorConfigSHA256: hash(prior), ConfigSHA256: hash(next), ConfigIdentity: identity,
		Guard: platformGuard, UpdateMethod: "append-owned-suffix;rollback-truncate-owned-suffix", Durability: platformDurability}
	r.ReceiptPath = filepath.Join(spec.InstallDir, spec.Server+"-"+r.ConfigSHA256+".rollback.json")
	return &Plan{spec: spec, source: source, prior: prior, next: next, receipt: r}, nil
}

// Preview returns the exact intended launch and config hashes. ReceiptSHA256 is
// empty until Install persists the rollback record.
func (p *Plan) Preview() Receipt {
	r := p.receipt
	r.Args = append([]string{}, r.Args...)
	return r
}

// Install is supported only where a native config handle excludes ordinary
// writers and replacers throughout verification, append and flush. Unix fails
// closed. The existing config prefix is never rewritten or replaced. A synced
// rollback record is published before appending. Append is not an atomic config
// replacement: a process crash or I/O failure can leave a partial owned suffix.
// Errors after publication return a recovery receipt and require reconciliation;
// neither errors nor a power loss imply that installation had no effect.
// WorkingDir is revalidated under the config guard immediately before effects.
// This does not lock directory membership or pin the cwd of a later launch; the
// SDK does not start the server. Directory changes after the check remain possible.
func (p *Plan) Install() (Receipt, error) {
	return p.install(openGuardedConfig)
}

// openConfigGuard is internal so software fixtures can exercise the append
// algorithm separately from the Windows sharing-mode witness. No public advisory
// installation mode exists.
type openConfigGuard func(string) (*os.File, error)

func (p *Plan) install(open openConfigGuard) (Receipt, error) {
	if p == nil || p.spec.Server == "" {
		return Receipt{}, errors.New("mcpinstall: unprepared plan")
	}
	if err := plainPath(p.spec.ConfigPath); err != nil {
		return Receipt{}, err
	}
	f, err := open(p.spec.ConfigPath)
	if err != nil {
		return Receipt{}, err
	}
	defer f.Close()
	if err := verifyConfig(f, p.receipt.ConfigIdentity, p.prior); err != nil {
		return Receipt{}, err
	}
	current, _, err := readRegular(p.spec.SourcePath, 256<<20)
	if err != nil || !bytes.Equal(current, p.source) {
		return Receipt{}, errors.New("mcpinstall: source changed")
	}
	if err := plainPath(p.spec.InstallDir); err != nil {
		return Receipt{}, err
	}
	if err := workingDirectory(p.spec.WorkingDir); err != nil {
		return Receipt{}, err
	}
	if err := publish(p.receipt.Command, p.source, 0o555); err != nil {
		return Receipt{}, err
	}
	r := p.Preview()
	// Do not copy other servers' config/secrets into the install directory.
	// Prefix bytes remain only in the original config and this in-memory plan.
	record, err := json.Marshal(rollbackRecord{Schema: "fak.mcpinstall.v2", Receipt: r,
		PrefixLength: int64(len(p.prior)), OwnedSuffix: p.next[len(p.prior):]})
	if err != nil {
		return Receipt{}, err
	}
	if err := publish(r.ReceiptPath, record, 0o600); err != nil {
		return Receipt{}, err
	}
	r.ReceiptSHA256 = hash(record)
	if err := appendConfig(f, r.ConfigIdentity, p.prior, p.next); err != nil {
		// Return the recovery handle even when append/flush is uncertain.
		return r, err
	}
	return r, nil
}

// Rollback verifies the same config file identity, exact original prefix and
// complete owned suffix under the native guard, then truncates only that suffix
// and flushes. Changed or partial configs refuse; the retained record supports
// explicit reconciliation. Binaries and records remain inert. No process stops
// or reloads. Rollback is not a guarantee against arbitrary storage/power failure.
// Version-1 records are intentionally unsupported; v2 neither migrates them nor
// claims legacy recovery from their copied prior config.
func Rollback(receipt Receipt) error {
	return rollback(receipt, openGuardedConfig)
}

func rollback(receipt Receipt, open openConfigGuard) error {
	if !digest(receipt.ReceiptSHA256) {
		return errors.New("mcpinstall: missing rollback digest")
	}
	data, _, err := readRegular(receipt.ReceiptPath, 8<<20)
	if err != nil || hash(data) != receipt.ReceiptSHA256 {
		return errors.New("mcpinstall: rollback record mismatch")
	}
	var record rollbackRecord
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&record); err != nil {
		return err
	}
	if err := dec.Decode(new(any)); err != io.EOF {
		return errors.New("mcpinstall: trailing rollback data")
	}
	bound := receipt
	bound.ReceiptSHA256 = ""
	want, _ := json.Marshal(bound)
	got, _ := json.Marshal(record.Receipt)
	if !bytes.Equal(want, got) || record.Schema != "fak.mcpinstall.v2" ||
		record.PrefixLength < 0 || record.PrefixLength > 4<<20 || len(record.OwnedSuffix) == 0 || len(record.OwnedSuffix) > 1<<20 {
		return errors.New("mcpinstall: rollback identity mismatch")
	}
	if err := plainPath(receipt.ConfigPath); err != nil {
		return err
	}
	f, err := open(receipt.ConfigPath)
	if err != nil {
		return err
	}
	defer f.Close()
	current, err := readConfig(f, receipt.ConfigIdentity)
	if err != nil {
		return err
	}
	if int64(len(current)) != record.PrefixLength+int64(len(record.OwnedSuffix)) || hash(current) != receipt.ConfigSHA256 {
		return errors.New("mcpinstall: installed config changed")
	}
	prefix := current[:int(record.PrefixLength)]
	if hash(prefix) != receipt.PriorConfigSHA256 || !bytes.Equal(current[len(prefix):], record.OwnedSuffix) ||
		!bytes.Equal(current, appendSection(prefix, receipt.Server, receipt.Command, receipt.Args, receipt.WorkingDir)) {
		return errors.New("mcpinstall: rollback prefix or owned suffix mismatch")
	}
	if err := f.Truncate(record.PrefixLength); err != nil {
		return fmt.Errorf("mcpinstall: rollback persistence uncertain: %w", err)
	}
	if err := f.Sync(); err != nil {
		return fmt.Errorf("mcpinstall: rollback flush uncertain: %w", err)
	}
	return verifyConfig(f, receipt.ConfigIdentity, prefix)
}

// RenderCodexSection renders a plain stdio entry. Empty cwd preserves the
// existing public CLI renderer exactly. Validation belongs to Prepare.
func RenderCodexSection(server, command string, args []string, cwd string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "[mcp_servers.%s]\ncommand = %s\nargs = [", server, tomlString(command))
	for i, arg := range args {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(tomlString(arg))
	}
	b.WriteString("]\n")
	if cwd != "" {
		fmt.Fprintf(&b, "cwd = %s\n", tomlString(cwd))
	}
	return b.String()
}

func appendSection(prior []byte, server, command string, args []string, cwd string) []byte {
	// Preserve every prior byte, including final whitespace and comments.
	next := append([]byte{}, prior...)
	if len(next) > 0 && next[len(next)-1] != '\n' {
		next = append(next, '\n')
	}
	next = append(next, '\n')
	return append(next, RenderCodexSection(server, command, args, cwd)...)
}

func tomlString(s string) string {
	if !strings.ContainsAny(s, "'\n\r") {
		return "'" + s + "'"
	}
	b, _ := json.Marshal(s)
	return string(b)
}

func hash(data []byte) string { sum := sha256.Sum256(data); return hex.EncodeToString(sum[:]) }
func digest(s string) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == 32 && strings.ToLower(s) == s
}
func identifier(s string) bool {
	if len(s) == 0 || len(s) > 64 {
		return false
	}
	for _, c := range s {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_') {
			return false
		}
	}
	return true
}
func safeText(s string) bool {
	if !utf8.ValidString(s) {
		return false
	}
	for _, c := range s {
		if c < 0x20 || c == 0x7f {
			return false
		}
	}
	return true
}
func plainPath(path string) error {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path || !safeText(path) {
		return fmt.Errorf("mcpinstall: require a clean absolute path: %q", path)
	}
	for current := path; ; current = filepath.Dir(current) {
		info, err := os.Lstat(current)
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		if err == nil && info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("mcpinstall: symlink path: %s", current)
		}
		if filepath.Dir(current) == current {
			return nil
		}
	}
}

func workingDirectory(path string) error {
	if err := plainPath(path); err != nil {
		return err
	}
	if info, err := os.Stat(path); err != nil || !info.IsDir() {
		return errors.New("mcpinstall: working directory is unavailable")
	}
	return nil
}
func readRegular(path string, max int64) ([]byte, os.FileMode, error) {
	if err := plainPath(path); err != nil {
		return nil, 0, err
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() > max {
		return nil, 0, errors.New("mcpinstall: require bounded regular file")
	}
	data, err := io.ReadAll(io.LimitReader(f, max+1))
	if len(data) > int(max) {
		return nil, 0, errors.New("mcpinstall: file exceeds limit")
	}
	return data, info.Mode().Perm(), err
}
func readConfig(f *os.File, identity string) ([]byte, error) {
	actual, err := configIdentity(f)
	if err != nil {
		return nil, err
	}
	if actual != identity {
		return nil, errors.New("mcpinstall: config identity changed")
	}
	info, err := f.Stat()
	if err != nil || !info.Mode().IsRegular() {
		return nil, errors.New("mcpinstall: config is not a regular file")
	}
	pathInfo, err := os.Lstat(f.Name())
	if err != nil || !os.SameFile(info, pathInfo) {
		return nil, errors.New("mcpinstall: config pathname identity changed")
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	current, err := io.ReadAll(io.LimitReader(f, (8<<20)+1))
	if err != nil {
		return nil, err
	}
	if len(current) > 8<<20 {
		return nil, errors.New("mcpinstall: config exceeds limit")
	}
	return current, nil
}

func verifyConfig(f *os.File, identity string, expected []byte) error {
	current, err := readConfig(f, identity)
	if err != nil {
		return err
	}
	if !bytes.Equal(current, expected) {
		return errors.New("mcpinstall: config changed")
	}
	return nil
}

func appendConfig(f *os.File, identity string, prior, next []byte) error {
	if !bytes.HasPrefix(next, prior) || len(next) <= len(prior) {
		return errors.New("mcpinstall: invalid append plan")
	}
	if err := verifyConfig(f, identity, prior); err != nil {
		return err
	}
	if _, err := f.Seek(int64(len(prior)), io.SeekStart); err != nil {
		return err
	}
	n, err := f.Write(next[len(prior):])
	if err != nil || n != len(next)-len(prior) {
		return fmt.Errorf("mcpinstall: append persistence uncertain: %w", errors.Join(err, io.ErrShortWrite))
	}
	if err := f.Sync(); err != nil {
		return fmt.Errorf("mcpinstall: append flush uncertain: %w", err)
	}
	return verifyConfig(f, identity, next)
}

// publish never replaces an existing destination. Namespace publication is
// flushed by the platform helper before config mutation is permitted.
func publish(path string, data []byte, mode os.FileMode) error {
	if err := plainPath(path); err != nil {
		return err
	}
	if old, oldMode, err := readRegular(path, 256<<20); err == nil {
		if bytes.Equal(old, data) && samePermissions(oldMode.Perm(), mode) {
			return syncPublished(path)
		}
		return errors.New("mcpinstall: immutable destination collision")
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	tmp, err := stage(path, data, mode)
	if err != nil {
		return err
	}
	defer func() { _ = os.Chmod(tmp, 0o600); _ = os.Remove(tmp) }()
	return publishNew(tmp, path)
}

func samePermissions(actual, expected os.FileMode) bool {
	if runtime.GOOS == "windows" {
		// Windows exposes only the read-only attribute through Go mode bits.
		return (actual&0o222 != 0) == (expected&0o222 != 0)
	}
	return actual == expected
}
func stage(path string, data []byte, mode os.FileMode) (name string, err error) {
	f, err := os.CreateTemp(filepath.Dir(path), ".mcpinstall-*")
	if err != nil {
		return "", err
	}
	name = f.Name()
	defer func() {
		_ = f.Close()
		if err != nil {
			_ = os.Remove(name)
		}
	}()
	if err = f.Chmod(mode); err == nil {
		_, err = f.Write(data)
	}
	if err == nil {
		err = f.Sync()
	}
	if err == nil {
		err = f.Close()
	}
	return name, err
}

// absentNamespace accepts ordinary TOML tables, including quoted dotted keys.
// Inline root MCP maps and multiline strings are conservatively unsupported:
// treating unrecognized namespace syntax as absent would permit collisions.
func absentNamespace(data []byte, server string) error {
	if !utf8.Valid(data) || bytes.IndexByte(data, 0) >= 0 || bytes.Contains(data, []byte(`"""`)) || bytes.Contains(data, []byte("'''")) {
		return errors.New("mcpinstall: unsupported config encoding or multiline string")
	}
	var section []string
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || line[0] == '#' {
			continue
		}
		var key []string
		if line[0] == '[' {
			array := strings.HasPrefix(line, "[[")
			start, closing := 1, "]"
			if array {
				start, closing = 2, "]]"
			}
			var rest string
			var err error
			key, rest, err = keyPath(line[start:])
			if err != nil || !strings.HasPrefix(rest, closing) || strings.TrimSpace(strings.SplitN(strings.TrimPrefix(rest, closing), "#", 2)[0]) != "" {
				return errors.New("mcpinstall: unsupported table declaration")
			}
			section = key
			if array && len(key) == 1 && key[0] == "mcp_servers" {
				return errors.New("mcpinstall: ambiguous MCP array table")
			}
		} else {
			parts, rest, err := keyPath(line)
			if err != nil || !strings.HasPrefix(rest, "=") {
				continue // non-header array continuations cannot introduce a table
			}
			key = append(append([]string{}, section...), parts...)
		}
		if len(key) > 0 && key[0] == "mcp_servers" && (len(key) == 1 && !strings.HasPrefix(line, "[") || len(key) > 1 && key[1] == server) {
			return errors.New("mcpinstall: server namespace collision or ambiguous MCP map")
		}
	}
	return nil
}
func keyPath(text string) (keys []string, rest string, err error) {
	for {
		text = strings.TrimSpace(text)
		if text == "" {
			return nil, "", errors.New("empty key")
		}
		var key string
		if text[0] == '\'' || text[0] == '"' {
			quote := text[0]
			end := 1
			for end < len(text) {
				if text[end] == quote {
					break
				}
				if text[end] == '\\' && quote == '"' {
					end++
				}
				end++
			}
			if end >= len(text) {
				return nil, "", errors.New("unterminated key")
			}
			key = text[1:end]
			if quote == '"' {
				key, err = strconv.Unquote(text[:end+1])
				if err != nil {
					return nil, "", err
				}
			}
			text = text[end+1:]
		} else {
			end := 0
			for end < len(text) && (text[end] >= 'a' && text[end] <= 'z' || text[end] >= 'A' && text[end] <= 'Z' || text[end] >= '0' && text[end] <= '9' || text[end] == '-' || text[end] == '_') {
				end++
			}
			if end == 0 {
				return nil, "", errors.New("unsupported key")
			}
			key, text = text[:end], text[end:]
		}
		keys = append(keys, key)
		text = strings.TrimSpace(text)
		if !strings.HasPrefix(text, ".") {
			return keys, text, nil
		}
		text = text[1:]
	}
}
