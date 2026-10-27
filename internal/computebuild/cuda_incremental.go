package computebuild

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/processlock"
)

type cudaLockContext struct{}

func lockCUDABuild(ctx context.Context, dir string) (context.Context, func(), error) {
	dir, err := filepath.Abs(dir)
	if err != nil {
		return ctx, nil, err
	}
	if ctx.Value(cudaLockContext{}) == dir {
		return ctx, func() {}, nil
	}
	for {
		unlock, err := processlock.Try(cudaCacheDir(dir), "build.lock")
		if err == nil {
			return context.WithValue(ctx, cudaLockContext{}, dir), unlock, nil
		}
		if !errors.Is(err, processlock.ErrBusy) {
			return ctx, nil, err
		}
		select {
		case <-ctx.Done():
			return ctx, nil, ctx.Err()
		case <-time.After(25 * time.Millisecond):
		}
	}
}

func cudaCacheDir(pkgDir string) string {
	return filepath.Join(pkgDir, "..", "..", ".fak", "cuda-cache")
}
func cudaManifestPath(object string) string {
	return filepath.Join(cudaCacheDir(filepath.Dir(object)), filepath.Base(object)+".json")
}

type cudaObjectManifest struct {
	Schema string `json:"schema"`
	Input  string `json:"input_sha256"`
	Output string `json:"output_sha256"`
}

func cudaJSONDigest(v any) string {
	b, _ := json.Marshal(v)
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func cudaFileDigest(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return "", err
	}
	if !info.Mode().IsRegular() {
		return "", errors.New("cuda_digest_nonregular")
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func cudaWriteJSON(path string, value any) error {
	b, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".cuda-json-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	if _, err := f.Write(append(b, '\n')); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}

// nvcc -M includes system headers, unlike -MM. Rescanning also detects a newly
// shadowing include that did not exist when the previous object was compiled.
func cudaDependencies(ctx context.Context, tc *Toolchain, dir, source string, args, env []string) (map[string]string, error) {
	var scan []string
	for i := 0; i < len(args); i++ {
		if args[i] == "-o" {
			i++
			continue
		}
		if args[i] != "-c" && args[i] != source {
			scan = append(scan, args[i])
		}
	}
	scan = append(scan, "-M", "-MT", "fak_cuda_dependency_target", source)
	var out, diagnostic bytes.Buffer
	if err := RunCmd(ctx, tc.NVCC, scan, dir, env, &out, &diagnostic); err != nil {
		return nil, fmt.Errorf("dependency_scan_failed: %w", err)
	}
	paths, err := parseCUDADependencies(out.String())
	if err != nil {
		return nil, err
	}
	deps := make(map[string]string, len(paths))
	for _, path := range paths {
		if !filepath.IsAbs(path) {
			path = filepath.Join(dir, path)
		}
		path = filepath.Clean(path)
		raw, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("dependency_unreadable: %w", err)
		}
		for _, macro := range []string{"__DATE__", "__TIME__", "__TIMESTAMP__"} {
			if bytes.Contains(raw, []byte(macro)) {
				return nil, errors.New("dependency_time_macro_uncacheable")
			}
		}
		h := sha256.Sum256(raw)
		deps[path] = hex.EncodeToString(h[:])
	}
	abs, err := filepath.Abs(filepath.Join(dir, source))
	if err != nil {
		return nil, err
	}
	if _, ok := deps[abs]; !ok {
		return nil, errors.New("dependency_source_missing")
	}
	return deps, nil
}

func parseCUDADependencies(raw string) ([]string, error) {
	raw = strings.ReplaceAll(raw, "\\\r\n", "")
	raw = strings.ReplaceAll(raw, "\\\n", "")
	target, body, ok := strings.Cut(raw, ":")
	if !ok || strings.TrimSpace(target) != "fak_cuda_dependency_target" {
		return nil, errors.New("dependency_manifest_invalid")
	}
	var paths []string
	var word strings.Builder
	flush := func() {
		if word.Len() > 0 {
			paths = append(paths, word.String())
			word.Reset()
		}
	}
	for i := 0; i < len(body); i++ {
		c := body[i]
		if c == '\\' && i+1 < len(body) && strings.ContainsRune(" \t#:$\\", rune(body[i+1])) {
			i++
			word.WriteByte(body[i])
			continue
		}
		if c == '$' && i+1 < len(body) && body[i+1] == '$' {
			i++
			word.WriteByte('$')
			continue
		}
		if c == '\n' || c == '\r' || c == ' ' || c == '\t' {
			flush()
			continue
		}
		if c == '#' || c == '\x00' {
			return nil, errors.New("dependency_manifest_invalid")
		}
		word.WriteByte(c)
	}
	flush()
	if len(paths) == 0 {
		return nil, errors.New("dependency_manifest_empty")
	}
	return paths, nil
}

func cudaToolPath(name string, env []string) (string, error) {
	if filepath.IsAbs(name) {
		return filepath.EvalSymlinks(name)
	}
	if strings.ContainsAny(name, `/\`) {
		abs, err := filepath.Abs(name)
		if err != nil {
			return "", err
		}
		return filepath.EvalSymlinks(abs)
	}
	for _, entry := range env {
		key, value, ok := strings.Cut(entry, "=")
		if !ok || (key != "PATH" && (runtime.GOOS != "windows" || !strings.EqualFold(key, "PATH"))) {
			continue
		}
		for _, dir := range filepath.SplitList(value) {
			for _, suffix := range []string{"", ".exe"} {
				path := filepath.Join(dir, name+suffix)
				if info, err := os.Stat(path); err == nil && info.Mode().IsRegular() {
					return filepath.EvalSymlinks(path)
				}
			}
		}
	}
	return "", fmt.Errorf("tool_identity_unresolved: %s", name)
}

func cudaToolchainDigest(ctx context.Context, tc *Toolchain, env []string) (string, error) {
	files := map[string]string{}
	add := func(path string) error {
		digest, err := cudaFileDigest(path)
		if err != nil {
			return err
		}
		files[path] = digest
		return nil
	}
	nvcc, err := cudaToolPath(tc.NVCC, env)
	if err != nil {
		return "", err
	}
	if err := add(nvcc); err != nil {
		return "", err
	}
	host := tc.CXX
	if tc.IsWindows {
		// The Windows nvcc invocation uses its default cl, not the cgo CXX.
		host = ""
	}
	seenEnv := map[string]bool{}
	for _, entry := range env {
		key, value, _ := strings.Cut(entry, "=")
		if tc.IsWindows || runtime.GOOS == "windows" {
			key = strings.ToUpper(key)
		}
		if key != "" && seenEnv[key] {
			return "", errors.New("tool_identity_duplicate_environment_uncacheable")
		}
		seenEnv[key] = true
		if value != "" && (key == "NVCC_PREPEND_FLAGS" || key == "NVCC_APPEND_FLAGS") {
			return "", errors.New("tool_identity_external_nvcc_flags_uncacheable")
		}
		if host == "" && key == "NVCC_CCBIN" {
			host = value
		}
	}
	if host == "" {
		host = "gcc"
		if tc.IsWindows {
			host = "cl"
		}
	}
	host, err = cudaToolPath(host, env)
	if err != nil {
		return "", err
	}
	if err := add(host); err != nil {
		return "", err
	}
	if tc.IsWindows {
		// MSVC loads its code generators and supporting DLLs beside cl.exe.
		if err := filepath.WalkDir(filepath.Dir(host), func(path string, entry os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if ctx.Err() != nil {
				return ctx.Err()
			}
			if entry.IsDir() {
				return nil
			}
			return add(path)
		}); err != nil {
			return "", err
		}
	}
	var version bytes.Buffer
	versionArgs := []string{"--version"}
	if tc.IsWindows {
		versionArgs = []string{"/?"}
	}
	if err := RunCmd(ctx, host, versionArgs, "", env, &version, &version); err != nil {
		return "", err
	}
	if !tc.IsWindows {
		for _, tool := range []string{"cc1plus", "as"} {
			var out bytes.Buffer
			if err := RunCmd(ctx, host, []string{"-print-prog-name=" + tool}, "", env, &out, io.Discard); err != nil {
				return "", err
			}
			path, err := cudaToolPath(strings.TrimSpace(out.String()), env)
			if err != nil {
				return "", err
			}
			if err := add(path); err != nil {
				return "", err
			}
		}
	}
	home := tc.CUDAHome
	if home == "" {
		home = filepath.Dir(filepath.Dir(nvcc))
	}
	homes := []string{home}
	compilerHome := filepath.Dir(filepath.Dir(nvcc))
	if filepath.Clean(home) != filepath.Clean(compilerHome) {
		homes = append(homes, compilerHome)
	}
	for _, sdkHome := range homes {
		libdevice, err := os.ReadDir(filepath.Join(sdkHome, "nvvm", "libdevice"))
		if err != nil || len(libdevice) == 0 {
			return "", errors.New("tool_identity_libdevice_incomplete")
		}
		for _, subtree := range []string{"bin", "nvvm"} {
			root := filepath.Join(sdkHome, subtree)
			if info, err := os.Stat(root); err != nil || !info.IsDir() {
				return "", fmt.Errorf("tool_identity_sdk_incomplete: %s", root)
			}
			if err := filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
				if ctx.Err() != nil {
					return ctx.Err()
				}
				if err != nil {
					return err
				}
				if entry.IsDir() {
					return nil
				}
				info, err := os.Stat(path)
				if err != nil {
					return err
				}
				if !info.Mode().IsRegular() {
					return fmt.Errorf("tool_identity_nonregular: %s", path)
				}
				return add(path)
			}); err != nil {
				return "", err
			}
		}
	}
	return cudaJSONDigest(struct {
		Files     map[string]string
		Version   string
		Env       []string
		Toolchain *Toolchain
		OS, Arch  string
	}{files, version.String(), env, tc, runtime.GOOS, runtime.GOARCH}), nil
}

func compileCUDAObject(ctx context.Context, tc *Toolchain, dir, source, object string, args, env []string, receipt *CUDANativeBuildReceipt, stdout io.Writer) error {
	record := CUDAObjectBuildReceipt{Source: source, Object: object, Action: "executed", EffectiveFlags: append([]string(nil), args...)}
	defer func() { receipt.Objects = append(receipt.Objects, record) }()
	started := time.Now()
	deps, depErr := cudaDependencies(ctx, tc, dir, source, args, env)
	record.DependencyScanMS = time.Since(started).Milliseconds()
	if depErr == nil {
		record.SourceClosureSHA256 = cudaJSONDigest(deps)
	}
	started = time.Now()
	toolchain, toolErr := cudaToolchainDigest(ctx, tc, env)
	record.ToolchainIdentityMS = time.Since(started).Milliseconds()
	receipt.ToolchainSHA256 = toolchain
	cacheable := depErr == nil && toolErr == nil
	if cacheable {
		record.InputSHA256 = cudaJSONDigest(struct {
			Dependencies map[string]string
			Args         []string
			Toolchain    string
			NCCL         bool
		}{deps, args, toolchain, receipt.NCCL})
	} else {
		record.MissReason = "uncacheable_dependency_scan"
		if toolErr != nil {
			record.MissReason = "uncacheable_toolchain_identity"
		}
	}
	manifestPath := cudaManifestPath(object)
	if cacheable {
		var old cudaObjectManifest
		raw, err := os.ReadFile(manifestPath)
		switch {
		case errors.Is(err, os.ErrNotExist):
			record.MissReason = "manifest_missing"
		case err != nil || json.Unmarshal(raw, &old) != nil || old.Schema != "fak.cuda-object-cache.v1" || len(old.Input) != 64 || len(old.Output) != 64:
			record.MissReason = "manifest_invalid"
		case old.Input != record.InputSHA256:
			record.MissReason = "inputs_changed"
		default:
			digest, err := cudaFileDigest(object)
			if err == nil && digest == old.Output {
				record.Action, record.MissReason, record.OutputSHA256 = "reused", "", digest
				if stdout != nil {
					fmt.Fprintf(stdout, "[cuda] reused %s\n", source)
				}
				return nil
			}
			record.MissReason = "output_missing_or_corrupt"
		}
	}
	// Invalidate first: a failed compile must never leave a reusable manifest.
	if err := os.Remove(manifestPath); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".cuda-object-*"+filepath.Ext(object))
	if err != nil {
		return err
	}
	name := tmp.Name()
	if err := tmp.Close(); err != nil {
		os.Remove(name)
		return err
	}
	defer os.Remove(name)
	compileArgs := append([]string(nil), args...)
	for i := range compileArgs {
		if compileArgs[i] == "-o" && i+1 < len(compileArgs) {
			compileArgs[i+1] = name
		}
	}
	started = time.Now()
	compileErr := RunCmd(ctx, tc.NVCC, compileArgs, dir, env, stdout, os.Stderr)
	record.CompileMS = time.Since(started).Milliseconds()
	if compileErr != nil {
		return compileErr
	}
	info, err := os.Stat(name)
	if err != nil || info.Size() == 0 {
		return errors.New("cuda_object_output_missing")
	}
	if cacheable {
		after, err := cudaDependencies(ctx, tc, dir, source, args, env)
		if err != nil || cudaJSONDigest(after) != cudaJSONDigest(deps) {
			return errors.New("cuda_dependencies_changed_during_compile")
		}
		afterTool, err := cudaToolchainDigest(ctx, tc, env)
		if err != nil || afterTool != toolchain {
			return errors.New("cuda_toolchain_changed_during_compile")
		}
	}
	record.OutputSHA256, err = cudaFileDigest(name)
	if err != nil {
		return err
	}
	if err := os.Rename(name, object); err != nil {
		return err
	}
	if cacheable {
		return cudaWriteJSON(manifestPath, cudaObjectManifest{"fak.cuda-object-cache.v1", record.InputSHA256, record.OutputSHA256})
	}
	return nil
}
