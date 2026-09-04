// Package harnessrelease captures a release-backed external harness clean-room witness.
package harnessrelease

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// Schema identifies the clean-room release witness receipt specification.
const Schema = "fak.harness-release-witness/v1alpha1"

// Options configures a clean-room harness release verification execution.
type Options struct {
	Archive         string
	Checksum        string
	Target          string
	ProductDir      string
	Module          string
	Receipt         string
	RollbackCommand string
}

// CommandReceipt records the execution outcome and timing of a single command
// invoked during the clean-room verification workflow.
type CommandReceipt struct {
	Command        []string `json:"command"`
	ExitCode       int      `json:"exit_code"`
	ElapsedSeconds float64  `json:"elapsed_seconds"`
	Output         string   `json:"output,omitempty"`
}

// Receipt contains the complete verification outcome, cryptographic hashes,
// metadata, and command history for a released harness archive.
type Receipt struct {
	Schema              string           `json:"schema"`
	Outcome             string           `json:"outcome"`
	Target              string           `json:"target"`
	Archive             string           `json:"archive"`
	ArchiveSHA256       string           `json:"archive_sha256"`
	BinarySHA256        string           `json:"binary_sha256"`
	ProductDir          string           `json:"product_dir"`
	Module              string           `json:"module"`
	Generator           string           `json:"generator"`
	ContractVersion     string           `json:"contract_version"`
	FAKModule           string           `json:"fak_module"`
	FAKVersion          string           `json:"fak_version"`
	UpgradeCommand      string           `json:"upgrade_command"`
	RollbackCommand     string           `json:"rollback_command"`
	UserConfigSHA256    string           `json:"user_config_sha256"`
	UserConfigPreserved bool             `json:"user_config_preserved"`
	ElapsedSeconds      float64          `json:"elapsed_seconds"`
	Commands            []CommandReceipt `json:"commands"`
}

type lockFile struct {
	Generator       string `json:"generator"`
	ContractVersion string `json:"contract_version"`
	FAKModule       string `json:"fak_module"`
	FAKVersion      string `json:"fak_version"`
	Upgrade         string `json:"upgrade"`
}

// Run executes the clean-room release witness flow: validating checksums,
// unpacking the archive, generating a product harness, verifying config preservation,
// and recording all executed commands into a persistent receipt.
func Run(ctx context.Context, opts Options) (Receipt, error) {
	started := time.Now()
	r := Receipt{Schema: Schema, Outcome: "failure", Target: opts.Target, Archive: opts.Archive, ProductDir: opts.ProductDir, Module: opts.Module, RollbackCommand: opts.RollbackCommand}
	if opts.Module == "" {
		r.Module = "example.test/released-harness"
	}
	if opts.Target == "" {
		opts.Target = runtime.GOOS + "_" + runtime.GOARCH
		r.Target = opts.Target
	}
	if opts.Archive == "" || opts.Checksum == "" || opts.ProductDir == "" || opts.Receipt == "" || opts.RollbackCommand == "" {
		return r, errors.New("--archive, --checksum, --dir, --receipt, and --rollback-command are required")
	}
	if opts.Target != runtime.GOOS+"_"+runtime.GOARCH {
		return r, fmt.Errorf("target %q cannot execute on host %s_%s", opts.Target, runtime.GOOS, runtime.GOARCH)
	}
	archiveHash, err := VerifyChecksum(opts.Archive, opts.Checksum)
	if err != nil {
		return r, err
	}
	r.ArchiveSHA256 = archiveHash
	extractDir, err := os.MkdirTemp("", "fak-release-witness-")
	if err != nil {
		return r, err
	}
	defer os.RemoveAll(extractDir)
	if err := Extract(opts.Archive, extractDir); err != nil {
		return r, err
	}
	binary, err := findBinary(extractDir)
	if err != nil {
		return r, err
	}
	r.BinarySHA256, err = fileSHA256(binary)
	if err != nil {
		return r, err
	}
	if err := os.MkdirAll(filepath.Dir(opts.ProductDir), 0o755); err != nil {
		return r, err
	}
	if _, err := os.Stat(opts.ProductDir); err == nil {
		return r, fmt.Errorf("product directory already exists: %s", opts.ProductDir)
	} else if !os.IsNotExist(err) {
		return r, err
	}

	r.Commands = append(r.Commands, run(ctx, "", binary, "harness", "init", "--dir", opts.ProductDir, "--module", r.Module))
	if lastFailed(r.Commands) {
		return writeFailure(r, opts.Receipt, started, fmt.Errorf("released harness init failed: %s", commandFailure(r.Commands[len(r.Commands)-1])))
	}
	configPath := filepath.Join(opts.ProductDir, "product", "config.go")
	if err := os.WriteFile(configPath, []byte(customConfig), 0o644); err != nil {
		return writeFailure(r, opts.Receipt, started, err)
	}
	before, err := fileSHA256(configPath)
	if err != nil {
		return writeFailure(r, opts.Receipt, started, err)
	}
	productBinary := filepath.Join(opts.ProductDir, "product-bin")
	if runtime.GOOS == "windows" {
		productBinary += ".exe"
	}
	r.Commands = append(r.Commands, run(ctx, opts.ProductDir, "go", "build", "-o", productBinary, "./cmd/product"))
	if lastFailed(r.Commands) {
		return writeFailure(r, opts.Receipt, started, errors.New("generated product build failed"))
	}
	r.Commands = append(r.Commands, run(ctx, opts.ProductDir, productBinary, "--selfcheck"))
	if lastFailed(r.Commands) {
		return writeFailure(r, opts.Receipt, started, errors.New("generated product selfcheck failed"))
	}
	body, err := os.ReadFile(filepath.Join(opts.ProductDir, "harness.lock.json"))
	if err != nil {
		return writeFailure(r, opts.Receipt, started, err)
	}
	var lock lockFile
	if err := json.Unmarshal(body, &lock); err != nil {
		return writeFailure(r, opts.Receipt, started, err)
	}
	r.Generator, r.ContractVersion, r.FAKModule, r.FAKVersion, r.UpgradeCommand = lock.Generator, lock.ContractVersion, lock.FAKModule, lock.FAKVersion, lock.Upgrade
	r.Commands = append(r.Commands, run(ctx, "", binary, "harness", "init", "--dir", opts.ProductDir, "--module", r.Module, "--fak-version", lock.FAKVersion))
	if lastFailed(r.Commands) {
		return writeFailure(r, opts.Receipt, started, errors.New("released harness regeneration failed"))
	}
	after, err := fileSHA256(configPath)
	if err != nil {
		return writeFailure(r, opts.Receipt, started, err)
	}
	r.UserConfigSHA256 = after
	r.UserConfigPreserved = before == after
	if !r.UserConfigPreserved {
		return writeFailure(r, opts.Receipt, started, errors.New("regeneration changed user-owned config"))
	}
	r.Outcome = "success"
	r.ElapsedSeconds = seconds(time.Since(started))
	return r, writeReceipt(opts.Receipt, r)
}

func run(ctx context.Context, dir, name string, args ...string) CommandReceipt {
	started := time.Now()
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	exit := 0
	if err != nil {
		exit = 1
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			exit = ee.ExitCode()
		}
	}
	return CommandReceipt{Command: append([]string{name}, args...), ExitCode: exit, ElapsedSeconds: seconds(time.Since(started)), Output: string(out)}
}

func commandFailure(command CommandReceipt) string {
	if output := strings.TrimSpace(command.Output); output != "" {
		return output
	}
	return fmt.Sprintf("exit_code=%d", command.ExitCode)
}
func lastFailed(commands []CommandReceipt) bool {
	return len(commands) > 0 && commands[len(commands)-1].ExitCode != 0
}
func seconds(d time.Duration) float64 {
	return float64(d.Round(time.Millisecond)) / float64(time.Second)
}
func writeFailure(r Receipt, path string, started time.Time, cause error) (Receipt, error) {
	r.ElapsedSeconds = seconds(time.Since(started))
	_ = writeReceipt(path, r)
	return r, cause
}
func writeReceipt(path string, r Receipt) error {
	body, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	body = append(body, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, body, 0o644)
}

// VerifyChecksum validates that the SHA-256 hash of archive matches the expected
// digest recorded in the sidecar file, returning the verified lowercase hexadecimal hash.
func VerifyChecksum(archive, sidecar string) (string, error) {
	got, err := fileSHA256(archive)
	if err != nil {
		return "", err
	}
	body, err := os.ReadFile(sidecar)
	if err != nil {
		return "", err
	}
	fields := strings.Fields(string(body))
	if len(fields) == 0 || len(fields[0]) != sha256.Size*2 {
		return "", errors.New("invalid checksum sidecar")
	}
	want := strings.ToLower(fields[0])
	if _, err := hex.DecodeString(want); err != nil {
		return "", errors.New("invalid checksum sidecar")
	}
	if got != want {
		return "", fmt.Errorf("archive checksum mismatch: got %s want %s", got, want)
	}
	return got, nil
}

func fileSHA256(path string) (string, error) {
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

// Extract unpacks a supported release archive (.zip, .tar.gz, .tgz) into dir,
// enforcing strict path bounds to prevent directory traversal escapes.
func Extract(archive, dir string) error {
	lower := strings.ToLower(archive)
	switch {
	case strings.HasSuffix(lower, ".zip"):
		return extractZip(archive, dir)
	case strings.HasSuffix(lower, ".tar.gz"), strings.HasSuffix(lower, ".tgz"):
		return extractTarGZ(archive, dir)
	default:
		return fmt.Errorf("unsupported release archive: %s", archive)
	}
}

func safePath(root, name string) (string, error) {
	p := filepath.Join(root, filepath.FromSlash(name))
	rel, err := filepath.Rel(root, p)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
		return "", fmt.Errorf("unsafe archive path %q", name)
	}
	return p, nil
}
func extractZip(path, dir string) error {
	z, err := zip.OpenReader(path)
	if err != nil {
		return err
	}
	defer z.Close()
	for _, f := range z.File {
		dst, err := safePath(dir, f.Name)
		if err != nil {
			return err
		}
		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(dst, 0o755); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return err
		}
		src, err := f.Open()
		if err != nil {
			return err
		}
		mode := f.Mode().Perm()
		if mode == 0 {
			mode = 0o755
		}
		out, err := os.OpenFile(dst, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, mode)
		if err == nil {
			_, err = io.Copy(out, src)
		}
		closeErr := out.Close()
		src.Close()
		if err != nil {
			return err
		}
		if closeErr != nil {
			return closeErr
		}
	}
	return nil
}
func extractTarGZ(path, dir string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		h, err := tr.Next()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		dst, err := safePath(dir, h.Name)
		if err != nil {
			return err
		}
		switch h.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(dst, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
				return err
			}
			out, err := os.OpenFile(dst, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, os.FileMode(h.Mode).Perm())
			if err != nil {
				return err
			}
			_, copyErr := io.Copy(out, tr)
			closeErr := out.Close()
			if copyErr != nil {
				return copyErr
			}
			if closeErr != nil {
				return closeErr
			}
		default:
			return fmt.Errorf("unsupported archive entry %q", h.Name)
		}
	}
}
func findBinary(root string) (string, error) {
	names := map[string]bool{"fak": true, "fak.exe": true}
	var found string
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && names[strings.ToLower(d.Name())] {
			if found != "" {
				return errors.New("release archive contains multiple fak binaries")
			}
			found = path
		}
		return nil
	})
	if err != nil {
		return "", err
	}
	if found == "" {
		return "", errors.New("release archive contains no fak binary")
	}
	return found, nil
}

const customConfig = `package product

type Config struct {
 ID string
 Version string
 Profile string
 SystemPrompt string
 Task string
}

func DefaultConfig() Config {
 return Config{
  ID: "released-clean-room-harness",
  Version: "0.1.0",
  Profile: "release-witness",
  SystemPrompt: "Answer only from admitted context.",
  Task: "Prove the released harness works.",
 }
}

func OfflineReply(prompt string) string { return "released harness: " + prompt }
`
