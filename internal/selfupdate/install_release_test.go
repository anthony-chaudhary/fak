package selfupdate

import (
	"archive/tar"
	"compress/gzip"
	"crypto/sha256"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// The test executable supplies controlled host/download commands; the actual
// installer still downloads, verifies, extracts, and installs the archive.
func TestMain(m *testing.M) {
	if os.Getenv("FAK_INSTALL_FIXTURE") == "1" {
		switch filepath.Base(os.Args[0]) {
		case "uname":
			if os.Args[1] == "-s" {
				fmt.Println(os.Getenv("FIXTURE_OS"))
			} else {
				fmt.Println(os.Getenv("FIXTURE_ARCH"))
			}
			os.Exit(0)
		case "getconf":
			version := os.Getenv("FIXTURE_LIBC")
			if version == "" {
				version = "glibc 2.39"
			}
			fmt.Println(version)
			os.Exit(0)
		case "nvidia-smi":
			if os.Getenv("FIXTURE_NVIDIA") != "1" {
				os.Exit(1)
			}
			if len(os.Args) > 1 && strings.HasPrefix(os.Args[1], "--query-gpu=compute_cap") {
				fmt.Println("12.0")
			}
			os.Exit(0)
		case "docker":
			if os.Getenv("FIXTURE_DOCKER") == "1" {
				os.Exit(0)
			}
			os.Exit(1)
		case "git":
			// Record each call; a clone materializes the checkout it names last.
			appendFixtureLog(strings.Join(os.Args[1:], " "))
			if len(os.Args) > 1 && os.Args[1] == "clone" {
				if err := os.MkdirAll(filepath.Join(os.Args[len(os.Args)-1], ".git"), 0700); err != nil {
					os.Exit(1)
				}
			}
			os.Exit(0)
		case "go":
			appendFixtureLog("go " + strings.Join(os.Args[1:], " ") + " CGO_LDFLAGS=" + os.Getenv("CGO_LDFLAGS"))
			for i, arg := range os.Args {
				if arg == "-o" && i+1 < len(os.Args) {
					self, err := os.Executable()
					if err != nil {
						os.Exit(1)
					}
					data, err := os.ReadFile(self)
					if err != nil || os.WriteFile(os.Args[i+1], data, 0755) != nil {
						os.Exit(1)
					}
				}
			}
			os.Exit(0)
		case "c++":
			os.Exit(0)
		case "brew":
			_ = os.WriteFile(os.Getenv("FIXTURE_BREW_LOG"), []byte("called"), 0600)
			os.Exit(1)
		case "curl":
			if len(os.Args) == 3 {
				fmt.Println(`{"tag_name":"v9.9.9"}`)
				os.Exit(0)
			}
			data, err := os.ReadFile(filepath.Join(os.Getenv("FIXTURE_ASSETS"), filepath.Base(os.Args[2])))
			if err != nil {
				os.Exit(22)
			}
			if err := os.WriteFile(os.Args[4], data, 0600); err != nil {
				os.Exit(23)
			}
			os.Exit(0)
		case "fak":
			if os.Getenv("FIXTURE_EXEC_FAIL") == "1" {
				os.Exit(127)
			}
			fmt.Println("9.9.9")
			os.Exit(0)
		}
	}
	os.Exit(m.Run())
}

func appendFixtureLog(line string) {
	f, err := os.OpenFile(os.Getenv("FIXTURE_CMD_LOG"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
	if err != nil {
		os.Exit(1)
	}
	defer f.Close()
	fmt.Fprintln(f, line)
}

func TestGPUFirstInstaller(t *testing.T) {
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	binary, err := os.ReadFile(executable)
	if err != nil {
		t.Fatal(err)
	}
	script, err := filepath.Abs("../../install.sh")
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, host, arch, variant, asset, want, libc string
		nvidia, docker, corrupt, missing, execFail   bool
	}{
		{name: "apple-default", host: "Darwin", arch: "arm64", asset: "darwin_arm64_metal"},
		{name: "linux-default", host: "Linux", arch: "x86_64", asset: "linux_amd64_vulkan"},
		{name: "cpu-reference", host: "Darwin", arch: "arm64", variant: "cpu", asset: "darwin_arm64"},
		{name: "nvidia", host: "Linux", arch: "x86_64", nvidia: true, docker: true, want: "docker run --rm --gpus all ghcr.io/anthony-chaudhary/fak:cuda-latest version"},
		{name: "nvidia-no-docker", host: "Linux", arch: "x86_64", nvidia: true, asset: "cuda-source"},
		{name: "old-glibc", host: "Linux", arch: "x86_64", libc: "glibc 2.35", want: "this host has 2.35"},
		{name: "musl", host: "Linux", arch: "x86_64", libc: "musl", want: "requires GNU glibc 2.39+"},
		{name: "unsupported-default", host: "Linux", arch: "aarch64", want: "explicitly select --variant cpu"},
		{name: "wrong-metal-host", host: "Linux", arch: "x86_64", variant: "metal", want: "supports darwin/arm64 only"},
		{name: "missing-gpu", host: "Darwin", arch: "arm64", asset: "darwin_arm64_metal", missing: true, want: "GPU asset"},
		{name: "binary-execution-fails", host: "Darwin", arch: "arm64", asset: "darwin_arm64_metal", execFail: true, want: "installed binary failed to run"},
		{name: "bad-checksum", host: "Linux", arch: "x86_64", asset: "linux_amd64_vulkan", corrupt: true, want: "checksum mismatch"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			bin := filepath.Join(dir, "bin")
			assets := filepath.Join(dir, "assets")
			if err := os.MkdirAll(bin, 0700); err != nil {
				t.Fatal(err)
			}
			if err := os.MkdirAll(assets, 0700); err != nil {
				t.Fatal(err)
			}
			for _, name := range []string{"uname", "curl", "brew", "nvidia-smi", "getconf", "docker", "git", "go", "c++"} {
				if err := os.Symlink(executable, filepath.Join(bin, name)); err != nil {
					t.Fatal(err)
				}
			}
			archive := "fak_9.9.9_" + tc.asset + ".tar.gz"
			cudaHome := filepath.Join(dir, "cuda")
			cudaLib := filepath.Join(cudaHome, "targets", "x86_64-linux", "lib")
			if err := os.MkdirAll(cudaLib, 0700); err != nil {
				t.Fatal(err)
			}
			if err := os.MkdirAll(filepath.Join(cudaHome, "bin"), 0700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(cudaHome, "bin", "nvcc"), []byte("#!/bin/sh\n"), 0700); err != nil {
				t.Fatal(err)
			}
			cmdLog := filepath.Join(dir, "cmd.log")
			if tc.asset != "" && tc.asset != "cuda-source" && !tc.missing {
				f, err := os.Create(filepath.Join(assets, archive))
				if err != nil {
					t.Fatal(err)
				}
				gz := gzip.NewWriter(f)
				tw := tar.NewWriter(gz)
				if err := tw.WriteHeader(&tar.Header{Name: "fak", Mode: 0755, Size: int64(len(binary))}); err != nil {
					t.Fatal(err)
				}
				if _, err := tw.Write(binary); err != nil {
					t.Fatal(err)
				}
				if tc.asset == "linux_amd64_vulkan" {
					if err := tw.WriteHeader(&tar.Header{Name: "spirv/probe.spv", Mode: 0644, Size: 4}); err != nil {
						t.Fatal(err)
					}
					if _, err := tw.Write([]byte("SPIR")); err != nil {
						t.Fatal(err)
					}
				}
				if err := tw.Close(); err != nil {
					t.Fatal(err)
				}
				if err := gz.Close(); err != nil {
					t.Fatal(err)
				}
				if err := f.Close(); err != nil {
					t.Fatal(err)
				}
				data, err := os.ReadFile(filepath.Join(assets, archive))
				if err != nil {
					t.Fatal(err)
				}
				hash := fmt.Sprintf("%x", sha256.Sum256(data))
				if tc.corrupt {
					hash = strings.Repeat("0", 64)
				}
				if err := os.WriteFile(filepath.Join(assets, "SHA256SUMS"), []byte(hash+" *"+archive+"\n"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			args := []string{script}
			if tc.variant != "" {
				args = append(args, "--variant", tc.variant)
			}
			cmd := exec.Command("sh", args...)
			// HOME and the destination are private fixtures.
			dest := filepath.Join(dir, "installed")
			cmd.Env = []string{"PATH=" + bin + ":" + os.Getenv("PATH"), "HOME=" + dir, "FAK_INSTALL_FIXTURE=1", "FIXTURE_OS=" + tc.host, "FIXTURE_ARCH=" + tc.arch, "FIXTURE_ASSETS=" + assets, "FIXTURE_BREW_LOG=" + filepath.Join(dir, "brew-called"), "FAK_INSTALL_DIR=" + dest}
			cmd.Env = append(cmd.Env, "FIXTURE_LIBC="+tc.libc, "FIXTURE_CMD_LOG="+cmdLog, "CUDA_HOME="+cudaHome)
			if tc.docker {
				cmd.Env = append(cmd.Env, "FIXTURE_DOCKER=1")
			}
			if tc.execFail {
				cmd.Env = append(cmd.Env, "FIXTURE_EXEC_FAIL=1")
			}
			if tc.nvidia {
				cmd.Env = append(cmd.Env, "FIXTURE_NVIDIA=1")
			}
			out, err := cmd.CombinedOutput()
			if tc.want != "" {
				if err == nil || !strings.Contains(string(out), tc.want) {
					t.Fatalf("want failure %q; err=%v output=%s", tc.want, err, out)
				}
				if _, err := os.Stat(filepath.Join(dest, "fak")); !tc.execFail && !os.IsNotExist(err) {
					t.Fatal("failed installer installed a binary")
				}
				return
			}
			if err != nil {
				t.Fatalf("installer: %v\n%s", err, out)
			}
			if tc.asset == "cuda-source" {
				calls, err := os.ReadFile(cmdLog)
				if err != nil {
					t.Fatal(err)
				}
				for _, want := range []string{
					"clone --depth 1 --branch main https://github.com/anthony-chaudhary/fak.git",
					"go run ./cmd/fak-dev build-cuda build --arch sm_120f --cuda-home " + cudaHome,
					"go build -tags cuda -o",
					"-Wl,-rpath," + cudaLib,
				} {
					if !strings.Contains(string(calls), want) {
						t.Fatalf("CUDA source build missing %q; calls:\n%s\noutput:\n%s", want, calls, out)
					}
				}
			} else if !strings.Contains(string(out), archive) || !strings.Contains(string(out), "checksum OK") {
				t.Fatalf("wrong asset or unverified install: %s", out)
			}
			installed, err := os.ReadFile(filepath.Join(dest, "fak"))
			if err != nil {
				t.Fatal(err)
			}
			if tc.asset == "linux_amd64_vulkan" {
				resource, err := os.ReadFile(filepath.Join(dest, "spirv", "probe.spv"))
				if err != nil || string(resource) != "SPIR" {
					t.Fatalf("missing installed Vulkan resource: %v", err)
				}
			}
			if sha256.Sum256(installed) != sha256.Sum256(binary) {
				t.Fatal("installed binary differs")
			}
		})
	}
}
