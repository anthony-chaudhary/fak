package oci

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/sandbox"
)

// SpecVersion is the supported OCI Runtime Specification version.
const SpecVersion = "1.0.2"

// MinimalLinuxCapabilities defines the minimal capability set without CAP_SYS_ADMIN.
var MinimalLinuxCapabilities = []string{
	"CAP_CHOWN",
	"CAP_DAC_OVERRIDE",
	"CAP_FOWNER",
	"CAP_FSETID",
	"CAP_KILL",
	"CAP_SETGID",
	"CAP_SETUID",
	"CAP_SETPCAP",
	"CAP_NET_BIND_SERVICE",
	"CAP_SYS_CHROOT",
	"CAP_SETFCAP",
}

// Spec represents the standard OCI Runtime Specification v1.0+.
type Spec struct {
	OCIVersion string   `json:"ociVersion"`
	Root       *Root    `json:"root,omitempty"`
	Process    *Process `json:"process,omitempty"`
	Mounts     []Mount  `json:"mounts,omitempty"`
	Linux      *Linux   `json:"linux,omitempty"`
}

// Root specifies the container's root filesystem.
type Root struct {
	Path     string `json:"path"`
	Readonly bool   `json:"readonly"`
}

// Process specifies the container execution process.
type Process struct {
	Terminal     bool               `json:"terminal"`
	User         User               `json:"user"`
	Args         []string           `json:"args"`
	Env          []string           `json:"env,omitempty"`
	Cwd          string             `json:"cwd"`
	Capabilities *LinuxCapabilities `json:"capabilities,omitempty"`
	Rlimits      []POSIXRlimit      `json:"rlimits,omitempty"`
}

// User specifies the user and group IDs for the container process.
type User struct {
	UID uint32 `json:"uid"`
	GID uint32 `json:"gid"`
}

// LinuxCapabilities specifies Linux capability sets.
type LinuxCapabilities struct {
	Bounding    []string `json:"bounding,omitempty"`
	Effective   []string `json:"effective,omitempty"`
	Inheritable []string `json:"inheritable,omitempty"`
	Permitted   []string `json:"permitted,omitempty"`
	Ambient     []string `json:"ambient,omitempty"`
}

// POSIXRlimit specifies resource limit bounds.
type POSIXRlimit struct {
	Type string `json:"type"`
	Hard uint64 `json:"hard"`
	Soft uint64 `json:"soft"`
}

// Mount specifies a filesystem mount point inside the container.
type Mount struct {
	Destination string   `json:"destination"`
	Type        string   `json:"type"`
	Source      string   `json:"source"`
	Options     []string `json:"options,omitempty"`
}

// Linux specifies Linux-specific container configuration.
type Linux struct {
	Namespaces []LinuxNamespace `json:"namespaces,omitempty"`
	Resources  *LinuxResources  `json:"resources,omitempty"`
}

// LinuxNamespace represents a Linux kernel namespace configuration.
type LinuxNamespace struct {
	Type string `json:"type"`
	Path string `json:"path,omitempty"`
}

// LinuxResources specifies container resource constraints.
type LinuxResources struct {
	Memory *LinuxMemory `json:"memory,omitempty"`
	CPU    *LinuxCPU    `json:"cpu,omitempty"`
}

// LinuxMemory specifies container memory limits.
type LinuxMemory struct {
	Limit *int64 `json:"limit,omitempty"`
}

// LinuxCPU specifies container CPU limits.
type LinuxCPU struct {
	Shares *uint64 `json:"shares,omitempty"`
}

// BuildSpec constructs an OCI runtime specification struct from sandbox.Spec and ExecutionRequest.
func BuildSpec(spec sandbox.Spec, req sandbox.ExecutionRequest, bundleDir string) Spec {
	rootPath := "rootfs"
	if spec.Rootfs != "" {
		rootPath = spec.Rootfs
	}

	args := req.Argv
	if len(args) == 0 {
		if strings.TrimSpace(req.Command) != "" {
			args = strings.Fields(req.Command)
		} else {
			args = []string{"/bin/sh"}
		}
	}

	cwd := req.WorkingDir
	if strings.TrimSpace(cwd) == "" {
		cwd = "/workspace"
	}

	var env []string
	if len(req.Env) > 0 {
		env = req.Env
	} else if len(spec.Env) > 0 {
		env = spec.Env
	} else {
		env = []string{
			"PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
			"TERM=xterm",
			"HOME=/workspace",
			"FAK_SANDBOX=1",
		}
	}

	capsCopy := append([]string(nil), MinimalLinuxCapabilities...)
	caps := &LinuxCapabilities{
		Bounding:    capsCopy,
		Effective:   capsCopy,
		Inheritable: capsCopy,
		Permitted:   capsCopy,
	}

	rlimits := []POSIXRlimit{
		{Type: "RLIMIT_NOFILE", Hard: 1024, Soft: 1024},
	}

	wsClean := filepath.Clean(spec.WorkspaceDir)
	wsReadOnly := false
	for _, ro := range spec.ReadOnlyPaths {
		if filepath.Clean(ro) == wsClean || ro == "/workspace" {
			wsReadOnly = true
			break
		}
	}

	wsOptions := []string{"rbind", "rw"}
	if wsReadOnly {
		wsOptions = []string{"rbind", "ro"}
	}

	mounts := []Mount{
		{
			Destination: "/proc",
			Type:        "proc",
			Source:      "proc",
		},
		{
			Destination: "/dev/pts",
			Type:        "devpts",
			Source:      "devpts",
			Options:     []string{"nosuid", "noexec", "newinstance", "ptmxmode=0666", "mode=0620"},
		},
		{
			Destination: "/dev/shm",
			Type:        "tmpfs",
			Source:      "shm",
			Options:     []string{"nosuid", "noexec", "nodev", "mode=1777", "size=65536k"},
		},
		{
			Destination: "/dev/mqueue",
			Type:        "mqueue",
			Source:      "mqueue",
			Options:     []string{"nosuid", "noexec", "nodev"},
		},
		{
			Destination: "/sys",
			Type:        "sysfs",
			Source:      "sysfs",
			Options:     []string{"nosuid", "noexec", "nodev", "ro"},
		},
		{
			Destination: "/tmp",
			Type:        "tmpfs",
			Source:      "tmpfs",
			Options:     []string{"nosuid", "nodev", "mode=1777", "size=67108864"},
		},
		{
			Destination: "/workspace",
			Type:        "bind",
			Source:      wsClean,
			Options:     wsOptions,
		},
	}

	// Mount additional read-only paths if specified
	for _, ro := range spec.ReadOnlyPaths {
		cleanRO := filepath.Clean(ro)
		destRO := filepath.ToSlash(cleanRO)
		if cleanRO != wsClean && destRO != "/workspace" {
			mounts = append(mounts, Mount{
				Destination: destRO,
				Type:        "bind",
				Source:      cleanRO,
				Options:     []string{"rbind", "ro"},
			})
		}
	}

	// Mount additional writable paths as tmpfs CoW mounts if specified
	for _, wp := range spec.WritablePaths {
		cleanWP := filepath.Clean(wp)
		destWP := filepath.ToSlash(cleanWP)
		if cleanWP != wsClean && destWP != "/workspace" && destWP != "/tmp" {
			mounts = append(mounts, Mount{
				Destination: destWP,
				Type:        "tmpfs",
				Source:      "tmpfs",
				Options:     []string{"nosuid", "nodev", "mode=0755"},
			})
		}
	}

	namespaces := []LinuxNamespace{
		{Type: "pid"},
		{Type: "network"},
		{Type: "ipc"},
		{Type: "uts"},
		{Type: "mount"},
		{Type: "user"},
	}

	var res *LinuxResources
	if spec.MemoryLimitBytes > 0 || spec.CPULimitPercent > 0 {
		res = &LinuxResources{}
		if spec.MemoryLimitBytes > 0 {
			memLimit := spec.MemoryLimitBytes
			res.Memory = &LinuxMemory{Limit: &memLimit}
		}
		if spec.CPULimitPercent > 0 {
			shares := uint64(spec.CPULimitPercent * 1024 / 100)
			res.CPU = &LinuxCPU{Shares: &shares}
		}
	}

	return Spec{
		OCIVersion: SpecVersion,
		Root: &Root{
			Path:     rootPath,
			Readonly: true,
		},
		Process: &Process{
			Terminal:     false,
			User:         User{UID: 1000, GID: 1000},
			Args:         args,
			Env:          env,
			Cwd:          cwd,
			Capabilities: caps,
			Rlimits:      rlimits,
		},
		Mounts: mounts,
		Linux: &Linux{
			Namespaces: namespaces,
			Resources:  res,
		},
	}
}

// GenerateBundle creates the bundle directory, rootfs mount points, and writes config.json.
func GenerateBundle(spec sandbox.Spec, req sandbox.ExecutionRequest, bundleDir string) error {
	if strings.TrimSpace(bundleDir) == "" {
		return fmt.Errorf("bundleDir cannot be empty")
	}
	if err := os.MkdirAll(bundleDir, 0755); err != nil {
		return fmt.Errorf("failed to create bundle directory: %w", err)
	}

	rootfsDir := filepath.Join(bundleDir, "rootfs")
	if spec.Rootfs != "" && filepath.IsAbs(spec.Rootfs) {
		rootfsDir = spec.Rootfs
	}

	for _, sub := range []string{
		"proc",
		filepath.Join("dev", "pts"),
		filepath.Join("dev", "shm"),
		filepath.Join("dev", "mqueue"),
		"sys",
		"tmp",
		"workspace",
	} {
		if err := os.MkdirAll(filepath.Join(rootfsDir, sub), 0755); err != nil {
			return fmt.Errorf("failed to create rootfs mount point %q: %w", sub, err)
		}
	}

	ociSpec := BuildSpec(spec, req, bundleDir)
	raw, err := json.MarshalIndent(ociSpec, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to serialize OCI config.json: %w", err)
	}

	configPath := filepath.Join(bundleDir, "config.json")
	if err := os.WriteFile(configPath, raw, 0644); err != nil {
		return fmt.Errorf("failed to write config.json: %w", err)
	}
	return nil
}
