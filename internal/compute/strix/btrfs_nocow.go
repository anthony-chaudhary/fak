package strix

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
)

// Linux FS ioctl constants for file attributes (NoCoW).
const (
	// FSIocGetFlags: _IOR('f', 1, long) = 0x80086601 on 64-bit Linux.
	FSIocGetFlags uintptr = 0x80086601

	// FSIocSetFlags: _IOW('f', 2, long) = 0x40086602 on 64-bit Linux.
	FSIocSetFlags uintptr = 0x40086602

	// FSNoCoWFlag is the Copy-on-Write disabled flag bit (0x00800000 = bit 23), matching chattr +C.
	FSNoCoWFlag int64 = 0x00800000

	// BtrfsSuperMagic is the Linux statfs magic number for Btrfs filesystems (0x9123683e).
	BtrfsSuperMagic uint32 = 0x9123683e
)

var (
	// ErrNoCoWFailed is returned when NoCoW cannot be enabled on a directory.
	ErrNoCoWFailed = errors.New("strix/loader: failed to apply NoCoW attribute")

	// ErrPathNotFound is returned when the target path does not exist.
	ErrPathNotFound = errors.New("strix/loader: target path not found")
)

// CommandRunner executes an external command with context.
type CommandRunner func(ctx context.Context, name string, args ...string) ([]byte, error)

// DefaultCommandRunner executes commands via os/exec.
func DefaultCommandRunner(ctx context.Context, name string, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	return cmd.CombinedOutput()
}

// BtrfsNoCoWEnforcer inspects and enforces the Btrfs NoCoW (FS_NOCOW_FL / +C) attribute
// on model directories, preventing Copy-on-Write metadata fragmentation during weight mapping.
type BtrfsNoCoWEnforcer struct {
	mu            sync.RWMutex
	commandRunner CommandRunner
	ioctlGetter   func(fd uintptr) (int64, error)
	ioctlSetter   func(fd uintptr, flags int64) error
	statfsChecker func(path string) (bool, error)
}

// NewBtrfsNoCoWEnforcer constructs a new NoCoW enforcer with system defaults.
func NewBtrfsNoCoWEnforcer() *BtrfsNoCoWEnforcer {
	return &BtrfsNoCoWEnforcer{
		commandRunner: DefaultCommandRunner,
		ioctlGetter:   sysIoctlGetFlags,
		ioctlSetter:   sysIoctlSetFlags,
		statfsChecker: sysCheckBtrfs,
	}
}

// CheckNoCoW checks if the path has the NoCoW (FS_NOCOW_FL) attribute set.
func (e *BtrfsNoCoWEnforcer) CheckNoCoW(ctx context.Context, path string) (bool, error) {
	cleanPath := filepath.Clean(path)
	if _, err := os.Stat(cleanPath); err != nil {
		if os.IsNotExist(err) {
			return false, fmt.Errorf("%w: %s", ErrPathNotFound, cleanPath)
		}
		return false, err
	}

	// 1. Attempt fast native ioctl read if available.
	if e.ioctlGetter != nil {
		f, err := os.Open(cleanPath)
		if err == nil {
			flags, ioctlErr := e.ioctlGetter(f.Fd())
			_ = f.Close()
			if ioctlErr == nil {
				return (flags & FSNoCoWFlag) != 0, nil
			}
		}
	}

	// 2. Fall back to lsattr command execution.
	runner := e.commandRunner
	if runner == nil {
		runner = DefaultCommandRunner
	}

	out, err := runner(ctx, "lsattr", "-d", cleanPath)
	if err == nil {
		return parseLsattrNoCoW(string(out)), nil
	}

	return false, fmt.Errorf("check nocow attribute on %s: %w", cleanPath, err)
}

// EnsureNoCoW ensures that the given directory or file has the NoCoW attribute enabled.
// If it is not set, it attempts to set it via ioctl (FS_IOC_SETFLAGS) or chattr +C.
func (e *BtrfsNoCoWEnforcer) EnsureNoCoW(ctx context.Context, path string) error {
	cleanPath := filepath.Clean(path)
	active, err := e.CheckNoCoW(ctx, cleanPath)
	if err == nil && active {
		return nil // Already active
	}

	// Attempt to set via ioctl
	if e.ioctlSetter != nil && e.ioctlGetter != nil {
		f, openErr := os.OpenFile(cleanPath, os.O_RDONLY, 0)
		if openErr == nil {
			curFlags, getErr := e.ioctlGetter(f.Fd())
			if getErr == nil {
				newFlags := curFlags | FSNoCoWFlag
				setErr := e.ioctlSetter(f.Fd(), newFlags)
				_ = f.Close()
				if setErr == nil {
					return nil
				}
			} else {
				_ = f.Close()
			}
		}
	}

	// Fallback to chattr +C
	runner := e.commandRunner
	if runner == nil {
		runner = DefaultCommandRunner
	}

	out, chattrErr := runner(ctx, "chattr", "+C", cleanPath)
	if chattrErr != nil {
		return fmt.Errorf("%w on %s: %v (%s)", ErrNoCoWFailed, cleanPath, chattrErr, strings.TrimSpace(string(out)))
	}

	// Verify it was applied
	verified, verifyErr := e.CheckNoCoW(ctx, cleanPath)
	if verifyErr == nil && !verified {
		return fmt.Errorf("%w: verification after chattr +C showed flag unset on %s", ErrNoCoWFailed, cleanPath)
	}

	return nil
}

// IsBtrfs reports whether the given path resides on a Btrfs filesystem.
func (e *BtrfsNoCoWEnforcer) IsBtrfs(ctx context.Context, path string) (bool, error) {
	cleanPath := filepath.Clean(path)
	if e.statfsChecker != nil {
		isBtrfs, err := e.statfsChecker(cleanPath)
		if err == nil {
			return isBtrfs, nil
		}
	}

	// Fallback to findmnt or df command
	runner := e.commandRunner
	if runner == nil {
		runner = DefaultCommandRunner
	}

	out, err := runner(ctx, "findmnt", "-n", "-o", "FSTYPE", "-T", cleanPath)
	if err == nil {
		fstype := strings.ToLower(strings.TrimSpace(string(out)))
		return strings.Contains(fstype, "btrfs"), nil
	}

	out, err = runner(ctx, "df", "-T", cleanPath)
	if err == nil {
		return bytes.Contains(bytes.ToLower(out), []byte("btrfs")), nil
	}

	return false, nil
}

// parseLsattrNoCoW inspects output from `lsattr -d <path>` for the 'C' attribute flag.
// Example output: "---------------C------ /var/lib/fak/models"
func parseLsattrNoCoW(output string) bool {
	fields := strings.Fields(output)
	if len(fields) == 0 {
		return false
	}
	// The first column contains the attribute flags
	attrString := fields[0]
	return strings.ContainsRune(attrString, 'C')
}
