//go:build windows

package main

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const dispatchHelperDescendantRole = "FAK_DISPATCH_HELPER_DESCENDANT_ROLE"

func TestConfigureDispatchHelperCommandAllocatesHiddenConsole(t *testing.T) {
	cmd := exec.Command("cmd.exe", "/c", "exit", "0")
	configureDispatchHelperCommand(cmd)

	if cmd.SysProcAttr == nil {
		t.Fatal("dispatch helper has no Windows process attributes")
	}
	const (
		detachedProcess = 0x00000008
		createNoWindow  = 0x08000000
	)
	flags := cmd.SysProcAttr.CreationFlags
	if flags&createNoWindow == 0 {
		t.Fatalf("dispatch helper flags %#x omit CREATE_NO_WINDOW; descendant consoles can become visible", flags)
	}
	if flags&detachedProcess != 0 {
		t.Fatalf("dispatch helper flags %#x retain DETACHED_PROCESS; descendants cannot inherit the hidden console", flags)
	}
}

func TestConfigureDispatchHelperCommandKeepsDescendantWindowsHidden(t *testing.T) {
	role := os.Getenv(dispatchHelperDescendantRole)
	switch role {
	case "descendant":
		fmt.Printf("descendant=%d,%d\n", dispatchHelperConsoleProcessCount(), dispatchHelperConsoleWindow())
		return
	case "helper":
		fmt.Printf("helper=%d,%d\n", dispatchHelperConsoleProcessCount(), dispatchHelperConsoleWindow())
		cmd := exec.Command(os.Args[0], "-test.run=^TestConfigureDispatchHelperCommandKeepsDescendantWindowsHidden$")
		cmd.Env = dispatchHelperRoleEnv("descendant")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("descendant probe: %v: %s", err, out)
		}
		_, _ = os.Stdout.Write(out)
		return
	}

	cmd := exec.Command(os.Args[0], "-test.run=^TestConfigureDispatchHelperCommandKeepsDescendantWindowsHidden$")
	cmd.Env = dispatchHelperRoleEnv("helper")
	configureDispatchHelperCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("configured helper probe: %v: %s", err, out)
	}

	for _, wantRole := range []string{"helper", "descendant"} {
		count, hwnd := dispatchHelperConsoleProbe(t, string(out), wantRole)
		if count == 0 {
			t.Errorf("%s owns no hidden console; an ordinary console descendant can allocate a visible one", wantRole)
		}
		if hwnd != 0 {
			t.Errorf("%s console HWND=%d, want 0 (hidden)", wantRole, hwnd)
		}
	}
}

func dispatchHelperRoleEnv(role string) []string {
	prefix := dispatchHelperDescendantRole + "="
	env := make([]string, 0, len(os.Environ())+1)
	for _, item := range os.Environ() {
		if !strings.HasPrefix(item, prefix) {
			env = append(env, item)
		}
	}
	return append(env, prefix+role)
}

func dispatchHelperConsoleProbe(t *testing.T, output, role string) (uint32, uintptr) {
	t.Helper()
	prefix := role + "="
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, prefix) {
			continue
		}
		parts := strings.Split(strings.TrimPrefix(line, prefix), ",")
		if len(parts) != 2 {
			break
		}
		count, countErr := strconv.ParseUint(parts[0], 10, 32)
		hwnd, hwndErr := strconv.ParseUint(parts[1], 10, 64)
		if countErr == nil && hwndErr == nil {
			return uint32(count), uintptr(hwnd)
		}
		break
	}
	t.Fatalf("probe output has no valid %s row: %q", role, output)
	return 0, 0
}

func TestConfigureDispatchWorkerSpawnKeepsCodexDescendantsHidden(t *testing.T) {
	t.Run("codex inherits one hidden console", func(t *testing.T) {
		cmd := exec.Command("cmd.exe", "/c", "exit", "0")
		configureDispatchSpawn(cmd)
		configureDispatchWorkerConsole(cmd, "codex")
		if cmd.SysProcAttr == nil || !cmd.SysProcAttr.HideWindow {
			t.Fatalf("Codex worker attributes = %#v, want hidden window", cmd.SysProcAttr)
		}
		flags := cmd.SysProcAttr.CreationFlags
		if flags&windowgate.CreateNoWindow == 0 || flags&windowgate.DetachedProcess != 0 {
			t.Fatalf("Codex worker flags = %#x, want CREATE_NO_WINDOW without DETACHED_PROCESS", flags)
		}
	})

	t.Run("other backends remain detached", func(t *testing.T) {
		cmd := exec.Command("cmd.exe", "/c", "exit", "0")
		configureDispatchSpawn(cmd)
		configureDispatchWorkerConsole(cmd, "claude")
		flags := cmd.SysProcAttr.CreationFlags
		if flags&windowgate.DetachedProcess == 0 || flags&windowgate.CreateNoWindow != 0 {
			t.Fatalf("non-Codex worker flags = %#x, want DETACHED_PROCESS without CREATE_NO_WINDOW", flags)
		}
	})
}
