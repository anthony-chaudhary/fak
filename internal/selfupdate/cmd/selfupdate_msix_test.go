//go:build windows

package selfupdatecmd

import (
	"errors"
	"os/exec"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/selfupdate"
)

func TestSelfUpdateMSIXDifferentialFallsBackToFull(t *testing.T) {
	original := selfUpdateMSIXRun
	t.Cleanup(func() { selfUpdateMSIXRun = original })
	var commands []*exec.Cmd
	selfUpdateMSIXRun = func(command *exec.Cmd) error {
		commands = append(commands, command)
		if len(commands) == 1 {
			return errors.New("differential unavailable")
		}
		return nil
	}
	plan := selfupdate.MSIXPlan{
		Action: selfupdate.MSIXUpdate, Delivery: "differential", URI: "https://updates.example/fak.appinstaller",
		PackageIdentity: "Fak.Package", Publisher: "CN=FAK", ArtifactIdentity: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	}
	outcome, err := executeSelfUpdateMSIX(plan, "https://updates.example/fak.msixbundle", "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
	if err != nil {
		t.Fatal(err)
	}
	if !outcome.FallbackUsed || outcome.SelectedDelivery != "full" {
		t.Fatalf("outcome = %+v", outcome)
	}
	if len(commands) != 2 || !strings.Contains(strings.Join(commands[0].Args, " "), ".appinstaller") || !strings.Contains(strings.Join(commands[1].Args, " "), ".msixbundle") || !strings.Contains(strings.Join(commands[1].Args, " "), strings.Repeat("b", 64)) || strings.Contains(strings.Join(commands[1].Args, " "), strings.Repeat("a", 64)) {
		t.Fatalf("fallback commands = %#v", commands)
	}
}

func TestSelfUpdateMSIXCommandIsWindowlessAndNonInteractive(t *testing.T) {
	command := selfUpdateMSIXCommand(selfupdate.MSIXPlan{
		Action: selfupdate.MSIXUpdate, Delivery: "differential", URI: "https://updates.example/fak.appinstaller",
		PackageIdentity: "Fak.Package", Publisher: "CN=FAK Release", ArtifactIdentity: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	})
	joined := strings.Join(command.Args, " ")
	if !strings.Contains(joined, "-NonInteractive") || command.SysProcAttr == nil || !command.SysProcAttr.HideWindow || command.SysProcAttr.CreationFlags&0x08000000 == 0 {
		t.Fatalf("command is not windowless/noninteractive: args=%q attr=%+v", joined, command.SysProcAttr)
	}
}

func TestSelfUpdateMSIXScriptWitnessesSignatureAndPublisher(t *testing.T) {
	plan := selfupdate.MSIXPlan{
		Action: selfupdate.MSIXUpdate, Delivery: "differential", URI: "https://updates.example/fak.appinstaller",
		PackageIdentity: "Fak.Package_1.2.3.0_x64__publisher", Publisher: "CN=FAK Release",
	}
	script := selfUpdateMSIXScript(plan)
	for _, witness := range []string{"Get-FileHash", "artifact digest mismatch", "AppInstaller.MainPackage", "CN=FAK Release", "Fak.Package", "-AppInstallerFile"} {
		if !strings.Contains(script, witness) {
			t.Fatalf("script missing %q: %s", witness, script)
		}
	}
}

func TestSelfUpdateMSIXScriptUsesFullFallbackRepairAndUninstall(t *testing.T) {
	tests := []struct {
		name string
		plan selfupdate.MSIXPlan
		want string
	}{
		{"full", selfupdate.MSIXPlan{Action: selfupdate.MSIXUpdate, Delivery: "full", URI: "https://updates.example/fak.msixbundle", Publisher: "CN=FAK", ArtifactIdentity: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}, "Get-AuthenticodeSignature"},
		{"repair", selfupdate.MSIXPlan{Action: selfupdate.MSIXRepair, Delivery: "repair", URI: "https://updates.example/fak.appinstaller", PackageIdentity: "Fak.Package", Publisher: "CN=FAK", ArtifactIdentity: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}, "-ForceUpdateFromAnyVersion"},
		{"uninstall", selfupdate.MSIXPlan{Action: selfupdate.MSIXUninstall, Delivery: "uninstall", PackageIdentity: "Fak.Package"}, "Remove-AppxPackage -AllUsers"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if script := selfUpdateMSIXScript(test.plan); !strings.Contains(script, test.want) {
				t.Fatalf("script missing %q: %s", test.want, script)
			}
		})
	}
}
