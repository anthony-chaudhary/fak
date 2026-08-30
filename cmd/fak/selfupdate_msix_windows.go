//go:build windows

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"syscall"

	"github.com/anthony-chaudhary/fak/internal/selfupdate"
)

type selfUpdateMSIXOptions struct {
	CLIInstaller, ConfigInstaller, AppInstallerURI, FullFallbackURI                    string
	PackageIdentity, Publisher, ArtifactIdentity, FullArtifactIdentity, SourceIdentity string
	InstalledVersion, TargetVersion                                                    string
	Repair, Uninstall, Check, AllowDowngrade, Offline, JSON                            bool
}

var selfUpdateMSIXRun = func(command *exec.Cmd) error { return command.Run() }

func runSelfUpdateMSIX(options selfUpdateMSIXOptions) (bool, error) {
	installer, err := selfupdate.ResolveInstaller(options.CLIInstaller, options.ConfigInstaller)
	if err != nil {
		return true, err
	}
	if installer == selfupdate.InstallerNative {
		return false, nil
	}
	if options.Check {
		return true, fmt.Errorf("--check is not available for the MSIX adapter; no package action was run")
	}
	if options.Repair && options.Uninstall {
		return true, fmt.Errorf("repair and uninstall are mutually exclusive")
	}
	action := selfupdate.MSIXUpdate
	if options.Repair {
		action = selfupdate.MSIXRepair
	} else if options.Uninstall {
		action = selfupdate.MSIXUninstall
	}
	plan, err := selfupdate.PlanMSIX(selfupdate.MSIXRequest{
		Action: action, AppInstallerURI: options.AppInstallerURI, FullFallbackURI: options.FullFallbackURI,
		PackageIdentity: options.PackageIdentity, AppIdentity: "fak", Publisher: options.Publisher,
		ArtifactIdentity: options.ArtifactIdentity, SourceIdentity: options.SourceIdentity,
		InstalledVersion: options.InstalledVersion, TargetVersion: options.TargetVersion,
		Differential: true, Offline: options.Offline, AllowDowngrade: options.AllowDowngrade,
	})
	if err != nil {
		return true, err
	}
	outcome, err := executeSelfUpdateMSIX(plan, options.FullFallbackURI, options.FullArtifactIdentity)
	if err != nil {
		return true, err
	}
	if options.JSON {
		return true, json.NewEncoder(os.Stdout).Encode(struct {
			Schema  string                `json:"schema"`
			Engine  string                `json:"engine"`
			Plan    selfupdate.MSIXPlan   `json:"plan"`
			Outcome selfUpdateMSIXOutcome `json:"outcome"`
		}{Schema: "fak-self-update-msix/1", Engine: "msix", Plan: plan, Outcome: outcome})
	}
	fmt.Fprintf(os.Stdout, "self-update: MSIX %s complete (%s; fallback=%t)\n", plan.Action, outcome.SelectedDelivery, outcome.FallbackUsed)
	return true, nil
}

type selfUpdateMSIXOutcome struct {
	SelectedDelivery string `json:"selected_delivery"`
	FallbackUsed     bool   `json:"fallback_used"`
}

func executeSelfUpdateMSIX(plan selfupdate.MSIXPlan, fullFallbackURI, fullArtifactIdentity string) (selfUpdateMSIXOutcome, error) {
	command := selfUpdateMSIXCommand(plan)
	command.Stdout = os.Stdout
	command.Stderr = os.Stderr
	if err := selfUpdateMSIXRun(command); err == nil {
		return selfUpdateMSIXOutcome{SelectedDelivery: plan.Delivery}, nil
	} else if plan.Delivery != "differential" || strings.TrimSpace(fullFallbackURI) == "" {
		return selfUpdateMSIXOutcome{}, fmt.Errorf("%s failed: %w", plan.Action, err)
	}
	if !validMSIXDigest(fullArtifactIdentity) {
		return selfUpdateMSIXOutcome{}, fmt.Errorf("full fallback requires --msix-full-artifact-digest sha256:hex")
	}
	fallback := plan
	fallback.Delivery = "full"
	fallback.URI = strings.TrimSpace(fullFallbackURI)
	fallback.ArtifactIdentity = strings.TrimSpace(fullArtifactIdentity)
	command = selfUpdateMSIXCommand(fallback)
	command.Stdout = os.Stdout
	command.Stderr = os.Stderr
	if err := selfUpdateMSIXRun(command); err != nil {
		return selfUpdateMSIXOutcome{}, fmt.Errorf("%s differential and full fallback failed: %w", plan.Action, err)
	}
	return selfUpdateMSIXOutcome{SelectedDelivery: "full", FallbackUsed: true}, nil
}

func validMSIXDigest(value string) bool {
	value = strings.TrimSpace(value)
	if !strings.HasPrefix(value, "sha256:") || len(value) != len("sha256:")+64 {
		return false
	}
	for _, r := range value[len("sha256:"):] {
		if !strings.ContainsRune("0123456789abcdefABCDEF", r) {
			return false
		}
	}
	return true
}
func selfUpdateMSIXCommand(plan selfupdate.MSIXPlan) *exec.Cmd {
	script := selfUpdateMSIXScript(plan)
	command := exec.Command("powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script)
	command.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x08000000}
	return command
}

func selfUpdateMSIXScript(plan selfupdate.MSIXPlan) string {
	packageIdentity := powershellLiteral(plan.PackageIdentity)
	if plan.Action == selfupdate.MSIXUninstall {
		return "$ErrorActionPreference='Stop'; $p=Get-AppxPackage -AllUsers -Name '" + packageIdentity + "'; if(-not $p){throw 'MSIX package not installed'}; $p | Remove-AppxPackage -AllUsers"
	}
	uri := powershellLiteral(plan.URI)
	publisher := powershellLiteral(plan.Publisher)
	digest := powershellLiteral(strings.TrimPrefix(strings.ToLower(plan.ArtifactIdentity), "sha256:"))
	download := "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; $u='" + uri + "'; $x=Join-Path ([IO.Path]::GetTempPath()) ('fak-'+[guid]::NewGuid().ToString('N')+'" + selfUpdateMSIXSuffix(plan) + "'); Invoke-WebRequest -UseBasicParsing -Uri $u -OutFile $x; try { "
	verify := "$hash=(Get-FileHash -Algorithm SHA256 -LiteralPath $x).Hash.ToLowerInvariant(); if($hash -ne '" + digest + "'){throw 'MSIX artifact digest mismatch'}; "
	if plan.Delivery == "full" {
		verify += "$sig=Get-AuthenticodeSignature -LiteralPath $x; if($sig.Status -ne 'Valid'){throw 'MSIX signature is not valid'}; if($sig.SignerCertificate.Subject -ne '" + publisher + "'){throw 'MSIX publisher mismatch'}; "
	} else {
		verify += "[xml]$manifest=Get-Content -LiteralPath $x; $identity=$manifest.AppInstaller.MainPackage; if(-not $identity){$identity=$manifest.AppInstaller.MainBundle}; if(-not $identity){throw 'App Installer package identity missing'}; if($identity.Publisher -ne '" + publisher + "'){throw 'MSIX publisher mismatch'}; if($identity.Name -ne '" + packageIdentity + "'){throw 'MSIX package identity mismatch'}; "
	}
	install := "Add-AppxPackage -Path $x -ForceApplicationShutdown"
	if plan.Delivery == "differential" || plan.Delivery == "repair" {
		install = "Add-AppxPackage -AppInstallerFile $x -ForceApplicationShutdown"
	}
	if plan.Action == selfupdate.MSIXRepair {
		install += " -ForceUpdateFromAnyVersion"
	}
	return download + verify + install + " } finally { Remove-Item -LiteralPath $x -Force -ErrorAction SilentlyContinue }"
}

func selfUpdateMSIXSuffix(plan selfupdate.MSIXPlan) string {
	if plan.Delivery == "differential" || plan.Delivery == "repair" {
		return ".appinstaller"
	}
	return ".msixbundle"
}

func powershellLiteral(value string) string {
	return strings.ReplaceAll(value, "'", "''")
}
