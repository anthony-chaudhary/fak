package selfupdate

import (
	"encoding/hex"
	"fmt"
	"net/url"
	"strings"
)

type Installer string

const (
	InstallerNative Installer = "native"
	InstallerMSIX   Installer = "msix"
)

func ResolveInstaller(cli, configured string) (Installer, error) {
	selected := strings.TrimSpace(cli)
	if selected == "" {
		selected = strings.TrimSpace(configured)
	}
	if selected == "" {
		return InstallerNative, nil
	}
	switch Installer(strings.ToLower(selected)) {
	case InstallerNative:
		return InstallerNative, nil
	case InstallerMSIX:
		return InstallerMSIX, nil
	default:
		return "", fmt.Errorf("unknown installer %q (want native or msix)", selected)
	}
}

type MSIXAction string

const (
	MSIXUpdate    MSIXAction = "update"
	MSIXRepair    MSIXAction = "repair"
	MSIXUninstall MSIXAction = "uninstall"
)

type MSIXRequest struct {
	Action           MSIXAction
	AppInstallerURI  string
	FullFallbackURI  string
	PackageIdentity  string
	AppIdentity      string
	ArtifactIdentity string
	SourceIdentity   string
	Publisher        string
	InstalledVersion string
	TargetVersion    string
	Differential     bool
	Offline          bool
	AllowDowngrade   bool
}

type MSIXPlan struct {
	Action           MSIXAction `json:"action"`
	Delivery         string     `json:"delivery"`
	URI              string     `json:"uri,omitempty"`
	PackageIdentity  string     `json:"package_identity"`
	AppIdentity      string     `json:"app_identity"`
	ArtifactIdentity string     `json:"artifact_identity,omitempty"`
	SourceIdentity   string     `json:"source_identity,omitempty"`
	Publisher        string     `json:"publisher,omitempty"`
	InstalledVersion string     `json:"installed_version,omitempty"`
	TargetVersion    string     `json:"target_version,omitempty"`
}

func PlanMSIX(request MSIXRequest) (MSIXPlan, error) {
	action := request.Action
	if action == "" {
		action = MSIXUpdate
	}
	plan := MSIXPlan{
		Action: action, PackageIdentity: strings.TrimSpace(request.PackageIdentity), AppIdentity: strings.TrimSpace(request.AppIdentity),
		ArtifactIdentity: strings.TrimSpace(request.ArtifactIdentity), SourceIdentity: strings.TrimSpace(request.SourceIdentity),
		Publisher: strings.TrimSpace(request.Publisher), InstalledVersion: strings.TrimSpace(request.InstalledVersion),
		TargetVersion: strings.TrimSpace(request.TargetVersion),
	}
	if plan.AppIdentity == "" {
		plan.AppIdentity = "fak"
	}
	if plan.PackageIdentity == "" {
		return MSIXPlan{}, fmt.Errorf("package identity is required")
	}
	if action == MSIXUninstall {
		plan.Delivery = "uninstall"
		return plan, nil
	}
	if action != MSIXUpdate && action != MSIXRepair {
		return MSIXPlan{}, fmt.Errorf("unsupported MSIX action %q", action)
	}
	if plan.Publisher == "" || plan.ArtifactIdentity == "" || plan.SourceIdentity == "" {
		return MSIXPlan{}, fmt.Errorf("signed provenance requires publisher, artifact identity, and source identity")
	}
	digest := strings.TrimPrefix(strings.ToLower(plan.ArtifactIdentity), "sha256:")
	if !strings.HasPrefix(strings.ToLower(plan.ArtifactIdentity), "sha256:") || len(digest) != 64 {
		return MSIXPlan{}, fmt.Errorf("artifact identity must be a sha256 digest")
	}
	if _, err := hex.DecodeString(digest); err != nil {
		return MSIXPlan{}, fmt.Errorf("artifact identity must be a sha256 digest")
	}
	identities := []string{plan.PackageIdentity, plan.AppIdentity, plan.ArtifactIdentity, plan.SourceIdentity}
	for left := range identities {
		for right := left + 1; right < len(identities); right++ {
			if identities[left] == identities[right] {
				return MSIXPlan{}, fmt.Errorf("package, app, artifact, and source identities must remain separate")
			}
		}
	}
	if fallbackURI := strings.TrimSpace(request.FullFallbackURI); fallbackURI != "" {
		if err := requireHTTPS(fallbackURI, "MSIX full fallback"); err != nil {
			return MSIXPlan{}, err
		}
	}
	if action == MSIXRepair {
		plan.Delivery = "repair"
		plan.URI = strings.TrimSpace(request.AppInstallerURI)
		return plan, requireHTTPS(plan.URI, "App Installer")
	}
	if plan.InstalledVersion != "" && plan.TargetVersion != "" {
		comparison, err := CompareReleaseVersions(plan.TargetVersion, plan.InstalledVersion)
		if err != nil {
			return MSIXPlan{}, fmt.Errorf("compare MSIX versions: %w", err)
		}
		if comparison < 0 && !request.AllowDowngrade {
			return MSIXPlan{}, fmt.Errorf("downgrade from %s to %s refused", plan.InstalledVersion, plan.TargetVersion)
		}
	}
	if request.Differential && !request.Offline {
		plan.Delivery = "differential"
		plan.URI = strings.TrimSpace(request.AppInstallerURI)
	} else {
		plan.Delivery = "full"
		plan.URI = strings.TrimSpace(request.FullFallbackURI)
		if plan.URI == "" {
			plan.URI = strings.TrimSpace(request.AppInstallerURI)
		}
	}
	return plan, requireHTTPS(plan.URI, "MSIX update")
}

func requireHTTPS(raw, label string) error {
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Scheme != "https" || parsed.Host == "" {
		return fmt.Errorf("%s URI must be absolute HTTPS", label)
	}
	return nil
}
