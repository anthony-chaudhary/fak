//go:build !windows

package selfupdatecmd

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/selfupdate"
)

type selfUpdateMSIXOptions struct {
	CLIInstaller, ConfigInstaller, AppInstallerURI, FullFallbackURI                    string
	PackageIdentity, Publisher, ArtifactIdentity, FullArtifactIdentity, SourceIdentity string
	InstalledVersion, TargetVersion                                                    string
	Repair, Uninstall, Check, AllowDowngrade, Offline, JSON                            bool
}

func runSelfUpdateMSIX(options selfUpdateMSIXOptions) (bool, error) {
	installer, err := selfupdate.ResolveInstaller(options.CLIInstaller, options.ConfigInstaller)
	if err != nil {
		return true, err
	}
	if installer == selfupdate.InstallerNative {
		return false, nil
	}
	return true, fmt.Errorf("MSIX installer is available only on Windows")
}
