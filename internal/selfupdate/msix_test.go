package selfupdate

import (
	"strings"
	"testing"
)

func TestResolveInstallerPrecedence(t *testing.T) {
	tests := []struct {
		name, cli, configured string
		want                  Installer
	}{
		{"native default", "", "", InstallerNative},
		{"config opt in", "", "msix", InstallerMSIX},
		{"cli overrides config", "native", "msix", InstallerNative},
		{"cli selects msix", "MSIX", "native", InstallerMSIX},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := ResolveInstaller(test.cli, test.configured)
			if err != nil || got != test.want {
				t.Fatalf("ResolveInstaller(%q, %q) = %q, %v; want %q", test.cli, test.configured, got, err, test.want)
			}
		})
	}
}

func TestPlanMSIXSeparatesIdentitiesAndRequiresSignedProvenance(t *testing.T) {
	request := validMSIXRequest()
	plan, err := PlanMSIX(request)
	if err != nil {
		t.Fatal(err)
	}
	if plan.PackageIdentity == plan.AppIdentity || plan.PackageIdentity == plan.ArtifactIdentity || plan.PackageIdentity == plan.SourceIdentity || plan.ArtifactIdentity == plan.SourceIdentity {
		t.Fatalf("identities conflated: %+v", plan)
	}
	request.Publisher = ""
	if _, err := PlanMSIX(request); err == nil || !strings.Contains(err.Error(), "signed provenance") {
		t.Fatalf("missing publisher error = %v", err)
	}
	request = validMSIXRequest()
	request.ArtifactIdentity = "sha256:" + strings.Repeat("z", 64)
	if _, err := PlanMSIX(request); err == nil || !strings.Contains(err.Error(), "sha256") {
		t.Fatalf("non-hex digest error = %v", err)
	}
	request = validMSIXRequest()
	request.AppIdentity = request.SourceIdentity
	if _, err := PlanMSIX(request); err == nil || !strings.Contains(err.Error(), "separate") {
		t.Fatalf("conflated app/source error = %v", err)
	}
}

func TestPlanMSIXDeliveryFallbackRepairDowngradeAndUninstall(t *testing.T) {
	t.Run("differential", func(t *testing.T) {
		plan, err := PlanMSIX(validMSIXRequest())
		if err != nil || plan.Delivery != "differential" || plan.URI != "https://updates.example/fak.appinstaller" {
			t.Fatalf("plan = %+v, %v", plan, err)
		}
	})
	t.Run("full fallback offline", func(t *testing.T) {
		request := validMSIXRequest()
		request.Offline = true
		plan, err := PlanMSIX(request)
		if err != nil || plan.Delivery != "full" || plan.URI != "https://updates.example/fak.msixbundle" {
			t.Fatalf("plan = %+v, %v", plan, err)
		}
	})
	t.Run("fallback must be HTTPS", func(t *testing.T) {
		request := validMSIXRequest()
		request.FullFallbackURI = "http://updates.example/fak.msixbundle"
		if _, err := PlanMSIX(request); err == nil || !strings.Contains(err.Error(), "HTTPS") {
			t.Fatalf("fallback URI error = %v", err)
		}
	})
	t.Run("repair", func(t *testing.T) {
		request := validMSIXRequest()
		request.Action = MSIXRepair
		plan, err := PlanMSIX(request)
		if err != nil || plan.Delivery != "repair" {
			t.Fatalf("plan = %+v, %v", plan, err)
		}
	})
	t.Run("downgrade refused", func(t *testing.T) {
		request := validMSIXRequest()
		request.InstalledVersion = "2.0.0"
		request.TargetVersion = "1.9.0"
		if _, err := PlanMSIX(request); err == nil || !strings.Contains(err.Error(), "downgrade") {
			t.Fatalf("downgrade error = %v", err)
		}
		request.AllowDowngrade = true
		if _, err := PlanMSIX(request); err != nil {
			t.Fatalf("explicit downgrade refused: %v", err)
		}
	})
	t.Run("uninstall", func(t *testing.T) {
		request := validMSIXRequest()
		request.Action = MSIXUninstall
		request.AppInstallerURI = ""
		request.Publisher = ""
		request.ArtifactIdentity = ""
		request.SourceIdentity = ""
		plan, err := PlanMSIX(request)
		if err != nil || plan.Delivery != "uninstall" || plan.PackageIdentity == "" {
			t.Fatalf("plan = %+v, %v", plan, err)
		}
	})
}

func validMSIXRequest() MSIXRequest {
	return MSIXRequest{
		AppInstallerURI: "https://updates.example/fak.appinstaller", FullFallbackURI: "https://updates.example/fak.msixbundle",
		PackageIdentity: "Fak.Package_1.2.3.0_x64__publisher", AppIdentity: "fak",
		ArtifactIdentity: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", SourceIdentity: "0123456789abcdef", Publisher: "CN=FAK Release",
		InstalledVersion: "1.2.2", TargetVersion: "1.2.3", Differential: true,
	}
}
