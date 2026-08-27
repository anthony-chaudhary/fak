package localapphelper

import "testing"

func TestBindingIsInstallAndSignatureScoped(t *testing.T) {
	host := HostIdentity{TeamID: "TEAM", BundleID: "com.example.JobApply", InstallID: "install-a", HelperBuild: "r1"}
	secret := []byte("0123456789abcdef0123456789abcdef")
	b, err := Bind(host, secret)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Authorize(host, secret); err != nil {
		t.Fatal(err)
	}
	changed := host
	changed.InstallID = "install-b"
	if err := b.Authorize(changed, secret); err == nil {
		t.Fatal("cross-install capability accepted")
	}
	changed = host
	changed.HelperBuild = "r0"
	if err := b.Authorize(changed, secret); err == nil {
		t.Fatal("downgraded helper accepted")
	}
	bad := append([]byte(nil), secret...)
	bad[0] ^= 1
	if err := b.Authorize(host, bad); err == nil {
		t.Fatal("wrong capability accepted")
	}
}

func TestBindingRejectsWeakSecretAndNeverStoresIt(t *testing.T) {
	host := HostIdentity{TeamID: "T", BundleID: "B", InstallID: "I", HelperBuild: "R"}
	if _, err := Bind(host, []byte("weak")); err == nil {
		t.Fatal("weak capability accepted")
	}
}
