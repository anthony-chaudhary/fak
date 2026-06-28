package compute

import (
	"os"
	"strings"
	"testing"
)

// TestClassifyFSMagic pins the network-vs-local decision for the load-path probe (#1062): an
// NFS/CIFS/SMB magic must classify as network (the ~50–100x load-time tax), the local-disk
// magics as local, and any unrecognized magic must fail open as unknown (never a false
// network warn on an exotic local fs).
func TestClassifyFSMagic(t *testing.T) {
	cases := []struct {
		name     string
		magic    int64
		wantKind LoadPathKind
		wantFS   string
	}{
		{"nfs", fsMagicNFS, LoadPathNetwork, "nfs"},
		{"smb", fsMagicSMB, LoadPathNetwork, "smb"},
		{"cifs", fsMagicCIFS, LoadPathNetwork, "cifs"},
		{"ext4", fsMagicEXT, LoadPathLocal, "ext"},
		{"xfs", fsMagicXFS, LoadPathLocal, "xfs"},
		{"btrfs", fsMagicBTRFS, LoadPathLocal, "btrfs"},
		{"tmpfs", fsMagicTMPFS, LoadPathLocal, "tmpfs"},
		{"zfs", fsMagicZFS, LoadPathLocal, "zfs"},
		{"unrecognized-fails-open", 0x12345678, LoadPathUnknown, ""},
		{"zero", 0, LoadPathUnknown, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotKind, gotFS := classifyFSMagic(tc.magic)
			if gotKind != tc.wantKind {
				t.Fatalf("classifyFSMagic(%#x) kind = %d, want %d", tc.magic, gotKind, tc.wantKind)
			}
			if gotFS != tc.wantFS {
				t.Fatalf("classifyFSMagic(%#x) fs = %q, want %q", tc.magic, gotFS, tc.wantFS)
			}
		})
	}
}

// TestWarnSlowLoadPath asserts the advisory fires ONLY for a network path and names the
// filesystem + the issue, and stays empty (fail-open) for local and unknown — so a serve on
// a local NVMe or an unclassifiable mount is never spuriously warned.
func TestWarnSlowLoadPath(t *testing.T) {
	if w := WarnSlowLoadPath(LoadPathInfo{Kind: LoadPathNetwork, FSName: "nfs", Magic: fsMagicNFS, Known: true}); w == "" {
		t.Fatal("WarnSlowLoadPath(network) = empty, want a non-empty advisory")
	} else {
		if !strings.Contains(w, "nfs") {
			t.Errorf("network advisory %q does not name the filesystem", w)
		}
		if !strings.Contains(w, "#1062") {
			t.Errorf("network advisory %q does not cite the tracking issue", w)
		}
	}
	if w := WarnSlowLoadPath(LoadPathInfo{Kind: LoadPathLocal, FSName: "ext", Known: true}); w != "" {
		t.Errorf("WarnSlowLoadPath(local) = %q, want empty", w)
	}
	if w := WarnSlowLoadPath(LoadPathInfo{Kind: LoadPathUnknown}); w != "" {
		t.Errorf("WarnSlowLoadPath(unknown) = %q, want empty", w)
	}
}

// TestProbeLoadPathLocalNeverWarns is the cross-platform safety invariant: probing a real
// local temp dir must NEVER classify as network (so it never spuriously warns). It may come
// back Local (Linux ext/xfs/…) or Unknown (non-Linux fail-open, or an unlisted local magic
// like overlayfs/9p under WSL) — both are acceptable; Network is not.
func TestProbeLoadPathLocalNeverWarns(t *testing.T) {
	dir := t.TempDir()
	info := ProbeLoadPath(dir)
	if info.Kind == LoadPathNetwork {
		t.Fatalf("ProbeLoadPath(%q) classified a local temp dir as network (magic=%#x, fs=%q)", dir, info.Magic, info.FSName)
	}
	if w := WarnSlowLoadPath(info); w != "" {
		t.Fatalf("local temp dir produced a slow-load-path warning: %q", w)
	}
}

// TestProbeLoadPathMissingFailsOpen pins fail-open on a path that cannot be statfs'd: a
// non-existent path must yield unknown/unknown and no warning, never a panic or a false
// network verdict.
func TestProbeLoadPathMissingFailsOpen(t *testing.T) {
	missing := t.TempDir() + string(os.PathSeparator) + "does-not-exist-1062"
	info := ProbeLoadPath(missing)
	if info.Kind == LoadPathNetwork {
		t.Fatalf("ProbeLoadPath(missing) = network, want unknown/local fail-open")
	}
	if w := WarnSlowLoadPath(info); w != "" {
		t.Fatalf("missing path produced a warning: %q", w)
	}
}
