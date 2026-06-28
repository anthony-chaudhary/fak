package compute

import "fmt"

// loadpath.go — the weights load-path probe (#1062). A serve launched against a large GGUF
// on a NETWORK filesystem (NFS/CIFS) silently pays a ~50–100x time-to-ready tax vs a local
// NVMe/SSD: da33 measured 0.063 GB/s reading GLM-5.2 (434 GiB) off NFS — ~82 minutes to
// mmap — where the same weights staged on a local ext4 NVMe load at disk speed (minutes).
// The load-path's filesystem type is therefore a first-class performance dimension, not an
// afterthought. This classifies the storage backing a weights path so `serve` can warn the
// operator BEFORE eating the slow path, instead of silently loading for an hour.
//
// The classification (magic -> kind) is pure and lives here so it is testable without a real
// NFS mount; the per-OS syscall that reads the filesystem magic lives behind a build tag
// (loadpath_linux.go / loadpath_other.go), mirroring the disk_*.go split. It is fail-open:
// any path whose backing store cannot be classified yields LoadPathUnknown and an empty
// advisory, so the load proceeds exactly as before.

// LoadPathKind classifies the storage backing a weights path for load-throughput purposes.
type LoadPathKind int

const (
	// LoadPathUnknown — the backing filesystem could not be determined (unsupported OS, a
	// statfs error, or an unrecognized magic). Fail-open: callers treat it as "no warning".
	LoadPathUnknown LoadPathKind = iota
	// LoadPathLocal — a recognized local-disk filesystem (ext4, xfs, btrfs, tmpfs, …). Loads
	// at device speed; no load-path tax.
	LoadPathLocal
	// LoadPathNetwork — a recognized network/cluster filesystem (NFS, CIFS/SMB, plus the
	// HPC/lab class: Lustre, Ceph, GFS2, OCFS2). Reads at the network's mercy; the source of
	// the ~50–100x load-time tax on a large model. The HPC class matters here because #1062's
	// host is a lab box, and lab/HPC weight stores ride Lustre/Ceph far more than plain NFS.
	LoadPathNetwork
)

// Linux statfs f_type magic numbers (see man statfs(2) / linux/magic.h). Listed here as the
// raw constants so the classifier is self-contained and unit-testable on any platform.
const (
	fsMagicNFS    int64 = 0x6969     // NFS_SUPER_MAGIC      — network
	fsMagicSMB    int64 = 0x517B     // SMB_SUPER_MAGIC      — network
	fsMagicCIFS   int64 = 0xFF534D42 // CIFS_MAGIC_NUMBER    — network
	fsMagicCEPH   int64 = 0x00C36400 // CEPH_SUPER_MAGIC     — network (distributed; common HPC/lab weight store)
	fsMagicLUSTRE int64 = 0x0BD00BD0 // LL_SUPER_MAGIC       — network (Lustre; the canonical HPC scratch fs)
	fsMagicGFS2   int64 = 0x01161970 // GFS2_MAGIC           — network (shared-disk cluster fs)
	fsMagicOCFS2  int64 = 0x7461636F // OCFS2_SUPER_MAGIC    — network (shared-disk cluster fs)
	fsMagicEXT    int64 = 0xEF53     // EXT2/3/4_SUPER_MAGIC — local
	fsMagicXFS    int64 = 0x58465342 // XFS_SUPER_MAGIC      — local
	fsMagicBTRFS  int64 = 0x9123683E // BTRFS_SUPER_MAGIC    — local
	fsMagicTMPFS  int64 = 0x01021994 // TMPFS_MAGIC          — local (RAM-backed)
	fsMagicZFS    int64 = 0x2FC12FC1 // ZFS_SUPER_MAGIC      — local
)

// LoadPathInfo is the result of probing a weights path's backing filesystem.
type LoadPathInfo struct {
	Kind   LoadPathKind // network / local / unknown
	FSName string       // human label for the recognized filesystem ("nfs", "ext", …); "" when unknown
	Magic  int64        // the raw f_type magic on Linux, 0 elsewhere — for the operator log
	Known  bool         // true when the OS reported a magic (even if the magic itself is unrecognized)
}

// classifyFSMagic maps a Linux statfs f_type magic to a LoadPathKind + a human label. Pure
// (no syscall) so the network-vs-local decision is unit-testable without a real mount. An
// unrecognized magic returns LoadPathUnknown so a new/exotic filesystem fails open (no warn).
func classifyFSMagic(magic int64) (LoadPathKind, string) {
	switch magic {
	case fsMagicNFS:
		return LoadPathNetwork, "nfs"
	case fsMagicSMB:
		return LoadPathNetwork, "smb"
	case fsMagicCIFS:
		return LoadPathNetwork, "cifs"
	case fsMagicCEPH:
		return LoadPathNetwork, "ceph"
	case fsMagicLUSTRE:
		return LoadPathNetwork, "lustre"
	case fsMagicGFS2:
		return LoadPathNetwork, "gfs2"
	case fsMagicOCFS2:
		return LoadPathNetwork, "ocfs2"
	case fsMagicEXT:
		return LoadPathLocal, "ext"
	case fsMagicXFS:
		return LoadPathLocal, "xfs"
	case fsMagicBTRFS:
		return LoadPathLocal, "btrfs"
	case fsMagicTMPFS:
		return LoadPathLocal, "tmpfs"
	case fsMagicZFS:
		return LoadPathLocal, "zfs"
	default:
		return LoadPathUnknown, ""
	}
}

// ProbeLoadPath classifies the filesystem backing a weights path. It is fail-open: on an OS
// that cannot report the filesystem magic (anything but Linux today) or on a statfs error it
// returns {Kind: LoadPathUnknown, Known: false}, and the caller proceeds without a warning.
func ProbeLoadPath(path string) LoadPathInfo {
	magic, known := loadPathFSMagic(path)
	if !known {
		return LoadPathInfo{Kind: LoadPathUnknown, Known: false}
	}
	kind, name := classifyFSMagic(magic)
	return LoadPathInfo{Kind: kind, FSName: name, Magic: magic, Known: true}
}

// WarnSlowLoadPath returns a one-line operator advisory when the weights load path is on a
// recognized NETWORK filesystem — the source of the ~50–100x time-to-ready tax — and "" for
// a local or unclassified path. The message is fak-owned text (no upstream content) naming
// the filesystem and the remedy (stage the GGUF on a local NVMe/SSD mount). It only WARNS:
// a network weights path is slow, not wrong, so the load still proceeds (#1062 item 1, warn
// arm — the refuse-and-suggest-a-faster-local-mount arm is the tracked follow-on).
func WarnSlowLoadPath(info LoadPathInfo) string {
	if info.Kind != LoadPathNetwork {
		return ""
	}
	return fmt.Sprintf("WARNING: weights are on a network filesystem (%s) — large-model load "+
		"reads at network speed and can take ~50-100x longer than a local disk (da33 saw "+
		"~82 min for GLM-5.2 over NFS). Stage the GGUF on a local NVMe/SSD mount for a "+
		"minutes-not-hours time-to-ready (#1062).", info.FSName)
}
