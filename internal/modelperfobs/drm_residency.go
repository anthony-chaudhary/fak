// Copyright (c) 2026 fak contributors
// SPDX-License-Identifier: MIT
//
// Adapted from sybil-solutions/inference-optimisation-skill
// (skills/inference-optimisation/scripts/mem.sh:21@fda968e7):
// native Go Linux DRM fdinfo resident-memory collector and watchdog for unified
// memory GPU allocations invisible to cgroups.
package modelperfobs

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// DRMFDInfo holds parsed memory-footprint fields from one /proc/<pid>/fdinfo/<fd> file.
type DRMFDInfo struct {
	Driver        string
	ClientID      string
	ResidentBytes int64
	GTTBytes      int64
	VRAMBytes     int64
}

// DRMResidencySnapshot aggregates unified/device DRM residency across a process's
// open DRM clients and evaluates an optional byte threshold watchdog.
type DRMResidencySnapshot struct {
	PID                int
	TotalResidentBytes int64
	GTTBytes           int64
	VRAMBytes          int64
	Clients            int
	LimitBytes         int64
	Exceeded           bool
}

// ParseDRMFDInfo parses Linux DRM fdinfo key-value lines (drm-driver,
// drm-client-id, drm-resident-memory, drm-memory-gtt, drm-memory-vram).
func ParseDRMFDInfo(r io.Reader) (DRMFDInfo, error) {
	var info DRMFDInfo
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		colon := strings.IndexByte(line, ':')
		if colon < 0 {
			continue
		}
		key := strings.TrimSpace(line[:colon])
		val := strings.TrimSpace(line[colon+1:])
		switch key {
		case "drm-driver":
			info.Driver = val
		case "drm-client-id":
			info.ClientID = val
		case "drm-resident-memory":
			n, err := parseDRMSizeBytes(val)
			if err != nil {
				return DRMFDInfo{}, fmt.Errorf("parse %s %q: %w", key, val, err)
			}
			info.ResidentBytes += n
		case "drm-memory-gtt":
			n, err := parseDRMSizeBytes(val)
			if err != nil {
				return DRMFDInfo{}, fmt.Errorf("parse %s %q: %w", key, val, err)
			}
			info.GTTBytes += n
		case "drm-memory-vram":
			n, err := parseDRMSizeBytes(val)
			if err != nil {
				return DRMFDInfo{}, fmt.Errorf("parse %s %q: %w", key, val, err)
			}
			info.VRAMBytes += n
		}
	}
	if err := sc.Err(); err != nil {
		return DRMFDInfo{}, err
	}
	return info, nil
}

func parseDRMSizeBytes(val string) (int64, error) {
	fields := strings.Fields(val)
	if len(fields) == 0 {
		return 0, fmt.Errorf("empty size")
	}
	n, err := strconv.ParseInt(fields[0], 10, 64)
	if err != nil {
		return 0, err
	}
	if len(fields) == 1 {
		return n, nil
	}
	switch strings.ToLower(fields[1]) {
	case "b", "bytes":
		return n, nil
	case "kib", "kb":
		return n << 10, nil
	case "mib", "mb":
		return n << 20, nil
	case "gib", "gb":
		return n << 30, nil
	default:
		return 0, fmt.Errorf("unknown unit %q", fields[1])
	}
}

// CollectDRMResidencyForPID scans <procRoot>/<pid>/fdinfo/*, deduplicates by
// (drm-driver, drm-client-id) so shared FDs are not double-counted, and checks
// whether total resident bytes exceed limitBytes (when limitBytes > 0).
func CollectDRMResidencyForPID(procRoot string, pid int, limitBytes int64) (DRMResidencySnapshot, error) {
	if procRoot == "" {
		procRoot = "/proc"
	}
	fdDir := filepath.Join(procRoot, strconv.Itoa(pid), "fdinfo")
	entries, err := os.ReadDir(fdDir)
	if err != nil {
		return DRMResidencySnapshot{}, err
	}
	type clientKey struct {
		driver, clientID string
	}
	seen := make(map[clientKey]DRMFDInfo)
	anonCount := 0
	for _, ent := range entries {
		if ent.IsDir() {
			continue
		}
		f, err := os.Open(filepath.Join(fdDir, ent.Name()))
		if err != nil {
			continue
		}
		info, parseErr := ParseDRMFDInfo(f)
		f.Close()
		if parseErr != nil || info.Driver == "" {
			continue
		}
		ck := clientKey{driver: info.Driver, clientID: info.ClientID}
		if info.ClientID == "" {
			anonCount++
			ck.clientID = fmt.Sprintf("anon-%d", anonCount)
		}
		if prev, ok := seen[ck]; !ok || info.ResidentBytes > prev.ResidentBytes {
			seen[ck] = info
		}
	}
	snap := DRMResidencySnapshot{PID: pid, LimitBytes: limitBytes, Clients: len(seen)}
	for _, info := range seen {
		snap.TotalResidentBytes += info.ResidentBytes
		snap.GTTBytes += info.GTTBytes
		snap.VRAMBytes += info.VRAMBytes
	}
	if limitBytes > 0 && snap.TotalResidentBytes > limitBytes {
		snap.Exceeded = true
	}
	return snap, nil
}
