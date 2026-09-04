package compute

import (
	"bytes"
	"encoding/binary"
	"errors"
	"reflect"
	"testing"
)

func TestTargetFallbackChain(t *testing.T) {
	tests := []struct {
		name     string
		arch     string
		expected []string
	}{
		{
			name:     "gfx1151 with features",
			arch:     "gfx1151:sramecc-:xnack-",
			expected: []string{"gfx1151:sramecc-:xnack-", "gfx1151", "gfx11-generic", "generic"},
		},
		{
			name:     "gfx1151 bare",
			arch:     "gfx1151",
			expected: []string{"gfx1151", "gfx11-generic", "generic"},
		},
		{
			name:     "gfx1150 bare",
			arch:     "gfx1150",
			expected: []string{"gfx1150", "gfx11-generic", "generic"},
		},
		{
			name:     "gfx1100 bare",
			arch:     "gfx1100",
			expected: []string{"gfx1100", "gfx11-generic", "generic"},
		},
		{
			name:     "gfx1102 bare",
			arch:     "gfx1102",
			expected: []string{"gfx1102", "gfx11-generic", "generic"},
		},
		{
			name:     "gfx1200 bare",
			arch:     "gfx1200",
			expected: []string{"gfx1200", "gfx12-generic", "generic"},
		},
		{
			name:     "gfx1201 with features",
			arch:     "gfx1201:sramecc-:xnack-",
			expected: []string{"gfx1201:sramecc-:xnack-", "gfx1201", "gfx12-generic", "generic"},
		},
		{
			name:     "gfx942 CDNA with features",
			arch:     "gfx942:sramecc+:xnack-",
			expected: []string{"gfx942:sramecc+:xnack-", "gfx942", "gfx9-generic"},
		},
		{
			name:     "gfx942 CDNA bare",
			arch:     "gfx942",
			expected: []string{"gfx942", "gfx9-generic"},
		},
		{
			name:     "gfx90a CDNA bare",
			arch:     "gfx90a",
			expected: []string{"gfx90a", "gfx9-generic"},
		},
		{
			name:     "gfx908 CDNA bare",
			arch:     "gfx908",
			expected: []string{"gfx908", "gfx9-generic"},
		},
		{
			name:     "gfx11-generic",
			arch:     "gfx11-generic",
			expected: []string{"gfx11-generic", "generic"},
		},
		{
			name:     "gfx9-generic",
			arch:     "gfx9-generic",
			expected: []string{"gfx9-generic"},
		},
		{
			name:     "generic",
			arch:     "generic",
			expected: []string{"generic"},
		},
		{
			name:     "empty arch",
			arch:     "",
			expected: nil,
		},
		{
			name:     "whitespace and case normalization",
			arch:     "  GFX1151:SRAMECC-:XNACK-  ",
			expected: []string{"gfx1151:sramecc-:xnack-", "gfx1151", "gfx11-generic", "generic"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := TargetFallbackChain(tc.arch)
			if !reflect.DeepEqual(got, tc.expected) {
				t.Fatalf("TargetFallbackChain(%q) = %v, want %v", tc.arch, got, tc.expected)
			}
		})
	}
}

func TestResolveTarget(t *testing.T) {
	t.Run("exact match", func(t *testing.T) {
		available := []string{"gfx1151:sramecc-:xnack-", "gfx1151", "gfx11-generic"}
		resolved, ok := ResolveTarget("gfx1151:sramecc-:xnack-", available)
		if !ok || resolved != "gfx1151:sramecc-:xnack-" {
			t.Fatalf("expected exact match gfx1151:sramecc-:xnack-, got %q (ok=%v)", resolved, ok)
		}
	})

	t.Run("feature variant match subset", func(t *testing.T) {
		available := []string{"gfx942:sramecc+", "gfx9-generic"}
		resolved, ok := ResolveTarget("gfx942:sramecc+:xnack-", available)
		if !ok || resolved != "gfx942:sramecc+" {
			t.Fatalf("expected feature variant match gfx942:sramecc+, got %q (ok=%v)", resolved, ok)
		}
	})

	t.Run("feature variant incompatible skips to generic", func(t *testing.T) {
		// Available has sramecc-, but query requires sramecc+
		available := []string{"gfx942:sramecc-", "gfx9-generic"}
		resolved, ok := ResolveTarget("gfx942:sramecc+:xnack-", available)
		if !ok || resolved != "gfx9-generic" {
			t.Fatalf("expected fallback to gfx9-generic due to incompatible feature, got %q (ok=%v)", resolved, ok)
		}
	})

	t.Run("generic family fallback", func(t *testing.T) {
		available := []string{"gfx11-generic", "generic"}
		resolved, ok := ResolveTarget("gfx1151", available)
		if !ok || resolved != "gfx11-generic" {
			t.Fatalf("expected fallback gfx11-generic, got %q (ok=%v)", resolved, ok)
		}
	})

	t.Run("global generic fallback", func(t *testing.T) {
		available := []string{"generic"}
		resolved, ok := ResolveTarget("gfx1100", available)
		if !ok || resolved != "generic" {
			t.Fatalf("expected fallback generic, got %q (ok=%v)", resolved, ok)
		}
	})

	t.Run("cdna does not fallback to cross-family generic", func(t *testing.T) {
		available := []string{"generic"}
		resolved, ok := ResolveTarget("gfx942", available)
		if ok {
			t.Fatalf("expected CDNA gfx942 not to match generic, got %q", resolved)
		}
	})

	t.Run("no match available", func(t *testing.T) {
		available := []string{"gfx1030", "gfx1010"}
		resolved, ok := ResolveTarget("gfx1151", available)
		if ok {
			t.Fatalf("expected no match, got %q", resolved)
		}
	})

	t.Run("empty inputs", func(t *testing.T) {
		if _, ok := ResolveTarget("", []string{"gfx1151"}); ok {
			t.Fatal("expected false for empty arch")
		}
		if _, ok := ResolveTarget("gfx1151", nil); ok {
			t.Fatal("expected false for nil availableTargets")
		}
	})
}

func TestKPACKSerializationDeserialization(t *testing.T) {
	entries := []KPACKEntry{
		{
			Target:           "gfx1151",
			Compression:      CompressionNone,
			Data:             []byte("kernel-gfx1151-bytecode-sample"),
			UncompressedSize: uint64(len("kernel-gfx1151-bytecode-sample")),
		},
		{
			Target:           "gfx942:sramecc+:xnack-",
			Compression:      CompressionZSTD,
			Data:             []byte{0x28, 0xb5, 0x2f, 0xfd, 0x00, 0x40, 0x21, 0x00},
			UncompressedSize: 8192,
		},
		{
			Target:           "gfx11-generic",
			Compression:      CompressionNone,
			Data:             []byte("kernel-rdna3-generic-portable-binary"),
			UncompressedSize: uint64(len("kernel-rdna3-generic-portable-binary")),
		},
	}

	var buf bytes.Buffer
	if err := WriteKPACK(&buf, entries); err != nil {
		t.Fatalf("WriteKPACK failed: %v", err)
	}

	archiveBytes := buf.Bytes()
	if len(archiveBytes) < KPACKHeaderSize {
		t.Fatalf("serialized archive too small: %d bytes", len(archiveBytes))
	}

	// Verify header magic
	magic := binary.BigEndian.Uint32(archiveBytes[0:4])
	if magic != KPACKMagic {
		t.Fatalf("expected magic 0x%X (KPAK), got 0x%X", KPACKMagic, magic)
	}

	// Deserialize archive
	archive, err := ReadKPACK(bytes.NewReader(archiveBytes), int64(len(archiveBytes)))
	if err != nil {
		t.Fatalf("ReadKPACK failed: %v", err)
	}

	if archive.Header.Magic != KPACKMagic {
		t.Fatalf("archive header magic mismatch: 0x%X", archive.Header.Magic)
	}
	if archive.Header.Version != KPACKVersion1 {
		t.Fatalf("archive header version mismatch: %d", archive.Header.Version)
	}
	if archive.Header.EntryCount != uint32(len(entries)) {
		t.Fatalf("archive entry count mismatch: %d vs %d", archive.Header.EntryCount, len(entries))
	}
	if len(archive.Entries) != len(entries) {
		t.Fatalf("entries length mismatch: %d vs %d", len(archive.Entries), len(entries))
	}

	// Verify entries
	for i, want := range entries {
		got := archive.Entries[i]
		if got.Target != want.Target {
			t.Errorf("entry[%d].Target = %q, want %q", i, got.Target, want.Target)
		}
		if got.Compression != want.Compression {
			t.Errorf("entry[%d].Compression = %d, want %d", i, got.Compression, want.Compression)
		}
		if got.CompressedSize != uint64(len(want.Data)) {
			t.Errorf("entry[%d].CompressedSize = %d, want %d", i, got.CompressedSize, len(want.Data))
		}
		if got.UncompressedSize != want.UncompressedSize {
			t.Errorf("entry[%d].UncompressedSize = %d, want %d", i, got.UncompressedSize, want.UncompressedSize)
		}
		if !bytes.Equal(got.Data, want.Data) {
			t.Errorf("entry[%d].Data mismatch: got %v, want %v", i, got.Data, want.Data)
		}
	}

	// Verify archive lookup and resolve methods
	targets := archive.Targets()
	expectedTargets := []string{"gfx1151", "gfx942:sramecc+:xnack-", "gfx11-generic"}
	if !reflect.DeepEqual(targets, expectedTargets) {
		t.Fatalf("archive.Targets() = %v, want %v", targets, expectedTargets)
	}

	entry1151, ok := archive.Get("gfx1151")
	if !ok || entry1151 == nil || entry1151.Target != "gfx1151" {
		t.Fatalf("archive.Get('gfx1151') failed: %v", entry1151)
	}

	if _, ok := archive.Get("nonexistent"); ok {
		t.Fatal("expected false for nonexistent target")
	}

	// Resolve gfx1151:sramecc-:xnack- -> falls back to gfx1151
	resEntry, matched, ok := archive.Resolve("gfx1151:sramecc-:xnack-")
	if !ok || matched != "gfx1151" || resEntry == nil {
		t.Fatalf("archive.Resolve('gfx1151:sramecc-:xnack-') failed: matched=%q, ok=%v", matched, ok)
	}

	// Resolve gfx942:sramecc+:xnack- -> exact match
	resEntry, matched, ok = archive.Resolve("gfx942:sramecc+:xnack-")
	if !ok || matched != "gfx942:sramecc+:xnack-" || resEntry == nil {
		t.Fatalf("archive.Resolve('gfx942:sramecc+:xnack-') failed: matched=%q, ok=%v", matched, ok)
	}

	// Resolve gfx1100 -> falls back to gfx11-generic
	resEntry, matched, ok = archive.Resolve("gfx1100")
	if !ok || matched != "gfx11-generic" || resEntry == nil {
		t.Fatalf("archive.Resolve('gfx1100') failed: matched=%q, ok=%v", matched, ok)
	}
}

func TestKPACKEmptyArchive(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteKPACK(&buf, []KPACKEntry{}); err != nil {
		t.Fatalf("WriteKPACK with empty entries failed: %v", err)
	}

	archive, err := ReadKPACK(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatalf("ReadKPACK on empty archive failed: %v", err)
	}

	if len(archive.Entries) != 0 {
		t.Fatalf("expected 0 entries, got %d", len(archive.Entries))
	}
	if archive.Header.EntryCount != 0 {
		t.Fatalf("expected 0 EntryCount in header, got %d", archive.Header.EntryCount)
	}
}

func TestKPACKCorruptionDetection(t *testing.T) {
	entries := []KPACKEntry{
		{
			Target: "gfx1151",
			Data:   []byte("sample-kernel-payload"),
		},
	}
	var buf bytes.Buffer
	if err := WriteKPACK(&buf, entries); err != nil {
		t.Fatalf("WriteKPACK failed: %v", err)
	}
	validBytes := buf.Bytes()

	t.Run("bad magic number", func(t *testing.T) {
		corrupt := make([]byte, len(validBytes))
		copy(corrupt, validBytes)
		binary.BigEndian.PutUint32(corrupt[0:4], 0xDEADBEEF)

		_, err := ReadKPACK(bytes.NewReader(corrupt), int64(len(corrupt)))
		if !errors.Is(err, ErrKPACKInvalidMagic) {
			t.Fatalf("expected ErrKPACKInvalidMagic, got %v", err)
		}
	})

	t.Run("invalid version", func(t *testing.T) {
		corrupt := make([]byte, len(validBytes))
		copy(corrupt, validBytes)
		binary.BigEndian.PutUint32(corrupt[4:8], 99)

		_, err := ReadKPACK(bytes.NewReader(corrupt), int64(len(corrupt)))
		if !errors.Is(err, ErrKPACKInvalidVersion) {
			t.Fatalf("expected ErrKPACKInvalidVersion, got %v", err)
		}
	})

	t.Run("truncated header", func(t *testing.T) {
		_, err := ReadKPACK(bytes.NewReader(validBytes[:16]), 16)
		if !errors.Is(err, ErrKPACKTruncated) {
			t.Fatalf("expected ErrKPACKTruncated for header < 32 bytes, got %v", err)
		}
	})

	t.Run("truncated TOC", func(t *testing.T) {
		// Cut file mid-TOC
		cutSize := int64(KPACKHeaderSize + 5)
		_, err := ReadKPACK(bytes.NewReader(validBytes[:cutSize]), cutSize)
		if !errors.Is(err, ErrKPACKTruncated) {
			t.Fatalf("expected ErrKPACKTruncated for cut TOC, got %v", err)
		}
	})

	t.Run("truncated payload data", func(t *testing.T) {
		// Cut off the last few bytes of payload
		cutSize := int64(len(validBytes) - 5)
		_, err := ReadKPACK(bytes.NewReader(validBytes[:cutSize]), cutSize)
		if !errors.Is(err, ErrKPACKTruncated) {
			t.Fatalf("expected ErrKPACKTruncated for missing payload, got %v", err)
		}
	})

	t.Run("invalid TOC offset in header", func(t *testing.T) {
		corrupt := make([]byte, len(validBytes))
		copy(corrupt, validBytes)
		// Set tocOffset to 100000 (past end of file)
		binary.BigEndian.PutUint32(corrupt[20:24], 100000)

		_, err := ReadKPACK(bytes.NewReader(corrupt), int64(len(corrupt)))
		if !errors.Is(err, ErrKPACKTruncated) {
			t.Fatalf("expected ErrKPACKTruncated for out-of-bounds TOC, got %v", err)
		}
	})
}

func TestKPACKMessagePackNestingLimit(t *testing.T) {
	// Construct a TOC map with a field containing 33 nested fixarrays (exceeding MaxMsgpackDepth = 32)
	var deepMsgpack bytes.Buffer
	// Top-level map with 2 entries: "entries" and "deep"
	deepMsgpack.WriteByte(0x82) // fixmap of 2 pairs

	// Key 1: "entries" -> empty array
	deepMsgpack.WriteByte(0xa7)
	deepMsgpack.WriteString("entries")
	deepMsgpack.WriteByte(0x90) // fixarray of 0

	// Key 2: "deep" -> 33 nested fixarrays
	deepMsgpack.WriteByte(0xa4)
	deepMsgpack.WriteString("deep")
	for i := 0; i < 33; i++ {
		deepMsgpack.WriteByte(0x91) // fixarray of 1 element
	}
	deepMsgpack.WriteByte(0x00) // leaf integer 0

	tocBytes := deepMsgpack.Bytes()
	tocSize := uint32(len(tocBytes))
	payloadOffset := uint64(KPACKHeaderSize + tocSize)

	var hdr [KPACKHeaderSize]byte
	binary.BigEndian.PutUint32(hdr[0:4], KPACKMagic)
	binary.BigEndian.PutUint32(hdr[4:8], KPACKVersion1)
	binary.BigEndian.PutUint32(hdr[8:12], 0)
	binary.BigEndian.PutUint32(hdr[12:16], 0)
	binary.BigEndian.PutUint32(hdr[16:20], tocSize)
	binary.BigEndian.PutUint32(hdr[20:24], KPACKHeaderSize)
	binary.BigEndian.PutUint64(hdr[24:32], payloadOffset)

	var archiveBuf bytes.Buffer
	archiveBuf.Write(hdr[:])
	archiveBuf.Write(tocBytes)

	data := archiveBuf.Bytes()
	_, err := ReadKPACK(bytes.NewReader(data), int64(len(data)))
	if !errors.Is(err, ErrKPACKNestingTooDeep) {
		t.Fatalf("expected ErrKPACKNestingTooDeep for >32 depth, got %v", err)
	}
}

func TestKPACKNilArguments(t *testing.T) {
	if err := WriteKPACK(nil, []KPACKEntry{}); err == nil {
		t.Fatal("expected error for nil writer")
	}
	if _, err := ReadKPACK(nil, 100); err == nil {
		t.Fatal("expected error for nil reader")
	}
}
