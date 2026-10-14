package computebuild

import (
	"bytes"
	"crypto/sha256"
	"debug/elf"
	"encoding/binary"
	"encoding/hex"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// fak-test:runtime fast est=20ms
// Tiny synthetic ELF data only: no compiler, executable, helper, or native load.
func TestV41EPOwnerLinkedPinELFContract(t *testing.T) {
	helper := strings.Repeat("d", 64)
	t.Run("content pinned snapshot", func(t *testing.T) {
		if runtime.GOOS != "linux" {
			t.Skip("filesystem snapshot qualification requires Linux")
		}
		image := v41EPOwnerELFFixture()
		name := filepath.Join(t.TempDir(), "owner-data")
		if err := os.WriteFile(name, image, 0600); err != nil {
			t.Fatal(err)
		}
		identity := v41EPOwnerELFIdentity(image)
		pin, err := InspectV41EPOwnerLinkedPinELF(name, identity, helper)
		if err != nil || pin.Policy != V41EPOwnerBuildPinPolicy || pin.Symbol != V41EPOwnerBuildPinSymbol || pin.Value != helper || pin.Owner != identity {
			t.Fatalf("snapshot diagnostic: %+v, %v", pin, err)
		}
		wrong := identity
		wrong.SHA256 = strings.Repeat("f", 64)
		if _, err := InspectV41EPOwnerLinkedPinELF(name, wrong, helper); err == nil {
			t.Fatal("unapproved image bytes accepted")
		}
		for _, size := range []int64{0, identity.SizeBytes - 1, identity.SizeBytes + 1, V41EPOwnerBuildImageMaxSize + 1} {
			wrong = identity
			wrong.SizeBytes = size
			if _, err := InspectV41EPOwnerLinkedPinELF(name, wrong, helper); err == nil {
				t.Fatalf("invalid independent size %d accepted", size)
			}
		}
		if _, err := InspectV41EPOwnerLinkedPinELF(name, identity, strings.Repeat("D", 64)); err == nil {
			t.Fatal("noncanonical helper expectation accepted")
		}
		if _, err := InspectV41EPOwnerLinkedPinELF("relative-owner", identity, helper); err == nil {
			t.Fatal("relative input path accepted")
		}
		if _, err := InspectV41EPOwnerLinkedPinELF(filepath.Dir(name), identity, helper); err == nil {
			t.Fatal("directory input accepted")
		}
		link := name + "-link"
		if err := os.Symlink(name, link); err != nil {
			t.Fatalf("Linux snapshot qualifier requires its symlink control: %v", err)
		}
		if _, err := InspectV41EPOwnerLinkedPinELF(link, identity, helper); err == nil {
			t.Fatal("symlink input accepted")
		}
		image[320] = 'e'
		if err := os.WriteFile(name, image, 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := InspectV41EPOwnerLinkedPinELF(name, identity, helper); err == nil {
			t.Fatal("later replacement reused the old snapshot's identity")
		}
	})
	t.Run("external linking local symbol", func(t *testing.T) {
		image := v41EPOwnerELFFixture()
		image[424+4] = elf.ST_INFO(elf.STB_LOCAL, elf.STT_OBJECT)
		if err := inspectV41EPOwnerPinSnapshot(image, helper); err != nil {
			t.Fatalf("ordinary external-link local Go object rejected: %v", err)
		}
	})
	t.Run("malformed ELF controls", func(t *testing.T) {
		bo := binary.LittleEndian
		cases := []struct {
			name string
			edit func([]byte)
		}{
			{"wrong class", func(b []byte) { b[4] = byte(elf.ELFCLASS32) }},
			{"wrong endian", func(b []byte) { b[5] = byte(elf.ELFDATA2MSB) }},
			{"wrong ABI", func(b []byte) { b[7] = byte(elf.ELFOSABI_FREEBSD) }},
			{"PIE", func(b []byte) { bo.PutUint16(b[16:18], uint16(elf.ET_DYN)) }},
			{"wrong machine", func(b []byte) { bo.PutUint16(b[18:20], uint16(elf.EM_AARCH64)) }},
			{"wrong header size", func(b []byte) { bo.PutUint16(b[52:54], 63) }},
			{"program count", func(b []byte) { bo.PutUint16(b[56:58], v41EPOwnerELFMaxPrograms+1) }},
			{"section count", func(b []byte) { bo.PutUint16(b[60:62], v41EPOwnerELFMaxSections+1) }},
			{"extended count", func(b []byte) { bo.PutUint16(b[60:62], 0) }},
			{"extended names", func(b []byte) { bo.PutUint16(b[62:64], uint16(elf.SHN_XINDEX)) }},
			{"overflow program offset", func(b []byte) { bo.PutUint64(b[32:40], ^uint64(0)-7) }},
			{"overlapping tables", func(b []byte) { bo.PutUint64(b[40:48], 64) }},
			{"overflow load address", func(b []byte) { bo.PutUint64(b[64+16:64+24], ^uint64(0)-7) }},
			{"overflow load size", func(b []byte) { bo.PutUint64(b[64+32:64+40], ^uint64(0)) }},
			{"load file exceeds memory", func(b []byte) { bo.PutUint64(b[64+40:64+48], 8) }},
			{"BSS-only load", func(b []byte) { bo.PutUint64(b[64+32:64+40], 0) }},
			{"header wrong load permissions", func(b []byte) { bo.PutUint32(b[64+4:64+8], uint32(elf.PF_R)) }},
			{"value executable load", func(b []byte) { bo.PutUint32(b[120+4:120+8], uint32(elf.PF_R|elf.PF_X)) }},
			{"duplicate load mapping", func(b []byte) { bo.PutUint16(b[56:58], 3); copy(b[176:232], b[64:120]) }},
			{"overflow section address", func(b []byte) { bo.PutUint64(b[1088+16:1088+24], ^uint64(0)-7) }},
			{"overflow section size", func(b []byte) { bo.PutUint64(b[1088+32:1088+40], ^uint64(0)) }},
			{"BSS-only section", func(b []byte) { bo.PutUint32(b[1088+4:1088+8], uint32(elf.SHT_NOBITS)) }},
			{"compressed header", func(b []byte) {
				bo.PutUint64(b[1088+8:1088+16], uint64(elf.SHF_ALLOC|elf.SHF_WRITE|elf.SHF_COMPRESSED))
			}},
			{"header wrong section permissions", func(b []byte) { bo.PutUint64(b[1088+8:1088+16], uint64(elf.SHF_ALLOC)) }},
			{"value writable section", func(b []byte) { bo.PutUint64(b[1152+8:1152+16], uint64(elf.SHF_ALLOC|elf.SHF_WRITE)) }},
			{"header section truncated", func(b []byte) { bo.PutUint64(b[1088+32:1088+40], 8) }},
			{"section and segment disagree", func(b []byte) { bo.PutUint64(b[1088+24:1088+32], 257) }},
			{"overlapping section mapping", func(b []byte) { copy(b[1408:1472], b[1088:1152]) }},
			{"compressed names", func(b []byte) { bo.PutUint64(b[1344+8:1344+16], uint64(elf.SHF_COMPRESSED)) }},
			{"oversize names", func(b []byte) { bo.PutUint64(b[1344+32:1344+40], v41EPOwnerELFMaxNames+1) }},
			{"bad section name index", func(b []byte) { bo.PutUint32(b[1088:1092], 65535) }},
			{"stripped symbols", func(b []byte) { bo.PutUint32(b[1216+4:1216+8], uint32(elf.SHT_PROGBITS)) }},
			{"duplicate symbol table", func(b []byte) { copy(b[1408:1472], b[1216:1280]) }},
			{"compressed symbols", func(b []byte) { bo.PutUint64(b[1216+8:1216+16], uint64(elf.SHF_COMPRESSED)) }},
			{"oversize symbols", func(b []byte) { bo.PutUint64(b[1216+32:1216+40], v41EPOwnerELFMaxSymbols+1) }},
			{"wrong symbol entry size", func(b []byte) { bo.PutUint64(b[1216+56:1216+64], 16) }},
			{"wrong string table link", func(b []byte) { bo.PutUint32(b[1216+40:1216+44], 7) }},
			{"compressed strings", func(b []byte) { bo.PutUint64(b[1280+8:1280+16], uint64(elf.SHF_COMPRESSED)) }},
			{"duplicate named symbol", func(b []byte) { copy(b[448:472], b[424:448]) }},
			{"missing named symbol", func(b []byte) { b[513] = 'x' }},
			{"invalid symbol name index", func(b []byte) { bo.PutUint32(b[424:428], ^uint32(0)) }},
			{"wrong object type", func(b []byte) { b[428] = elf.ST_INFO(elf.STB_GLOBAL, elf.STT_FUNC) }},
			{"weak binding", func(b []byte) { b[428] = elf.ST_INFO(elf.STB_WEAK, elf.STT_OBJECT) }},
			{"undefined symbol", func(b []byte) { bo.PutUint16(b[430:432], 0) }},
			{"wrong object section", func(b []byte) { bo.PutUint16(b[430:432], 2) }},
			{"wrong object size", func(b []byte) { bo.PutUint64(b[440:448], 8) }},
			{"unaligned object", func(b []byte) { bo.PutUint64(b[432:440], 0x400101) }},
			{"wrong string length", func(b []byte) { bo.PutUint64(b[264:272], 63) }},
			{"overflow string pointer", func(b []byte) { bo.PutUint64(b[256:264], ^uint64(0)-31) }},
			{"wrong linked value", func(b []byte) { b[320] = 'e' }},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				image := v41EPOwnerELFFixture()
				tc.edit(image)
				if err := inspectV41EPOwnerPinSnapshot(image, helper); err == nil {
					t.Fatal("malformed or unsupported ELF accepted")
				}
			})
		}
		for _, length := range []int{0, 4, 63, 175, 1471} {
			if err := inspectV41EPOwnerPinSnapshot(v41EPOwnerELFFixture()[:length], helper); err == nil {
				t.Fatalf("truncated %d-byte ELF accepted", length)
			}
		}
	})
}

func v41EPOwnerELFIdentity(image []byte) V41EPOwnerBuildFileIdentity {
	digest := sha256.Sum256(image)
	return V41EPOwnerBuildFileIdentity{SHA256: hex.EncodeToString(digest[:]), SizeBytes: int64(len(image))}
}

func v41EPOwnerELFFixture() []byte {
	// This file is deliberately non-executable test data, not a built program.
	image := make([]byte, 1536)
	bo := binary.LittleEndian
	copy(image, "\x7fELF")
	image[4], image[5], image[6] = byte(elf.ELFCLASS64), byte(elf.ELFDATA2LSB), byte(elf.EV_CURRENT)
	bo.PutUint16(image[16:18], uint16(elf.ET_EXEC))
	bo.PutUint16(image[18:20], uint16(elf.EM_X86_64))
	bo.PutUint32(image[20:24], uint32(elf.EV_CURRENT))
	bo.PutUint64(image[32:40], 64)
	bo.PutUint64(image[40:48], 1024)
	bo.PutUint16(image[52:54], 64)
	bo.PutUint16(image[54:56], 56)
	bo.PutUint16(image[56:58], 2)
	bo.PutUint16(image[58:60], 64)
	bo.PutUint16(image[60:62], 7)
	bo.PutUint16(image[62:64], 5)
	program := func(off int, flags elf.ProgFlag, file, address, size uint64) {
		p := image[off : off+56]
		bo.PutUint32(p[:4], uint32(elf.PT_LOAD))
		bo.PutUint32(p[4:8], uint32(flags))
		bo.PutUint64(p[8:16], file)
		bo.PutUint64(p[16:24], address)
		bo.PutUint64(p[24:32], address)
		bo.PutUint64(p[32:40], size)
		bo.PutUint64(p[40:48], size)
		bo.PutUint64(p[48:56], 4096)
	}
	program(64, elf.PF_R|elf.PF_W, 256, 0x400100, 16)
	program(120, elf.PF_R, 320, 0x500140, 64)
	bo.PutUint64(image[256:264], 0x500140)
	bo.PutUint64(image[264:272], 64)
	copy(image[320:384], strings.Repeat("d", 64))
	copy(image[512:], "\x00"+V41EPOwnerBuildPinSymbol+"\x00")
	bo.PutUint32(image[424:428], 1)
	image[428] = elf.ST_INFO(elf.STB_GLOBAL, elf.STT_OBJECT)
	bo.PutUint16(image[430:432], 1)
	bo.PutUint64(image[432:440], 0x400100)
	bo.PutUint64(image[440:448], 16)
	names := []byte("\x00.data\x00.rodata\x00.symtab\x00.strtab\x00.shstrtab\x00.other\x00")
	copy(image[768:], names)
	section := func(index int, name string, typ elf.SectionType, flags elf.SectionFlag, address, file, size uint64) {
		s := image[1024+index*64 : 1024+(index+1)*64]
		bo.PutUint32(s[:4], uint32(bytes.Index(names, []byte(name+"\x00"))))
		bo.PutUint32(s[4:8], uint32(typ))
		bo.PutUint64(s[8:16], uint64(flags))
		bo.PutUint64(s[16:24], address)
		bo.PutUint64(s[24:32], file)
		bo.PutUint64(s[32:40], size)
		bo.PutUint64(s[48:56], 8)
	}
	section(1, ".data", elf.SHT_PROGBITS, elf.SHF_ALLOC|elf.SHF_WRITE, 0x400100, 256, 16)
	section(2, ".rodata", elf.SHT_PROGBITS, elf.SHF_ALLOC, 0x500140, 320, 64)
	section(3, ".symtab", elf.SHT_SYMTAB, 0, 0, 400, 72)
	bo.PutUint32(image[1216+40:1216+44], 4)
	bo.PutUint32(image[1216+44:1216+48], 1)
	bo.PutUint64(image[1216+56:1216+64], elf.Sym64Size)
	section(4, ".strtab", elf.SHT_STRTAB, 0, 0, 512, uint64(len(V41EPOwnerBuildPinSymbol)+2))
	section(5, ".shstrtab", elf.SHT_STRTAB, 0, 0, 768, uint64(len(names)))
	section(6, ".other", elf.SHT_PROGBITS, 0, 0, 0, 0)
	return image
}
