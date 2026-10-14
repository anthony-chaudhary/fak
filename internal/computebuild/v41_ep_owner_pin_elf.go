package computebuild

import (
	"bytes"
	"crypto/sha256"
	"debug/elf"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"os"
)

const (
	v41EPOwnerELFMaxPrograms = 64
	v41EPOwnerELFMaxSections = 512
	v41EPOwnerELFMaxNames    = 64 << 10
	v41EPOwnerELFMaxName     = 128
	v41EPOwnerELFMaxSymbols  = 16 << 20
	v41EPOwnerELFMaxStrings  = 32 << 20
)

// InspectV41EPOwnerLinkedPinELF inspects one owned, independently content-pinned
// snapshot without executing or loading the image. The returned public value is
// a diagnostic observation, not native authority or a build-causality receipt.
// Path/handle checks can detect races but cannot reserve a pathname or attest a
// running process's mapped image. Only the exact hashed snapshot is inspected.
// ELFOSABI_NONE is compatible with the fixed Linux build profile; it does not
// prove GOOS provenance. That remains the independent plan/session's scope.
func InspectV41EPOwnerLinkedPinELF(ownerPath string, expectedOwner V41EPOwnerBuildFileIdentity, expectedHelperSHA256 string) (*V41EPOwnerLinkedPin, error) {
	if !validV41EPOwnerBuildFile(expectedOwner) || !validLowerSHA256(expectedHelperSHA256) {
		return nil, fmt.Errorf("owner ELF inspection requires bounded independent owner and helper identities")
	}
	clean, before, err := strictAbsoluteRegularFile(ownerPath, "owner ELF")
	if err != nil {
		return nil, err
	}
	f, err := os.Open(clean)
	if err != nil {
		return nil, fmt.Errorf("open owner ELF: %w", err)
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil || !sameFileMetadata(before, opened) || opened.Size() != expectedOwner.SizeBytes {
		return nil, fmt.Errorf("owner ELF identity or size changed before snapshot")
	}
	// Allocate exactly the independently expected bounded size, not a size read
	// from the image, an unbounded ReadAll, or a growing file's current length.
	image := make([]byte, int(expectedOwner.SizeBytes))
	if _, err := io.ReadFull(f, image); err != nil {
		return nil, fmt.Errorf("read complete owner ELF snapshot: %w", err)
	}
	var extra [1]byte
	if n, err := f.Read(extra[:]); n != 0 || err != io.EOF {
		return nil, fmt.Errorf("owner ELF exceeds the expected snapshot size")
	}
	afterHandle, err := f.Stat()
	if err != nil || !sameFileMetadata(opened, afterHandle) {
		return nil, fmt.Errorf("owner ELF handle changed during snapshot")
	}
	_, afterPath, err := strictAbsoluteRegularFile(clean, "owner ELF")
	if err != nil || !sameFileMetadata(opened, afterPath) {
		return nil, fmt.Errorf("owner ELF path changed during snapshot")
	}
	digest := sha256.Sum256(image)
	if hex.EncodeToString(digest[:]) != expectedOwner.SHA256 {
		return nil, fmt.Errorf("owner ELF snapshot digest differs from independent expectation")
	}
	if err := inspectV41EPOwnerPinSnapshot(image, expectedHelperSHA256); err != nil {
		return nil, err
	}
	return &V41EPOwnerLinkedPin{Policy: V41EPOwnerBuildPinPolicy, Symbol: V41EPOwnerBuildPinSymbol, Value: expectedHelperSHA256, Owner: expectedOwner}, nil
}

func inspectV41EPOwnerPinSnapshot(image []byte, helper string) error {
	if err := preflightV41EPOwnerELF(image); err != nil {
		return err
	}
	f, err := elf.NewFile(bytes.NewReader(image))
	if err != nil {
		return fmt.Errorf("parse bounded owner ELF: %w", err)
	}
	defer f.Close()
	var symtab *elf.Section
	for _, section := range f.Sections {
		if section.Type == elf.SHT_SYMTAB {
			if symtab != nil {
				return fmt.Errorf("owner ELF has multiple static symbol tables")
			}
			symtab = section
		}
	}
	if symtab == nil || symtab.Flags&elf.SHF_COMPRESSED != 0 || symtab.Entsize != elf.Sym64Size || symtab.Size < 2*elf.Sym64Size || symtab.Size%elf.Sym64Size != 0 || symtab.Size > v41EPOwnerELFMaxSymbols || symtab.Link == 0 || uint64(symtab.Link) >= uint64(len(f.Sections)) {
		return fmt.Errorf("owner ELF requires one bounded uncompressed static ELF64 symbol table")
	}
	strtab := f.Sections[symtab.Link]
	if strtab.Type != elf.SHT_STRTAB || strtab.Flags&elf.SHF_COMPRESSED != 0 || strtab.Size == 0 || strtab.Size > v41EPOwnerELFMaxStrings {
		return fmt.Errorf("owner ELF symbol string table is unsupported")
	}
	syms, ok := v41EPOwnerELFSlice(image, symtab.Offset, symtab.Size)
	if !ok {
		return fmt.Errorf("owner ELF symbol table is outside the snapshot")
	}
	names, ok := v41EPOwnerELFSlice(image, strtab.Offset, strtab.Size)
	if !ok || names[0] != 0 || names[len(names)-1] != 0 || !bytes.Equal(syms[:elf.Sym64Size], make([]byte, elf.Sym64Size)) {
		return fmt.Errorf("owner ELF symbol/string framing is invalid")
	}
	// Symbols() allocates every symbol name. Repeated long names can multiply
	// memory use; scan bounded raw entries and compare only the fixed name.
	targetName := []byte(V41EPOwnerBuildPinSymbol)
	var target []byte
	for off := elf.Sym64Size; off < len(syms); off += elf.Sym64Size {
		symbol := syms[off : off+elf.Sym64Size]
		nameOffset := uint64(binary.LittleEndian.Uint32(symbol[:4]))
		if nameOffset >= uint64(len(names)) {
			return fmt.Errorf("owner ELF symbol name index is invalid")
		}
		name, ok := v41EPOwnerELFSlice(names, nameOffset, uint64(len(targetName)+1))
		if !ok || name[len(name)-1] != 0 || !bytes.Equal(name[:len(name)-1], targetName) {
			continue
		}
		if target != nil {
			return fmt.Errorf("owner ELF contains duplicate named helper pins")
		}
		target = symbol
	}
	if target == nil {
		return fmt.Errorf("owner ELF named helper pin is absent")
	}
	bind := elf.ST_BIND(target[4])
	sectionIndex := binary.LittleEndian.Uint16(target[6:8])
	address := binary.LittleEndian.Uint64(target[8:16])
	// External CGO linking makes ordinary Go symbols local; both local and
	// global object bindings are valid. Weak or undefined aliases are refused.
	if elf.ST_TYPE(target[4]) != elf.STT_OBJECT || (bind != elf.STB_LOCAL && bind != elf.STB_GLOBAL) || target[5] != 0 || binary.LittleEndian.Uint64(target[16:24]) != 16 || address%8 != 0 || sectionIndex == 0 || int(sectionIndex) >= len(f.Sections) {
		return fmt.Errorf("owner ELF named helper pin is not a defined 16-byte Go string object")
	}
	headerOffset, mappedSection, err := mapV41EPOwnerELFRange(f, address, 16, true)
	if err != nil || mappedSection != int(sectionIndex) {
		return fmt.Errorf("owner ELF string header has no unique writable file-backed mapping")
	}
	header, ok := v41EPOwnerELFSlice(image, headerOffset, 16)
	if !ok || binary.LittleEndian.Uint64(header[8:]) != 64 {
		return fmt.Errorf("owner ELF helper pin string length must be 64")
	}
	valueOffset, _, err := mapV41EPOwnerELFRange(f, binary.LittleEndian.Uint64(header[:8]), 64, false)
	if err != nil {
		return fmt.Errorf("owner ELF helper value has no unique read-only file-backed mapping")
	}
	value, ok := v41EPOwnerELFSlice(image, valueOffset, 64)
	if !ok || !validLowerSHA256(helper) || string(value) != helper {
		return fmt.Errorf("owner ELF linked helper value differs from independent expectation")
	}
	return nil
}

// Preflight everything NewFile can size or allocate from. Extended counts and
// compressed name tables are outside the fixed profile. Other compressed debug
// sections may exist, but no debug data or compressed payload is ever decoded.
func preflightV41EPOwnerELF(image []byte) error {
	if len(image) < 64 || len(image) > V41EPOwnerBuildImageMaxSize || string(image[:4]) != "\x7fELF" || image[4] != byte(elf.ELFCLASS64) || image[5] != byte(elf.ELFDATA2LSB) || image[6] != byte(elf.EV_CURRENT) || image[7] != byte(elf.ELFOSABI_NONE) || image[8] != 0 {
		return fmt.Errorf("owner ELF requires the bounded ELF64 little-endian SYSV profile")
	}
	bo := binary.LittleEndian
	if bo.Uint16(image[16:18]) != uint16(elf.ET_EXEC) || bo.Uint16(image[18:20]) != uint16(elf.EM_X86_64) || bo.Uint32(image[20:24]) != uint32(elf.EV_CURRENT) || bo.Uint32(image[48:52]) != 0 || bo.Uint16(image[52:54]) != 64 || bo.Uint16(image[54:56]) != 56 || bo.Uint16(image[58:60]) != 64 {
		return fmt.Errorf("owner ELF type, machine, version, or header sizes differ from the fixed profile")
	}
	phoff, shoff := bo.Uint64(image[32:40]), bo.Uint64(image[40:48])
	phnum, shnum, shstr := uint64(bo.Uint16(image[56:58])), uint64(bo.Uint16(image[60:62])), uint64(bo.Uint16(image[62:64]))
	if phnum == 0 || phnum > v41EPOwnerELFMaxPrograms || shnum == 0 || shnum > v41EPOwnerELFMaxSections || shstr == 0 || shstr >= shnum || phoff < 64 || shoff < 64 || phoff%8 != 0 || shoff%8 != 0 {
		return fmt.Errorf("owner ELF requires bounded ordinary header counts and offsets")
	}
	ph, phOK := v41EPOwnerELFSlice(image, phoff, phnum*56)
	sh, shOK := v41EPOwnerELFSlice(image, shoff, shnum*64)
	if !phOK || !shOK || v41EPOwnerELFOverlap(phoff, phnum*56, shoff, shnum*64) || !bytes.Equal(sh[:64], make([]byte, 64)) {
		return fmt.Errorf("owner ELF header tables are invalid, overlapping, or extended")
	}
	for off := 0; off < len(ph); off += 56 {
		p := ph[off : off+56]
		fileOffset, fileSize, address, memSize := bo.Uint64(p[8:16]), bo.Uint64(p[32:40]), bo.Uint64(p[16:24]), bo.Uint64(p[40:48])
		if _, ok := v41EPOwnerELFSlice(image, fileOffset, fileSize); !ok || address > ^uint64(0)-memSize {
			return fmt.Errorf("owner ELF program range is invalid")
		}
		if elf.ProgType(bo.Uint32(p[:4])) == elf.PT_LOAD && fileSize > memSize {
			return fmt.Errorf("owner ELF load file size exceeds memory size")
		}
	}
	nameHeader := sh[shstr*64 : (shstr+1)*64]
	nameOffset, nameSize := bo.Uint64(nameHeader[24:32]), bo.Uint64(nameHeader[32:40])
	if elf.SectionType(bo.Uint32(nameHeader[4:8])) != elf.SHT_STRTAB || elf.SectionFlag(bo.Uint64(nameHeader[8:16]))&elf.SHF_COMPRESSED != 0 || nameSize == 0 || nameSize > v41EPOwnerELFMaxNames {
		return fmt.Errorf("owner ELF section-name table is unsupported")
	}
	names, ok := v41EPOwnerELFSlice(image, nameOffset, nameSize)
	if !ok || names[0] != 0 {
		return fmt.Errorf("owner ELF section-name table is invalid")
	}
	for off := 0; off < len(sh); off += 64 {
		s := sh[off : off+64]
		name, flags, typ := uint64(bo.Uint32(s[:4])), elf.SectionFlag(bo.Uint64(s[8:16])), elf.SectionType(bo.Uint32(s[4:8]))
		fileOffset, fileSize, address := bo.Uint64(s[24:32]), bo.Uint64(s[32:40]), bo.Uint64(s[16:24])
		if fileOffset > uint64(len(image)) || (flags&elf.SHF_ALLOC != 0 && address > ^uint64(0)-fileSize) {
			return fmt.Errorf("owner ELF section range is invalid")
		}
		if typ != elf.SHT_NOBITS {
			if _, ok := v41EPOwnerELFSlice(image, fileOffset, fileSize); !ok {
				return fmt.Errorf("owner ELF section is outside the snapshot")
			}
		}
		if flags&elf.SHF_COMPRESSED != 0 && (typ == elf.SHT_NOBITS || flags&elf.SHF_ALLOC != 0 || fileSize < 24) {
			return fmt.Errorf("owner ELF compressed section framing is invalid")
		}
		if name >= uint64(len(names)) {
			return fmt.Errorf("owner ELF section name index is invalid")
		}
		tail := names[name:]
		if len(tail) > v41EPOwnerELFMaxName+1 {
			tail = tail[:v41EPOwnerELFMaxName+1]
		}
		if bytes.IndexByte(tail, 0) < 0 {
			return fmt.Errorf("owner ELF section name is unbounded")
		}
	}
	return nil
}

func v41EPOwnerELFSlice(image []byte, offset, size uint64) ([]byte, bool) {
	if offset > uint64(len(image)) || size > uint64(len(image))-offset {
		return nil, false
	}
	return image[offset : offset+size], true
}

// Callers first establish non-overflowing endpoints. Empty ranges do not map.
func v41EPOwnerELFOverlap(a, n, b, m uint64) bool {
	return n != 0 && m != 0 && a < b+m && b < a+n
}

func mapV41EPOwnerELFRange(f *elf.File, address, size uint64, writable bool) (uint64, int, error) {
	if address == 0 || size == 0 || address > ^uint64(0)-size {
		return 0, 0, fmt.Errorf("invalid virtual range")
	}
	var segment *elf.Prog
	for _, p := range f.Progs {
		if p.Type != elf.PT_LOAD || !v41EPOwnerELFOverlap(address, size, p.Vaddr, p.Memsz) {
			continue
		}
		if segment != nil {
			return 0, 0, fmt.Errorf("ambiguous load mapping")
		}
		segment = p
	}
	wantProg, wantSection := elf.PF_R, elf.SHF_ALLOC
	if writable {
		wantProg |= elf.PF_W
		wantSection |= elf.SHF_WRITE
	}
	if segment == nil || segment.Flags != wantProg || address < segment.Vaddr || address-segment.Vaddr > segment.Filesz || size > segment.Filesz-(address-segment.Vaddr) {
		return 0, 0, fmt.Errorf("range lacks the required file-backed load permissions")
	}
	fileOffset := segment.Off + (address - segment.Vaddr)
	sectionIndex := -1
	for i, s := range f.Sections {
		if s.Flags&elf.SHF_ALLOC == 0 || !v41EPOwnerELFOverlap(address, size, s.Addr, s.Size) {
			continue
		}
		if sectionIndex != -1 {
			return 0, 0, fmt.Errorf("ambiguous allocated section mapping")
		}
		if s.Type != elf.SHT_PROGBITS || s.Flags != wantSection || address < s.Addr || address-s.Addr > s.Size || size > s.Size-(address-s.Addr) || s.Offset+(address-s.Addr) != fileOffset {
			return 0, 0, fmt.Errorf("range lacks a matching uncompressed file-backed section")
		}
		sectionIndex = i
	}
	if sectionIndex == -1 {
		return 0, 0, fmt.Errorf("range has no allocated section")
	}
	return fileOffset, sectionIndex, nil
}
