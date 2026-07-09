//go:build windows

package windowgate

import (
	"fmt"
	"runtime"
	"syscall"
	"unsafe"
)

// The VERSIONINFO reader. Kept on syscall.NewLazyDLL rather than a typed wrapper
// so this package stays standard-library-only, matching the house pattern in
// internal/compute/hostmem_windows.go.
var (
	versionDLL                  = syscall.NewLazyDLL("version.dll")
	procGetFileVersionInfoSizeW = versionDLL.NewProc("GetFileVersionInfoSizeW")
	procGetFileVersionInfoW     = versionDLL.NewProc("GetFileVersionInfoW")
	procVerQueryValueW          = versionDLL.NewProc("VerQueryValueW")
)

// fixedFileInfoSignature is VS_FIXEDFILEINFO.dwSignature, which every valid
// block carries. Checked before the struct is trusted.
const fixedFileInfoSignature = 0xFEEF04BD

// fixedFileInfo mirrors VS_FIXEDFILEINFO. Only the two file-version words are
// read; the trailing flags/OS/type/date fields are skipped.
type fixedFileInfo struct {
	signature        uint32
	strucVersion     uint32
	fileVersionMS    uint32
	fileVersionLS    uint32
	productVersionMS uint32
	productVersionLS uint32
	_                [7]uint32
}

// ReadVersionInfo returns the StringFileInfo "FileVersion" and "ProductVersion"
// of a PE image.
//
// The string block is authoritative, not VS_FIXEDFILEINFO: the fixed struct packs
// each component into 16 bits, so it cannot represent the ConPTY floor's
// 260402001 build — which is exactly why ProductVersion exists as a string. The
// fixed struct is only a fallback for images carrying no string block.
func ReadVersionInfo(path string) (ConPTYVersionInfo, error) {
	var out ConPTYVersionInfo
	p, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return out, fmt.Errorf("windowgate: %s: %w", path, err)
	}
	size, _, callErr := procGetFileVersionInfoSizeW.Call(uintptr(unsafe.Pointer(p)), 0)
	if size == 0 {
		return out, fmt.Errorf("windowgate: %s: no version resource: %w", path, callErr)
	}
	buf := make([]byte, size)
	ok, _, callErr := procGetFileVersionInfoW.Call(
		uintptr(unsafe.Pointer(p)), 0, size, uintptr(unsafe.Pointer(&buf[0])))
	if ok == 0 {
		return out, fmt.Errorf("windowgate: %s: GetFileVersionInfo: %w", path, callErr)
	}
	defer runtime.KeepAlive(buf)

	out.FileVersion = queryVersionString(buf, "FileVersion")
	out.ProductVersion = queryVersionString(buf, "ProductVersion")
	if out.FileVersion == "" {
		if v, ok := queryFixedVersion(buf); ok {
			out.FileVersion = v
		}
	}
	if out.FileVersion == "" && out.ProductVersion == "" {
		return out, fmt.Errorf("windowgate: %s: version resource carries no FileVersion", path)
	}
	return out, nil
}

// queryVersionString reads one named value out of the StringFileInfo block,
// trying the translations the image declares before the two codepages that ship
// on almost every binary.
func queryVersionString(buf []byte, name string) string {
	for _, t := range translations(buf) {
		key := fmt.Sprintf(`\StringFileInfo\%04x%04x\%s`, t[0], t[1], name)
		if v := queryString(buf, key); v != "" {
			return v
		}
	}
	for _, cp := range []string{
		"040904b0", // US English, Unicode
		"040904e4", // US English, Windows Multilingual
	} {
		if v := queryString(buf, `\StringFileInfo\`+cp+`\`+name); v != "" {
			return v
		}
	}
	return ""
}

// translations reads \VarFileInfo\Translation: an array of (langID, codepage).
func translations(buf []byte) [][2]uint16 {
	var ptr unsafe.Pointer
	var n uint32
	sub, err := syscall.UTF16PtrFromString(`\VarFileInfo\Translation`)
	if err != nil {
		return nil
	}
	ok, _, _ := procVerQueryValueW.Call(
		uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(sub)),
		uintptr(unsafe.Pointer(&ptr)), uintptr(unsafe.Pointer(&n)))
	runtime.KeepAlive(buf)
	if ok == 0 || ptr == nil || n < 4 {
		return nil
	}
	// n is a byte count over (langID, codepage) pairs, each 4 bytes wide.
	raw := unsafe.Slice((*uint16)(ptr), int(n)/2)
	out := make([][2]uint16, 0, len(raw)/2)
	for i := 0; i+1 < len(raw); i += 2 {
		out = append(out, [2]uint16{raw[i], raw[i+1]})
	}
	return out
}

// queryString reads one NUL-terminated UTF-16 value out of the version block.
func queryString(buf []byte, sub string) string {
	key, err := syscall.UTF16PtrFromString(sub)
	if err != nil {
		return ""
	}
	var ptr unsafe.Pointer
	var n uint32
	ok, _, _ := procVerQueryValueW.Call(
		uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(key)),
		uintptr(unsafe.Pointer(&ptr)), uintptr(unsafe.Pointer(&n)))
	runtime.KeepAlive(buf)
	if ok == 0 || ptr == nil || n == 0 {
		return ""
	}
	// n is the length in UTF-16 code units, including the terminating NUL.
	u := unsafe.Slice((*uint16)(ptr), int(n))
	for i, c := range u {
		if c == 0 {
			u = u[:i]
			break
		}
	}
	return syscall.UTF16ToString(u)
}

// queryFixedVersion falls back to VS_FIXEDFILEINFO's packed 4x16-bit tuple.
func queryFixedVersion(buf []byte) (string, bool) {
	root, err := syscall.UTF16PtrFromString(`\`)
	if err != nil {
		return "", false
	}
	var ptr unsafe.Pointer
	var n uint32
	ok, _, _ := procVerQueryValueW.Call(
		uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(root)),
		uintptr(unsafe.Pointer(&ptr)), uintptr(unsafe.Pointer(&n)))
	runtime.KeepAlive(buf)
	if ok == 0 || ptr == nil || uintptr(n) < unsafe.Sizeof(fixedFileInfo{}) {
		return "", false
	}
	ffi := (*fixedFileInfo)(ptr)
	if ffi.signature != fixedFileInfoSignature {
		return "", false
	}
	ms, ls := ffi.fileVersionMS, ffi.fileVersionLS
	if ms == 0 && ls == 0 {
		return "", false
	}
	return fmt.Sprintf("%d.%d.%d.%d", ms>>16, ms&0xffff, ls>>16, ls&0xffff), true
}
