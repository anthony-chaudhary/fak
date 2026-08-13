//go:build windows

package main

import (
	"fmt"
	"io"
	"os"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

// runSessionConPTY executes one deterministic command through the real Windows
// pseudoconsole API and returns the exact VT byte stream emitted by ConHost.
func runSessionConPTY(command string, timeout time.Duration) ([]byte, error) {
	var pseudoInputRead, pseudoInputWrite windows.Handle
	if err := windows.CreatePipe(&pseudoInputRead, &pseudoInputWrite, nil, 0); err != nil {
		return nil, err
	}
	defer windows.CloseHandle(pseudoInputRead)
	defer windows.CloseHandle(pseudoInputWrite)
	var pseudoOutputRead, pseudoOutputWrite windows.Handle
	if err := windows.CreatePipe(&pseudoOutputRead, &pseudoOutputWrite, nil, 0); err != nil {
		return nil, err
	}
	defer windows.CloseHandle(pseudoOutputRead)
	defer windows.CloseHandle(pseudoOutputWrite)
	var pseudo windows.Handle
	if err := windows.CreatePseudoConsole(windows.Coord{X: 120, Y: 30}, pseudoInputRead, pseudoOutputWrite, 0, &pseudo); err != nil {
		return nil, fmt.Errorf("CreatePseudoConsole: %w", err)
	}
	defer func() {
		if pseudo != 0 {
			windows.ClosePseudoConsole(pseudo)
		}
	}()
	windows.CloseHandle(pseudoInputRead)
	pseudoInputRead = 0
	windows.CloseHandle(pseudoOutputWrite)
	pseudoOutputWrite = 0
	attrs, err := windows.NewProcThreadAttributeList(1)
	if err != nil {
		return nil, err
	}
	defer attrs.Delete()
	if err := attrs.Update(windows.PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE, *(*unsafe.Pointer)(unsafe.Pointer(&pseudo)), unsafe.Sizeof(pseudo)); err != nil {
		return nil, err
	}
	si := windows.StartupInfoEx{StartupInfo: windows.StartupInfo{Cb: uint32(unsafe.Sizeof(windows.StartupInfoEx{})), Flags: windows.STARTF_USESTDHANDLES}, ProcThreadAttributeList: attrs.List()}
	exePath := os.Getenv("ComSpec")
	if exePath == "" {
		exePath = `C:\Windows\System32\cmd.exe`
	}
	line, err := windows.UTF16PtrFromString(exePath + " /d /s /c " + command)
	if err != nil {
		return nil, err
	}
	pi := new(windows.ProcessInformation)
	flags := uint32(windows.EXTENDED_STARTUPINFO_PRESENT | windows.CREATE_UNICODE_ENVIRONMENT)
	var sec windows.SecurityAttributes
	sec.Length = uint32(unsafe.Sizeof(sec))
	sec.InheritHandle = 1
	if err := windows.CreateProcess(nil, line, &sec, &sec, false, flags, nil, nil, &si.StartupInfo, pi); err != nil {
		return nil, fmt.Errorf("CreateProcess ConPTY: %w", err)
	}
	windows.CloseHandle(pi.Thread)
	done := make(chan error, 1)
	go func() { _, err := windows.WaitForSingleObject(pi.Process, windows.INFINITE); done <- err }()
	file := os.NewFile(uintptr(pseudoOutputRead), "conpty-output")
	read := make(chan struct {
		b   []byte
		err error
	}, 1)
	go func() {
		b, err := io.ReadAll(file)
		read <- struct {
			b   []byte
			err error
		}{b, err}
	}()
	select {
	case err := <-done:
		if err != nil {
			return nil, err
		}
	case <-time.After(timeout):
		windows.TerminateProcess(pi.Process, 124)
		return nil, fmt.Errorf("ConPTY command timed out after %s", timeout)
	}
	var exit uint32
	_ = windows.GetExitCodeProcess(pi.Process, &exit)
	windows.CloseHandle(pi.Process)
	windows.CloseHandle(pseudoInputWrite)
	pseudoInputWrite = 0
	windows.ClosePseudoConsole(pseudo)
	pseudo = 0
	result := <-read
	_ = file.Close()
	pseudoOutputRead = 0
	if result.err != nil {
		return result.b, result.err
	}
	if exit != 0 {
		return result.b, fmt.Errorf("ConPTY command exit=%d transcript=%q", exit, result.b)
	}
	return result.b, nil
}
