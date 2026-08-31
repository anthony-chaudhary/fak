//go:build windows

package codexresume

import (
	"fmt"
	"path/filepath"
	"syscall"
	"time"
	"unsafe"
)

const (
	errorMoreData                  = syscall.Errno(234)
	processQueryLimitedInformation = 0x1000
)

var (
	rstrtmgr                   = syscall.NewLazyDLL("rstrtmgr.dll")
	procRmStartSession         = rstrtmgr.NewProc("RmStartSession")
	procRmRegisterResources    = rstrtmgr.NewProc("RmRegisterResources")
	procRmGetList              = rstrtmgr.NewProc("RmGetList")
	procRmEndSession           = rstrtmgr.NewProc("RmEndSession")
	procQueryFullProcessImageW = syscall.NewLazyDLL("kernel32.dll").NewProc("QueryFullProcessImageNameW")
)

type nativeOwnershipProbe struct{}

type rmUniqueProcess struct {
	PID       uint32
	StartTime syscall.Filetime
}

type rmProcessInfo struct {
	Process          rmUniqueProcess
	AppName          [256]uint16
	ServiceShortName [64]uint16
	ApplicationType  uint32
	AppStatus        uint32
	SessionID        uint32
	Restartable      int32
}

func (nativeOwnershipProbe) inspect(lockPath string) (ownershipWitness, error) {
	const source = "windows_restart_manager"
	var session uint32
	key := make([]uint16, 33)
	if code, _, _ := procRmStartSession.Call(uintptr(unsafe.Pointer(&session)), 0, uintptr(unsafe.Pointer(&key[0]))); code != 0 {
		return ownershipWitness{source: source}, fmt.Errorf("RmStartSession: %w", syscall.Errno(code))
	}
	defer procRmEndSession.Call(uintptr(session))

	path, err := syscall.UTF16PtrFromString(filepath.Clean(lockPath))
	if err != nil {
		return ownershipWitness{source: source}, err
	}
	paths := []*uint16{path}
	if code, _, _ := procRmRegisterResources.Call(
		uintptr(session), 1, uintptr(unsafe.Pointer(&paths[0])), 0, 0, 0, 0,
	); code != 0 {
		return ownershipWitness{source: source}, fmt.Errorf("RmRegisterResources: %w", syscall.Errno(code))
	}

	infos, err := rmOwners(session)
	if err != nil {
		return ownershipWitness{source: source}, err
	}
	owners := make([]processOwner, 0, len(infos))
	for _, info := range infos {
		owner, ok, err := corroborateRMOwner(info.Process)
		if err != nil {
			return ownershipWitness{source: source}, err
		}
		if !ok {
			// The process exited between Restart Manager and OpenProcess. Re-query
			// the resource rather than turning that race into a stale verdict.
			refreshed, refreshErr := rmOwners(session)
			if refreshErr != nil {
				return ownershipWitness{source: source}, refreshErr
			}
			if len(refreshed) == 0 {
				return ownershipWitness{source: source, conclusive: true}, nil
			}
			return ownershipWitness{source: source}, fmt.Errorf("resource owner changed during inspection")
		}
		owners = append(owners, owner)
	}
	return ownershipWitness{source: source, conclusive: true, owners: owners}, nil
}

func rmOwners(session uint32) ([]rmProcessInfo, error) {
	var needed, count, reboot uint32
	code, _, _ := procRmGetList.Call(
		uintptr(session), uintptr(unsafe.Pointer(&needed)), uintptr(unsafe.Pointer(&count)), 0, uintptr(unsafe.Pointer(&reboot)),
	)
	if code == 0 && needed == 0 {
		return nil, nil
	}
	if syscall.Errno(code) != errorMoreData {
		return nil, fmt.Errorf("RmGetList(size): %w", syscall.Errno(code))
	}
	infos := make([]rmProcessInfo, needed)
	count = needed
	code, _, _ = procRmGetList.Call(
		uintptr(session), uintptr(unsafe.Pointer(&needed)), uintptr(unsafe.Pointer(&count)), uintptr(unsafe.Pointer(&infos[0])), uintptr(unsafe.Pointer(&reboot)),
	)
	if code != 0 {
		return nil, fmt.Errorf("RmGetList(data): %w", syscall.Errno(code))
	}
	return infos[:count], nil
}

func corroborateRMOwner(process rmUniqueProcess) (processOwner, bool, error) {
	if process.PID == 0 {
		return processOwner{}, false, fmt.Errorf("Restart Manager returned PID 0")
	}
	h, err := syscall.OpenProcess(processQueryLimitedInformation, false, process.PID)
	if err != nil {
		if err == syscall.Errno(87) {
			return processOwner{}, false, nil
		}
		return processOwner{}, false, fmt.Errorf("open owner PID %d: %w", process.PID, err)
	}
	defer syscall.CloseHandle(h)

	var created, exited, kernel, user syscall.Filetime
	if err := syscall.GetProcessTimes(h, &created, &exited, &kernel, &user); err != nil {
		return processOwner{}, false, fmt.Errorf("read owner PID %d start time: %w", process.PID, err)
	}
	rmToken := filetimeToken(process.StartTime)
	createdToken := filetimeToken(created)
	if rmToken == 0 || createdToken != rmToken {
		return processOwner{}, false, fmt.Errorf("owner PID %d identity changed during inspection", process.PID)
	}
	image, err := processImage(h)
	if err != nil {
		return processOwner{}, false, fmt.Errorf("read owner PID %d image: %w", process.PID, err)
	}
	return processOwner{
		pid:        int(process.PID),
		startTime:  time.Unix(0, created.Nanoseconds()).UTC().Format(time.RFC3339Nano),
		startToken: createdToken,
		image:      image,
	}, true, nil
}

func processImage(h syscall.Handle) (string, error) {
	buf := make([]uint16, 32768)
	size := uint32(len(buf))
	r, _, callErr := procQueryFullProcessImageW.Call(
		uintptr(h), 0, uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(&size)),
	)
	if r == 0 {
		return "", callErr
	}
	return syscall.UTF16ToString(buf[:size]), nil
}

func filetimeToken(ft syscall.Filetime) uint64 {
	return uint64(ft.HighDateTime)<<32 | uint64(ft.LowDateTime)
}
