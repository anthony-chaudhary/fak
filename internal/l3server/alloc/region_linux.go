//go:build linux

package alloc

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"unsafe"
)

const (
	mapHuge2MB = 21 << 26
	mapHuge1GB = 30 << 26
)

func (r *Region) allocate() error {
	flags := syscall.MAP_PRIVATE | syscall.MAP_ANONYMOUS | 0x8000
	if r.useHuge {
		flags |= syscall.MAP_HUGETLB
		switch r.hugePageSizeKB {
		case 1048576:
			flags |= mapHuge1GB
		case 2048:
			flags |= mapHuge2MB
		}
	}
	data, err := syscall.Mmap(-1, 0, int(r.size), syscall.PROT_READ|syscall.PROT_WRITE, flags)
	if err != nil {
		if r.useHuge {
			if r.hugePageSizeKB == 1048576 {
				fallbackFlags := syscall.MAP_PRIVATE | syscall.MAP_ANONYMOUS | 0x8000 |
					syscall.MAP_HUGETLB | mapHuge2MB
				data, err = syscall.Mmap(-1, 0, int(r.size), syscall.PROT_READ|syscall.PROT_WRITE, fallbackFlags)
				if err == nil {
					r.data = data
					r.isMapped = true
					r.gotHuge = true
					r.gotHugeSizeKB = 2048
					return nil
				}
			}
			data, err = syscall.Mmap(-1, 0, int(r.size), syscall.PROT_READ|syscall.PROT_WRITE,
				syscall.MAP_PRIVATE|syscall.MAP_ANONYMOUS|0x8000)
			if err != nil {
				return err
			}
			r.data = data
			r.isMapped = true
			mode := GetTHPMode()
			if mode == "madvise" || mode == "always" {
				madviseHugepage(data)
				r.thpHinted = true
			}
			return nil
		}
		return err
	}
	r.data = data
	r.isMapped = true
	if r.useHuge {
		r.gotHuge = true
		r.gotHugeSizeKB = r.hugePageSizeKB
	}
	return nil
}

func (r *Region) munmap() error {
	if r.data != nil {
		return syscall.Munmap(r.data)
	}
	return nil
}

func TryMlockall() error {
	const MCL_CURRENT = 1
	const MCL_FUTURE = 2
	_, _, errno := syscall.Syscall(syscall.SYS_MLOCKALL, MCL_CURRENT|MCL_FUTURE, 0, 0)
	if errno != 0 {
		return fmt.Errorf("mlockall(MCL_CURRENT|MCL_FUTURE) failed: %v (ensure ulimit -l unlimited)", errno)
	}
	return nil
}

func (r *Region) PinPages() error {
	if len(r.data) == 0 || r.pinned {
		return nil
	}
	if err := syscall.Mlock(r.data); err != nil {
		return fmt.Errorf("mlock failed (%d bytes): %w", len(r.data), err)
	}
	r.pinned = true
	const MADV_DONTFORK = 10
	ptr := unsafe.Pointer(&r.data[0])
	_, _, errno := syscall.Syscall(syscall.SYS_MADVISE, uintptr(ptr), uintptr(len(r.data)), MADV_DONTFORK)
	if errno != 0 {
		return fmt.Errorf("madvise(MADV_DONTFORK) failed: %v (pages still locked)", errno)
	}
	return nil
}

func NewDevdaxRegion(devPath string, offset, size uint64) (*Region, error) {
	const align2MB = 2 * 1024 * 1024
	if size == 0 {
		return nil, fmt.Errorf("devdax region size must be > 0")
	}
	if offset%align2MB != 0 {
		return nil, fmt.Errorf("devdax offset %d is not 2MB-aligned", offset)
	}
	if size%align2MB != 0 {
		return nil, fmt.Errorf("devdax size %d is not 2MB-aligned", size)
	}

	fd, err := syscall.Open(devPath, syscall.O_RDWR, 0)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", devPath, err)
	}

	data, err := syscall.Mmap(fd, int64(offset), int(size),
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_SHARED|0x8000)
	syscall.Close(fd)
	if err != nil {
		return nil, fmt.Errorf("mmap %s (offset=%d, size=%d): %w", devPath, offset, size, err)
	}

	return &Region{
		data:         data,
		size:         size,
		isMapped:     true,
		gotHuge:      true,
		devdaxFd:     -1,
		devdaxOffset: offset,
		devdaxPath:   devPath,
	}, nil
}

func queryDevdaxCapacityImpl(devPath string) (uint64, error) {
	devName := filepath.Base(devPath)
	sysPath := fmt.Sprintf("/sys/class/dax/%s/size", devName)
	data, err := os.ReadFile(sysPath)
	if err != nil {
		return 0, fmt.Errorf("cannot read devdax capacity from %s: %w", sysPath, err)
	}
	s := strings.TrimSpace(string(data))
	cap, err := strconv.ParseUint(s, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid devdax capacity %q from %s: %w", s, sysPath, err)
	}
	return cap, nil
}
