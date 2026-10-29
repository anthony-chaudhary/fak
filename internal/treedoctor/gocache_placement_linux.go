//go:build linux

package treedoctor

import (
	"fmt"
	"os"
	"syscall"
)

func samePlacementFilesystem(a, b string) (bool, error) {
	identify := func(path string) (uint64, int64, syscall.Fsid, error) {
		fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_DIRECTORY|syscall.O_CLOEXEC|syscall.O_NOFOLLOW, 0)
		if err != nil {
			return 0, 0, syscall.Fsid{}, err
		}
		f := os.NewFile(uintptr(fd), path)
		defer f.Close()
		info, err := f.Stat()
		if err != nil {
			return 0, 0, syscall.Fsid{}, err
		}
		st, ok := info.Sys().(*syscall.Stat_t)
		if !ok {
			return 0, 0, syscall.Fsid{}, fmt.Errorf("filesystem identity unavailable for %s", path)
		}
		var fs syscall.Statfs_t
		if err := syscall.Fstatfs(fd, &fs); err != nil {
			return 0, 0, syscall.Fsid{}, err
		}
		return uint64(st.Dev), int64(fs.Type), fs.Fsid, nil
	}
	ad, at, ai, err := identify(a)
	if err != nil {
		return false, err
	}
	bd, bt, bi, err := identify(b)
	if err != nil {
		return false, err
	}
	return ad == bd && at == bt && ai == bi, nil
}

// Open nonblocking before testing type: a FIFO/device must not hang a
// diagnostic, and NOFOLLOW avoids accepting a retargeted final symlink.
func openPlacementPlan(path string) (*os.File, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("plan must be a regular file")
	}
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_NONBLOCK|syscall.O_CLOEXEC|syscall.O_NOFOLLOW, 0)
	if err != nil {
		return nil, err
	}
	f := os.NewFile(uintptr(fd), path)
	info, err = f.Stat()
	if err != nil || !info.Mode().IsRegular() {
		f.Close()
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("plan must be a regular file")
	}
	return f, nil
}
