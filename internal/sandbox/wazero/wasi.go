package wazero

import (
	"bytes"
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"time"
)

// Standard WASI preview1 errno values.
const (
	wasiSuccess    uint64 = 0
	wasiBadF       uint64 = 8
	wasiInval      uint64 = 28
	wasiNoSys      uint64 = 52
	wasiPipe       uint64 = 58
	wasiPerm       uint64 = 63
	wasiNotCapable uint64 = 76
)

type procExitError struct {
	Code int
}

func (e *procExitError) Error() string {
	return fmt.Sprintf("proc_exit(%d)", e.Code)
}

// WASIContext manages isolated streams, arguments, environment, and in-memory VFS.
type WASIContext struct {
	Stdin    []byte
	stdinOff int
	Stdout   bytes.Buffer
	Stderr   bytes.Buffer
	Env      []string
	Argv     []string
	VFS      map[string][]byte
}

// NewWASIContext initializes a WASI execution context.
func NewWASIContext(stdin []byte, env, argv []string, vfs map[string][]byte) *WASIContext {
	if vfs == nil {
		vfs = make(map[string][]byte)
	}
	return &WASIContext{
		Stdin: stdin,
		Env:   env,
		Argv:  argv,
		VFS:   vfs,
	}
}

// Dispatch routes a wasi_snapshot_preview1 host function invocation.
func (w *WASIContext) Dispatch(vm *VM, field string, args []uint64) (uint64, error) {
	mem := vm.memory

	switch field {
	case "fd_write":
		if len(args) < 4 {
			return wasiInval, nil
		}
		fd := int32(args[0])
		iovsPtr := uint32(args[1])
		iovsLen := uint32(args[2])
		nwrittenPtr := uint32(args[3])
		return w.fdWrite(mem, fd, iovsPtr, iovsLen, nwrittenPtr)

	case "fd_read":
		if len(args) < 4 {
			return wasiInval, nil
		}
		fd := int32(args[0])
		iovsPtr := uint32(args[1])
		iovsLen := uint32(args[2])
		nreadPtr := uint32(args[3])
		return w.fdRead(mem, fd, iovsPtr, iovsLen, nreadPtr)

	case "proc_exit":
		if len(args) < 1 {
			return wasiInval, nil
		}
		rval := int32(args[0])
		return 0, &procExitError{Code: int(rval)}

	case "environ_sizes_get":
		if len(args) < 2 {
			return wasiInval, nil
		}
		countPtr := uint32(args[0])
		sizePtr := uint32(args[1])
		return w.environSizesGet(mem, countPtr, sizePtr)

	case "environ_get":
		if len(args) < 2 {
			return wasiInval, nil
		}
		ptrsPtr := uint32(args[0])
		bufPtr := uint32(args[1])
		return w.environGet(mem, ptrsPtr, bufPtr)

	case "args_sizes_get":
		if len(args) < 2 {
			return wasiInval, nil
		}
		argcPtr := uint32(args[0])
		sizePtr := uint32(args[1])
		return w.argsSizesGet(mem, argcPtr, sizePtr)

	case "args_get":
		if len(args) < 2 {
			return wasiInval, nil
		}
		argvPtrsPtr := uint32(args[0])
		argvBufPtr := uint32(args[1])
		return w.argsGet(mem, argvPtrsPtr, argvBufPtr)

	case "clock_time_get":
		if len(args) < 3 {
			return wasiInval, nil
		}
		timePtr := uint32(args[2])
		return w.clockTimeGet(mem, timePtr)

	case "random_get":
		if len(args) < 2 {
			return wasiInval, nil
		}
		bufPtr := uint32(args[0])
		bufLen := uint32(args[1])
		return w.randomGet(mem, bufPtr, bufLen)

	case "fd_close":
		return wasiSuccess, nil

	case "fd_seek":
		return wasiSuccess, nil

	case "fd_fdstat_get":
		if len(args) < 2 {
			return wasiInval, nil
		}
		fd := int32(args[0])
		statPtr := uint32(args[1])
		return w.fdFdstatGet(mem, fd, statPtr)

	case "fd_prestat_get", "fd_prestat_dir_name":
		// Host filesystem isolation: zero pre-opened directories exposed
		return wasiBadF, nil

	case "sched_yield":
		return wasiSuccess, nil

	case "poll_oneoff":
		return wasiSuccess, nil

	case "path_open", "path_filestat_get", "fd_filestat_get", "path_create_directory", "path_unlink_file":
		// Host filesystem access is 100% disabled. Fails closed.
		return wasiNotCapable, nil

	default:
		return wasiNoSys, nil
	}
}

func (w *WASIContext) fdWrite(mem []byte, fd int32, iovsPtr, iovsLen, nwrittenPtr uint32) (uint64, error) {
	var totalWritten uint32
	for i := uint32(0); i < iovsLen; i++ {
		iovOffset := iovsPtr + i*8
		if int(iovOffset)+8 > len(mem) {
			return wasiInval, nil
		}
		ptr := binary.LittleEndian.Uint32(mem[iovOffset : iovOffset+4])
		length := binary.LittleEndian.Uint32(mem[iovOffset+4 : iovOffset+8])
		if int(ptr)+int(length) > len(mem) {
			return wasiInval, nil
		}
		data := mem[ptr : ptr+length]
		if fd == 1 {
			w.Stdout.Write(data)
			totalWritten += length
		} else if fd == 2 {
			w.Stderr.Write(data)
			totalWritten += length
		} else {
			return wasiBadF, nil
		}
	}
	if int(nwrittenPtr)+4 <= len(mem) {
		binary.LittleEndian.PutUint32(mem[nwrittenPtr:nwrittenPtr+4], totalWritten)
	}
	return wasiSuccess, nil
}

func (w *WASIContext) fdRead(mem []byte, fd int32, iovsPtr, iovsLen, nreadPtr uint32) (uint64, error) {
	if fd != 0 {
		return wasiBadF, nil
	}
	var totalRead uint32
	for i := uint32(0); i < iovsLen; i++ {
		if w.stdinOff >= len(w.Stdin) {
			break
		}
		iovOffset := iovsPtr + i*8
		if int(iovOffset)+8 > len(mem) {
			return wasiInval, nil
		}
		ptr := binary.LittleEndian.Uint32(mem[iovOffset : iovOffset+4])
		length := binary.LittleEndian.Uint32(mem[iovOffset+4 : iovOffset+8])
		if int(ptr)+int(length) > len(mem) {
			return wasiInval, nil
		}
		avail := len(w.Stdin) - w.stdinOff
		toRead := int(length)
		if toRead > avail {
			toRead = avail
		}
		copy(mem[ptr:ptr+uint32(toRead)], w.Stdin[w.stdinOff:w.stdinOff+toRead])
		w.stdinOff += toRead
		totalRead += uint32(toRead)
	}
	if int(nreadPtr)+4 <= len(mem) {
		binary.LittleEndian.PutUint32(mem[nreadPtr:nreadPtr+4], totalRead)
	}
	return wasiSuccess, nil
}

func (w *WASIContext) environSizesGet(mem []byte, countPtr, sizePtr uint32) (uint64, error) {
	if int(countPtr)+4 > len(mem) || int(sizePtr)+4 > len(mem) {
		return wasiInval, nil
	}
	binary.LittleEndian.PutUint32(mem[countPtr:], uint32(len(w.Env)))
	var totalSize uint32
	for _, e := range w.Env {
		totalSize += uint32(len(e) + 1)
	}
	binary.LittleEndian.PutUint32(mem[sizePtr:], totalSize)
	return wasiSuccess, nil
}

func (w *WASIContext) environGet(mem []byte, ptrsPtr, bufPtr uint32) (uint64, error) {
	currBuf := bufPtr
	for i, e := range w.Env {
		ptrOffset := ptrsPtr + uint32(i*4)
		if int(ptrOffset)+4 > len(mem) {
			return wasiInval, nil
		}
		binary.LittleEndian.PutUint32(mem[ptrOffset:], currBuf)

		eBytes := []byte(e)
		if int(currBuf)+len(eBytes)+1 > len(mem) {
			return wasiInval, nil
		}
		copy(mem[currBuf:], eBytes)
		mem[currBuf+uint32(len(eBytes))] = 0
		currBuf += uint32(len(eBytes) + 1)
	}
	return wasiSuccess, nil
}

func (w *WASIContext) argsSizesGet(mem []byte, argcPtr, sizePtr uint32) (uint64, error) {
	if int(argcPtr)+4 > len(mem) || int(sizePtr)+4 > len(mem) {
		return wasiInval, nil
	}
	binary.LittleEndian.PutUint32(mem[argcPtr:], uint32(len(w.Argv)))
	var totalSize uint32
	for _, a := range w.Argv {
		totalSize += uint32(len(a) + 1)
	}
	binary.LittleEndian.PutUint32(mem[sizePtr:], totalSize)
	return wasiSuccess, nil
}

func (w *WASIContext) argsGet(mem []byte, argvPtrsPtr, argvBufPtr uint32) (uint64, error) {
	currBuf := argvBufPtr
	for i, a := range w.Argv {
		ptrOffset := argvPtrsPtr + uint32(i*4)
		if int(ptrOffset)+4 > len(mem) {
			return wasiInval, nil
		}
		binary.LittleEndian.PutUint32(mem[ptrOffset:], currBuf)

		aBytes := []byte(a)
		if int(currBuf)+len(aBytes)+1 > len(mem) {
			return wasiInval, nil
		}
		copy(mem[currBuf:], aBytes)
		mem[currBuf+uint32(len(aBytes))] = 0
		currBuf += uint32(len(aBytes) + 1)
	}
	return wasiSuccess, nil
}

func (w *WASIContext) clockTimeGet(mem []byte, timePtr uint32) (uint64, error) {
	if int(timePtr)+8 > len(mem) {
		return wasiInval, nil
	}
	nowNano := uint64(time.Now().UnixNano())
	binary.LittleEndian.PutUint64(mem[timePtr:], nowNano)
	return wasiSuccess, nil
}

func (w *WASIContext) randomGet(mem []byte, bufPtr, bufLen uint32) (uint64, error) {
	if int(bufPtr)+int(bufLen) > len(mem) {
		return wasiInval, nil
	}
	_, _ = rand.Read(mem[bufPtr : bufPtr+bufLen])
	return wasiSuccess, nil
}

func (w *WASIContext) fdFdstatGet(mem []byte, fd int32, statPtr uint32) (uint64, error) {
	if int(statPtr)+24 > len(mem) {
		return wasiInval, nil
	}
	if fd >= 0 && fd <= 2 {
		mem[statPtr] = 2 // character device
		binary.LittleEndian.PutUint16(mem[statPtr+2:], 0)
		binary.LittleEndian.PutUint64(mem[statPtr+8:], 0x1ffff)
		binary.LittleEndian.PutUint64(mem[statPtr+16:], 0x1ffff)
		return wasiSuccess, nil
	}
	return wasiBadF, nil
}
