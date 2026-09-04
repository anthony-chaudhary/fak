package compute

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// CUDAOpError represents a typed runtime failure encountered during a CUDA backend operation.
// It declares the operation (Op), choke point or stage (Site), failure message (Msg),
// status code (Code), memory class (Class), and underlying cause (Err) if applicable.
type CUDAOpError struct {
	Op    string      // Operation name, e.g. "GraphEndLaunch", "UploadConstantParam", "MatMul", "BatchedMatMul"
	Site  string      // Specific choke point or stage, e.g. "cuda-graph-launch", "upload-constant-param"
	Msg   string      // Descriptive failure message
	Code  int         // CUDA driver or runtime status code (0 = none)
	Class MemoryClass // Memory class if applicable, e.g. MemoryWeights, MemoryActivation
	Err   error       // Underlying cause, if any
}

// CUDAError is an alias for CUDAOpError.
type CUDAError = CUDAOpError

// CUDABackendError is an alias for CUDAOpError.
type CUDABackendError = CUDAOpError

func (e *CUDAOpError) Error() string {
	if e == nil {
		return "compute: nil cuda op error"
	}
	msg := e.Msg
	if msg == "" && e.Err != nil {
		msg = e.Err.Error()
	}
	if msg == "" {
		msg = "cuda operation failed"
	}
	var b strings.Builder
	if !strings.HasPrefix(msg, "compute:") && !strings.HasPrefix(msg, "compute/") {
		b.WriteString("compute: cuda ")
		if e.Op != "" {
			b.WriteString(e.Op)
			b.WriteString(": ")
		}
	}
	b.WriteString(msg)
	if e.Site != "" && !strings.Contains(b.String(), e.Site) {
		b.WriteString(" (site: ")
		b.WriteString(e.Site)
		b.WriteString(")")
	}
	if e.Code != 0 && !strings.Contains(b.String(), "code "+strconv.Itoa(e.Code)) {
		b.WriteString(" (code ")
		b.WriteString(strconv.Itoa(e.Code))
		b.WriteString(")")
	}
	if e.Class != "" && e.Class != MemoryUnknown && !strings.Contains(b.String(), string(e.Class)) {
		b.WriteString(" [class: ")
		b.WriteString(string(e.Class))
		b.WriteString("]")
	}
	return b.String()
}

// Unwrap returns the underlying error, if any.
func (e *CUDAOpError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// CUDALaunchError represents a kernel or graph launch rejection or failure.
type CUDALaunchError struct {
	CUDAOpError
}

func (e *CUDALaunchError) Error() string {
	if e == nil {
		return "compute: nil cuda launch error"
	}
	return e.CUDAOpError.Error()
}

func (e *CUDALaunchError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.CUDAOpError.Err
}

func (e *CUDALaunchError) As(target any) bool {
	if t, ok := target.(**CUDAOpError); ok {
		*t = &e.CUDAOpError
		return true
	}
	return false
}

// AsCUDAError inspects an error or recovered panic value and extracts or converts it
// into a *CUDAOpError if it represents a CUDA failure.
// If r is already *CUDAOpError or *CUDALaunchError (or wraps one), it is returned.
// If r is a bare string or error indicating a CUDA failure, it converts it into a typed *CUDAOpError.
// If r is not a recognized CUDA failure, it returns (nil, false).
func AsCUDAError(r any) (*CUDAOpError, bool) {
	if r == nil {
		return nil, false
	}
	switch v := r.(type) {
	case *CUDAOpError:
		return v, true
	case *CUDALaunchError:
		if v != nil {
			return &v.CUDAOpError, true
		}
		return nil, false
	case error:
		var opErr *CUDAOpError
		if errors.As(v, &opErr) {
			return opErr, true
		}
		var launchErr *CUDALaunchError
		if errors.As(v, &launchErr) {
			return &launchErr.CUDAOpError, true
		}
		var dae *DeviceAllocError
		if errors.As(v, &dae) {
			return nil, false
		}
		var dfe *DeviceFaultError
		if errors.As(v, &dfe) {
			return nil, false
		}
		return parseCUDAErrorString(v.Error())
	case string:
		return parseCUDAErrorString(v)
	default:
		return nil, false
	}
}

func parseCUDAErrorString(s string) (*CUDAOpError, bool) {
	trimmed := strings.TrimSpace(s)
	isCUDA := strings.HasPrefix(trimmed, "compute: cuda") ||
		strings.HasPrefix(trimmed, "compute/cuda:") ||
		strings.HasPrefix(trimmed, "compute: UploadConstantParam") ||
		strings.HasPrefix(trimmed, "cuda:") ||
		strings.Contains(trimmed, "cuda graph")
	if !isCUDA {
		return nil, false
	}
	op := "op"
	site := "cuda"
	if strings.Contains(trimmed, "graph") {
		op = "GraphEndLaunch"
		site = "cuda-graph-launch"
	} else if strings.Contains(trimmed, "UploadConstantParam") {
		op = "UploadConstantParam"
		site = "upload-constant-param"
	} else if strings.Contains(trimmed, "BatchedMatMul") {
		op = "BatchedMatMul"
		site = "batched-matmul"
	} else if strings.Contains(trimmed, "MatMul") {
		op = "MatMul"
		site = "matmul"
	} else if strings.Contains(trimmed, "Upload") {
		op = "Upload"
		site = "upload"
	} else if strings.Contains(trimmed, "sigmoid") {
		op = "SigmoidMulInPlace"
		site = "sigmoid"
	} else if strings.Contains(trimmed, "query/gate") || strings.Contains(trimmed, "Qwen query/gate") {
		op = "SplitQwen35QueryGate"
		site = "split-qg"
	}
	return &CUDAOpError{
		Op:   op,
		Site: site,
		Msg:  trimmed,
	}, true
}

// ConvertCUDAPanic converts a recovered panic value into a typed backend error.
// If the recovered value is already a typed compute error (e.g. *CUDAOpError, *CUDALaunchError,
// *DeviceAllocError, *DeviceFaultError), it is returned. If it is a recognized CUDA failure
// string or error, it is converted into *CUDAOpError. Otherwise it returns an unclassified *CUDAOpError
// if op or site is non-empty, or nil if not recognized.
func ConvertCUDAPanic(r any, op, site string) (error, bool) {
	if r == nil {
		return nil, false
	}
	if err, ok := AsCUDAError(r); ok {
		if op != "" && err.Op == "op" {
			err.Op = op
		}
		if site != "" && err.Site == "cuda" {
			err.Site = site
		}
		return err, true
	}
	if dae, ok := r.(*DeviceAllocError); ok {
		return dae, true
	}
	if dfe, ok := r.(*DeviceFaultError); ok {
		return dfe, true
	}
	if err, ok := r.(error); ok {
		var dae *DeviceAllocError
		if errors.As(err, &dae) {
			return dae, true
		}
		var dfe *DeviceFaultError
		if errors.As(err, &dfe) {
			return dfe, true
		}
	}
	if op != "" || site != "" {
		msg := fmt.Sprint(r)
		return &CUDAOpError{
			Op:   op,
			Site: site,
			Msg:  msg,
		}, true
	}
	return nil, false
}

// RunGuardedCUDA executes fn, converting any CUDA panic into a typed error.
func RunGuardedCUDA(op, site string, fn func()) (err error) {
	defer func() {
		if r := recover(); r != nil {
			if converted, ok := ConvertCUDAPanic(r, op, site); ok {
				err = converted
				return
			}
			panic(r)
		}
	}()
	fn()
	return nil
}
