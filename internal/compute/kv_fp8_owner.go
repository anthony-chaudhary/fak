package compute

import (
	"errors"
	"fmt"
	"math"
	"sync/atomic"
	"unsafe"
)

var (
	errKVFP8Owner      = errors.New("compute: FP8 KV owner is not usable")
	errKVFP8Completion = errors.New("compute: FP8 KV completion is uncertain")
	errKVFP8Retirement = errors.New("compute: FP8 KV retirement is uncertain")
	errKVFP8Budget     = errors.New("compute: FP8 KV replacement exceeds admitted peak budget")
)

// kvFP8ReceiptError preserves the exact native status together with the
// semantic failure class. A zero status still cannot certify missing completion
// or a missing retirement disposition.
type kvFP8ReceiptError struct {
	cause  error
	status int
}

func (e *kvFP8ReceiptError) Error() string {
	return fmt.Sprintf("%v (native status %d)", e.cause, e.status)
}
func (e *kvFP8ReceiptError) Unwrap() error { return e.cause }

const (
	kvFP8Raw = iota
	kvFP8Key
	kvFP8Value
	kvFP8KeyScale
	kvFP8ValueScale
	kvFP8Planes
)

// kvFP8BackendIdentity is embedded once in the CUDA backend. Its address, not
// a display name or device number, identifies the backend owning a descriptor.
// stopped is monotonic. The future native fault bridge MUST set it for every
// terminal shared-stream outcome; no adapter may clear or replace this identity
// to revive existing owners. That bridge is an activation prerequisite.
// The nonzero size avoids the unspecified equality of zero-sized Go objects.
type kvFP8BackendIdentity struct{ stopped atomic.Bool }

type kvFP8Allocation struct {
	ptr   unsafe.Pointer
	bytes int
}

type kvFP8Image struct {
	positions int
	layers    [][kvFP8Planes]kvFP8Allocation
	workspace []kvFP8Allocation // status and intermediate staging, attached before allocation
}

// kvFP8Owner owns five physical planes per layer, including exact normalized
// pre-RoPE Kraw. This is an internal eager lifecycle primitive, not a KVStore.
// All methods except borrowed Ready require the caller's CUDA serialization
// domain. Callbacks run in that same domain and must not acquire it recursively.
type kvFP8Owner struct {
	backend                        *kvFP8BackendIdentity
	device, layers, heads, headDim int
	rotary                         []kvFP8RotaryContract
	image                          *kvFP8Image
	stage                          *kvFP8Image
	retiring                       *kvFP8Image

	// Low bits: 1 usable, 0 operation in progress, 2 terminal owner-local poison.
	// Upper bits are a nonwrapping generation. Only checked-complete images
	// become usable; there is no independent unverified "ready" publication.
	stamp atomic.Uint64
	// Preserve a rejection before cleanup can panic or Goexit. Diagnostics
	// are metadata only and never authorize reuse or a cleanup retry.
	primaryFailure     error
	numericalFailure   uint32 // most recent checked numerical mask, preserved through cleanup
	cleanupFailure     error
	cleanupInterrupted bool
	uncertain          bool
	retained           bool
	retainNext         *kvFP8Owner
}

// Quarantine is allocation-free and roots every still-owned plane. Its lock is
// the caller's CUDA serialization domain. A native adapter must ALSO close the
// shared stream after uncertain execution. This local list cannot do that.
var kvFP8RetainedOwners *kvFP8Owner

type kvFP8Retirement struct {
	retired bool // ownership transferred or physical free conclusively completed
	status  int
}

type kvFP8Completion struct {
	completed        bool // checked fence covers every staged write and source read
	numericalChecked bool
	numerical        uint32
	status           int
}

// allocate must store a returned pointer in dst before any later failure/panic.
// Allocations must be fresh, nonoverlapping and recorded in native accounting.
// release must report disposition even on failure; retired pointers are never
// retried. A void free is NOT a suitable implementation of this interface.
type kvFP8OwnerOps struct {
	allocate func(dst *kvFP8Allocation, bytes, layer, plane int) error
	release  func(kvFP8Allocation) kvFP8Retirement
}

func newKVFP8Owner(backend *kvFP8BackendIdentity, device, layers, heads, hd int) (*kvFP8Owner, error) {
	if backend != nil && backend.stopped.Load() {
		return nil, errKVFP8Owner
	}
	if backend == nil || device < 0 || layers <= 0 || int64(layers) > math.MaxInt32 {
		return nil, ErrKVFP8Geometry
	}
	if _, _, err := kvFP8Shape(0, heads, hd); err != nil {
		return nil, err
	}
	o := &kvFP8Owner{backend: backend, device: device, layers: layers, heads: heads, headDim: hd}
	o.image = &kvFP8Image{layers: make([][kvFP8Planes]kvFP8Allocation, layers)}
	o.rotary = make([]kvFP8RotaryContract, layers)
	o.stamp.Store(1)
	return o, nil
}

func (o *kvFP8Owner) usable() error {
	if o == nil || o.backend == nil || o.backend.stopped.Load() || o.stamp.Load()&3 != 1 || o.uncertain || o.retained || o.stage != nil || o.retiring != nil {
		return errKVFP8Owner
	}
	return nil
}

func (o *kvFP8Owner) retainUncertain() {
	if o.backend != nil {
		o.backend.stopped.Store(true)
	}
	o.uncertain = true
	o.stamp.Store(o.stamp.Load()&^3 | 2)
	if !o.retained {
		o.retained = true
		o.retainNext, kvFP8RetainedOwners = kvFP8RetainedOwners, o
	}
}

// poisonNumerical is owner-local. The hybrid caller must also invalidate its
// GDN states and any dependent graph plan. This does not declare device loss.
func (o *kvFP8Owner) poisonNumerical() {
	if o != nil {
		o.stamp.Store(o.stamp.Load()&^3 | 2)
	}
}

func (o *kvFP8Owner) planeBytes(positions int) ([kvFP8Planes]int, int64, error) {
	var sizes [kvFP8Planes]int
	rows, elements, err := kvFP8Shape(positions, o.heads, o.headDim)
	if err != nil {
		return sizes, 0, err
	}
	maxInt := int(^uint(0) >> 1)
	if elements > maxInt/4 || rows > maxInt/4 {
		return sizes, 0, ErrKVFP8Geometry
	}
	sizes = [kvFP8Planes]int{elements * 4, elements, elements, rows * 4, rows * 4}
	perLayer := int64(elements)*6 + int64(rows)*8
	if perLayer > 0 && int64(o.layers) > math.MaxInt64/perLayer {
		return sizes, 0, ErrKVFP8Geometry
	}
	return sizes, perLayer * int64(o.layers), nil
}

// allocatedBytes remains available after poison/quarantine and includes staged
// and incompletely retired allocations. It is physical ownership, not the
// logical KVStore ResidentBytes report (which also charges position metadata).
func (o *kvFP8Owner) allocatedBytes() int64 {
	if o == nil {
		return 0
	}
	var n int64
	for _, image := range []*kvFP8Image{o.image, o.stage, o.retiring} {
		if image == nil {
			continue
		}
		for _, layer := range image.layers {
			for _, a := range layer {
				if a.ptr != nil {
					n += int64(a.bytes)
				}
			}
		}
		for _, a := range image.workspace {
			if a.ptr != nil {
				n += int64(a.bytes)
			}
		}
	}
	return n
}

func (o *kvFP8Owner) retireWorkspace(image *kvFP8Image, release func(kvFP8Allocation) kvFP8Retirement) error {
	for i := range image.workspace {
		a := &image.workspace[i]
		if a.ptr == nil {
			continue
		}
		result := release(*a)
		if result.retired {
			*a = kvFP8Allocation{}
		}
		if result.status != 0 || !result.retired {
			o.retainUncertain()
			return &kvFP8ReceiptError{cause: errKVFP8Retirement, status: result.status}
		}
	}
	return nil
}

func (o *kvFP8Owner) retireImage(image *kvFP8Image, release func(kvFP8Allocation) kvFP8Retirement) error {
	if image == nil {
		return nil
	}
	for l := range image.layers {
		for p := range image.layers[l] {
			a := &image.layers[l][p]
			if a.ptr == nil {
				continue
			}
			result := release(*a)
			if result.retired {
				*a = kvFP8Allocation{}
			}
			if result.status != 0 || !result.retired {
				o.retainUncertain()
				return &kvFP8ReceiptError{cause: errKVFP8Retirement, status: result.status}
			}
		}
	}
	return o.retireWorkspace(image, release)
}

// replace installs a COMPLETE new image. It is intentionally not the per-layer
// AppendKV path. Reserve, clone and staged eviction adapters must fill all five
// planes and preserve exact raw/V/code/scale bytes where no re-RoPE is needed.
// fill must check completion AND numerical status before any dependent read of
// rejected/unwritten staging, not only before returning its final receipt.
//
// peakBudget bounds this owner's old+new live device allocations, not global
// device free memory. A caller also needs current allocator/device admission.
// workspaceSizes declares EVERY status/intermediate allocation before effects;
// fill may not create unrooted scratch. Layer -1 in allocate identifies workspace.
// The whole-image baseline may nearly double residency; it earns no speed claim.
func (o *kvFP8Owner) replace(positions int, workspaceSizes []int, peakBudget int64, ops kvFP8OwnerOps,
	fill func(dst, src *kvFP8Image) kvFP8Completion, poisonOnReject bool) (err error) {
	if err = o.usable(); err != nil {
		return err
	}
	if ops.allocate == nil || ops.release == nil || fill == nil {
		return errKVFP8Owner
	}
	oldStamp := o.stamp.Load()
	if oldStamp > ^uint64(0)-4 {
		return errKVFP8Owner
	}
	sizes, nextBytes, err := o.planeBytes(positions)
	if err != nil {
		return err
	}
	for _, bytes := range workspaceSizes {
		if bytes <= 0 || int64(bytes) > math.MaxInt64-nextBytes {
			return ErrKVFP8Geometry
		}
		nextBytes += int64(bytes)
	}
	current := o.allocatedBytes()
	if peakBudget < current || nextBytes > peakBudget-current {
		return errKVFP8Budget
	}
	stage := &kvFP8Image{positions: positions, layers: make([][kvFP8Planes]kvFP8Allocation, o.layers), workspace: make([]kvFP8Allocation, len(workspaceSizes))}
	// Attach all destination slots before the first native allocation. Ready is
	// false while an operation owns potential references to the current image.
	o.stage = stage
	o.stamp.Store(oldStamp &^ 3)
	normalReturn := false
	defer func() {
		// Goexit and panic(nil) also execute defers. Do not use recover()!=nil
		// as evidence of abnormal unwind: every non-normal exit is uncertain.
		if !normalReturn {
			o.retainUncertain()
		}
	}()
	err = func() error {
		discard := func(reason error) error {
			o.primaryFailure, o.cleanupFailure = reason, nil
			o.cleanupInterrupted = false
			cleanupReturned := false
			defer func() {
				// Do not recover: retain the primary cause separately while
				// the original cleanup panic/Goexit continues unchanged.
				if !cleanupReturned {
					o.cleanupInterrupted = true
				}
			}()
			retireErr := o.retireImage(o.stage, ops.release)
			cleanupReturned = true
			if retireErr != nil {
				o.cleanupFailure = retireErr
				return errors.Join(reason, retireErr)
			}
			o.stage = nil
			if poisonOnReject {
				o.poisonNumerical()
			} else {
				o.stamp.Store(oldStamp)
			}
			return reason
		}
		for l := range stage.layers {
			for p, size := range sizes {
				if size == 0 {
					continue
				}
				a := &stage.layers[l][p]
				a.bytes = size
				if allocErr := ops.allocate(a, size, l, p); allocErr != nil {
					// Allocation-error completion belongs to the allocator's
					// contract: no queued writes may reference these fresh slots.
					return discard(allocErr)
				}
				if a.ptr == nil || a.bytes != size {
					return discard(errKVFP8Owner)
				}
			}
		}
		for i, bytes := range workspaceSizes {
			a := &stage.workspace[i]
			a.bytes = bytes
			if allocErr := ops.allocate(a, bytes, -1, i); allocErr != nil {
				return discard(allocErr)
			}
			if a.ptr == nil || a.bytes != bytes {
				return discard(errKVFP8Owner)
			}
		}
		result := fill(stage, o.image)
		if result.completed && result.numericalChecked && result.numerical != 0 {
			o.numericalFailure = result.numerical
			o.primaryFailure = ErrKVFP8NonFinite
		}
		if result.status != 0 || !result.completed {
			o.retainUncertain()
			failure := &kvFP8ReceiptError{cause: errKVFP8Completion, status: result.status}
			if result.completed && result.numericalChecked && result.numerical != 0 {
				o.cleanupFailure = failure
				return errors.Join(o.primaryFailure, failure)
			}
			return failure
		}
		if !result.numericalChecked {
			// Completion alone cannot authorize a read/publication of bytes that
			// a rejected-row kernel may deliberately have left unwritten.
			return discard(errKVFP8Owner)
		}
		if result.numerical != 0 {
			return discard(ErrKVFP8NonFinite)
		}
		if err = o.retireWorkspace(o.stage, ops.release); err != nil {
			return err
		}
		o.retiring, o.image, o.stage = o.image, o.stage, nil
		o.stamp.Store(oldStamp + 4)
		if err = o.retireImage(o.retiring, ops.release); err != nil {
			return err
		}
		o.retiring = nil
		return nil
	}()
	normalReturn = true
	return err
}

// free permits retirement after known-complete numerical poisoning, but never
// after uncertain work. Every stale view becomes unusable BEFORE retirement.
func (o *kvFP8Owner) free(ops kvFP8OwnerOps) (err error) {
	if o == nil {
		return nil
	}
	if o.backend == nil {
		return errKVFP8Owner
	}
	if o.uncertain || o.retained || o.backend.stopped.Load() {
		o.retainUncertain()
		return errKVFP8Completion
	}
	if ops.release == nil || o.stage != nil || o.retiring != nil {
		return errKVFP8Owner
	}
	o.poisonNumerical()
	normalReturn := false
	defer func() {
		if !normalReturn {
			o.retainUncertain()
		}
	}()
	err = o.retireImage(o.image, ops.release)
	if err != nil {
		normalReturn = true
		return err
	}
	o.image.positions = 0
	normalReturn = true
	return nil
}

// clone copies only complete images and creates a distinct owner, descriptor
// identity and rotary-contract backing array. copyComplete must bit-copy all
// five planes, never widen/requantize them, and return a checked receipt.
func (o *kvFP8Owner) clone(peakBudget int64, ops kvFP8OwnerOps,
	copyComplete func(dst, src *kvFP8Image) kvFP8Completion) (*kvFP8Owner, error) {
	if err := o.usable(); err != nil {
		return nil, err
	}
	if copyComplete == nil {
		return nil, errKVFP8Owner
	}
	current := o.allocatedBytes()
	if peakBudget < current || current > peakBudget-current {
		return nil, errKVFP8Budget
	}
	next, err := newKVFP8Owner(o.backend, o.device, o.layers, o.heads, o.headDim)
	if err != nil {
		return nil, err
	}
	copy(next.rotary, o.rotary)
	normalReturn := false
	defer func() {
		// Even Goexit after submission must retain both copy participants.
		if !normalReturn {
			o.retainUncertain()
			next.retainUncertain()
		}
	}()
	err = next.replace(o.image.positions, nil, peakBudget-current, ops,
		func(dst, _ *kvFP8Image) kvFP8Completion { return copyComplete(dst, o.image) }, false)
	normalReturn = true
	if err != nil {
		if next.uncertain {
			// The source was referenced by the uncertain copy too.
			o.retainUncertain()
		}
		return nil, err
	}
	return next, nil
}

// kvFP8Borrowed is not a cudaBuf, HostBuffer or owning Tensor. Logical F32 view
// metadata must be validated at the Read ingress before any effect. Native
// attention accesses the owner's explicit code+scale path, not this buffer.
type kvFP8Borrowed struct {
	owner                          *kvFP8Owner
	stamp                          uint64
	layer, plane, positions, width int
}

func (o *kvFP8Owner) borrow(layer, plane int) (*kvFP8Borrowed, error) {
	if err := o.usable(); err != nil {
		return nil, err
	}
	if layer < 0 || layer >= o.layers || (plane != kvFP8Key && plane != kvFP8Value) {
		return nil, ErrKVFP8Geometry
	}
	return &kvFP8Borrowed{o, o.stamp.Load(), layer, plane, o.image.positions, o.heads * o.headDim}, nil
}

// Ready never takes CUDA locks, calls native code or certifies unrelated work.
// All published images have a checked producer completion; equality is enough.
func (v *kvFP8Borrowed) Ready() bool {
	return v != nil && v.owner != nil && v.owner.backend != nil && !v.owner.backend.stopped.Load() && v.stamp&3 == 1 && v.owner.stamp.Load() == v.stamp
}

// resolve must be called under the same CUDA serialization domain that protects
// pointer use, not followed by an unlock before enqueue/read. The caller checks
// logical Tensor dtype==F32, RowMajor, Quant==nil, and exact two-dimensional shape.
// Descriptor Free is a no-op BEFORE ordinary CUDA drain or cleanup paths.
func (v *kvFP8Borrowed) resolve(backend *kvFP8BackendIdentity, device, positions, width int) (codes, scales kvFP8Allocation, err error) {
	if !v.Ready() || v.owner.usable() != nil || backend != v.owner.backend || device != v.owner.device ||
		positions != v.positions || width != v.width || v.layer < 0 || v.layer >= v.owner.layers ||
		(v.plane != kvFP8Key && v.plane != kvFP8Value) {
		return codes, scales, errKVFP8Owner
	}
	plane := kvFP8KeyScale
	if v.plane == kvFP8Value {
		plane = kvFP8ValueScale
	}
	return v.owner.image.layers[v.layer][v.plane], v.owner.image.layers[v.layer][plane], nil
}
