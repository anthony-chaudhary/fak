package strix

import (
	"runtime"
	"testing"
	"unsafe"
)

// fak-test:runtime fast est=1s
func TestHostTensorViewRetainsZeroCopyBacking(t *testing.T) {
	const byteLength = 128
	view, err := func() (*TensorView, error) {
		backing := make([]byte, byteLength)
		backing[0] = 0x5a
		return NewTensorViewFromHostSlice(backing, DTypeFP32, []int{byteLength / 4})
	}()
	if err != nil {
		t.Fatal(err)
	}
	// Assert the strong slice edge directly; GC timing is not the safety oracle.
	if len(view.hostBacking) != byteLength || uintptr(unsafe.Pointer(&view.hostBacking[0])) != view.HostPtr() {
		t.Fatal("view did not retain the original host backing")
	}
	if view.ParentBuffer() != nil || view.DevPtr() != view.HostPtr() {
		t.Fatal("host view changed native ownership or pointer identity")
	}
	runtime.GC()
	bytes := view.ByteSlice()
	if len(bytes) != byteLength || bytes[0] != 0x5a || &bytes[0] != &view.hostBacking[0] {
		t.Fatal("host byte view lost zero-copy backing")
	}
	bytes[1] = 0xa5
	if view.hostBacking[1] != 0xa5 {
		t.Fatal("host view unexpectedly copied storage")
	}
	floats := view.Float32Slice()
	floats[2] = 3.5
	if view.Float32Slice()[2] != 3.5 {
		t.Fatal("float view lost aliasing")
	}
	view.KeepAlive()
}

// fak-test:runtime fast est=1s
func TestHostTensorChildRetainsRootBacking(t *testing.T) {
	for _, mode := range []string{"slice", "subslice"} {
		t.Run(mode, func(t *testing.T) {
			const offset, length = 16, 32
			child, err := func() (*TensorView, error) {
				backing := make([]byte, offset+length+16)
				backing[offset] = 0x6b
				root, err := NewTensorViewFromHostSlice(backing, DTypeFP32, nil)
				if err != nil {
					return nil, err
				}
				if mode == "slice" {
					return root.Slice(SliceDescriptor{ByteOffset: offset, ByteLength: length, DType: DTypeFP32})
				}
				return root.SubSlice(offset, length)
			}()
			if err != nil {
				t.Fatal(err)
			}
			root := child.parentView
			if root == nil || len(root.hostBacking) < offset+length {
				t.Fatal("child has no strong path to original host backing")
			}
			if child.HostPtr() != root.HostPtr()+offset || child.ByteOffset() != offset || child.ByteLength() != length {
				t.Fatal("child geometry changed")
			}
			runtime.GC()
			bytes := child.ByteSlice()
			if len(bytes) != length || bytes[0] != 0x6b || &bytes[0] != &root.hostBacking[offset] {
				t.Fatal("child lost root alias")
			}
			bytes[1] = 0x7c
			if root.ByteSlice()[offset+1] != 0x7c {
				t.Fatal("child writes no longer reach root")
			}
			child.KeepAlive()
		})
	}
}

// fak-test:runtime fast est=1s
func TestUMATensorViewDoesNotAcquireHostSliceOwnership(t *testing.T) {
	const byteLength = 128
	parent, err := AllocateUMABuffer(byteLength, GFX1151VectorAlignment)
	if err != nil {
		t.Fatal(err)
	}
	defer parent.Close()
	view, err := parent.AsTensorView(DTypeFP32, []int{byteLength / 4})
	if err != nil {
		t.Fatal(err)
	}
	if view.ParentBuffer() != parent || view.hostBacking != nil {
		t.Fatal("UMA view acquired unrelated host ownership")
	}
	view.ByteSlice()[0] = 0x31
	if parent.Slice()[0] != 0x31 {
		t.Fatal("UMA view aliasing changed")
	}
}
