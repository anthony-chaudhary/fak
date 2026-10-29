package ggufload

import (
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/newmodel"
)

func manifestAuxHeaderFixture(t *testing.T) *File {
	t.Helper()
	ws, _, _ := tensorEngramAuxFixture(t)
	f := loadV41HeaderFixture(t)
	f.Tensors = append([]TensorInfo(nil), f.Tensors...)
	for _, info := range ws.File.Tensors {
		if deepSeek41EngramAuxTensor(info.Name) {
			f.Tensors = append(f.Tensors, info)
		}
	}
	if f.readEngramAux != nil {
		t.Fatal("metadata header unexpectedly has a payload reader")
	}
	return f
}

func manifestHasRefusal(p ManifestPreflight, reason newmodel.RefusalReason, axis string) bool {
	for _, r := range p.Refusals {
		if r.Reason == reason && r.Axis == axis {
			return true
		}
	}
	return false
}

// fak-test:justify why=regression when=changed:internal/ggufload/manifest_preflight.go
// fak-test:runtime medium est=2s lane=default
func TestManifestPreflightEngramAuxiliaryRoles(t *testing.T) {
	for _, typ := range []TensorType{26, 27} {
		if HostDequantSupported(typ) {
			t.Fatalf("integer metadata falsely acquired float decoding: %v", typ)
		}
	}
	for _, metadataOnly := range []bool{false, true} {
		t.Run(fmt.Sprintf("admitted_metadata_only=%t", metadataOnly), func(t *testing.T) {
			ws, _, _ := tensorEngramAuxFixture(t)
			f := ws.File
			if metadataOnly {
				f = manifestAuxHeaderFixture(t)
			}
			if _, err := f.Config(); err != nil {
				t.Fatalf("fixture Config prerequisite: %v", err)
			}
			p := BuildManifestPreflight(ManifestPreflightInput{File: f})
			for _, r := range p.Refusals {
				if r.Reason == newmodel.RefusalTensorOrphan && deepSeek41EngramAuxTensor(r.Axis) {
					t.Errorf("admitted configuration auxiliary reported as orphan: %s", r.Axis)
				}
				if r.Reason == newmodel.RefusalUnsupportedDtype && (r.Axis == OpHostDequant+"(TensorType(26))" || r.Axis == OpHostDequant+"(TensorType(27))") {
					t.Errorf("admitted integer configuration auxiliary requires float decoding: %s", r.Axis)
				}
			}
			if p.DTypes[TensorType(26).String()] != 1 || p.DTypes[TensorType(27).String()] != 3 {
				t.Fatalf("auxiliary dtype census was dropped: %v", p.DTypes)
			}
		})
	}
	for _, mode := range []string{"wrong_type", "wrong_shape", "near_name"} {
		t.Run(mode, func(t *testing.T) {
			f := manifestAuxHeaderFixture(t)
			name := "engram_token_map.weight"
			for i := range f.Tensors {
				if f.Tensors[i].Name != name {
					continue
				}
				switch mode {
				case "wrong_type":
					f.Tensors[i].Type = TensorF32
				case "wrong_shape":
					f.Tensors[i].Dims = []uint64{^uint64(0)}
				case "near_name":
					name += ".extra"
					f.Tensors[i].Name = name
				}
				break
			}
			if _, err := f.Config(); err != nil {
				t.Fatalf("metadata admission prerequisite: %v", err)
			}
			p := BuildManifestPreflight(ManifestPreflightInput{File: f})
			if !manifestHasRefusal(p, newmodel.RefusalTensorOrphan, name) {
				t.Fatalf("malformed/unrecognized tensor hidden by metadata role: %s: %v", name, p.Refusals)
			}
		})
	}
	for _, typ := range []TensorType{26, 27} {
		t.Run(fmt.Sprintf("ordinary_%d", typ), func(t *testing.T) {
			f := syntheticPreflightFile()
			f.Tensors[0].Type = typ
			p := BuildManifestPreflight(ManifestPreflightInput{File: f})
			if !manifestHasRefusal(p, newmodel.RefusalUnsupportedDtype, OpHostDequant+"("+typ.String()+")") {
				t.Fatal("ordinary integer weight was exempted from host decoding")
			}
		})
	}
	t.Run("foreign_architecture", func(t *testing.T) {
		f := syntheticPreflightFile()
		f.Tensors = append(f.Tensors, TensorInfo{Name: "engram_token_map.weight", Type: TensorType(26), Dims: []uint64{4}})
		p := BuildManifestPreflight(ManifestPreflightInput{File: f})
		if !manifestHasRefusal(p, newmodel.RefusalTensorOrphan, "engram_token_map.weight") || !manifestHasRefusal(p, newmodel.RefusalUnsupportedDtype, OpHostDequant+"(TensorType(26))") {
			t.Fatal("foreign architecture gained Engram metadata role")
		}
	})
	t.Run("readerless_tensor_only_header", func(t *testing.T) {
		ws, _, _ := tensorEngramAuxFixture(t)
		if _, err := ws.File.Config(); err != nil {
			t.Fatal(err)
		}
		f := *ws.File // Deliberately carries a stale admitted pointer.
		f.readEngramAux = nil
		p := BuildManifestPreflight(ManifestPreflightInput{File: &f})
		if !manifestHasRefusal(p, newmodel.RefusalHeaderConfigInvalid, "config") || !manifestHasRefusal(p, newmodel.RefusalUnsupportedDtype, OpHostDequant+"(TensorType(26))") {
			t.Fatalf("readerless header's Config refusal was hidden: %v", p.Refusals)
		}
	})
	t.Run("partial_engram_before_topology_failure", func(t *testing.T) {
		ws, _, _ := tensorEngramAuxFixture(t)
		removed := false
		for i := range ws.File.Tensors {
			if strings.HasSuffix(ws.File.Tensors[i].Name, ".indexer.attn_k.weight") {
				ws.File.Tensors[i].Name = "unmapped.removed_index_key"
				removed = true
				break
			}
		}
		if !removed {
			t.Fatal("fixture has no topology key owner")
		}
		if _, err := ws.File.Config(); err == nil || ws.File.DeepSeek41Engram == nil {
			t.Fatalf("did not isolate post-Engram topology failure: %v", err)
		}
		p := BuildManifestPreflight(ManifestPreflightInput{File: ws.File})
		if !manifestHasRefusal(p, newmodel.RefusalHeaderConfigInvalid, "config") || !manifestHasRefusal(p, newmodel.RefusalUnsupportedDtype, OpHostDequant+"(TensorType(26))") {
			t.Fatalf("partial Config incorrectly admitted metadata role: %v", p.Refusals)
		}
	})
}
