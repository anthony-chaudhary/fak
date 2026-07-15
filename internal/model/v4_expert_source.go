package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

var (
	// ErrV4ExpertNotFound reports a well-formed routed-expert tensor that is not
	// present in the indexed safetensors source.
	ErrV4ExpertNotFound = errors.New("v4 routed expert tensor not found")
	// ErrV4ExpertNotRouted reports a request for a tensor outside the routed-
	// expert class. Callers must keep always-resident tensors on their own path.
	ErrV4ExpertNotRouted = errors.New("v4 tensor is not a routed expert")
	// ErrV4ExpertMetadata reports malformed routed-expert identity or bounds in
	// the source header. Construction fails before any tensor payload is read.
	ErrV4ExpertMetadata = errors.New("invalid v4 routed expert metadata")
)

// v4ExpertTensor is one selectively-read routed-expert tensor. Bytes contains
// exactly the safetensors data range and never includes another tensor.
type v4ExpertTensor struct {
	Name  string
	Dtype string
	Shape []int
	Bytes []byte
}

// v4ExpertSource indexes routed-expert metadata while retaining the existing
// ReaderAt-backed safetensors file as the payload tier. Constructing the source
// reads only the already-loaded header; read faults exactly one requested range.
type v4ExpertSource struct {
	file    *safetensorsFile
	entries map[string]stEntry
}

func newV4ExpertSource(sf *safetensorsFile) (*v4ExpertSource, error) {
	if sf == nil {
		return nil, fmt.Errorf("%w: nil safetensors file", ErrV4ExpertMetadata)
	}
	s := &v4ExpertSource{file: sf, entries: make(map[string]stEntry)}
	for name, raw := range sf.hdr {
		class, ok := classifyV4Tensor(name)
		if !ok || class != V4ClassRoutedExpert {
			continue
		}
		if err := validateV4ExpertIdentity(name); err != nil {
			return nil, err
		}
		var entry stEntry
		if err := json.Unmarshal(raw, &entry); err != nil {
			return nil, fmt.Errorf("%w: %s header: %v", ErrV4ExpertMetadata, name, err)
		}
		if entry.Dtype == "" || len(entry.Shape) == 0 {
			return nil, fmt.Errorf("%w: %s missing dtype or shape", ErrV4ExpertMetadata, name)
		}
		for _, dim := range entry.Shape {
			if dim <= 0 {
				return nil, fmt.Errorf("%w: %s has non-positive shape", ErrV4ExpertMetadata, name)
			}
		}
		if _, _, err := safetensorsDataBounds(sf.dataBase, sf.size, entry); err != nil {
			return nil, fmt.Errorf("%w: %s: %v", ErrV4ExpertMetadata, name, err)
		}
		s.entries[name] = entry
	}
	return s, nil
}

// read returns one exact routed-expert range. Classification and lookup happen
// before tensorBytes, so rejected requests perform no tensor-data IO.
func (s *v4ExpertSource) read(name string) (v4ExpertTensor, error) {
	class, ok := classifyV4Tensor(name)
	if !ok || class != V4ClassRoutedExpert {
		return v4ExpertTensor{}, fmt.Errorf("%w: %s", ErrV4ExpertNotRouted, name)
	}
	if err := validateV4ExpertIdentity(name); err != nil {
		return v4ExpertTensor{}, err
	}
	entry, ok := s.entries[name]
	if !ok {
		return v4ExpertTensor{}, fmt.Errorf("%w: %s", ErrV4ExpertNotFound, name)
	}
	payload, err := s.file.tensorBytes(entry)
	if err != nil {
		return v4ExpertTensor{}, fmt.Errorf("read v4 routed expert %s: %w", name, err)
	}
	return v4ExpertTensor{
		Name:  name,
		Dtype: entry.Dtype,
		Shape: append([]int(nil), entry.Shape...),
		Bytes: payload,
	}, nil
}

func (s *v4ExpertSource) len() int { return len(s.entries) }

func validateV4ExpertIdentity(name string) error {
	parts := strings.Split(name, ".")
	if len(parts) > 0 && parts[0] == "model" {
		parts = parts[1:]
	}
	// layers.<layer>.(ffn|mlp).experts.<expert>.<tensor path...>
	if len(parts) < 7 || parts[0] != "layers" || (parts[2] != "ffn" && parts[2] != "mlp") || parts[3] != "experts" {
		return fmt.Errorf("%w: malformed routed-expert name %q", ErrV4ExpertMetadata, name)
	}
	layer, layerErr := strconv.Atoi(parts[1])
	expert, expertErr := strconv.Atoi(parts[4])
	if layerErr != nil || expertErr != nil || layer < 0 || expert < 0 {
		return fmt.Errorf("%w: malformed routed-expert identity %q", ErrV4ExpertMetadata, name)
	}
	for _, component := range parts[5:] {
		if component == "" {
			return fmt.Errorf("%w: empty routed-expert tensor component in %q", ErrV4ExpertMetadata, name)
		}
	}
	return nil
}
