package strictjson

import (
	"encoding/json"
	"os"
)

// LoadFile reads a JSON file into a newly allocated typed value.
func LoadFile[T any](path string) (*T, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var value T
	if err := json.Unmarshal(data, &value); err != nil {
		return nil, err
	}
	return &value, nil
}
