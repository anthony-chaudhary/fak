package jsonlledger

import (
	"encoding/json"
	"io"
)

// AppendValidated marshals and appends one validated value as a JSONL row.
func AppendValidated[T any](writer io.Writer, value T, validate func(T) error) error {
	if err := validate(value); err != nil {
		return err
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return err
	}
	_, err = writer.Write(append(encoded, '\n'))
	return err
}
