package studyclass

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
)

func Marshal(out Output) ([]byte, error) {
	if err := Validate(out); err != nil {
		return nil, err
	}
	return marshalDeterministic(out)
}

func WriteJSON(w io.Writer, out Output) error {
	b, err := Marshal(out)
	if err != nil {
		return err
	}
	_, err = w.Write(b)
	return err
}

func ReadJSON(r io.Reader) (Output, error) {
	var out Output
	if err := decodeClosed(r, &out); err != nil {
		return Output{}, err
	}
	if err := Validate(out); err != nil {
		return Output{}, err
	}
	return out, nil
}

func MarshalCompact(index CompactIndex) ([]byte, error) {
	if err := ValidateCompact(index); err != nil {
		return nil, err
	}
	return marshalDeterministic(index)
}

func WriteCompactJSON(w io.Writer, index CompactIndex) error {
	b, err := MarshalCompact(index)
	if err != nil {
		return err
	}
	_, err = w.Write(b)
	return err
}

func ReadCompactJSON(r io.Reader) (CompactIndex, error) {
	var out CompactIndex
	if err := decodeClosed(r, &out); err != nil {
		return CompactIndex{}, err
	}
	if err := ValidateCompact(out); err != nil {
		return CompactIndex{}, err
	}
	return out, nil
}

func marshalDeterministic(v any) ([]byte, error) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}

func decodeClosed(r io.Reader, dst any) error {
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return err
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return fmt.Errorf("multiple JSON values")
		}
		return err
	}
	return nil
}

func canonicalOutputChecksum(out Output) string {
	b, err := marshalDeterministic(out)
	if err != nil {
		panic(err)
	}
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:])
}
