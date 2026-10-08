//go:build !windows

package main

import (
	"errors"
	"fmt"
	"os"
	"reflect"
)

func modelArtifactFileIdentity(_ *os.File, info os.FileInfo) (string, int64, error) {
	stat := reflect.ValueOf(info.Sys())
	if stat.Kind() == reflect.Pointer {
		stat = stat.Elem()
	}
	if !stat.IsValid() || stat.Kind() != reflect.Struct {
		return "", 0, errors.New("model artifact stat identity unavailable")
	}
	dev, ok := modelArtifactUintField(stat.FieldByName("Dev"))
	if !ok {
		return "", 0, errors.New("model artifact device identity unavailable")
	}
	ino, ok := modelArtifactUintField(stat.FieldByName("Ino"))
	if !ok {
		return "", 0, errors.New("model artifact inode identity unavailable")
	}
	changeTime, ok := modelArtifactChangeTime(stat)
	if !ok {
		return "", 0, errors.New("model artifact change time unavailable")
	}
	return fmt.Sprintf("%d:%d", dev, ino), changeTime, nil
}

func modelArtifactChangeTime(stat reflect.Value) (int64, bool) {
	for _, name := range []string{"Ctim", "Ctimespec"} {
		value := stat.FieldByName(name)
		if !value.IsValid() {
			continue
		}
		if value.Kind() == reflect.Pointer {
			value = value.Elem()
		}
		if !value.IsValid() || value.Kind() != reflect.Struct {
			continue
		}
		sec, secOK := modelArtifactIntField(value.FieldByName("Sec"))
		nsec, nsecOK := modelArtifactIntField(value.FieldByName("Nsec"))
		if secOK && nsecOK {
			return sec*1e9 + nsec, true
		}
	}
	sec, secOK := modelArtifactIntField(stat.FieldByName("Ctime"))
	nsec, nsecOK := modelArtifactIntField(stat.FieldByName("Ctimensec"))
	if secOK && nsecOK {
		return sec*1e9 + nsec, true
	}
	return 0, false
}

func modelArtifactUintField(value reflect.Value) (uint64, bool) {
	if !value.IsValid() {
		return 0, false
	}
	switch value.Kind() {
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return value.Uint(), true
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		v := value.Int()
		return uint64(v), v >= 0
	default:
		return 0, false
	}
}

func modelArtifactIntField(value reflect.Value) (int64, bool) {
	if !value.IsValid() {
		return 0, false
	}
	switch value.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return value.Int(), true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		v := value.Uint()
		return int64(v), v <= uint64(^uint64(0)>>1)
	default:
		return 0, false
	}
}
