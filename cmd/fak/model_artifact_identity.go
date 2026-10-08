package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
)

type modelArtifactStamp struct {
	path        string
	incarnation string
	changeTime  int64
	size        int64
	modTime     int64
}

func stampModelArtifact(path string) modelArtifactStamp {
	canonical, err := filepath.Abs(path)
	if err != nil {
		return modelArtifactStamp{}
	}
	if resolved, err := filepath.EvalSymlinks(canonical); err == nil {
		canonical = resolved
	}
	canonical = filepath.Clean(canonical)
	f, err := os.Open(canonical)
	if err != nil {
		return modelArtifactStamp{}
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil || !info.Mode().IsRegular() {
		return modelArtifactStamp{}
	}
	incarnation, changeTime, err := modelArtifactFileIdentity(f, info)
	if err != nil || incarnation == "" {
		return modelArtifactStamp{}
	}
	return modelArtifactStamp{
		path:        canonical,
		incarnation: incarnation,
		changeTime:  changeTime,
		size:        info.Size(),
		modTime:     info.ModTime().UnixNano(),
	}
}

func (before modelArtifactStamp) identityAfterLoad(path string) string {
	after := stampModelArtifact(path)
	if before.incarnation == "" || after.incarnation == "" || before.path != after.path ||
		before.incarnation != after.incarnation || before.changeTime != after.changeTime ||
		before.size != after.size || before.modTime != after.modTime {
		return ""
	}
	runtimeID := modelRuntimeIdentity()
	if runtimeID == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(fmt.Sprintf("fak-model-artifact/v2\x00%s\x00%s\x00%d\x00%d\x00%d\x00%s",
		after.path, after.incarnation, after.changeTime, after.size, after.modTime, runtimeID)))
	return hex.EncodeToString(sum[:])
}

var modelRuntimeIdentityOnce = sync.OnceValue(func() string {
	executable, err := os.Executable()
	if err != nil {
		return ""
	}
	f, err := os.Open(executable)
	if err != nil {
		return ""
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return ""
	}
	return hex.EncodeToString(h.Sum(nil))
})

func modelRuntimeIdentity() string { return modelRuntimeIdentityOnce() }
