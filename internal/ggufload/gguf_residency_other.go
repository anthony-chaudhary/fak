//go:build !linux

package ggufload

import "os"

func fileResidentBytes(*os.File) (int64, int64, error) {
	return 0, 0, errPageCacheResidencyUnsupported
}
