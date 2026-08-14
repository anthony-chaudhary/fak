//go:build !windows

package main

import "os"

func replaceMicroCacheWitness(src, dst string) error {
	return os.Rename(src, dst)
}
