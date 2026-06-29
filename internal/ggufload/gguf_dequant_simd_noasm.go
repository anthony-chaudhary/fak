//go:build !amd64

package ggufload

func dequantQ4KArch(out []float32, raw []byte) bool    { return false }
func dequantQ5KArch(out []float32, raw []byte) bool    { return false }
func dequantQ6KArch(out []float32, raw []byte) bool    { return false }
func dequantIQ3XXSArch(out []float32, raw []byte) bool { return false }
