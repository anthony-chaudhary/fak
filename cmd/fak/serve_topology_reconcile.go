package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func configureServeTopologyShadow(s *gateway.Server, path *string) error {
	if path == nil || *path == "" {
		return nil
	}
	info, err := os.Stat(*path)
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() || info.Size() > 16<<20 {
		return fmt.Errorf("topology shadow: input must be a regular file of at most 16 MiB")
	}
	f, err := os.Open(*path)
	if err != nil {
		return err
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil {
		return err
	}
	if !opened.Mode().IsRegular() || opened.Size() > 16<<20 {
		return fmt.Errorf("topology shadow: opened input is not a bounded regular file")
	}
	in, err := gateway.DecodeTopologyShadow(f)
	if err != nil {
		return err
	}
	return s.SetTopologyShadow(in, func(r gateway.TopologyShadowReport) {
		// Background lifecycle readback only. No new HTTP endpoint, credential,
		// request-path callback, hardware probe or persistent journal is introduced.
		b, err := json.Marshal(r)
		if err == nil {
			fmt.Fprintf(os.Stderr, "fak topology-shadow: %s\n", b)
		}
	})
}
