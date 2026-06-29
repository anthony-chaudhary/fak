package main

import "os"

func init() {
	if len(os.Args) < 2 {
		return
	}
	switch os.Args[1] {
	case "serve", "guard":
		watchdogAutohealOnStart(os.Args[1])
	}
}
