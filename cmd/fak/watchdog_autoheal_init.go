package main

import "os"

func init() {
	if len(os.Args) < 2 {
		return
	}
	switch os.Args[1] {
	case "serve":
		watchdogAutohealOnStart(os.Args[1])
	case "guard":
		// Guard children are numerous and inherit the environment. Only an explicit
		// opt-in may run host-global autoheal from each wrapper; serve remains default-on.
		if _, explicit := os.LookupEnv("FAK_WATCHDOG_AUTOHEAL"); explicit {
			watchdogAutohealOnStart(os.Args[1])
		}
	}
}
