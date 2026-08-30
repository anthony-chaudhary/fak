package main

import (
	"os"

	selfupdatecmd "github.com/anthony-chaudhary/fak/internal/selfupdate/cmd"
)

func main() { selfupdatecmd.Run(os.Args[1:]) }
