package normgate

import (
	"testing"
)

func TestPowerShellTargets(t *testing.T) {
	tests := []struct {
		name        string
		command     string
		wantTargets []PowerShellTarget
		wantErr     bool
	}{
		// Set-Content & alias sc
		{
			name:    "Set-Content with -Path and -Value",
			command: `Set-Content -Path "test.txt" -Value "hello"`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Set-Content", Path: "test.txt", Op: "write", TreeKnown: true},
			},
		},
		{
			name:    "Set-Content with -LiteralPath",
			command: `Set-Content -LiteralPath 'C:\My Folder\test.txt' -Value 'hello'`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Set-Content", Path: `C:\My Folder\test.txt`, Op: "write", TreeKnown: true},
			},
		},
		{
			name:    "Set-Content with -Path:colon",
			command: `Set-Content -Path:foo.txt -Value:bar`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Set-Content", Path: "foo.txt", Op: "write", TreeKnown: true},
			},
		},
		{
			name:    "Set-Content with -Path=equals",
			command: `Set-Content -Path="path with spaces/foo.txt"`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Set-Content", Path: "path with spaces/foo.txt", Op: "write", TreeKnown: true},
			},
		},
		{
			name:    "Set-Content positional path and value",
			command: `Set-Content "path with spaces/file.txt" "content"`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Set-Content", Path: "path with spaces/file.txt", Op: "write", TreeKnown: true},
			},
		},
		{
			name:    "Set-Content -Value before positional path",
			command: `Set-Content -Value "content" "file.txt"`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Set-Content", Path: "file.txt", Op: "write", TreeKnown: true},
			},
		},
		{
			name:    "Set-Content alias sc positional",
			command: `sc file.txt 'data'`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Set-Content", Path: "file.txt", Op: "write", TreeKnown: true},
			},
		},
		{
			name:    "Set-Content alias sc with -Path",
			command: `sc -Path 'file.txt' 'data'`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Set-Content", Path: "file.txt", Op: "write", TreeKnown: true},
			},
		},
		{
			name:    "Pipeline destination Set-Content positional",
			command: `"hello" | Set-Content "file.txt"`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Set-Content", Path: "file.txt", Op: "write", TreeKnown: true},
			},
		},
		{
			name:    "Pipeline destination Set-Content with -Path",
			command: `"hello" | Set-Content -Path "file.txt"`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Set-Content", Path: "file.txt", Op: "write", TreeKnown: true},
			},
		},
		{
			name:    "Pipeline destination alias sc",
			command: `"hello" | sc file.txt`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Set-Content", Path: "file.txt", Op: "write", TreeKnown: true},
			},
		},

		// Out-File
		{
			name:    "Out-File in pipeline positional",
			command: `... | Out-File path.txt`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Out-File", Path: "path.txt", Op: "write", TreeKnown: true},
			},
		},
		{
			name:    "Out-File in pipeline with -FilePath",
			command: `... | Out-File -FilePath path.txt`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Out-File", Path: "path.txt", Op: "write", TreeKnown: true},
			},
		},
		{
			name:    "Out-File in pipeline with -FilePath colon and quotes",
			command: `... | Out-File -FilePath:"path with spaces.txt"`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Out-File", Path: "path with spaces.txt", Op: "write", TreeKnown: true},
			},
		},
		{
			name:    "Out-File in pipeline with -Path",
			command: `Get-Process | Out-File -Path "procs.txt"`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Out-File", Path: "procs.txt", Op: "write", TreeKnown: true},
			},
		},
		{
			name:    "Out-File in pipeline with -LiteralPath",
			command: `Get-Process | Out-File -LiteralPath 'procs.txt'`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Out-File", Path: "procs.txt", Op: "write", TreeKnown: true},
			},
		},
		{
			name:    "Out-File with -Append flag",
			command: `... | Out-File path.txt -Append`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Out-File", Path: "path.txt", Op: "append", TreeKnown: true},
			},
		},
		{
			name:    "Out-File with -Append and -FilePath",
			command: `... | Out-File -FilePath log.txt -Append`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Out-File", Path: "log.txt", Op: "append", TreeKnown: true},
			},
		},
		{
			name:    "Out-File with -a short append",
			command: `echo "msg" | Out-File -a log.txt`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Out-File", Path: "log.txt", Op: "append", TreeKnown: true},
			},
		},
		{
			name:    "Out-File with -Encoding before path",
			command: `echo "msg" | Out-File -Encoding utf8 "out.txt"`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Out-File", Path: "out.txt", Op: "write", TreeKnown: true},
			},
		},
		{
			name:    "Out-File standalone without pipeline",
			command: `Out-File -FilePath "output.log" -InputObject "hello"`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Out-File", Path: "output.log", Op: "write", TreeKnown: true},
			},
		},

		// Add-Content & alias ac
		{
			name:    "Add-Content with -Path",
			command: `Add-Content -Path file.txt -Value 'more data'`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Add-Content", Path: "file.txt", Op: "append", TreeKnown: true},
			},
		},
		{
			name:    "Add-Content with -LiteralPath",
			command: `Add-Content -LiteralPath 'file.txt' -Value 'more data'`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Add-Content", Path: "file.txt", Op: "append", TreeKnown: true},
			},
		},
		{
			name:    "Add-Content positional path and value",
			command: `Add-Content "file.txt" "line 2"`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Add-Content", Path: "file.txt", Op: "append", TreeKnown: true},
			},
		},
		{
			name:    "Add-Content alias ac positional",
			command: `ac file.txt 'line 3'`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Add-Content", Path: "file.txt", Op: "append", TreeKnown: true},
			},
		},
		{
			name:    "Pipeline destination Add-Content",
			command: `"line" | Add-Content log.txt`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Add-Content", Path: "log.txt", Op: "append", TreeKnown: true},
			},
		},

		// Remove-Item & aliases rm, ri, del
		{
			name:    "Remove-Item with -Path",
			command: `Remove-Item -Path "file.txt"`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Remove-Item", Path: "file.txt", Op: "delete", TreeKnown: true},
			},
		},
		{
			name:    "Remove-Item with -LiteralPath",
			command: `Remove-Item -LiteralPath 'C:\temp\file[1].txt'`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Remove-Item", Path: `C:\temp\file[1].txt`, Op: "delete", TreeKnown: true},
			},
		},
		{
			name:    "Remove-Item with switches -Force -Recurse",
			command: `Remove-Item -Force -Recurse "build"`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Remove-Item", Path: "build", Op: "delete", TreeKnown: true},
			},
		},
		{
			name:    "Remove-Item alias rm",
			command: `rm file.txt`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Remove-Item", Path: "file.txt", Op: "delete", TreeKnown: true},
			},
		},
		{
			name:    "Remove-Item alias ri",
			command: `ri file.txt`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Remove-Item", Path: "file.txt", Op: "delete", TreeKnown: true},
			},
		},
		{
			name:    "Remove-Item alias del",
			command: `del "path with spaces/file.txt"`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Remove-Item", Path: "path with spaces/file.txt", Op: "delete", TreeKnown: true},
			},
		},
		{
			name:    "Remove-Item alias rm with -rf",
			command: `rm -rf "dist"`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Remove-Item", Path: "dist", Op: "delete", TreeKnown: true},
			},
		},
		{
			name:    "Remove-Item comma-separated paths positional",
			command: `Remove-Item file1.txt, file2.txt`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Remove-Item", Path: "file1.txt", Op: "delete", TreeKnown: true},
				{Cmdlet: "Remove-Item", Path: "file2.txt", Op: "delete", TreeKnown: true},
			},
		},
		{
			name:    "Remove-Item comma-separated paths with -Path",
			command: `Remove-Item -Path "a.txt", "b.txt"`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Remove-Item", Path: "a.txt", Op: "delete", TreeKnown: true},
				{Cmdlet: "Remove-Item", Path: "b.txt", Op: "delete", TreeKnown: true},
			},
		},
		{
			name:    "Remove-Item comma-separated paths without space",
			command: `Remove-Item file1.txt,file2.txt`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Remove-Item", Path: "file1.txt", Op: "delete", TreeKnown: true},
				{Cmdlet: "Remove-Item", Path: "file2.txt", Op: "delete", TreeKnown: true},
			},
		},

		// New-Item & alias ni
		{
			name:    "New-Item with -Path and -ItemType",
			command: `New-Item -Path "test.txt" -ItemType File`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "New-Item", Path: "test.txt", Op: "create", TreeKnown: true},
			},
		},
		{
			name:    "New-Item alias ni directory",
			command: `ni -ItemType Directory -Path "mydir"`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "New-Item", Path: "mydir", Op: "create", TreeKnown: true},
			},
		},
		{
			name:    "New-Item positional path",
			command: `New-Item file.txt -ItemType File`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "New-Item", Path: "file.txt", Op: "create", TreeKnown: true},
			},
		},
		{
			name:    "New-Item with -Name and -Path",
			command: `New-Item -ItemType File -Name "index.html" -Path "public"`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "New-Item", Path: "public/index.html", Op: "create", TreeKnown: true},
			},
		},
		{
			name:    "New-Item with -Name only",
			command: `New-Item -Name "standalone.txt" -ItemType File`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "New-Item", Path: "standalone.txt", Op: "create", TreeKnown: true},
			},
		},

		// Quotes handling
		{
			name:    "Escaped quotes inside double-quotes",
			command: `Set-Content -Path "dir/\"file\".txt" -Value 'data'`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Set-Content", Path: `dir/"file".txt`, Op: "write", TreeKnown: true},
			},
		},
		{
			name:    "Escaped quotes inside single-quotes",
			command: `Set-Content -Path 'dir/''file''.txt' -Value 'data'`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Set-Content", Path: `dir/'file'.txt`, Op: "write", TreeKnown: true},
			},
		},

		// TreeKnown evaluation
		{
			name:    "Dynamic environment variable in double quotes",
			command: `Set-Content -Path "$env:TEMP\foo.txt"`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Set-Content", Path: `$env:TEMP\foo.txt`, Op: "write", TreeKnown: false},
			},
		},
		{
			name:    "Variable in unquoted path",
			command: `Remove-Item $targetFile`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Remove-Item", Path: "$targetFile", Op: "delete", TreeKnown: false},
			},
		},
		{
			name:    "Subexpression path",
			command: `Out-File $(Join-Path $P "log.txt")`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Out-File", Path: `$(Join-Path $P "log.txt")`, Op: "write", TreeKnown: false},
			},
		},
		{
			name:    "Literal dollar in single quotes is known",
			command: `Set-Content -Path 'file$1.txt' -Value 'hello'`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Set-Content", Path: "file$1.txt", Op: "write", TreeKnown: true},
			},
		},
		{
			name:    "Missing path argument",
			command: `Set-Content`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Set-Content", Path: "", Op: "write", TreeKnown: false},
			},
		},

		// PowerShell wrapper unpacking
		{
			name:    "powershell -Command wrapper",
			command: `powershell -Command "Set-Content -Path file.txt -Value hello"`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Set-Content", Path: "file.txt", Op: "write", TreeKnown: true},
			},
		},
		{
			name:    "powershell.exe -Command wrapper",
			command: `powershell.exe -Command "Out-File -FilePath file.txt"`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Out-File", Path: "file.txt", Op: "write", TreeKnown: true},
			},
		},
		{
			name:    "pwsh -c wrapper",
			command: `pwsh -c "Remove-Item file.txt"`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Remove-Item", Path: "file.txt", Op: "delete", TreeKnown: true},
			},
		},

		// Multi-command / pipeline combinations
		{
			name:    "Multiple commands semicolon separated",
			command: `Set-Content a.txt '1'; Add-Content b.txt '2'`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Set-Content", Path: "a.txt", Op: "write", TreeKnown: true},
				{Cmdlet: "Add-Content", Path: "b.txt", Op: "append", TreeKnown: true},
			},
		},
		{
			name:    "Multiple commands newline separated",
			command: "Set-Content a.txt '1'\nAdd-Content b.txt '2'",
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Set-Content", Path: "a.txt", Op: "write", TreeKnown: true},
				{Cmdlet: "Add-Content", Path: "b.txt", Op: "append", TreeKnown: true},
			},
		},
		{
			name:    "Pipeline stages with non-mutating source",
			command: `Get-Content in.txt | Select-String "foo" | Out-File -Append log.txt`,
			wantTargets: []PowerShellTarget{
				{Cmdlet: "Out-File", Path: "log.txt", Op: "append", TreeKnown: true},
			},
		},

		// Non-PowerShell or empty commands
		{
			name:        "Empty command",
			command:     "",
			wantTargets: []PowerShellTarget{},
		},
		{
			name:        "Whitespace only",
			command:     "   ",
			wantTargets: []PowerShellTarget{},
		},
		{
			name:        "Git command",
			command:     "git status",
			wantTargets: []PowerShellTarget{},
		},
		{
			name:        "Bash echo",
			command:     `echo "hello world"`,
			wantTargets: []PowerShellTarget{},
		},

		// Syntax error
		{
			name:    "Unterminated quote",
			command: `Set-Content "unterminated`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			targets, err := ExtractPowerShellTargets(tt.command)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ExtractPowerShellTargets(%q) error = %v, wantErr %v", tt.command, err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			if len(targets) != len(tt.wantTargets) {
				t.Fatalf("ExtractPowerShellTargets(%q) len = %d, want %d; got %+v", tt.command, len(targets), len(tt.wantTargets), targets)
			}
			for i := range targets {
				got := targets[i]
				want := tt.wantTargets[i]
				if got.Cmdlet != want.Cmdlet || got.Path != want.Path || got.Op != want.Op || got.TreeKnown != want.TreeKnown {
					t.Errorf("ExtractPowerShellTargets(%q)[%d] = %+v, want %+v", tt.command, i, got, want)
				}
			}
		})
	}
}
