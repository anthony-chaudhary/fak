# build_vulkan.ps1 — compile the GLSL compute shaders to SPIR-V (glslc) and the Vulkan
# shim (clang++ -> libfakvulkan.a), then build/test the `-tags vulkan` variant of the
# compute package NATIVELY on Windows. The real AMD GPU is only reachable through the
# Windows Vulkan loader, so unlike the rest of fak's suite this lane does NOT run in WSL.
#
#   usage:  pwsh internal/compute/build_vulkan.ps1 [shaders|lib|build|test]   (default: test)
#           pwsh internal/compute/build_vulkan.ps1 binary <pkg> <out>
#
# Requires (installed via winget; see memory fak-vulkan-toolchain):
#   - Vulkan SDK at $env:VULKAN_SDK (default C:\VulkanSDK\1.4.350.0): glslc, headers, vulkan-1.lib
#   - clang/clang++ on PATH (or C:\Program Files\LLVM\bin)
#   - a Windows C++ standard-library environment (MSVC Build Tools / Developer PowerShell)
#   - for build/test: a Go-compatible Windows cgo toolchain (MinGW-w64/MSYS2 gcc + g++)
[CmdletBinding()]
param(
    [string]$cmd = "test",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest = @()
)

$ErrorActionPreference = "Stop"

$isWindowsHost = ($env:OS -eq "Windows_NT") -or ($PSVersionTable.Platform -eq "Win32NT") -or ($IsWindows -eq $true)

function Find-VsDevCmd {
    $candidates = @()
    $programFilesX86 = ${env:ProgramFiles(x86)}
    if ($programFilesX86) {
        $vswhere = Join-Path $programFilesX86 "Microsoft Visual Studio\Installer\vswhere.exe"
        if (Test-Path $vswhere) {
            $installPaths = @(& $vswhere -latest -products * -property installationPath 2>$null)
            foreach ($installPath in $installPaths) {
                if ($installPath) {
                    $candidates += Join-Path $installPath "Common7\Tools\VsDevCmd.bat"
                }
            }
        }
        $candidates += Join-Path $programFilesX86 "Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
        $candidates += Join-Path $programFilesX86 "Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
        $candidates += Join-Path $programFilesX86 "Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat"
        $candidates += Join-Path $programFilesX86 "Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat"
    }
    foreach ($candidate in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        if (Test-Path $candidate) { return $candidate }
    }
    return $null
}

function Import-VsDevEnv {
    if (-not $isWindowsHost) { return }
    if ($env:INCLUDE -and $env:LIB -and $env:LIBPATH) { return }

    $vsDevCmd = Find-VsDevCmd
    if (-not $vsDevCmd) { return }

    Write-Host "[vulkan] importing MSVC environment from $vsDevCmd"
    $cmdLine = "`"$vsDevCmd`" -no_logo -arch=amd64 -host_arch=amd64 >nul && set"
    $envLines = & cmd.exe /s /c $cmdLine
    if ($LASTEXITCODE -ne 0) {
        throw "VsDevCmd.bat failed while preparing the MSVC environment"
    }
    foreach ($line in $envLines) {
        if ($line -match "^([^=]+)=(.*)$") {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
}

$VK = if ($env:VULKAN_SDK) { $env:VULKAN_SDK } else { "C:\VulkanSDK\1.4.350.0" }
$glslc = Join-Path $VK "Bin\glslc.exe"
$vkInclude = Join-Path $VK "Include"
$vkLib = Join-Path $VK "Lib"
if (-not (Test-Path $glslc)) { throw "glslc not found at $glslc — set VULKAN_SDK" }

$llvmBin = "C:\Program Files\LLVM\bin"
if ($isWindowsHost -and (Test-Path $llvmBin) -and ($env:PATH -notlike "*$llvmBin*")) {
    $env:PATH = "$llvmBin;$env:PATH"
}
Import-VsDevEnv

$clangxxCmd = Get-Command "clang++" -ErrorAction SilentlyContinue
$clangxx = if ($clangxxCmd) { $clangxxCmd.Source } else { "C:\Program Files\LLVM\bin\clang++.exe" }
$llvmArCmd = Get-Command "llvm-ar" -ErrorAction SilentlyContinue
$llvmAr = if ($llvmArCmd) { $llvmArCmd.Source } else { "C:\Program Files\LLVM\bin\llvm-ar.exe" }
if (-not (Test-Path $clangxx)) { throw "clang++ not found on PATH or at $clangxx" }
if (-not (Test-Path $llvmAr)) { throw "llvm-ar not found on PATH or at $llvmAr" }
$script:libCxx = $clangxx
$script:libAr = $llvmAr
$script:cxxRuntime = if ($isWindowsHost) { "-lmsvcprt" } else { "-lc++" }

$pkgDir = $PSScriptRoot                                  # internal/compute
$modDir = (Resolve-Path (Join-Path $pkgDir "..\..")).Path # fak/
$shaderSrc = Join-Path $pkgDir "shaders"
$spvOut = Join-Path $pkgDir "spirv"                      # compiled .spv land here

function Test-CxxToolchain {
    $probe = Join-Path $env:TEMP "fak_vulkan_cxx_probe.cpp"
    $obj = Join-Path $env:TEMP "fak_vulkan_cxx_probe.obj"
    Set-Content -Encoding ASCII -Path $probe -Value "#include <vector>`n#include <cmath>`nint main(){std::vector<float> v{1.0f}; return std::isfinite(v[0]) ? 0 : 1;}`n"
    $out = & $script:libCxx -std=c++17 -c $probe -o $obj 2>&1
    $code = $LASTEXITCODE
    Remove-Item -ErrorAction SilentlyContinue $probe, $obj
    if ($code -ne 0) {
        throw "$script:libCxx is present but cannot compile C++ standard-library headers. Install the matching C++ toolchain environment before running build_vulkan.ps1.`n$out"
    }
}

function Use-GoCgoToolchain {
    if (-not $isWindowsHost) { return }

    $gccCmd = Get-Command "gcc" -ErrorAction SilentlyContinue
    $gxxCmd = Get-Command "g++" -ErrorAction SilentlyContinue
    if (-not $gccCmd -or -not $gxxCmd) {
        throw "go build -tags vulkan requires a GCC-compatible Windows cgo toolchain (MinGW-w64/MSYS2 gcc + g++). Go's Windows cgo path passes MinGW flags such as -mthreads, which the LLVM/MSVC clang toolchain cannot consume. Install MSYS2 UCRT64/MinGW-w64 and put gcc/g++ on PATH, then rerun build_vulkan.ps1."
    }

    $arCmd = Get-Command "ar" -ErrorAction SilentlyContinue
    $script:libCxx = $gxxCmd.Source
    $script:libAr = if ($arCmd) { $arCmd.Source } else { $llvmAr }
    $script:cxxRuntime = "-lstdc++"
    $env:CC = $gccCmd.Name
    $env:CXX = $gxxCmd.Name
}

function Build-Shaders {
    New-Item -ItemType Directory -Force -Path $spvOut | Out-Null
    $shaders = @("matmul","matmul_add","matmul_argmax","matmul_argmax_blocks","matmul2","matmul3","rmsnorm","rmsnorm_matmul","rmsnorm_matmul2","rmsnorm_matmul3","rmsnorm_matmul_argmax_blocks","rope","swiglu","swiglu_matmul_add","add","add_bias","attention","argmax","argmax_pairs","q8_matmul","q8_matmul2","q8_matmul3","rmsnorm_q8_matmul2","rmsnorm_q8_matmul3","swiglu_q8_matmul_add")
    foreach ($s in $shaders) {
        $src = Join-Path $shaderSrc "$s.comp"
        $dst = Join-Path $spvOut "$s.spv"
        Write-Host "[vulkan] glslc $s.comp -> $s.spv"
        & $glslc -O --target-env=vulkan1.2 -fshader-stage=comp $src -o $dst
        if ($LASTEXITCODE -ne 0) { throw "glslc failed on $s.comp" }
    }
    Write-Host "[vulkan] compiled $($shaders.Count) SPIR-V modules to $spvOut"
}

function Build-Lib {
    Push-Location $pkgDir
    try {
        Test-CxxToolchain
        Write-Host "[vulkan] C++ compile vulkan_shim.cpp -> libfakvulkan.a"
        $pic = if ($isWindowsHost) { @() } else { @("-fPIC") }
        $crt = if ($isWindowsHost) { @("-D_CRT_SECURE_NO_WARNINGS") } else { @() }
        $cxxArgs = @("-O3", "-std=c++17") + $pic + $crt + @("-I$vkInclude", "-c", "vulkan_shim.cpp", "-o", "vulkan_shim.o")
        & $script:libCxx @cxxArgs
        if ($LASTEXITCODE -ne 0) { throw "C++ compiler failed compiling vulkan_shim.cpp" }
        & $script:libAr rcs libfakvulkan.a vulkan_shim.o
        if ($LASTEXITCODE -ne 0) { throw "llvm-ar failed" }
        Write-Host "[vulkan] built libfakvulkan.a ($((Get-Item libfakvulkan.a).Length) bytes)"
    } finally { Pop-Location }
}

# cgo env so `go build -tags vulkan` finds the header, the lib, and the loader import lib.
function Set-CgoEnv {
    $env:CGO_ENABLED = "1"
    if ($isWindowsHost) {
        Use-GoCgoToolchain
    } else {
        $env:CC = ($clangxx -replace 'clang\+\+\.exe$', 'clang.exe')
        $env:CXX = $clangxx
    }
    $env:CGO_CFLAGS = "-I$vkInclude -I$pkgDir"
    # Link our static shim + the Vulkan loader import lib + the host C++ runtime.
    $env:CGO_LDFLAGS = "-L$pkgDir -lfakvulkan -L$vkLib -lvulkan-1 $script:cxxRuntime"
    $env:FAK_VULKAN_SPIRV = $spvOut
}

switch ($cmd) {
    "shaders" { Build-Shaders }
    "lib"     { Build-Lib }
    "build" {
        Set-CgoEnv; Build-Shaders; Build-Lib
        Push-Location $modDir
        try {
            Write-Host "[vulkan] go build -tags vulkan ./internal/compute/"
            & go build -tags vulkan ./internal/compute/
            if ($LASTEXITCODE -ne 0) { throw "go build -tags vulkan failed" }
            Write-Host "[vulkan] OK build"
        } finally { Pop-Location }
    }
    "binary" {
        if ($Rest.Count -ne 2) {
            throw "usage: build_vulkan.ps1 binary <pkg> <out>"
        }
        Set-CgoEnv; Build-Shaders; Build-Lib
        Push-Location $modDir
        try {
            $pkg = $Rest[0]
            $out = $Rest[1]
            Write-Host "[vulkan] go build -tags vulkan -o $out $pkg"
            & go build -tags vulkan -o $out $pkg
            if ($LASTEXITCODE -ne 0) { throw "go build -tags vulkan failed" }
            Write-Host "[vulkan] OK binary $out"
        } finally { Pop-Location }
    }
    "test" {
        Set-CgoEnv; Build-Shaders; Build-Lib
        Push-Location $modDir
        try {
            Write-Host "[vulkan] go test -tags vulkan (compute + model HAL on the real GPU)"
            & go test -tags vulkan -count=1 -v -run 'Vulkan|HALDevice' ./internal/compute/ ./internal/model/
            if ($LASTEXITCODE -ne 0) { throw "go test -tags vulkan failed" }
        } finally { Pop-Location }
    }
    default { throw "unknown subcommand: $cmd (use shaders|lib|build|binary|test)" }
}
