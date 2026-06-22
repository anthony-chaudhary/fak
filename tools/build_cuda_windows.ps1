<#
build_cuda_windows.ps1 - native-Windows `-tags cuda` build + Authenticode sign (issue #481).

Ports internal/compute/build_cuda.sh OFF the WSL workaround onto a native Windows host: it
compiles the CUDA kernels with the Windows CUDA Toolkit's nvcc into an in-tree static archive,
builds the `-tags cuda` variant of cmd/fak against it (cgo, CGO_CFLAGS/CGO_LDFLAGS injected here
so internal/compute/cuda_windows.go does not have to bake the space-containing toolkit path into
a #cgo literal), and then code-signs the produced binary. Signing is load-bearing, not cosmetic:
the WDAC / Smart-App-Control policy on the reference host refuses to fork/exec freshly compiled
UNSIGNED binaries from %TEMP% (the reason the suite runs in WSL today), so an unsigned fak.exe -
and any test binary it spawns - is blocked. The DEFAULT `go build ./cmd/fak` needs NONE of this
and stays pure-Go (DIRECTION reviewer check 3); this script is the opt-in build/CI seam only.

REQUIREMENTS (a Windows host - this script does not provision them):
  - NVIDIA CUDA Toolkit for Windows (provides nvcc.exe; sets the CUDA_PATH env var).
  - A cgo C compiler that understands the GNU-style -l/-L flags inherited from cuda.go
    (mingw-w64 gcc - the toolchain Go documents for cgo on Windows - or clang in GNU-driver mode),
    plus `ar` from the same toolchain to build libfakcuda.a. Falls back to `nvcc --lib` if no `ar`.
  - Go (the toolchain pinned by go.mod; GOTOOLCHAIN=auto fetches it if the host's Go is older).
  - signtool.exe from the Windows SDK (skip with -SkipSign for a build-only smoke run).

ENV (cert is ALWAYS from the environment, NEVER hardcoded):
  FAK_CUDA_ARCH            sm_89 (default) | sm_80 | sm_90 | sm_100   ("89" also accepted)
  CUDA_PATH               CUDA Toolkit root (default: the var the Windows installer sets)
  FAK_SIGN_CERT_THUMBPRINT SHA-1 thumbprint of a cert already in the Windows cert store, OR
  FAK_SIGN_PFX             path to a .pfx, with FAK_SIGN_PFX_PASSWORD for its password
  FAK_TIMESTAMP_URL        RFC-3161 timestamp server (default: http://timestamp.digicert.com)

EXAMPLES:
  pwsh tools/build_cuda_windows.ps1                       # build + sign (cert from env)
  pwsh tools/build_cuda_windows.ps1 -SkipSign            # build only (WDAC will still block run)
  $env:FAK_CUDA_ARCH='sm_90'; pwsh tools/build_cuda_windows.ps1 -Output fak-cuda.exe

RESIDUAL / HANDOFF: the produced native-Windows cuda binary, its signature, and the Approx gate
(`go test -tags cuda`) can only be realized on a Windows host that actually has the CUDA Toolkit,
a GPU, and a signing cert. This script is the wiring; running it there is the explicit follow-up.
#>
[CmdletBinding()]
param(
  [string] $Arch = $(if ($env:FAK_CUDA_ARCH) { $env:FAK_CUDA_ARCH } else { 'sm_89' }),
  [string] $CudaPath = $env:CUDA_PATH,
  [string] $Output = 'fak.exe',
  [switch] $SkipSign
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Module root = this script's parent's parent (tools/.. == clone root, the Go module root).
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ModuleRoot = Split-Path -Parent $ScriptDir
$PkgDir = Join-Path $ModuleRoot 'internal\compute'

function Invoke-Checked {
  # Run an external command and throw on a non-zero exit so the pipeline fails closed.
  param([Parameter(Mandatory)] [string] $Exe, [Parameter(ValueFromRemainingArguments)] [string[]] $Rest)
  Write-Host "[cuda-win] $Exe $($Rest -join ' ')"
  & $Exe @Rest
  if ($LASTEXITCODE -ne 0) { throw "$Exe failed (exit $LASTEXITCODE)" }
}

function Resolve-Nvcc {
  # Prefer $CudaPath\bin\nvcc.exe; fall back to nvcc already on PATH (DLVM/dev image).
  param([string] $Root)
  if ($Root) {
    $cand = Join-Path $Root 'bin\nvcc.exe'
    if (Test-Path $cand) { return $cand }
  }
  $onPath = Get-Command nvcc.exe -ErrorAction SilentlyContinue
  if ($onPath) { return $onPath.Source }
  throw "nvcc.exe not found. Install the CUDA Toolkit for Windows and set CUDA_PATH (got '$Root')."
}

function Resolve-SignTool {
  # signtool ships in the Windows SDK bin; search PATH then the usual SDK locations.
  $onPath = Get-Command signtool.exe -ErrorAction SilentlyContinue
  if ($onPath) { return $onPath.Source }
  $roots = @("${env:ProgramFiles(x86)}\Windows Kits\10\bin", "${env:ProgramFiles}\Windows Kits\10\bin")
  foreach ($r in $roots) {
    if (Test-Path $r) {
      $found = Get-ChildItem -Path $r -Recurse -Filter signtool.exe -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match '\\x64\\' } | Select-Object -Last 1
      if ($found) { return $found.FullName }
    }
  }
  throw "signtool.exe not found. Install the Windows SDK, or pass -SkipSign for a build-only run."
}

# --- 1. resolve toolkit + normalize arch ----------------------------------------------------
$nvcc = Resolve-Nvcc -Root $CudaPath
if (-not $CudaPath) { $CudaPath = Split-Path -Parent (Split-Path -Parent $nvcc) }  # ...\bin\nvcc -> root
if ($Arch -notmatch '^sm_') { $Arch = "sm_$Arch" }
Write-Host "[cuda-win] CUDA_PATH=$CudaPath  nvcc=$nvcc  arch=$Arch"

$cudaInc = Join-Path $CudaPath 'include'
$cudaLib = Join-Path $CudaPath 'lib\x64'
foreach ($d in @($cudaInc, $cudaLib)) {
  if (-not (Test-Path $d)) { throw "expected CUDA dir missing: $d" }
}

# --- 2. nvcc compile kernels -> in-tree static archive (matches cuda.go's -lfakcuda) --------
$obj = Join-Path $PkgDir 'cuda_kernels.obj'
$lib = Join-Path $PkgDir 'libfakcuda.a'
Push-Location $PkgDir
try {
  Invoke-Checked $nvcc '-O3' '-std=c++14' "-arch=$Arch" '-Xcompiler' '/MD' '-c' 'cuda_kernels.cu' '-o' $obj
  $ar = Get-Command ar.exe -ErrorAction SilentlyContinue
  if (-not $ar) { $ar = Get-Command llvm-ar.exe -ErrorAction SilentlyContinue }
  if ($ar) {
    Invoke-Checked $ar.Source 'rcs' $lib $obj
  }
  else {
    # No GNU archiver on PATH: let nvcc drive the host archiver into the same filename so
    # cuda.go's `-lfakcuda` still resolves. (mingw ld reads a COFF .lib under this name too.)
    Write-Host '[cuda-win] no ar/llvm-ar on PATH; archiving via nvcc --lib'
    Invoke-Checked $nvcc '--lib' '--output-file' $lib $obj
  }
  Write-Host "[cuda-win] built $lib ($((Get-Item $lib).Length) bytes)"
}
finally {
  Pop-Location
}

# --- 3. cgo build of the -tags cuda variant -------------------------------------------------
# Inject the toolkit include/lib paths via env (cuda_windows.go keeps these OUT of its #cgo
# literal because the default install path has spaces). Quote the paths so Go's CGO_* parser
# keeps a space-containing path as one argument.
$env:CGO_ENABLED = '1'
if (-not $env:GOTOOLCHAIN) { $env:GOTOOLCHAIN = 'auto' }
$env:CGO_CFLAGS = "-I`"$cudaInc`""
$env:CGO_LDFLAGS = "-L`"$cudaLib`""
Push-Location $ModuleRoot
try {
  Invoke-Checked 'go' 'build' '-tags' 'cuda' '-o' $Output './cmd/fak'
  Write-Host "[cuda-win] built $Output"
}
finally {
  Pop-Location
}
$artifact = Join-Path $ModuleRoot $Output

# --- 4. Authenticode sign (cert strictly from env) ------------------------------------------
if ($SkipSign) {
  Write-Host '[cuda-win] -SkipSign set: leaving binary UNSIGNED (WDAC will block fork/exec).'
}
else {
  $signtool = Resolve-SignTool
  $tsUrl = if ($env:FAK_TIMESTAMP_URL) { $env:FAK_TIMESTAMP_URL } else { 'http://timestamp.digicert.com' }
  if ($env:FAK_SIGN_CERT_THUMBPRINT) {
    $signArgs = @('sign', '/sha1', $env:FAK_SIGN_CERT_THUMBPRINT, '/fd', 'SHA256', '/tr', $tsUrl, '/td', 'SHA256', $artifact)
  }
  elseif ($env:FAK_SIGN_PFX) {
    $signArgs = @('sign', '/f', $env:FAK_SIGN_PFX, '/fd', 'SHA256', '/tr', $tsUrl, '/td', 'SHA256')
    if ($env:FAK_SIGN_PFX_PASSWORD) { $signArgs += @('/p', $env:FAK_SIGN_PFX_PASSWORD) }
    $signArgs += $artifact
  }
  else {
    throw 'No signing cert in env. Set FAK_SIGN_CERT_THUMBPRINT or FAK_SIGN_PFX (+FAK_SIGN_PFX_PASSWORD), or pass -SkipSign. WDAC blocks unsigned fork/exec - we do NOT ship an unsigned binary silently.'
  }
  Invoke-Checked $signtool @signArgs
  $sig = Get-AuthenticodeSignature -FilePath $artifact
  if ($sig.Status -ne 'Valid') { throw "signature not Valid after signing: $($sig.Status) - $($sig.StatusMessage)" }
  Write-Host "[cuda-win] signed: $($sig.SignerCertificate.Subject) status=$($sig.Status)"
}

Write-Host "[cuda-win] done -> $artifact"
Write-Host '[cuda-win] next: run the Approx gate on a GPU host:  go test -tags cuda ./internal/compute/ ./internal/model/'
