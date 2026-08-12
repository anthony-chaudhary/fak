@echo off
rem One tick of the bench-fleet loop, as committed tooling (#6503).
rem
rem Usage: fak-bench-fleet-tick.cmd <fak.exe> <workspace-root>
rem
rem This payload used to be written to %TEMP% by `fak bench-loop install`, so the
rem scheduled action depended on an ephemeral file that any cleanup could delete out
rem from under it. It lives in the tree now, and it PROPAGATES the dispatch result:
rem the dispatcher exits nonzero when the durable queue holds no successful benchmark
rem measurement, and the scheduler must see that instead of a healthy 0.
setlocal
set "FAK_BIN=%~1"
set "FAK_ROOT=%~2"
if "%FAK_BIN%"=="" goto :usage
if "%FAK_ROOT%"=="" goto :usage

"%FAK_BIN%" bench-loop fleet --apply --json --workspace "%FAK_ROOT%"
if errorlevel 1 exit /b %errorlevel%

"%FAK_BIN%" bench-loop fleet dispatch --json --workspace "%FAK_ROOT%"
exit /b %errorlevel%

:usage
echo usage: fak-bench-fleet-tick.cmd ^<fak.exe^> ^<workspace-root^> 1>&2
exit /b 2
