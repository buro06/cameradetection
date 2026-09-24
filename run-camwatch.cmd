@echo off
rem camwatch watcher: activates .venv, runs camwatch, and restarts it in this window whenever it exits.
rem
rem   run-camwatch                   live dashboard (default)
rem   run-camwatch run --headless    no dashboard, logs to the console
rem
rem Arguments are passed straight to camwatch. To stop for good, press Q at the restart prompt
rem (or Ctrl+C, then Y). Restarts are logged to data\logs\watcher.log.

setlocal
cd /d "%~dp0"
title camwatch (watcher)

if not exist ".venv\Scripts\activate.bat" (
    echo [watcher] No .venv in %CD% - create it first, see README "Install".
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
if not exist "data\logs" mkdir "data\logs"
set "WATCHLOG=data\logs\watcher.log"
set "QUICK_EXITS=0"

:loop
call :now
set "STARTED=%NOW_EPOCH%"
call :log "starting camwatch"
python -m camwatch %*
set "CODE=%ERRORLEVEL%"
call :now
set /a RAN=NOW_EPOCH-STARTED

rem Back off when it keeps dying right after starting (camera/GPU/network not ready): 10s, 20s, ... 5 min.
if %RAN% LSS 60 (set /a QUICK_EXITS+=1) else (set "QUICK_EXITS=0")
set /a DELAY=10*QUICK_EXITS
if %DELAY% LSS 10 set "DELAY=10"
if %DELAY% GTR 300 set "DELAY=300"

call :log "camwatch exited with code %CODE% after %RAN%s, restarting in %DELAY%s"
echo.
choice /c RQ /t %DELAY% /d R /n /m "[watcher] Press R to restart now, Q to quit: "
if %ERRORLEVEL%==2 goto stopped
goto loop

:stopped
call :now
call :log "watcher stopped by user"
exit /b %CODE%

rem Sets NOW_EPOCH (seconds) and NOW_TEXT (local time, 12-hour clock like the rest of camwatch).
:now
set "NOW_EPOCH=0"
set "NOW_TEXT=%DATE% %TIME%"
for /f "tokens=1,*" %%a in ('python -c "import time; from camwatch.timefmt import stamp; t = time.time(); print(int(t), stamp(t))" 2^>nul') do (
    set "NOW_EPOCH=%%a"
    set "NOW_TEXT=%%b"
)
exit /b

:log
echo [watcher %NOW_TEXT%] %~1
>>"%WATCHLOG%" echo %NOW_TEXT%  %~1
exit /b
