@echo off

REM Run pylint and redirect output to pylint_output.txt
pylint src src/checkers > pylint_output.txt

REM Run flake8 with --show-source option and redirect output to flake8_output.txt
flake8 > flake8_output.txt

REM Parse the output of pylint to get the number of reported issues
set PYLINT_SCORE=
for /f "tokens=7 delims= " %%i in ('findstr /r "Your code has been rated at [0-9]*\.[0-9]*\/10" pylint_output.txt') do (
    set "PYLINT_SCORE=%%i"
)
REM Parse the output of flake8 to get the number of reported issues
set FLAKE8_ISSUES=
for /f %%j in ('flake8 --count') do set FLAKE8_ISSUES=%%j

REM Display the pylint quality score
if "%PYLINT_SCORE%"=="10.00/10" (
    echo Pylint Quality Score: 10/10
) else (
    echo Pylint Quality Score: %PYLINT_SCORE% - Look at pylint_output.txt for more details
)

REM Display the number of flake8 issues
if %FLAKE8_ISSUES% GTR 0 (
    echo Flake8 Issues: %FLAKE8_ISSUES% - Look at flake8_output.txt for more details
) else (
    echo Flake8 Issues: 0
)

