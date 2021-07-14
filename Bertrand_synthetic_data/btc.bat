:: ============================================================
:: Daimensions Docker Windows Batch Script
::
:: This code is copyrighted (c) 2020 by Brainome, Inc. All Rights Reserved.
:: Please contact support@brainome.ai with any questions.
::
:: Usage: 
::   btc-docker -update            : update to latest version
::   btc-docker -update-beta       : update to beta version
::   btc-docker arg1 [arg2 [...]]  : run Daimensions with arguments
::
:: Version 1.01 - username converted to lowercase and remove spaces
::
:: ============================================================

@echo off

setlocal

set DOCKERIMAGE=brainome/btc_local_gpu:alpha
set USERNAMELC=%username%
call :LoCase USERNAMELC
set USERIMAGE=btc-%USERNAMELC%
set UPDATE=0
set docker=docker.exe

:: ============================================================
:: Check that docker is installed
:: ============================================================
where /q %docker%
if ERRORLEVEL 1 (
    echo docker is not installed on this machine.
    echo Please install first and restart this script.
    exit /B
) 

:: Check if user wants to update
if "%1%" == "-update" ( 
    set UPDATE=1 
)

:: Check if user wants to update to a beta release
if "%1%" == "-update-beta" (
    set DOCKERIMAGE=brainome/btc_local_gpu:beta
    set UPDATE=1
)

if %UPDATE% == 1 ( 
   goto :updatedocker 
) 

:: ============================================================
:: If the user image is missing, have the user run -update
:: ============================================================
set "UIexist="
for /f "delims=" %%A in ('%docker% images -q %USERIMAGE%') do set "UIexist=%%A"
if "%UIexist%"=="" (
    echo The docker image %USERIMAGE% is not present on this machine.
    echo Please run:  %0 -update
    exit /B
)

goto :rundocker

:: ============================================================
:: Update Daimensions docker image
:: ============================================================
:updatedocker

echo Updating %DOCKERIMAGE%
%docker% pull %DOCKERIMAGE%

for /F "delims=" %%C in ('%docker% images -q %DOCKERIMAGE%') do set "DIexist=%%C"
if "%DIexist%"=="" (
   echo Docker image %DOCKERIMAGE% was not downloaded properly. Terminating.
   echo Make sure that your docker credential are authorized by Brainome.
   exit /B
)

set "UIexist="
for /f "delims=" %%A in ('%docker% images -q %USERIMAGE%') do set "UIexist=%%A"
if not "%UIexist%"=="" (
   echo Deleting user docker image %USERIMAGE%
   %docker% rmi %USERIMAGE%
)

echo Creating user image %USERIMAGE% from %DOCKERIMAGE%
%docker% tag %DOCKERIMAGE% %USERIMAGE%

set "UIexist="
for /f "delims=" %%A in ('%docker% images -q %USERIMAGE%') do set "UIexist=%%A"
if "%UIexist%"=="" (
   echo Docker image %USERIMAGE% was not created properly. Terminating.
   exit /B
)

echo Docker image %USERIMAGE% was created successfully.
exit /B

:: ============================================================
:: Run Daimensions docker
:: ============================================================
:rundocker 

:: GPU Check is not implemented on Windows yet
set USEGPU=

:: Get all args after the first - it's complicated in windows...
set RESTARGS=
:argloop
if "%1"=="" goto :argdone
set RESTARGS=%RESTARGS% %1
shift
goto argloop

:argdone

%docker% run --rm %USEGPU% -it --mount type=bind,source="%cd%",target=/btc %USERIMAGE% %RESTARGS%
:: echo %docker% run --rm %USEGPU% -it --mount type=bind,source="%cd%",target=/btc %USERIMAGE% %RESTARGS%

endlocal
GOTO:EOF

:LoCase
:: Subroutine to convert a variable VALUE to all lower case.
:: The argument for this subroutine is the variable NAME.
FOR %%i IN ("A=a" "B=b" "C=c" "D=d" "E=e" "F=f" "G=g" "H=h" "I=i" "J=j" "K=k" "L=l" "M=m" "N=n" "O=o" "P=p" "Q=q" "R=r" "S=s" "T=t" "U=u" "V=v" "W=w" "X=x" "Y=y" "Z=z" " =" ) DO CALL SET "%1=%%%1:%%~i%%"
GOTO:EOF
