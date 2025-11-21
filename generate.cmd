@echo off
set UnrealEditorPath=E:\UnrealEngine\UE_4.27\Engine\Binaries\DotNET\UnrealBuildTool.exe
%UnrealEditorPath% -ProjectFiles -project=%~dp0TestPlugin.uproject -game -progress -engine
pause