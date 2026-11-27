; Inno Setup script for the Windows installer. Build with packaging/build.ps1, which runs
; PyInstaller first and passes the version:  ISCC /DAppVersion=0.1.0 packaging\installer.iss
;
; Installs per user into %LOCALAPPDATA%\Programs (no admin prompt), so the app can write
; config.json, models\ and .setup_done next to the exe. The speech model is downloaded on
; first launch by the setup wizard (lingo_overlay/setup_wizard.py), which also guides the
; VB-Cable install and runs audio_setup.ps1.

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{6B8E2C4A-3F1D-4E7B-9A52-1C0D8F6E4B21}
AppName=Lingo Overlay
AppVersion={#AppVersion}
AppPublisher=Lingo Overlay contributors
AppPublisherURL=https://github.com/dmunasingha/lingo-overlay
DefaultDirName={localappdata}\Programs\Lingo Overlay
DefaultGroupName=Lingo Overlay
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=LingoOverlay-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\LICENSE

[Files]
Source: "..\dist\LingoOverlay\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion
Source: "..\glossary\*"; DestDir: "{app}\glossary"; Flags: recursesubdirs ignoreversion
Source: "..\audio_setup.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
; Keep the user's settings on upgrade.
Source: "..\config.json"; DestDir: "{app}"; Flags: onlyifdoesntexist uninsneveruninstall

[Icons]
Name: "{group}\Lingo Overlay"; Filename: "{app}\LingoOverlay.exe"
Name: "{group}\Lingo Overlay setup (model, VB-Cable, audio)"; Filename: "{app}\LingoOverlay.exe"; Parameters: "--setup"
Name: "{group}\Uninstall Lingo Overlay"; Filename: "{uninstallexe}"
Name: "{userdesktop}\Lingo Overlay"; Filename: "{app}\LingoOverlay.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Run]
Filename: "{app}\LingoOverlay.exe"; Description: "Start Lingo Overlay (first start downloads the speech model)"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Downloaded models are hundreds of MB: remove them with the app.
Type: filesandordirs; Name: "{app}\models"
Type: files; Name: "{app}\.setup_done"
