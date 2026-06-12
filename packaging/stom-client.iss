; Inno Setup script — wraps dist/stom-client/ into StomClientSetup.exe.
; Compiled in CI by ISCC.exe (Inno Setup 6).

[Setup]
AppName=Stom CBCT Viewer
AppVersion=0.1.6
AppPublisher=Stom
DefaultDirName={autopf}\StomClient
DefaultGroupName=Stom
DisableProgramGroupPage=yes
OutputDir=installer
OutputBaseFilename=StomClientSetup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
; In-app updater downloads + launches this installer while the app may still be
; running; let Setup close the running instance so it can replace the files.
CloseApplications=yes
RestartApplications=no

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "dist\stom-client\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Stom CBCT Viewer"; Filename: "{app}\stom-client.exe"
Name: "{group}\Uninstall Stom"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Stom CBCT Viewer"; Filename: "{app}\stom-client.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\stom-client.exe"; Description: "Launch Stom CBCT Viewer"; Flags: nowait postinstall skipifsilent
