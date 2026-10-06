$ErrorActionPreference = 'Stop'

$launcherPath = Join-Path $PSScriptRoot 'launch_eventlens_online.bat'
if (-not (Test-Path -LiteralPath $launcherPath -PathType Leaf)) {
    throw "Launcher not found: $launcherPath"
}

$desktopPath = [Environment]::GetFolderPath('Desktop')
$shortcutPath = Join-Path $desktopPath 'EventLens AI.lnk'
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $launcherPath
$shortcut.WorkingDirectory = $PSScriptRoot
$shortcut.Description = 'Open EventLens AI in your default browser'
$shortcut.Save()

Write-Output "Created desktop shortcut: $shortcutPath"
