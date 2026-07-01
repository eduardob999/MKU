<#
Compact the WSL2 virtual disk (ext4.vhdx) to return freed space to Windows C:.

WHY: WSL's disk is a dynamically-growing file on C:. Deleting files *inside* WSL
frees space internally but the .vhdx does not shrink on its own, so C: stays full.
This compacts it back down.

IMPORTANT — the .vhdx must be released first. It is held open by ANYTHING using
the WSL filesystem: a running distro, VS Code Remote-WSL, Docker Desktop's WSL2
backend, Windows Explorer windows on \\wsl.localhost, and any Claude Code session
whose working directory is on \\wsl.localhost. If diskpart reports
"the file is being used by another process", something is still holding it.

MOST RELIABLE: reboot Windows, then run this in an elevated PowerShell BEFORE
opening VS Code / a terminal / Claude on the WSL filesystem.

Run as Administrator:  powershell -ExecutionPolicy Bypass -File compact_wsl_disk.ps1
#>

$ErrorActionPreference = "Stop"

# --- must be elevated ---
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
         ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { Write-Error "Run this in an ELEVATED PowerShell (Run as Administrator)."; exit 1 }

$distro = "Ubuntu"

# --- locate the distro's ext4.vhdx from the registry (robust to path changes) ---
$vhdx = $null
Get-ChildItem "HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss" | ForEach-Object {
    $p = Get-ItemProperty $_.PSPath
    if ($p.DistributionName -eq $distro) {
        $base = [Environment]::ExpandEnvironmentVariables($p.BasePath)
        $vhdx = Join-Path $base "ext4.vhdx"
    }
}
if (-not $vhdx -or -not (Test-Path $vhdx)) { Write-Error "Could not find $distro ext4.vhdx."; exit 1 }

function Show-State($label) {
    $sizeGB = [math]::Round((Get-Item $vhdx).Length / 1GB, 1)
    $c = Get-Volume C
    "{0}: vhdx = {1} GB | C: free = {2} GB" -f $label, $sizeGB,
        [math]::Round($c.SizeRemaining / 1GB, 1)
}
Write-Output "vhdx: $vhdx"
Write-Output (Show-State "before")

# --- stop WSL and wait for the lightweight VM to fully release the file ---
Write-Output "Shutting down WSL..."
wsl --shutdown
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep 2
    if (-not (Get-Process -Name "vmmemWSL","vmmem" -ErrorAction SilentlyContinue)) { break }
}
Start-Sleep 5

if (Get-Process -Name "Code","docker" -ErrorAction SilentlyContinue) {
    Write-Warning "VS Code / Docker still running — they may re-open the WSL disk and cause "
    Write-Warning "a 'file in use' error. Close them (and any Claude/Explorer on \\wsl.localhost) if so."
}

# --- compact (attach read-only so the ext4 fs is untouched, just the container shrinks) ---
$dp = @"
select vdisk file="$vhdx"
attach vdisk readonly
compact vdisk
detach vdisk
exit
"@
$tmp = Join-Path $env:TEMP "wsl_compact_$(Get-Date -Format HHmmss).txt"
$dp | Set-Content -Encoding Ascii $tmp
Write-Output "Compacting (this can take a few minutes)..."
diskpart /s $tmp
Remove-Item $tmp -ErrorAction SilentlyContinue

Write-Output (Show-State "after")
Write-Output "Done. If it errored with 'file in use', reboot and run this before opening anything WSL."
