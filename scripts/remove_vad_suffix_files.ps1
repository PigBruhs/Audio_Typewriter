param(
    [string]$TargetDir = "E:\Audios\max",
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $TargetDir -PathType Container)) {
    throw "Target directory not found: $TargetDir"
}

$files = Get-ChildItem -LiteralPath $TargetDir -Recurse -File | Where-Object { $_.Name -like "*_vad.wav" }

if (-not $files -or $files.Count -eq 0) {
    Write-Host "No files matched '*_vad.wav' in: $TargetDir"
    exit 0
}

$totalBytes = ($files | Measure-Object -Property Length -Sum).Sum
Write-Host "Matched $($files.Count) file(s), total size $totalBytes bytes"

foreach ($file in $files) {
    Remove-Item -LiteralPath $file.FullName -Force -WhatIf:$WhatIf.IsPresent
}

if ($WhatIf) {
    Write-Host "Dry run complete. No files were deleted."
} else {
    Write-Host "Deleted $($files.Count) file(s)."
}

