param(
    [string]$TaskName = "Audio2MBTI-Full-CNN-Pipeline",
    [string]$StartTime = "02:00",
    [string]$ProjectRoot = "D:\project"
)

$batchPath = Join-Path $ProjectRoot "run_full_cnn_pipeline.bat"
if (-not (Test-Path $batchPath)) {
    throw "Batch file not found: $batchPath"
}

$createArgs = @(
    "/Create",
    "/TN", $TaskName,
    "/SC", "DAILY",
    "/ST", $StartTime,
    "/TR", "`"$batchPath`"",
    "/F"
)

Write-Host "Registering task $TaskName at $StartTime"
schtasks.exe @createArgs

if ($LASTEXITCODE -ne 0) {
    throw "Failed to create scheduled task."
}

Write-Host "Task created successfully."
