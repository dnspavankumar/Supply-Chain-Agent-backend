[CmdletBinding()]
param(
    [Alias("Host")]
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$NoReload,
    [switch]$InstallDeps
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptRoot

function Resolve-Python {
    $venvPython = Join-Path $scriptRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return @{
            Command = $venvPython
            PrefixArgs = @()
        }
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        return @{
            Command = $pythonCmd.Source
            PrefixArgs = @()
        }
    }

    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        return @{
            Command = $pyCmd.Source
            PrefixArgs = @("-3")
        }
    }

    throw "Python was not found. Install Python or create .venv first."
}

function Resolve-AppTarget {
    if (Test-Path (Join-Path $scriptRoot "main.py")) {
        return "main:app"
    }

    throw "Could not find a FastAPI entrypoint. Expected main.py in the agent root."
}

$python = Resolve-Python

if ($InstallDeps) {
    Write-Host "Installing dependencies from requirements.txt..."
    & $python.Command @($python.PrefixArgs + @("-m", "pip", "install", "-r", "requirements.txt"))
}

$appTarget = Resolve-AppTarget
$reloadArgs = @()
if (-not $NoReload) {
    $reloadArgs = @("--reload")
}

$uvicornArgs = @(
    "-m", "uvicorn",
    $appTarget,
    "--host", $BindHost,
    "--port", $Port
) + $reloadArgs

Write-Host "Starting $appTarget on http://$BindHost`:$Port ..."
& $python.Command @($python.PrefixArgs + $uvicornArgs)
