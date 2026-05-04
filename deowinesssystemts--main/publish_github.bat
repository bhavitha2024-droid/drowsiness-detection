@echo off
setlocal
cd /d "%~dp0"

where gh >nul 2>&1
if errorlevel 1 (
  echo GitHub CLI is not installed. Install it first from https://cli.github.com/
  exit /b 1
)

gh auth status >nul 2>&1
if errorlevel 1 (
  echo GitHub CLI is not logged in.
  echo Starting login flow...
  gh auth login
)

if not exist ".git" (
  git init
)

git add .
git commit -m "Initial commit: personalized driver drowsiness detection system" 2>nul
if errorlevel 1 (
  echo Commit may already exist or git user identity may not be configured.
)

git branch -M main
gh repo create driveai --public --source . --remote origin --push

echo Publish flow complete.
endlocal
