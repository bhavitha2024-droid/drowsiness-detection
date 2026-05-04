# Publish To GitHub

This project is ready to publish, but GitHub CLI authentication is not configured on this machine yet.

## One-time setup

```powershell
gh auth login
```

Follow the browser-based login flow.

## Create the repository and push

Run these commands from the project root:

```powershell
git init
git add .
git commit -m "Initial commit: personalized driver drowsiness detection system"
gh repo create driveai --public --source . --remote origin --push
```

## If you already created a repo on GitHub

```powershell
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

## Recommended repository description

```text
Real-time personalized driver drowsiness detection using MediaPipe, OpenCV, low-light enhancement, temporal LSTM modeling, and live dashboard monitoring.
```
