# Packaging — Windows installer for the desktop client

The Windows installer is built **in the cloud by GitHub Actions** (a Windows
runner), because PyInstaller cannot cross-compile a Windows `.exe` from Linux.
You don't need Python or any build tools on your PC — just download the result.

## Files
- `launch.py` — PyInstaller entry script (calls `stomclient.__main__.main`).
- `stom-client.spec` — PyInstaller one-folder build (bundles PySide6 + SimpleITK).
- `stom-client.iss` — Inno Setup script → `StomClientSetup.exe`.
- `../.github/workflows/build-windows.yml` — CI that runs both on a Windows runner.

## How to get the installer

### Option A — tagged release (recommended)
Push a version tag; CI builds the installer and attaches it to a GitHub Release:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Then open the repo on GitHub → **Releases** → download `StomClientSetup.exe`,
run it, and launch **Stom CBCT Viewer** from the Start menu / desktop.

### Option B — manual run (just the artifact, no release)
GitHub → **Actions** → *build-windows-installer* → **Run workflow**. When it
finishes, download the `StomClientSetup` artifact from the run page.

## Notes
- The app launches standalone. For the full DICOM→segmentation→mask cycle it
  needs a running `stomserver` backend; set its URL + token in **Settings**.
- Bump `AppVersion` in `stom-client.iss` (and the tag) for each release.
- First CI run takes a few minutes (downloads PySide6 + SimpleITK).
