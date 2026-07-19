# Private deployment data

Before a local production deploy, place the private catalog in this directory:

- `deployment/catalog/`
- `deployment/catalog-index.npz`
- optional `deployment/BUILD_INFO.json`

These files are intentionally excluded from Git. The Docker image seeds an empty Fly
volume from them, while confirmed phone-photo references persist on that volume.
