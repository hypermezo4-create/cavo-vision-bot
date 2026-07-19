# CAVO Vision V3

Production-oriented Telegram product recognition for the private Cavo & Diva catalog.
The bot identifies one exact product/color, rejects uncertain matches, and reads live sizes
from the existing Google Sheet.

## What V3 changes

- Loads the prebuilt ResNet18 shape-plus-color index instead of rebuilding a different
  embedding index on every restart.
- Requires both an absolute similarity score and a top-two margin.
- Filters disabled inventory rows before returning a result.
- Shows the closest references when a photo is uncertain; only an admin confirmation can
  add a new reference to the production index.
- Rejects invalid, tiny, dark, overexposed, or heavily blurred photos.
- Processes Telegram updates concurrently while limiting expensive model inference.
- Defaults to an explicit user allowlist and drops stale Telegram updates after a restart.
- Uses deterministic stock answers. Gemini is optional and never decides the product or
  receives the full inventory snapshot.
- Seeds and validates an empty Fly volume before the bot starts.

## Recognition pipeline

1. Validate the Telegram image and measure basic quality.
2. Normalize EXIF orientation and RGB input.
3. Extract ResNet18 shape features plus an explicit color histogram.
4. Search every reference and keep the strongest reference per product ID.
5. Accept only when `MIN_MATCH_SCORE` and `MIN_MATCH_MARGIN` both pass.
6. Otherwise show `TOP_K` reference images for human selection.
7. Persist an admin-confirmed photo with an atomic index replacement.

The included model is a strong safe baseline for the existing 463-reference deployment.
Reaching measured 99% accepted precision requires a separately captured, held-out phone
photo dataset. Do not lower thresholds to make every image return a forced result.

## Sheet contract

The configured tab must contain these exact headers:

| Column | Header | Meaning |
| --- | --- | --- |
| A | Product image | Existing over-grid reference |
| B | `المقاسات` | Repeated available sizes |
| C | `الكمية` | Displayed total quantity |
| D | `BOT_PRODUCT_ID` | Stable `CAVO-0001` identifier |
| E | `BOT_ENABLED` | Whether the row can be returned |
| F | `BOT_VALIDATION` | Validation state |

The bot calculates the quantity from the repeated values in column B. A disagreement with
column C produces a warning instead of changing the calculated result.

## Access control

V3 is private by default:

- `ADMIN_USER_IDS`: users allowed to confirm and teach uncertain matches.
- `ALLOWED_USER_IDS`: additional users allowed to search; defaults to the admin list.
- `ALLOW_PUBLIC=false`: required production default.

The process refuses to start with an empty allowlist unless `ALLOW_PUBLIC=true` is set
explicitly.

## Local setup

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
set -a
. ./.env
set +a
cavo-validate-deployment \
  --catalog deployment/catalog \
  --index deployment/catalog-index.npz \
  --build-info deployment/BUILD_INFO.json \
  --strict-hashes
cavo-bot
```

Required secrets:

- `TELEGRAM_BOT_TOKEN`
- `ADMIN_USER_IDS` or `ALLOWED_USER_IDS`
- `SHEET_ID`

`GEMINI_API_KEY` is optional. When absent, deterministic inventory mode remains fully
functional.

## Build a catalog from XLSX

```bash
cavo-extract-xlsx Cavo-Store.xlsx \
  --output /data/catalog \
  --sheet "الورقة1" \
  --strict

cavo-build-index \
  --catalog /data/catalog \
  --output /data/catalog-index.npz
```

Product IDs are restricted to `CAVO-XXXX`. Extraction and truncated-workbook recovery
reject unsafe archive, product, and drawing-relationship paths.

## Calibrate recognition

Use phone photos that were never used to build the reference index. Measure:

- accepted top-1 precision;
- automatic coverage;
- top-3 recall;
- false acceptance for out-of-catalog photos;
- per-product confusion pairs;
- p50 and p95 latency.

The production target is at least 99% precision among automatically accepted images. Low
confidence images should be rejected or confirmed by a human; coverage and precision must
be reported separately.

For the next model generation, collect 10-15 real phone photos per product and fine-tune a
metric-learning model with hard negatives. The runtime API and index validation are kept
separate so a DINOv2/SigLIP2 retrieval service can replace the baseline without rewriting
the Telegram and inventory layers.

## Fly deployment

The deployment bundle must contain:

```text
deployment/catalog/catalog-manifest.json
deployment/catalog/CAVO-XXXX/*.jpg
deployment/catalog-index.npz
deployment/BUILD_INFO.json
```

Create the app and a persistent volume once:

```bash
fly apps create cavo-vision-bot
fly volumes create cavo_data --region fra --size 2 --app cavo-vision-bot
fly secrets set \
  TELEGRAM_BOT_TOKEN=... \
  ADMIN_USER_IDS=... \
  ALLOWED_USER_IDS=... \
  SHEET_ID=... \
  GEMINI_API_KEY=... \
  --app cavo-vision-bot
fly deploy --depot=false --ha=false --app cavo-vision-bot
```

The Docker image contains the model weights and private seed bundle. On the first boot the
entrypoint copies the catalog and index to the empty volume, validates every image and the
index checksum, then drops root privileges. Later boots perform a fast structural check.

Seed data is initialization-only. Updating an existing production volume is an explicit
catalog migration so confirmed references are not overwritten silently.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m compileall -q src tests
```

GitHub Actions runs lint, tests, and a coverage threshold on Python 3.11 and 3.12.

## Private data

The following paths are intentionally excluded from Git:

- `deployment/catalog/`
- `deployment/catalog-index.npz`
- `deployment/BUILD_INFO.json`
- `.env`
- customer or admin-confirmed photos

Never move the private deployment bundle into the public repository.
