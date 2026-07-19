# CAVO Vision Bot

Telegram bot that recognizes one CAVO product/color from a phone photo and returns the live sizes from the existing Google Sheet.

## Current CAVO contract

The bot reads the existing `الورقة1` tab. The visible columns remain unchanged:

| Column | Meaning |
| --- | --- |
| A | Product reference image (over-grid image) |
| B | Repeated available sizes |
| C | Displayed total quantity |

The live sheet also has three hidden bot columns:

| Column | Meaning |
| --- | --- |
| D | Stable `CAVO-0001` product/color ID |
| E | Whether the product is enabled |
| F | Data validation state |

Each row/image is an independent product color. Similar shapes are never grouped.

## Safe recognition flow

1. Normalize the phone photo and compute a shape-plus-color embedding.
2. Search the private CAVO reference index.
3. Return the live inventory only when both the score and top-two margin pass calibrated thresholds.
4. When uncertain, show the closest reference images and ask an admin to select the exact color.
5. An admin-confirmed phone photo is immediately added as another reference angle and persisted in the index.

The bot calculates total quantity from column B. It does not trust a manually typed total in column C.

## One-time image import

Google Sheets does not expose over-grid images through its values API. Download the prepared sheet once as Microsoft Excel (`.xlsx`), then run:

```bash
python -m pip install .
cavo-extract-xlsx Cavo-Store.xlsx --output /data/catalog --sheet "الورقة1"
cavo-build-index --catalog /data/catalog --output /data/catalog-index.npz
```

The extractor maps each image anchor to its sheet row and stable product ID. It produces `catalog-manifest.json` with missing images and quantity mismatches.

If an uploaded workbook is truncated, `cavo-recover-xlsx` safely salvages every complete ZIP entry, maps recovered media through the drawing anchors, and emits an exact missing-image report. A partial catalog must never be treated as production-ready.

## Run locally

```bash
cp .env.example .env
set -a
. ./.env
set +a
cavo-bot
```

Required secrets:

- `TELEGRAM_BOT_TOKEN`
- `ADMIN_USER_IDS` (comma-separated Telegram numeric user IDs)
- `SHEET_ID` (Google Sheet ID used for live inventory)

Do not commit tokens or credentials.

## Deploy to Fly

Create a separate app and persistent volume; the ML runtime is intentionally isolated from any existing DeadZone bot:

```bash
fly apps create cavo-vision-bot
fly volumes create cavo_data --region fra --size 5 --app cavo-vision-bot
fly secrets set TELEGRAM_BOT_TOKEN=... ADMIN_USER_IDS=... SHEET_ID=... --app cavo-vision-bot
fly deploy --app cavo-vision-bot
```

Upload or build `/data/catalog` and `/data/catalog-index.npz` before starting the bot. Start with the provided thresholds, then calibrate them using real phone photos rather than lowering them to force matches.

## Tests

The tests use Python's standard library runner:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
