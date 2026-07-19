#!/bin/sh
set -eu

mkdir -p /data/catalog /data/confirmed
seeded=0

if [ ! -f /data/catalog/catalog-manifest.json ]; then
    if [ ! -f /seed/catalog/catalog-manifest.json ]; then
        echo "CAVO seed catalog is missing" >&2
        exit 1
    fi
    cp -a /seed/catalog/. /data/catalog/
    seeded=1
fi

if [ ! -f /data/catalog-index.npz ]; then
    if [ ! -f /seed/catalog-index.npz ]; then
        echo "CAVO seed index is missing" >&2
        exit 1
    fi
    cp /seed/catalog-index.npz /data/catalog-index.npz
    seeded=1
fi

chown -R cavo:cavo /data

if [ "$seeded" -eq 1 ]; then
    gosu cavo cavo-validate-deployment \
        --build-info /seed/BUILD_INFO.json \
        --strict-hashes
else
    gosu cavo cavo-validate-deployment --fast
fi

exec gosu cavo "$@"
