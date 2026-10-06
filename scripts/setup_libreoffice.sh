#!/usr/bin/env bash
#
# setup_libreoffice.sh - Install LibreOffice into the active Conda environment.
#
# Purpose:
#   Provides the `soffice` binary that doc2md needs for document conversion, without
#   root access or a system package manager: the official .deb release is unpacked
#   directly into the env, so it lives and dies with that env.
#
# Processing logic:
#   1. Require an activated Conda env ($CONDA_PREFIX); exit early if LibreOffice is
#      already installed at $CONDA_PREFIX/opt/libreoffice.
#   2. Map `uname -m` to the release's file naming (x86_64 -> x86-64, aarch64 as-is).
#   3. Download the deb tarball, trying each source in order until one succeeds:
#        a. $LIBREOFFICE_MIRROR (if set)
#        b. download.documentfoundation.org (TDF's redirector; picks a nearby mirror)
#        c. ftp.fau.de/tdf (direct mirror, for when the redirector is down)
#        d. downloadarchive.documentfoundation.org (releases no longer in stable/)
#      curl --fail rejects HTTP error pages, and a stall timeout skips hung hosts.
#   4. Check that the download is a valid gzip archive, then extract it.
#   5. Unpack every .deb with `dpkg-deb -x` (no install, no root) and move the
#      resulting /opt/libreoffice* tree into $CONDA_PREFIX/opt/libreoffice.
#   6. Symlink program/soffice into $CONDA_PREFIX/bin so it is on PATH, then warn if
#      another soffice (e.g. a system /usr/bin/soffice) would be found first.
#
# Environment variables:
#   LIBREOFFICE_VER     Release to install (default below; see
#                       https://download.documentfoundation.org/libreoffice/stable/).
#   LIBREOFFICE_MIRROR  Base URL of a preferred TDF mirror, e.g. https://ftp.fau.de/tdf.
#
# Requirements: curl, tar, gzip, file, dpkg-deb; Linux on x86_64 or aarch64.
#
# Usage:
#   conda activate <env>
#   bash scripts/setup_libreoffice.sh
#   LIBREOFFICE_MIRROR=https://ftp.fau.de/tdf bash scripts/setup_libreoffice.sh
#   soffice --version    # older version? run `rehash` (zsh) / `hash -r` (bash) first
#
set -euo pipefail

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "Error: Conda environment is not activated."
  exit 1
fi

# Override with e.g. LIBREOFFICE_VER=25.8.5 when the pinned release leaves the mirror.
LIBREOFFICE_VER="${LIBREOFFICE_VER:-26.8.1}"
ARCH=$(uname -m)
# The mirror directory uses uname-style names, but x86_64 tarballs are named "x86-64".
case "${ARCH}" in
  x86_64) FILE_ARCH="x86-64" ;;
  aarch64) FILE_ARCH="aarch64" ;;
  *)
    echo "Error: unsupported architecture '${ARCH}' (LibreOffice ships x86_64 and aarch64 debs)."
    exit 1
    ;;
esac

TARBALL="LibreOffice_${LIBREOFFICE_VER}_Linux_${FILE_ARCH}_deb.tar.gz"
STABLE_PATH="libreoffice/stable/${LIBREOFFICE_VER}/deb/${ARCH}/${TARBALL}"
# download.documentfoundation.org is a redirector that hosts no files, so a direct
# mirror is listed as a fallback for when it is down (it returns 504s during outages).
# Set LIBREOFFICE_MIRROR (e.g. https://ftp.fau.de/tdf) to try a preferred mirror first.
# stable/ only keeps the newest releases; older ones move to the archive host.
CANDIDATE_URLS=()
if [ -n "${LIBREOFFICE_MIRROR:-}" ]; then
  CANDIDATE_URLS+=("${LIBREOFFICE_MIRROR%/}/${STABLE_PATH}")
fi
CANDIDATE_URLS+=(
  "https://download.documentfoundation.org/${STABLE_PATH}"
  "https://ftp.fau.de/tdf/${STABLE_PATH}"
  "https://downloadarchive.documentfoundation.org/libreoffice/old/${LIBREOFFICE_VER}/deb/${ARCH}/${TARBALL}"
)

DEST_DIR="${CONDA_PREFIX}/opt/libreoffice"
ENV_SOFFICE="${CONDA_PREFIX}/bin/soffice"

# Warn when another soffice (e.g. Ubuntu's apt package in /usr/bin) wins the PATH lookup.
# This script runs in a fresh bash, so the result reflects PATH order only; the calling
# shell may still have an older location cached in its command hash table.
check_soffice_resolution() {
  local resolved
  resolved="$(command -v soffice || true)"
  if [ "${resolved}" != "${ENV_SOFFICE}" ]; then
    echo "Warning: 'soffice' resolves to '${resolved:-<not found>}', not '${ENV_SOFFICE}'."
    if [ -n "${resolved}" ]; then
      echo "  That copy reports: $("${resolved}" --version 2>/dev/null || echo unknown)"
    fi
    echo "  Make sure ${CONDA_PREFIX}/bin comes before /usr/bin in \$PATH."
  fi
  echo "If 'soffice --version' still shows an older version in your current shell,"
  echo "run 'hash -r' (bash) or 'rehash' (zsh), or open a new shell."
}

if [ -f "${DEST_DIR}/program/soffice" ]; then
  echo "LibreOffice is already installed in ${DEST_DIR}"
  check_soffice_resolution
  exit 0
fi

echo "Installing LibreOffice ${LIBREOFFICE_VER} into active Conda environment..."
TMP_DIR=$(mktemp -d)
# Cleanup temporary directory on exit
trap 'rm -rf "${TMP_DIR}"' EXIT

ARCHIVE_PATH="${TMP_DIR}/libreoffice.tar.gz"
downloaded=false
for url in "${CANDIDATE_URLS[@]}"; do
  echo "Downloading LibreOffice from ${url}..."
  # --fail turns HTTP errors into a non-zero exit instead of saving the error page as the tarball.
  # The speed limit aborts hosts that accept the connection but then stall, so fallback starts.
  if curl -fsSL --connect-timeout 15 --speed-limit 1024 --speed-time 30 \
    --retry 3 --retry-delay 2 "${url}" -o "${ARCHIVE_PATH}"; then
    downloaded=true
    break
  fi
  echo "Download failed from ${url}"
done

if [ "${downloaded}" != true ]; then
  echo "Error: could not download LibreOffice ${LIBREOFFICE_VER} for ${ARCH}. Tried:"
  printf '  %s\n' "${CANDIDATE_URLS[@]}"
  echo "Check https://download.documentfoundation.org/libreoffice/stable/ for current releases"
  echo "and rerun with LIBREOFFICE_VER=<version>."
  exit 1
fi

if ! gzip -t "${ARCHIVE_PATH}" 2>/dev/null; then
  echo "Error: downloaded file is not a valid gzip archive ($(file -b "${ARCHIVE_PATH}"))."
  exit 1
fi

echo "Download successful. Extracting LibreOffice..."
tar -xzf "${ARCHIVE_PATH}" -C "${TMP_DIR}"

shopt -s nullglob
deb_files=("${TMP_DIR}"/LibreOffice_*_deb/DEBS/*.deb)
shopt -u nullglob
if [ "${#deb_files[@]}" -eq 0 ]; then
  echo "Error: no .deb packages found in the extracted archive."
  exit 1
fi

mkdir -p "${DEST_DIR}"
for deb in "${deb_files[@]}"; do
  dpkg-deb -x "${deb}" "${TMP_DIR}/extracted"
done

# Move internal contents directly to DEST_DIR
mv "${TMP_DIR}/extracted/opt/libreoffice"*/* "${DEST_DIR}/"

# Expose 'soffice' via a symlink in the env's bin/, like any other conda
# package would. Putting the whole program/ dir on PATH instead would shadow
# the env's python with LibreOffice's own bundled python/python.bin.
ln -sf "${DEST_DIR}/program/soffice" "${ENV_SOFFICE}"

echo "LibreOffice installed successfully in ${DEST_DIR}"
check_soffice_resolution
