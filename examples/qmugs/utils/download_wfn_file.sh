#!/bin/bash
# downloads wfn tarball at index INDEX to DATA_DIR

set -e

INDEX=$1
DATA_DIR=$2
URL="https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM/download?path=%2Fwfns&files=wfns_$INDEX.tar.gz"

wget -O "$DATA_DIR/wfns_$INDEX.tar.gz" "$URL"
