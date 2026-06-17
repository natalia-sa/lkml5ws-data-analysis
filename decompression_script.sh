#!/bin/bash

TARGET_DIR="./LKML5Ws"
echo "Creating $TARGET_DIR for decompression"
mkdir -p $TARGET_DIR
mkdir -p ${TARGET_DIR}_lineage

for file in *dataset.tar.gz; do tar -zxf "$file" -C $TARGET_DIR; done

echo "Creating $TARGET_DIR_lineage for decompression"
mkdir -p ${TARGET_DIR}_lineage
for file in *data-lineage.tar.gz; do tar -zxf "$file" -C ${TARGET_DIR}_lineage; done
