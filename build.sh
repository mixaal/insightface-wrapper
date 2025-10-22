#!/bin/bash -xe
MODELS="$HOME/.insightface/models/"

ORG_DIR=$(pwd)
[ -d "$MODELS/buffalo_l" ] || {
	mkdir -p "$MODELS"
	cd $MODELS && rm -rf buffalo_l
 	curl -kv -L https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -o "$MODELS/buffalo_l.zip"
	cd $MODELS && mkdir buffalo_l && cd buffalo_l && unzip "$MODELS/buffalo_l.zip"
}

ls -l "$MODELS/buffalo_l/det_10g.onnx" && echo "Insightface models installed"

cd $ORG_DIR
cargo build --release
