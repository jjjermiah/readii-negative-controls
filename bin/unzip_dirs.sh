#!/bin/bash

# Set source and destination directories
SOURCE_DIR="data/nbia/images/zipped"
DEST_DIR="data/nbia/images/unzipped"

# Ensure the destination directory exists
mkdir -p "$DEST_DIR"

# Function to unzip a file
unzip_file() {
	local zip_file="$1"
	local rel_path="${zip_file#$SOURCE_DIR/}"
	local relative_dir=$(dirname "$rel_path")
	local filename=$(basename "$zip_file" .zip)
	local target_dir="$DEST_DIR/$relative_dir/$filename"
	mkdir -p "$target_dir"
	unzip -o "$zip_file" -d "$target_dir" > /dev/null
}

export -f unzip_file
export SOURCE_DIR DEST_DIR

# Find all zip files and process them in parallel
find "$SOURCE_DIR" -type f -name "*.zip" | parallel unzip_file