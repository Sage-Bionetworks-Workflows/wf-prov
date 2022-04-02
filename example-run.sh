#!/bin/bash

mkdir -p "manifests"

# Copy the staged BulkRNAseqLevel1 dataset into the Release folder
./wf-prov.py \
    copy \
    --results_dir "/path/to/results" \
    --parent_id "syn12345678" \
    --config_path "/path/to/data_model/config.yml" \
    --data_type "BulkRNAseqLevel1" \
    --constants "./etc/constants.tsv" \
    --metrics "./etc/metrics.tsv" \
    --manifest_id "syn12345678" \
    --output "./manifests/manifest.l1.csv"

# Upload the BulkRNAseqLevel2 data files to the Release folder
./wf-prov.py \
    upload \
    --results_dir "/path/to/results" \
    --parent_id "syn12345678" \
    --config_path "/path/to/data_model/config.yml" \
    --data_type "BulkRNAseqLevel2" \
    --constants "./etc/constants.tsv" \
    --metrics "./etc/metrics.tsv" \
    --outputs "./etc/outputs.tsv" \
    --parents "./manifests/manifest.l1.csv" \
    --provenance "./etc/provenance.tsv" \
    --output "./manifests/manifest.l2.csv"

# Subset the BulkRNAseqLevel2 manifest for the key files
(
    head -n 1 ./manifests/manifest.l2.csv \
    && \
    grep -w BAM ./manifests/manifest.l2.csv \
) > ./manifests/manifest.l2.bams.csv

# Upload the BulkRNAseqLevel3 data files to the Release folder
./wf-prov.py \
    upload \
    --results_dir "/path/to/results" \
    --parent_id "syn12345678" \
    --config_path "/path/to/data_model/config.yml" \
    --data_type "BulkRNAseqLevel3" \
    --constants "./etc/constants.tsv" \
    --metrics "./etc/metrics.tsv" \
    --outputs "./etc/outputs.tsv" \
    --parents "./manifests/manifest.l2.bams.csv" \
    --provenance "./etc/provenance.tsv" \
    --output "./manifests/manifest.l3.csv"

# Release Biospecimen data
./wf-prov.py \
    copy \
    --results_dir "/path/to/results" \
    --parent_id "syn12345678" \
    --config_path "/path/to/data_model/config.yml" \
    --data_type "Biospecimen" \
    --constants "./etc/constants.tsv" \
    --metrics "./etc/metrics.tsv" \
    --manifest_id "syn12345678" \
    --output "./manifests/biospecimen.csv"
