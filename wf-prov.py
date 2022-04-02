#!/usr/bin/env python3

import argparse
from collections import defaultdict
from fnmatch import fnmatch
import glob
import json
import os
import re
from tempfile import NamedTemporaryFile
from typing import Tuple, Sequence, Mapping, Dict

import numpy as np
import pandas as pd
from schematic import CONFIG
from schematic.manifest.generator import ManifestGenerator
from schematic.models.metadata import MetadataModel
import synapseclient
from synapseclient.entity import File
import synapseutils
from synapseutils.copy_functions import copy, changeFileMetaData
import yaml


MANIFEST_NAME = "manifest.csv"


def main():
    """Run the main program"""
    args = parse_args()
    syn = synapseclient.login(silent=True)
    metrics = read_table(args.metrics)
    constants = read_table(args.constants)
    template, generator = generate_template(args.config_path, args.data_type)
    qc_data = extract_qc_data(metrics, template, args.results_dir)
    filters = read_dict(args.filters)
    if args.subcommand == "upload":
        parents = read_table(args.parents)
        outputs = read_table(args.outputs)
        provenance = read_table(args.provenance)
        wildcards = read_table(args.wildcards)
        outputs = filter_outputs(outputs, args.data_type)
        outputs = expand_outputs(outputs, parents, args.results_dir, wildcards)
        outputs.to_csv(args.output, index=False)
        manifest = generate_sync_manifest(
            syn, args.results_dir, args.parent_id, outputs.path
        )
        metadata = fill_in_template(
            template, outputs, manifest, constants, qc_data, filters
        )
        metadata = add_provenance(metadata, provenance)
        validate_template(args.config_path, args.data_type, metadata)
        synapse_sync(syn, args.parent_id, metadata, template, generator, args.output)
    elif args.subcommand == "copy":
        manifest = load_manifest(syn, args.manifest_id)
        manifest = update_manifest(manifest, constants, qc_data, template, filters)
        copy_files(syn, manifest, template, generator, args.parent_id, args.output)


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments

    Args:
        args (argparse.Namespace): Parsed command-line arguments
    """
    assert args.parent_id.startswith("syn")
    assert os.path.isfile(args.config_path)
    assert os.path.isfile(args.constants)
    assert os.path.isfile(args.metrics)
    assert os.path.isfile(args.filters)
    assert normalize_path(args.output).endswith(".csv")
    if args.subcommand == "upload":
        assert os.path.isdir(args.results_dir)
        assert os.path.isfile(args.parents)
        assert os.path.isfile(args.outputs)
        assert os.path.isfile(args.provenance)
        assert os.path.isfile(args.wildcards)
    elif args.subcommand == "copy":
        assert args.manifest_id.startswith("syn")


def parse_args(args=None) -> argparse.Namespace:
    """Parse and validate command-line arguments

    Returns:
        argparse.Namespace: Validated command-line arguments
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")
    # Subcommand for new datasets (like Level-2 and Level-3)
    parser_upload = subparsers.add_parser("upload")
    parser_upload.add_argument("--results_dir", required=True)
    parser_upload.add_argument("--parents", required=True)
    parser_upload.add_argument("--outputs", required=True)
    parser_upload.add_argument("--constants", required=True)
    parser_upload.add_argument("--metrics", required=True)
    parser_upload.add_argument("--provenance", required=True)
    parser_upload.add_argument("--wildcards", required=True)
    parser_upload.add_argument("--filters", required=True)
    parser_upload.add_argument("--config_path", required=True)
    parser_upload.add_argument("--data_type", required=True)
    parser_upload.add_argument("--parent_id", required=True)
    parser_upload.add_argument("--output", required=True)
    # Subcommand for existing datasets (like Level-1)
    parser_copy = subparsers.add_parser("copy")
    parser_copy.add_argument("--parent_id", required=True)
    parser_copy.add_argument("--manifest_id", required=True)
    parser_copy.add_argument("--config_path", required=True)
    parser_copy.add_argument("--data_type", required=True)
    parser_copy.add_argument("--constants", required=True)
    parser_copy.add_argument("--metrics", required=True)
    parser_copy.add_argument("--filters", required=True)
    parser_copy.add_argument("--results_dir", required=True)
    parser_copy.add_argument("--output", required=True)
    args = parser.parse_args(args)
    validate_args(args)
    return args


def normalize_path(path: str) -> str:
    """Normalize file path for consistency

    Args:
        path (str): File path

    Returns:
        str: Normalized file path
    """
    path = os.path.expanduser(path)
    path = os.path.normpath(path)
    if "*" in path:
        matches = glob.glob(path)
        assert len(matches) == 1
        path = matches[0]
    path = os.path.realpath(path)
    return path


def read_table(path: str) -> pd.DataFrame:
    """Load tab- or csv-delimited file as Pandas data frame

    Args:
        path (str): Input file path

    Returns:
        pd.DataFrame: File contents as data frame
    """
    if path.endswith(".tsv") or path.endswith(".txt"):
        table = pd.read_table(path)
    elif path.endswith(".csv"):
        table = pd.read_csv(path)
    else:
        raise ValueError(f"Unknown file extension: {path}")
    return table


def read_dict(path: str) -> Dict[str, dict]:
    """Load tab- or csv-delimited file as Pandas data frame

    Args:
        path (str): Input file path

    Returns:
        pd.DataFrame: File contents as data frame
    """
    if path.endswith(".json"):
        with open(path) as f:
            dictionary = json.load(f)
    elif path.endswith(".yml") or path.endswith(".yaml"):
        with open(path) as f:
            dictionary = yaml.safe_load(f)
    else:
        raise ValueError(f"Unknown file extension: {path}")
    return dictionary


def load_manifest(synapse: synapseclient.Synapse, syn_id: str) -> pd.DataFrame:
    """Load a CSV manifest file from Synapse

    Args:
        synapse (synapseclient.Synapse): Synapse object
        syn_id (str): Synapse ID

    Returns:
        pd.DataFrame: Manifest table
    """
    manifest_file = synapse.get(syn_id)
    manifest = pd.read_csv(manifest_file.path)
    return manifest


def filter_outputs(outputs: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """Filter list of output paths and names for given data type

    Args:
        outputs (pd.DataFrame): Table of output paths and names
        data_type (str): Data type or component from data model

    Returns:
        pd.DataFrame: Subset of input table for given data type
    """
    outputs = outputs[outputs["component"] == data_type]
    return outputs


def list_files(
    outputs: pd.DataFrame,
    dirname: str,
    restrictions: Mapping[str, Sequence[str]] = None,
) -> pd.DataFrame:
    """Yield a tuple of existing filepaths for the given pattern.

    Wildcard values are yielded as the second tuple item.

    This function was adapted from the Snakemake repository:
    https://github.com/snakemake/snakemake/blob/cfd2f890a0af57628f7b9278d8d43f59b7006825/snakemake/utils.py

    Args:
        outputs (pd.DataFrame):
            Table of output paths and their names, where
            the paths contain placeholders for sample IDs
        dirname (str): Results directory path
        restrictions (Mapping[str : Sequence[str]]):
            Mapping between wildcards and possible values

    Yields:
        pd.DataFrame: Data frame of matching files and the placeholder values
    """

    def regex(filepattern):
        wildcard_regex = re.compile(
            r"""
            \{
                (?=(    # This lookahead assertion emulates an 'atomic group'
                        # which is required for performance
                    \s*(?P<name>\w+)                    # wildcard name
                    (\s*,\s*
                        (?P<constraint>                 # an optional constraint
                            ([^{}]+ | \{\d+(,\d+)?\})*  # allow curly braces to nest one level
                        )                               # ...  as in '{w,a{3,5}}'
                    )?\s*
                ))\1
            \}
            """,
            re.VERBOSE,
        )
        f = []
        last = 0
        wildcards = set()
        for match in wildcard_regex.finditer(filepattern):
            f.append(re.escape(filepattern[last : match.start()]))
            wildcard = match.group("name")
            if wildcard in wildcards:
                if match.group("constraint"):
                    raise ValueError(
                        "Constraint regex must be defined only in the first "
                        "occurence of the wildcard in a string."
                    )
                f.append("(?P={})".format(wildcard))
            else:
                wildcards.add(wildcard)
                f.append(
                    "(?P<{}>{})".format(
                        wildcard,
                        match.group("constraint")
                        if match.group("constraint")
                        else "[^/]+?",
                    )
                )
            last = match.end()
        f.append(re.escape(filepattern[last:]))
        f.append("$")  # ensure that the match spans the whole file
        return "".join(f)

    dirname = normalize_path(dirname)
    get_relpath = lambda x: os.path.relpath(x, dirname)

    regexes = list()
    for _, row in outputs.iterrows():
        row_dict = row.to_dict()
        pattern = row["path"]
        pattern = os.path.join(dirname, pattern)
        pattern_as_regex = re.compile(regex(pattern))
        row_dict["regex"] = pattern_as_regex
        regexes.append(row_dict)

    files = list()
    for dirpath, _, filenames in os.walk(dirname):
        for f in filenames:
            f = normalize_path(os.path.join(dirpath, f))
            files.append(f)

    all_matches = list()
    for row_dict in regexes:
        pattern = row_dict.pop("regex")
        for f in files:
            matches = pattern.match(f)
            if matches:
                wildcards = matches.groupdict()
                invalid = False
                if restrictions is not None:
                    for wildcard, value in wildcards.items():
                        if (
                            wildcard in restrictions
                            and value not in restrictions[wildcard]
                        ):
                            invalid = True
                if not invalid:
                    row_dict["path"] = f
                    row_dict["relpath"] = get_relpath(f)
                    row_dict.update(wildcards)
                    series = pd.Series(row_dict)
                    all_matches.append(series)

    return pd.DataFrame(all_matches)


def expand_outputs_old(
    outputs: pd.DataFrame, parents: Sequence[str], results_dir: str
) -> pd.DataFrame:
    """Expand

    Args:
        outputs (pd.DataFrame):
            Table of output paths and their names, where
            the paths contain placeholders for sample IDs
        parents (Sequence[str]): Table of sample IDs and parent IDs
        results_dir (str): Results directory prefix

    Returns:
        pd.DataFrame: Table of actual output paths and their names
    """
    all_outputs = list()
    samples = parents["HTAN Biospecimen ID"].unique()
    parents_grp = (
        parents.groupby("HTAN Biospecimen ID", as_index=False)
        .agg({"entityId": lambda x: ";".join(x)})
        .rename(
            columns={"HTAN Biospecimen ID": "sample_id", "entityId": "parent_ids"}
        )
    )
    for sample_id in samples:
        sample_outputs = outputs.copy()
        format_output = lambda x: x.format(sample_id=sample_id)
        prefix_output = lambda x: os.path.join(results_dir, x)
        check_output = lambda x: os.path.isfile(x)
        get_relpath = lambda x: os.path.relpath(x, normalize_path(results_dir))
        sample_outputs["path"] = sample_outputs["path"].map(format_output)
        sample_outputs["path"] = sample_outputs["path"].map(prefix_output)
        sample_outputs["path"] = sample_outputs["path"].map(normalize_path)
        sample_outputs["path"].map(check_output)
        sample_outputs["sample_id"] = sample_id
        sample_outputs["relpath"] = sample_outputs["path"].map(get_relpath)
        sample_outputs = sample_outputs.merge(parents_grp, on="sample_id")
        all_outputs.append(sample_outputs)
    all_outputs_df = pd.concat(all_outputs)
    return all_outputs_df


def expand_outputs(
    outputs: pd.DataFrame,
    parents: Sequence[str],
    results_dir: str,
    wildcards: pd.DataFrame,
) -> pd.DataFrame:
    """Expand

    Args:
        outputs (pd.DataFrame):
            Table of output paths and their names, where
            the paths contain placeholders for sample IDs
        parents (Sequence[str]): Table of sample IDs and parent IDs
        results_dir (str): Results directory prefix
        wildcards (pd.DataFrame): Mapping between wildcards and attributes

    Returns:
        pd.DataFrame: Table of actual output paths and their names
    """
    results_dir = normalize_path(results_dir)
    restrictions = dict()
    for _, row in wildcards.iterrows():
        wildcard = row["wildcard"]
        attribute = row["attribute"]
        restrictions[wildcard] = set(parents[attribute].unique())
    matches = list_files(outputs, results_dir, restrictions)
    mapping = dict(zip(wildcards.wildcard, wildcards.attribute))
    keys = wildcards["wildcard"].to_list()

    parents_grp = parents.groupby("HTAN Biospecimen ID", as_index=False).agg(
        {"entityId": lambda x: ",".join(x)}
    )
    parents_dict = dict(zip(parents_grp.iloc[:, 0], parents_grp.iloc[:, 1]))

    all_outputs = list()
    for _, match in matches.iterrows():
        output = defaultdict(set)
        match = match.dropna()
        for key, value in match.iteritems():
            if key in keys:
                attribute = mapping[key]
                output[attribute].add(value)
                if value in parents_dict:
                    output["parent_ids"].add(parents_dict[value])
            elif key in outputs.columns or key in {"relpath"}:
                output[key].add(match[key])
        output = {k: ",".join(v) for k, v in output.items()}
        all_outputs.append(output)
    all_outputs_df = pd.DataFrame(all_outputs)
    return all_outputs_df


def generate_template(
    config_path: str, data_type: str
) -> Tuple[pd.DataFrame, ManifestGenerator]:
    """Generate empty manifest template for given data model and component

    Args:
        config_path (str): Schematic configuration
        data_type (str): Data model component

    Returns:
        Tuple[pd.DataFrame, ManifestGenerator]:
            The empty manifest and the manifest generator used to create it
    """
    CONFIG.load_config(config_path)
    jsonld_model = CONFIG.normalize_path(CONFIG["model"]["input"]["location"])
    manifest_generator = ManifestGenerator(jsonld_model, root=data_type)
    manifest_url = manifest_generator.get_manifest(sheet_url=False)
    manifest_df = manifest_generator.get_dataframe_by_url(manifest_url)
    return manifest_df, manifest_generator


def validate_template(config_path: str, data_type: str, metadata: pd.DataFrame) -> None:
    """Validate metadata manifest against data model

    Args:
        config_path (str): Schematic configuration
        data_type (str): Data model component
        metadata (pd.DataFrame): Metadata manifest
    """
    CONFIG.load_config(config_path)
    jsonld_model = CONFIG.normalize_path(CONFIG["model"]["input"]["location"])
    metadata_model = MetadataModel(jsonld_model, "local")
    with NamedTemporaryFile() as fp:
        metadata.to_csv(fp, index=False)
        errors = metadata_model.validateModelManifest(fp.name, data_type)
    assert len(errors) == 0


def format_annotation_names(
    metadata: pd.DataFrame, template: pd.DataFrame, generator: ManifestGenerator
) -> pd.DataFrame:
    """Format annotation names for Synapse

    Args:
        metadata (pd.DataFrame): Filled-in metadata manifest (with new columns)
        template (pd.DataFrame): Empty metadata manifest
        generator (ManifestGenerator): Manifest generator from Schematic

    Returns:
        pd.DataFrame: Manifest with Synapse-compatible column names
    """
    label_map = dict()
    for display_name in template.columns:
        label_map[display_name] = generator.sg.get_node_label(display_name)
    metadata = metadata.rename(columns=label_map)
    return metadata


def extract_qc_data(
    metrics: pd.DataFrame,
    template: pd.DataFrame,
    results_dir: str,
) -> pd.DataFrame:
    """Extract relevant metrics from workflow outputs

    Args:
        metrics (pd.DataFrame): Table of QC metrics and where to find the values
        template (pd.DataFrame): Metadata manifest template based on data model
        results_dir (str): Results directory prefix

    Returns:
        pd.DataFrame: Table of QC metric values for all samples
    """
    qc_data_dict = dict()
    data_type = template.loc[0, "Component"]
    selected = metrics[metrics["attribute"].isin(template.columns)]
    selected = selected[selected["component"] == data_type]
    for filename, group in selected.groupby("filename"):
        full_filename = os.path.join(results_dir, filename)
        full_filename = normalize_path(full_filename)
        qc_data = read_table(full_filename)
        metric_names = group.colname.to_list()
        colnames = ["Sample"]
        colnames.extend(metric_names)
        subset = qc_data[colnames]
        rename_map = dict(zip(group.colname, group.attribute))
        subset = subset.rename(columns=rename_map)
        qc_data_dict[filename] = subset
    return qc_data_dict


def fill_in_template(
    template: pd.DataFrame,
    outputs: pd.DataFrame,
    manifest: pd.DataFrame,
    constants: pd.DataFrame,
    qc_data: pd.DataFrame,
    filters: Mapping[str, dict],
) -> pd.DataFrame:
    """Fill in manifest template with output files and their metadata

    Args:
        template (pd.DataFrame): Manifest template from data model
        outputs (pd.DataFrame): Table of output paths and names
        manifest (pd.DataFrame): Synapse sync manifest (path and parent)
        constants (pd.DataFrame): Table of constant values to fill in
        qc_data (pd.DataFrame): Table of QC metrics for all samples
        filters (Mapping[str, dict]): Filters to apply on matches when
            pairing QC metrics with output files

    Returns:
        pd.DataFrame: Filled-in manifest
    """
    data_type = template.loc[0, "Component"]
    options = {"", data_type}
    template = template[0:0]
    outputs = outputs.rename(
        columns={
            "component": "Component",
            "workflow": "WorkflowName",
            "name": "File Contents",
            "sample_id": "HTAN Biospecimen ID",
            "relpath": "Filename",
            "format": "File Format",
            "compression": "File Compression",
            "parent_ids": "Parent File IDs",
        }
    )
    for filename, data in qc_data.items():
        qc_data_list = list()
        outputs_filt = outputs.copy()
        matching_filters = [v for k, v in filters.items() if fnmatch(filename, k)]
        assert len(matching_filters) <= 1
        if len(matching_filters) == 1:
            queries = matching_filters[0]
            for colname, value in queries.items():
                outputs_filt = outputs_filt[outputs_filt[colname] == value]
        for _, row in data.iterrows():
            sample = row.pop("Sample")
            matches = outputs_filt["Filename"].str.contains(
                rf"(?:\b|_){sample}(?:\b|_)"
            )
            assert sum(matches) <= 1
            if sum(matches) == 1:
                row["Filename"] = outputs_filt.loc[matches, "Filename"].values[0]
                qc_data_list.append(row)
        qc_data_df = pd.DataFrame(qc_data_list)
        outputs = outputs.merge(qc_data_df, on="Filename", how="left")
    metadata = pd.concat([template, outputs])
    metadata = metadata.merge(manifest, on="path", how="inner")
    for _, row in constants.iterrows():
        name, value, component = row.fillna("").to_list()
        if component in options and name in metadata.columns:
            metadata[name] = value
    return metadata


def create_folder(syn, name, parent_id):
    """Create Synapse folder."""
    entity = {
        "name": name,
        "concreteType": "org.sagebionetworks.repo.model.Folder",
        "parentId": parent_id,
    }
    entity = syn.store(entity)
    return entity


def walk_directory_tree(syn, path, parent_id, subset=[]):
    """Replicate folder structure on Synapse and generate manifest
    rows for files using corresponding Synapse folders as parents.
    """
    rows = list()
    parents = {normalize_path(path): parent_id}
    subset = set(subset)
    for dirpath, dirnames, filenames in os.walk(path):
        # Replicate the folders on Synapse
        dirpath = normalize_path(dirpath)
        for dirname in dirnames:
            name = dirname
            folder_path = os.path.join(dirpath, dirname)
            folder_path = normalize_path(folder_path)
            if subset and not any(x.startswith(folder_path) for x in subset):
                continue
            parent_id = parents[dirpath]
            folder = create_folder(syn, name, parent_id)
            # Store Synapse ID for sub-folders/files
            parents[folder_path] = folder["id"]
        # Generate rows per file for the manifest
        for filename in filenames:
            # Add file to manifest if non-zero size
            folder_path = normalize_path(dirpath)
            filepath = os.path.join(dirpath, filename)
            filepath = normalize_path(filepath)
            if subset and filepath not in subset:
                continue
            manifest_row = {
                "path": filepath,
                "parent": parents[folder_path],
            }
            if os.stat(filepath).st_size > 0:
                rows.append(manifest_row)
    return rows


def generate_sync_manifest(
    synapse: synapseclient.Synapse,
    directory_path: str,
    parent_id: str,
    subset: Sequence[str],
) -> pd.DataFrame:
    """Generate manifest for syncToSynapse() from a local directory

    Args:
        synapse (synapseclient.Synapse): Synapse object
        directory_path (str): Local directory path containing results
        parent_id (str): Synapse ID of the parent folder
        subset (Sequence[str]): List of files to handle

    Returns:
        pd.DataFrame: [description]
    """
    rows = walk_directory_tree(synapse, directory_path, parent_id, subset)
    df = pd.DataFrame(rows)
    return df


def add_provenance(metadata: pd.DataFrame, provenance: pd.DataFrame) -> pd.DataFrame:
    """Add provenance metadata based on existing columns

    Args:
        metadata (pd.DataFrame): Metadata manifest
        provenance (pd.DataFrame): Mapping between manifest and provenance columns

    Returns:
        pd.DataFrame: Manifest with provenance metadata
    """
    data_type = metadata.loc[0, "Component"]
    options = {"", data_type}
    for _, row in provenance.iterrows():
        source, target, component = row.fillna("").to_list()
        if component in options and source in metadata.columns:
            metadata[target] = metadata[source].str.replace(",", ";")
    return metadata


def synapse_sync(
    synapse: synapseclient.Synapse,
    parent_id: str,
    metadata: pd.DataFrame,
    template: pd.DataFrame,
    generator: ManifestGenerator,
    output_path: str,
) -> None:
    """Upload manifest and files listed therein to Synapse

    Args:
        synapse (synapseclient.Synapse): Synapse object
        parent_id (str): Synapse ID of the parent folder
        metadata (pd.DataFrame): Metadata manifest
        template (pd.DataFrame): Empty metadata manifest
        generator (ManifestGenerator): Manifest generator from Schematic
        output_path (str): Output file path for manifest CSV file
    """
    # Upload files listed in manifest to Synapse
    with NamedTemporaryFile() as tsv:
        sync_manifest = format_annotation_names(metadata, template, generator)
        sync_manifest["forceVersion"] = "False"
        sync_manifest.to_csv(tsv, sep="\t", index=False)
        synapseutils.syncToSynapse(synapse, tsv.name, sendMessages=False)
    # Retrieve Synapse IDs for `entityId` column
    entity_ids = list()
    for root, dirs, files in synapseutils.walk(synapse, parent_id):
        rootname, rootid = root
        basename = rootname.partition("/")[2]
        # Delete empty folders
        if len(dirs) == 0 and len(files) == 0:
            synapse.delete(rootid)
        for file in files:
            filename, fileid = file
            fullname = f"{basename}/{filename}"
            entity_ids.append({"Filename": fullname, "entityId": fileid})
    entity_ids_df = pd.DataFrame.from_records(entity_ids)
    metadata = metadata.merge(entity_ids_df, on="Filename")
    # Upload manifest to Synapse
    metadata.to_csv(output_path, index=False)
    metadata_file = File(path=output_path, name=MANIFEST_NAME, parent=parent_id)
    metadata_file = synapse.store(metadata_file)
    synapseutils.changeFileMetaData(synapse, metadata_file, MANIFEST_NAME)


def update_manifest(
    manifest: pd.DataFrame,
    constants: pd.DataFrame,
    qc_data: pd.DataFrame,
    template: pd.DataFrame,
    filters: Mapping[str, dict],
) -> pd.DataFrame:
    """Update manifest with QC metrics and constants

    Args:
        manifest (pd.DataFrame): Synapse sync manifest (path and parent)
        constants (pd.DataFrame): Table of constant values to fill in
        qc_data (pd.DataFrame): Table of QC metrics for all samples
        template (pd.DataFrame): Manifest template
        filters (Mapping[str, dict]): Filters to apply on matches when
            pairing QC metrics with output files

    Returns:
        pd.DataFrame: Filled-in manifest
    """
    data_type = manifest.loc[0, "Component"]
    options = {"", data_type}

    for filename, data in qc_data.items():
        qc_data_list = list()
        manifest_filt = manifest.copy()
        matching_filters = [v for k, v in filters.items() if fnmatch(filename, k)]
        assert len(matching_filters) <= 1
        if len(matching_filters) == 1:
            queries = matching_filters[0]
            for colname, value in queries.items():
                manifest_filt = manifest_filt[manifest_filt[colname] == value]
        for _, row in data.iterrows():
            sample = row.pop("Sample")
            matches = manifest_filt["Filename"].str.contains(
                rf"(?:\b|_){sample}(?:\b|_)"
            )
            assert sum(matches) <= 1
            if sum(matches) == 1:
                row["Filename"] = manifest_filt.loc[matches, "Filename"].values[0]
                qc_data_list.append(row)
        qc_data_df = pd.DataFrame(qc_data_list)
        for name in qc_data_df.columns:
            if name != "Filename":
                manifest = manifest.rename(columns={name: f"{name} Self-Reported"})
        manifest = manifest.merge(qc_data_df, on="Filename", how="left")

    for _, row in constants.iterrows():
        name, value, component = row.fillna("").to_list()
        if component in options and name in manifest.columns:
            manifest = manifest.rename(columns={name: f"{name} Self-Reported"})
            manifest[name] = value
    # Move all self-reported columns to the end
    template_cols = template.columns.to_list() + ["entityId", "eTag"]
    for name in reversed(template_cols):
        if name not in manifest:
            continue
        popped = manifest.pop(name)
        manifest.insert(0, name, popped)
    return manifest


def copy_files(
    synapse: synapseclient.Synapse,
    manifest: pd.DataFrame,
    template: pd.DataFrame,
    generator: ManifestGenerator,
    parent_id: str,
    output_path: str,
) -> pd.DataFrame:
    """Copy the files listed in the manifest to a new Synapse folder

    Args:
        synapse (synapseclient.Synapse): Synapse object
        manifest (pd.DataFrame): Manifest table
        template (pd.DataFrame): Manifest template
        generator (ManifestGenerator): Manifest generator from Schematic
        parent_id (str): Synapse ID of destination folder
        output_path (str): Output file path for manifest CSV file

    Returns:
        pd.DataFrame: Manifest with updated Synapse IDs
    """
    for index, row in manifest.iterrows():
        source = synapse.get(row.entityId, downloadFile=False, followLink=False)
        target = synapse.findEntityId(source.name, parent=parent_id)
        if target is not None:
            print(f"Deleting existing target ({target}) to avoid version history")
            synapse.delete(target)
        mapping = copy(
            synapse,
            row.entityId,
            parent_id,
            updateExisting=False,
            setProvenance=None,
            skipCopyAnnotations=True,
        )
        new_entity_id = mapping[row.entityId]
        new_entity = synapse.get(new_entity_id, downloadFile=False)
        manifest.loc[index, "entityId"] = new_entity.id
        manifest.loc[index, "eTag"] = new_entity.etag
        print(f"Copied {row.entityId} to {new_entity_id}")
        # Add annotations from the manifest row
        annotations = dict()
        for display_name, value in row.to_dict().items():
            if display_name in template:
                label = generator.sg.get_node_label(display_name)
            else:
                label = generator.sg.se.get_class_label_from_display_name(display_name)
            if label == "":
                continue
            if label in ["Filename", "ETag", "eTag", "entityId", ""]:
                continue
            if isinstance(value, str) and len(value) >= 500:
                value = value[0:485] + "[truncated]"
            annotations[label] = value
        new_entity.annotations = annotations
        synapse.store(new_entity, forceVersion=False)
    # Output updated manifest and upload copy to Synapse
    manifest.to_csv(output_path, index=False)
    metadata_file = File(path=output_path, name=MANIFEST_NAME, parent=parent_id)
    metadata_file = synapse.store(metadata_file)
    changeFileMetaData(synapse, metadata_file, MANIFEST_NAME)


if __name__ == "__main__":
    main()
