# Workflow Provenance

This folder contains the script that can be used to upload a processed dataset to Synapse and prepare it for release. There are a lot of moving parts, hence the accessory files in the `etc/` subfolder. The `Pipfile` specifies the software dependencies for this script, which can be installed using the following command. 

```console
pipenv install
```

The included `example-run.sh` file demonstrates how a bulk RNA-seq dataset and the associated biospecimen dataset can be uploaded by the release script.
