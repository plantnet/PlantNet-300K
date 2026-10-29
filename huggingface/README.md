# Publishing the six Pl@ntNet-300K-v2 models

The publisher validates and uploads all six checkpoints to the `PlantNet`
organization. Repositories are private by default so their files and model
cards can be reviewed before release.

## Expected local files

Place the checkpoints produced by `main.py` in one directory:

```text
checkpoints/
├── plantnet300kv2_cnxt.pth
├── plantnet300kv2_cnxs.pth
├── plantnet300kv2_cnxb.pth
├── plantnet300kv2_vits.pth
├── plantnet300kv2_vitsplus.pth
└── plantnet300kv2_vitb.pth
```

You also need `species_metadata.csv` from Pl@ntNet-300K-v2.

## Authenticate

Use a write-scoped token belonging to an account that can create model
repositories in the `PlantNet` organization:

```bash
python -m pip install -r huggingface/requirements.txt
hf auth login
hf auth whoami
```

Do not put a Hugging Face token in this repository or pass it on the command
line.

## Validate without uploading

```bash
python huggingface/publish_models.py \
  --checkpoint-dir /path/to/checkpoints \
  --species-metadata /path/to/plantnet300kv2/species_metadata.csv \
  --dry-run
```

## Upload to private staging repositories

Remove `--dry-run` after reviewing the plan:

```bash
python huggingface/publish_models.py \
  --checkpoint-dir /path/to/checkpoints \
  --species-metadata /path/to/plantnet300kv2/species_metadata.csv
```

This creates or updates:

- `PlantNet/plantnet-300k-v2-cnxt`
- `PlantNet/plantnet-300k-v2-cnxs`
- `PlantNet/plantnet-300k-v2-cnxb`
- `PlantNet/plantnet-300k-v2-vits`
- `PlantNet/plantnet-300k-v2-vitsplus`
- `PlantNet/plantnet-300k-v2-vitb`

To validate or upload only selected models, use for example:

```bash
python huggingface/publish_models.py \
  --checkpoint-dir /path/to/checkpoints \
  --species-metadata /path/to/plantnet300kv2/species_metadata.csv \
  --models cnxb vitb
```

## Public release

After reviewing the six private repositories, rerun the same command with
`--public`. This explicitly changes any private target repository to public and
uploads the audited files again:

```bash
python huggingface/publish_models.py \
  --checkpoint-dir /path/to/checkpoints \
  --species-metadata /path/to/plantnet300kv2/species_metadata.csv \
  --public
```
