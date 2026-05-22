# MPStab Benchmark Example

This directory contains example assets for running simulations with `mssim`.

The provided files are intended as templates only:
- adapt the settings file to your simulation parameters,
- adapt the launcher script to your local environment or workflow.

## Files

- `settings.json`  
  Example benchmark configuration.

- `run.sh`  
  Example launcher script.

## Usage

Place the files inside your local `mssim` repository:

```text
mssim/
├── settings.json
└── script/
    └── run.sh
````

Run from the repository root:

```bash
./script/run.sh settings.json
```

## Configuration

You should edit:

* simulation parameters
* your path to python virtual enviroment at the beginning of the bash file, if not in mssim/.venv/
* other system specific options in bash file


## Output

Results are written to:

```text
results/result.jsonl
```

## Additional notes

* For SLURM execution, adapt `script/run_parallel.sh` to your cluster configuration.
* General installation, configuration format, and architecture documentation are available in the main project README.