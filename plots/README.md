# Plotting Launcher

This folder contains the plotting CLI and its settings file.

## Files

- [main_plotting.py](main_plotting.py): Python entrypoint for plotting.
- [run_plotting.sh](run_plotting.sh): Bash wrapper that activates the environment and launches the plotting CLI.
- [plot_settings.json](plot_settings.json): Plot configuration used by the CLI.

## Usage

Run the default magnetization plot pipeline with:

```bash
bash plots/run_plotting.sh
```

Or point it at another settings file:

```bash
bash plots/run_plotting.sh plots/plot_settings.json
```

The output filename is taken from `output.filename` in the settings file. Relative paths are saved inside this folder.