# XRB Lightcurve Simulation - Python Version

This is a Python implementation of the XRB (X-Ray Binary) Lightcurve simulation, migrated from the original R code. The simulation calculates column densities for eclipsing binary systems as the compact object eclipses its companion star.

## Features

- **Vectorized Operations**: Uses NumPy for efficient array operations instead of nested loops
- **Command Line Interface**: Full argparse support with all parameters configurable
- **Flexible Output**: Configurable output file with default naming
- **Modular Design**: Clean, well-documented functions for each component
- **Type Hints**: Full type annotations for better code maintainability

## Installation

1. Ensure you have Python 3.7+ installed
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage (Default Parameters)

```bash
python xrb_lightcurve.py
```

This will run the simulation with default parameters and save results to `xrb_lightcurve_output.csv`.

### Custom Parameters

```bash
python xrb_lightcurve.py --r 0.001 --R 2.0 --d1 11.0 --d2 8.0 --gma0 -90.0 --i0 26.0 --dth 1.0 --output my_results.csv
```

### Available Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--r` | 0.001 | Radius of smaller star B (compact object) in solar radii |
| `--R` | 2.0 | Radius of larger star A (companion) in solar radii |
| `--d1` | 11.0 | Distance of star B from COM in solar radii |
| `--d2` | 8.0 | Distance of star A from COM in solar radii |
| `--gma0` | -90.0 | Starting phase angle in degrees |
| `--i0` | 26.0 | Orbital inclination in degrees |
| `--dth` | 1.0 | Orbital increment in degrees |
| `--output` | xrb_lightcurve_output.csv | Output file name for results |

### Examples

**High-resolution simulation:**
```bash
python xrb_lightcurve.py --dth 0.5 --output high_res_simulation.csv
```

**Different binary configuration:**
```bash
python xrb_lightcurve.py --r 0.002 --R 3.0 --d1 15.0 --d2 10.0 --output large_binary.csv
```

**Custom phase range:**
```bash
python xrb_lightcurve.py --gma0 0.0 --dth 2.0 --output custom_phase.csv
```

## Output

The simulation generates a CSV file with the following columns:

- `deg`: Phase angle in degrees
- `ph`: Phase angle in radians
- `phase`: Normalized phase (0-1)
- `A2`: Area calculations
- `flx`: Flux calculations (accelerated wind)
- `flx2`: Flux calculations (constant velocity wind)
- `icd`: Integrated column density
- `time`: Time calculations
- `l3`, `L3`, `h3`: Geometric parameters
- `fl`, `fl2`: Scaled flux values
- `nfl_hard_av`, `nfl_hard_cv`: Hard band fluxes
- `nfl_soft_av`, `nfl_soft_cv`: Soft band fluxes
- `pho_count_hard_av`, `pho_count_soft_av`: Photon counts

## Code Structure

### Main Functions

1. **`simulate_lightcurve()`**: Main simulation function
2. **`create_grid()`**: Creates polar grid for wind integral calculations
3. **`density_function()`**: Calculates wind density along line of sight
4. **`wind_los_integral()`**: Calculates column integral along line of sight

### Key Improvements Over R Version

1. **Vectorization**: Uses NumPy arrays and vectorized operations instead of loops
2. **Modularity**: Each component is a separate, well-documented function
3. **Error Handling**: Better handling of edge cases and empty arrays
4. **Type Safety**: Full type hints for better code maintainability
5. **Command Line Interface**: Easy parameter configuration via argparse
6. **Flexible Output**: Configurable output file with default naming

## Performance

The Python version is significantly faster than the R version due to:
- Vectorized operations using NumPy
- Reduced nested loops
- More efficient array handling
- Better memory management

## Dependencies

- `numpy>=1.21.0`: For numerical computations and array operations
- `pandas>=1.3.0`: For data manipulation and CSV output

## Comparison with Original R Code

| Feature | R Version | Python Version |
|---------|-----------|----------------|
| Loops | Multiple nested for loops | Vectorized operations |
| Input | Interactive prompts | Command line arguments |
| Output | Manual CSV writing | Automatic DataFrame export |
| Modularity | Multiple source files | Single file with functions |
| Type Safety | None | Full type hints |
| Performance | Slower due to loops | Faster due to vectorization |

## Troubleshooting

**Memory Issues**: For very high-resolution simulations (small `dth` values), consider reducing the grid resolution or using a larger `dth` value.

**Convergence Issues**: If the simulation doesn't converge, try adjusting the geometric parameters (`r`, `R`, `d1`, `d2`) to more physically reasonable values.

**Output File Issues**: Ensure you have write permissions in the output directory.

## License

This code is provided as-is for educational and research purposes. 