# JSBSim Gym Environment

![image](sample_animation.gif)

## Installation

Required libraries are provided in the `requirements.txt` and are tested for Python 3.9.7.

## Usage

You can train a Soft Actor-Critic (SAC) agent on the environment by simply running:
```
python train.py
```
Once the agent is trained, you can watch it fly to a random goal using:
```
python test.py
```
A pretrained agent can also be downloaded [here](https://drive.google.com/file/d/1IujYzcj4hXwO4n2XLX7D5nnBemUFieRX/view?usp=share_link) to skip the training step.

## Variable-Width Canyon Demo

The project includes a canyon-focused environment with a narrow canyon whose
width changes as a function of downrange position.

- Env id: `JSBSimCanyon-v0`
- Environment file: `jsbsim_gym/canyon_env.py`
- Unified runner: `run_scenario.py`

Run the rendering demo:

```
uv run python run_scenario.py --controller altitude_hold --render
```

For RL training later, use the same environment id via Gymnasium:

```python
import gymnasium as gym
import jsbsim_gym.canyon_env  # registers JSBSimCanyon-v0

env = gym.make("JSBSimCanyon-v0")
```

### MPPI and Gatekeeper Setup

Before using `run_scenario.py` with `--controller mppi`, `--controller smooth_mppi`,
or `--gatekeeper`, generate the offline dynamics artifacts first.

1. Generate the F-16 dataset used for nominal-model fitting and uncertainty calibration:

```
uv run python -m jsbsim_gym.dataset
```

This writes:

```
f16_dataset.parquet
output/f16_dataset_coverage.json
```

2. Fit the nominal dynamics weights and uncertainty artifact from that dataset:

```
uv run python -m jsbsim_gym.calibration
```

This writes:

```
jsbsim_gym/nominal_coeff_weights.npz
f16_uncertainty_model.pkl
```

After those two steps:

```
uv run python run_scenario.py --controller mppi
uv run python run_scenario.py --controller mppi --gatekeeper
```

To tune the new offline-reference MPPI controller with Optuna:

```
uv run python optuna/tune_mppi.py --nominal-dyn-path air-racing-optimization/final_results/dyn.asb
```

This writes the best-parameter summary to:

```
output/canyon_mppi/mppi_optuna_best.json
```

`run_scenario.py --controller mppi` will auto-load that summary when it exists.

If you regenerate `f16_dataset.parquet`, rerun `jsbsim_gym.calibration` so MPPI
and the gatekeeper use artifacts fit from the current dataset.

Test the simple controller by running:

```
uv run python run_scenario.py --controller simple
```

### Real Canyon DEM Workflow (Black Canyon of the Gunnison)

The canyon environment also supports DEM-backed geometry extracted from real
elevation rasters.

1. Download DEM clips from OpenTopography (set your API key first):

```
export OT_API_KEY="<your_api_key>"
uv run python download_canyon_dem.py --preset black-canyon-gunnison
```

2. Plot DEM previews for verification:

```
uv run python plot_downloaded_dem.py
```

3. Run a canyon scenario (currently configured for the Black Canyon DEM):

```
uv run python run_scenario.py --controller altitude_hold --render
```

### Map Download and Setup Details

This is the exact workflow used to download and configure the DEM maps in this
repo.

1. Get an OpenTopography API key:
	- Create an account at https://opentopography.org
	- Generate an API key in your profile

2. Export the API key for the downloader script:

```
export OT_API_KEY="<your_api_key>"
```

3. Download a preset DEM clip (USGS 3DEP, GeoTIFF):

```
uv run python download_canyon_dem.py --preset black-canyon-gunnison
```

This writes the map to:

```
data/dem/black-canyon-gunnison_USGS10m.tif
```

You can also download the Grand Canyon preset:

```
uv run python download_canyon_dem.py --preset grand-canyon
```

4. Optional: inspect request metadata before downloading:

```
uv run python download_canyon_dem.py --preset black-canyon-gunnison --dry-run --print-url
```

5. Generate map previews to verify the raster before running simulation:

```
uv run python plot_downloaded_dem.py
```

Preview images are written under:

```
data/dem/plots/
```

6. Run the DEM canyon scenario (uses the downloaded map):

```
uv run python run_scenario.py --controller altitude_hold --render
```

The scenario runner is configured to use:
- DEM path: `data/dem/black-canyon-gunnison_USGS10m.tif`
- BBox: `(38.52, 38.62, -107.78, -107.65)`
- Start pixel: `(1400, 950)`

These are defined in `run_scenario.py` as `DEM_PATH`, `DEM_BBOX`, and
`DEM_START_PIXEL`.

7. Permanent trajectory overlay output:
	- Each run writes persistent artifacts under a controller-specific directory,
	  e.g. for `--controller altitude_hold`:

```
output/canyon_altitude_hold/
```

## Important Files

The main files defining the environment and feature transformation are `jsbsim_gym/jsbsim_gym.py` and `jsbsim_gym/features.py`. The files under `jsbsim_gym/visualization` are auxiliary files for rendering the environment. 

- `jsbsim_gym.py`: This file defines the environment which wraps a JSBSim simulation which runs an F-16 aerodynamics model. The environment class defines a goal and reward function for the agent. Additional shaping rewards are also defined in a Gym wrapper in this file. 
- `features.py`: This file defines a feature extractor for the JSBSim environment. This is the feature vector I found to be most beneficial for this task. Further details can be found in the comments in this file.
- `train.py`: This is a short script for training a SAC agent on the JSBSim environment. The hardcoded parameters should be sufficient to get decent results. The script takes about 12 hours to run on my desktop though time my vary depending on hardware.
- `test.py`: This script will run the trained agent for one episode while visualizing the environment. The visualization will automatically be saved to an MP4 video and GIF animation.
