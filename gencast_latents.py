# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import datetime
import math
from typing import Optional
import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray

from graphcast import rollout
from graphcast import xarray_jax
from graphcast import normalization
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import xarray_tree
from graphcast import gencast
from graphcast import denoiser
from graphcast import nan_cleaning


def select(
    data: xarray.Dataset,
    variable: str,
    level: Optional[int] = None,
    max_steps: Optional[int] = None
    ) -> xarray.Dataset:
  data = data[variable]
  if "batch" in data.dims:
    data = data.isel(batch=0)
  if max_steps is not None and "time" in data.sizes and max_steps < data.sizes["time"]:
    data = data.isel(time=range(0, max_steps))
  if level is not None and "level" in data.coords:
    data = data.sel(level=level)
  return data


def scale(
    data: xarray.Dataset,
    center: Optional[float] = None,
    robust: bool = False,
    ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
  vmin = np.nanpercentile(data, (2 if robust else 0))
  vmax = np.nanpercentile(data, (98 if robust else 100))
  if center is not None:
    diff = max(vmax - center, center - vmin)
    vmin = center - diff
    vmax = center + diff
  return (data, matplotlib.colors.Normalize(vmin, vmax),
          ("RdBu_r" if center is not None else "viridis"))


def plot_data(
    data: dict[str, xarray.Dataset],
    fig_title: str,
    plot_size: float = 5,
    robust: bool = False,
    cols: int = 4
    ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:

  first_data = next(iter(data.values()))[0]
  max_steps = first_data.sizes.get("time", 1)
  assert all(max_steps == d.sizes.get("time", 1) for d, _, _ in data.values())

  cols = min(cols, len(data))
  rows = math.ceil(len(data) / cols)
  figure = plt.figure(figsize=(plot_size * 2 * cols,
                               plot_size * rows))
  figure.suptitle(fig_title, fontsize=16)
  figure.subplots_adjust(wspace=0, hspace=0)
  figure.tight_layout()

  images = []
  for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
    ax = figure.add_subplot(rows, cols, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    im = ax.imshow(
        plot_data.isel(time=0, missing_dims="ignore"), norm=norm,
        origin="lower", cmap=cmap)
    plt.colorbar(
        mappable=im,
        ax=ax,
        orientation="vertical",
        pad=0.02,
        aspect=16,
        shrink=0.75,
        cmap=cmap,
        extend=("both" if robust else "neither"))
    images.append(im)

  def update(frame):
    if "time" in first_data.dims:
      td = datetime.timedelta(microseconds=first_data["time"][frame].item() / 1000)
      figure.suptitle(f"{fig_title}, {td}", fontsize=16)
    else:
      figure.suptitle(fig_title, fontsize=16)
    for im, (plot_data, norm, cmap) in zip(images, data.values()):
      im.set_data(plot_data.isel(time=frame, missing_dims="ignore"))

  ani = animation.FuncAnimation(
      fig=figure, func=update, frames=max_steps, interval=250)
  plt.close(figure.number)
  return ani


def main():
    # Set paths
    MODEL_PATH = "gencast-params-GenCast 0p25deg <2019.npz"
    DATA_PATH = "source-era5_date-2019-03-29_res-0.25_levels-13_steps-04.nc"
    STATS_DIR = "stats/"

    # Load the model
    with open(MODEL_PATH, "rb") as f:
        ckpt = checkpoint.load(f, gencast.CheckPoint)
        denoiser_architecture_config = ckpt.denoiser_architecture_config
        denoiser_architecture_config.sparse_transformer_config.attention_type = "triblockdiag_mha"
        denoiser_architecture_config.sparse_transformer_config.mask_type = "full"
    params = ckpt.params
    state = {}

    task_config = ckpt.task_config
    sampler_config = ckpt.sampler_config
    noise_config = ckpt.noise_config
    noise_encoder_config = ckpt.noise_encoder_config
    denoiser_architecture_config = ckpt.denoiser_architecture_config
    print("Model description:\n", ckpt.description, "\n")
    print("Model license:\n", ckpt.license, "\n")

    # Check example dataset matches model
    def parse_file_parts(file_name):
        return dict(part.split("-", 1) for part in file_name.split("_"))

    def data_valid_for_model(file_name: str, params_file_name: str):
        """Check data type and resolution matches."""
        data_file_parts = parse_file_parts(file_name.removesuffix(".nc"))
        res_matches = data_file_parts["res"].replace(".", "p") in params_file_name.lower()
        source_matches = "Operational" in params_file_name
        if data_file_parts["source"] == "era5":
            source_matches = not source_matches
        return res_matches and source_matches

    assert data_valid_for_model(DATA_PATH, MODEL_PATH)

    # Load weather data
    with open(DATA_PATH, "rb") as f:
        example_batch = xarray.load_dataset(f).compute()

    assert example_batch.dims["time"] >= 3  # 2 for input, >=1 for targets

    print(", ".join([f"{k}: {v}" for k, v in parse_file_parts(DATA_PATH.removesuffix(".nc")).items()]))
    print(example_batch)

    # Extract training and eval data
    train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=slice("12h", "12h"),  # Only 1AR training.
        **dataclasses.asdict(task_config))

    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=slice("12h", "24h"),  # Only predict 1 day (24 hours)
        **dataclasses.asdict(task_config))

    print("All Examples:  ", example_batch.dims.mapping)
    print("Train Inputs:  ", train_inputs.dims.mapping)
    print("Train Targets: ", train_targets.dims.mapping)
    print("Train Forcings:", train_forcings.dims.mapping)
    print("Eval Inputs:   ", eval_inputs.dims.mapping)
    print("Eval Targets:  ", eval_targets.dims.mapping)
    print("Eval Forcings: ", eval_forcings.dims.mapping)

    # Load normalization data
    with open(STATS_DIR + "diffs_stddev_by_level.nc", "rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with open(STATS_DIR + "mean_by_level.nc", "rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with open(STATS_DIR + "stddev_by_level.nc", "rb") as f:
        stddev_by_level = xarray.load_dataset(f).compute()
    with open(STATS_DIR + "min_by_level.nc", "rb") as f:
        min_by_level = xarray.load_dataset(f).compute()

    # Build jitted functions, and possibly initialize random weights
    def construct_wrapped_gencast():
        """Constructs and wraps the GenCast Predictor."""
        predictor = gencast.GenCast(
            sampler_config=sampler_config,
            task_config=task_config,
            denoiser_architecture_config=denoiser_architecture_config,
            noise_config=noise_config,
            noise_encoder_config=noise_encoder_config,
        )

        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level,
        )

        predictor = nan_cleaning.NaNCleaner(
            predictor=predictor,
            reintroduce_nans=True,
            fill_value=min_by_level,
            var_to_clean='sea_surface_temperature',
        )

        return predictor

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = construct_wrapped_gencast()
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    @hk.transform_with_state
    def loss_fn(inputs, targets, forcings):
        predictor = construct_wrapped_gencast()
        loss, diagnostics = predictor.loss(inputs, targets, forcings)
        return xarray_tree.map_structure(
            lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
            (loss, diagnostics),
        )

    def grads_fn(params, state, inputs, targets, forcings):
        def _aux(params, state, i, t, f):
            (loss, diagnostics), next_state = loss_fn.apply(
                params, state, jax.random.PRNGKey(0), i, t, f
            )
            return loss, (diagnostics, next_state)

        (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
            _aux, has_aux=True
        )(params, state, inputs, targets, forcings)
        return loss, diagnostics, next_state, grads

    if params is None:
        init_jitted = jax.jit(loss_fn.init)
        params, state = init_jitted(
            rng=jax.random.PRNGKey(0),
            inputs=train_inputs,
            targets=train_targets,
            forcings=train_forcings,
        )

    loss_fn_jitted = jax.jit(
        lambda rng, i, t, f: loss_fn.apply(params, state, rng, i, t, f)[0]
    )
    grads_fn_jitted = jax.jit(grads_fn)
    run_forward_jitted = jax.jit(
        lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
    )
    # We also produce a pmapped version for running in parallel.
    run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample")

    # The number of ensemble members should be a multiple of the number of devices.
    print(f"Number of local devices {len(jax.local_devices())}")

    # Autoregressive rollout (loop in python)
    print("Inputs:  ", eval_inputs.dims.mapping)
    print("Targets: ", eval_targets.dims.mapping)
    print("Forcings:", eval_forcings.dims.mapping)

    num_ensemble_members = 8
    rng = jax.random.PRNGKey(0)
    # We fold-in the ensemble member, this way the first N members should always
    # match across different runs which use take the same inputs
    # regardless of total ensemble size.
    rngs = np.stack(
        [jax.random.fold_in(rng, i) for i in range(num_ensemble_members)], axis=0)

    chunks = []
    latent_chunks = []
    for chunk, latent_chunk in rollout.chunked_prediction_generator_multiple_runs(
        # Use pmapped version to parallelise across devices.
        predictor_fn=run_forward_pmap,
        rngs=rngs,
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings,
        num_steps_per_chunk=1,
        num_samples=num_ensemble_members,
        pmap_devices=jax.local_devices()
    ):
        chunks.append(chunk)
        if latent_chunk is not None:
            latent_chunks.append(latent_chunk)
    
    predictions = xarray.combine_by_coords(chunks)
    latent_representations = xarray.combine_by_coords(latent_chunks)
    print(f"Predictions: {predictions.dims}")
    print(f"Latent representations: {latent_representations.dims if latent_representations else 'None'}")

    # Extract second timestep first, then compute ensemble mean
    if latent_representations and 'latent_representations' in latent_representations:
        # First, extract the second timestep from each ensemble member
        latent_timestep_2_all_samples = latent_representations['latent_representations'].isel(time=1)
        
        print("Original latent representations shape:", latent_representations['latent_representations'].shape)
        print("Second timestep shape (all samples):", latent_timestep_2_all_samples.shape)
        
        # Now compute the ensemble mean across the 'sample' dimension
        latent_timestep_2_ensemble_mean = latent_timestep_2_all_samples.mean(dim='sample')
        
        # Create a new dataset with the ensemble mean of the second timestep
        latent_timestep_2_dataset = xarray.Dataset({
            'latent_representations': latent_timestep_2_ensemble_mean
        })
        
        print("Final ensemble mean shape (second timestep only):", latent_timestep_2_ensemble_mean.shape)
        
        # Calculate memory usage (approximate)
        # Assuming float32 (4 bytes per element)
        memory_bytes = latent_timestep_2_ensemble_mean.size * 4
        memory_mb = memory_bytes / (1024 * 1024)
        
        # Print statistics for the ensemble mean of the second timestep
        print("\n--- Ensemble Mean of Second Timestep Latent Representations Statistics ---")
        total_elements = latent_timestep_2_ensemble_mean.size
        nan_count = int(latent_timestep_2_ensemble_mean.isnull().sum())
        
        if nan_count < total_elements:
            zero_count = int((latent_timestep_2_ensemble_mean == 0).sum())
            min_val = float(latent_timestep_2_ensemble_mean.min())
            max_val = float(latent_timestep_2_ensemble_mean.max())
            mean_val = float(latent_timestep_2_ensemble_mean.mean())
            std_val = float(latent_timestep_2_ensemble_mean.std())
        else:
            zero_count = 0
            min_val = max_val = mean_val = std_val = 'N/A (all NaNs)'
        
        print(f"Shape: {latent_timestep_2_ensemble_mean.shape}")
        print(f"Total number of elements: {total_elements:,}")
        print(f"Memory usage:             {memory_mb:.2f} MB")
        print(f"Number of NaN values:     {nan_count:,} ({nan_count/total_elements:.2%})")
        print(f"Number of zero values:    {zero_count:,} ({zero_count/total_elements:.2%})")
        print(f"Minimum value:            {min_val}")
        print(f"Maximum value:            {max_val}")
        print(f"Mean value:               {mean_val}")
        print(f"Standard deviation:       {std_val}")
        print("--------------------------------------------------")
        
        
    else:
        print("No latent representations were generated to extract second timestep.")

    print("done")


if __name__ == "__main__":
    main()