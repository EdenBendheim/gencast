# Copyright 2023 DeepMind Technologies Limited.
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
"""A predictor that cleans NaNs from inputs and optionally re-introduces them."""

from typing import Optional, Tuple

from graphcast import predictor_base
import numpy as np
import xarray

class NaNCleaner(predictor_base.Predictor):
  """A predictor that cleans NaNs from inputs and optionally re-introduces them."""

  def __init__(
      self,
      predictor: predictor_base.Predictor,
      var_to_clean: str,
      fill_value: xarray.Dataset,
      reintroduce_nans: bool = True,
  ):
    self._predictor = predictor
    self._var_to_clean = var_to_clean
    self._reintroduce_nans = reintroduce_nans
    self._fill_value = fill_value[var_to_clean]

  def _clean(self, data: xarray.Dataset) -> xarray.Dataset:
    """Replaces NaNs with the fill value."""
    data_array = data[self._var_to_clean]
    data = data.assign(
        {self._var_to_clean: data_array.fillna(self._fill_value)}
    )
    return data

  def _maybe_reintroduce_nans(
      self, stale_inputs: xarray.Dataset, predictions: xarray.Dataset
  ) -> xarray.Dataset:
    # NaN positions don't change between input frames, if they do then
    # we should be more careful about re-introducing them.
    if self._var_to_clean in predictions.keys():
      nan_mask = np.isnan(stale_inputs[self._var_to_clean]).any(dim='time')
      with_nan_values = predictions[self._var_to_clean].where(~nan_mask, np.nan)
      predictions = predictions.assign(
          {self._var_to_clean: with_nan_values})
    return predictions

  def __call__(
      self,
      inputs: xarray.Dataset,
      targets_template: xarray.Dataset,
      forcings: Optional[xarray.Dataset],
      **kwargs,
  ) -> Tuple[xarray.Dataset, Optional[xarray.Dataset]]:
    """Runs the predictor and returns the predictions."""
    original_inputs = inputs
    if self._var_to_clean in inputs.keys():
      inputs = self._clean(inputs)
    if forcings and self._var_to_clean in forcings.keys():
      forcings = self._clean(forcings)

    predictor_output = self._predictor(
        inputs, targets_template, forcings, **kwargs
    )
    
    if isinstance(predictor_output, tuple):
        predictions, latent_representations = predictor_output
    else:
        predictions = predictor_output
        latent_representations = None
    
    if self._reintroduce_nans:
      predictions = self._maybe_reintroduce_nans(original_inputs, predictions)

    return predictions, latent_representations

  def loss(self, *args, **kwargs):
    return self._predictor.loss(*args, **kwargs)

  def loss_and_predictions(self, *args, **kwargs):
    return self._predictor.loss_and_predictions(*args, **kwargs)
