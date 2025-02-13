# SPDX-License-Identifier: Apache-2.0
#
"""
Plotting routines.
"""
from functools import partial
from pathlib import Path

from monet import savefig as monet_savefig

__all__ = (
    "savefig",
    "surfplots",
    "aircraftplots",
    "xarray_plots",
)

LOGO_PATH = Path(__file__).parent / "../data/MM_logo.png"

savefig = partial(monet_savefig, logo=LOGO_PATH, loc=2, decorate=True, bbox_inches="tight", dpi=200)

from . import surfplots
from . import aircraftplots
from . import xarray_plots
