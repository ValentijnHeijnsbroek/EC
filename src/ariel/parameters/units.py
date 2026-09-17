"""Unit-aware parameter types, built on :mod:`astropy.units`.

Why
---
MuJoCo is unit-agnostic: it treats every number as being in a single coherent
system and never checks. ARIEL fixes that system to SI (metres, kilograms,
seconds, radians). Historically the parameter files carried the unit only in a
comment, and every unit error that convention permits actually happened:

    * ``*_DIMENSIONS`` documented as "(length, width, height) in meters" but
      passed to ``add_geom(size=...)``, which takes **half-extents** — so every
      module was silently twice its stated size.
    * A servo stall torque of ``13.5`` "kg*cm" left as a bare float, with the
      9.80665e-2 conversion to N*m applied by hand at the point of use.
    * Joint ranges written in radians into a spec whose compiler was in degree
      mode, turning +-90 degrees into +-1.57 degrees.

Declaring parameters as :class:`~astropy.units.Quantity` makes all three a
loud error instead of a silent one.

How
---
Declare parameters in whatever unit the datasheet or caliper uses::

    BRICK_SIZE: Length = [100, 100, 100] * u.mm
    STALL_TORQUE: Torque = 13.5 * kgf * u.cm
    MAX_ANGLE: Angle = 90 * u.deg

and convert once, at the MuJoCo boundary, with :func:`si`::

    size = si(cfg.BRICK_SIZE / 2, LENGTH)   # -> ndarray of metres

Quantities live only in the config layer. Everything handed to MuJoCo is a
plain ``float``/``ndarray``, so no per-step simulation cost is introduced.

Notes
-----
    * A bare number assigned to one of these fields is *assumed to already be
      in the canonical SI unit* and is accepted. This keeps the config
      env-var-overridable and backwards compatible; prefer explicit units.
    * ``kgf`` (kilogram-force) is defined here because hobby-servo datasheets
      quote stall torque in kgf*cm and astropy has no built-in for it.

References
----------
    [1] https://docs.astropy.org/en/stable/units/

"""

# Standard library
from typing import Annotated, Any

# Third-party libraries
import astropy.units as u
import numpy as np
from astropy.constants import g0
from pydantic import BeforeValidator, PlainSerializer
from pydantic_settings import SettingsConfigDict

__all__ = [
    "ANGLE",
    "ANGULAR_VELOCITY",
    "INERTIA",
    "LENGTH",
    "MASS",
    "QUANTITY_MODEL_CONFIG",
    "ROTATIONAL_DAMPING",
    "ROTATIONAL_STIFFNESS",
    "TIME",
    "TORQUE",
    "Angle",
    "AngularVelocity",
    "Inertia",
    "Length",
    "Mass",
    "RotationalDamping",
    "RotationalStiffness",
    "Time",
    "Torque",
    "kgf",
    "si",
    "u",
]

# --- CUSTOM UNITS --- #
# Kilogram-force: the unit hobby-servo datasheets quote torque in.
kgf = u.def_unit("kgf", u.kg * g0)
u.add_enabled_units([kgf])

# --- CANONICAL MUJOCO UNITS --- #
# Everything handed to MuJoCo is expressed in these.
LENGTH = u.m
MASS = u.kg
TIME = u.s
ANGLE = u.rad
TORQUE = u.N * u.m
ANGULAR_VELOCITY = u.rad / u.s
INERTIA = u.kg * u.m**2
ROTATIONAL_DAMPING = u.N * u.m * u.s / u.rad
ROTATIONAL_STIFFNESS = u.N * u.m / u.rad


def si(value: u.Quantity, unit: u.UnitBase) -> Any:
    """Convert a quantity to a plain number in the given unit.

    This is the boundary between the unit-checked config layer and MuJoCo,
    which wants bare floats and arrays.

    Parameters
    ----------
    value : u.Quantity
        The quantity to convert.
    unit : u.UnitBase
        Target unit, normally one of the canonical units in this module.

    Returns
    -------
    float | np.ndarray
        ``float`` for a scalar quantity, ``ndarray`` for an array quantity.

    Raises
    ------
    u.UnitConversionError
        If ``value`` is not convertible to ``unit``.
    """
    converted = u.Quantity(value).to_value(unit)
    if np.isscalar(converted) or converted.ndim == 0:
        return float(converted)
    return np.asarray(converted, dtype=float)


def _quantity_validator(unit: u.UnitBase) -> Any:
    """Build a pydantic validator accepting a Quantity convertible to ``unit``.

    Parameters
    ----------
    unit : u.UnitBase
        The canonical unit this field must be convertible to.

    Returns
    -------
    Callable
        A validator suitable for :class:`pydantic.BeforeValidator`.
    """

    def _validate(value: Any) -> u.Quantity:
        # A bare number is taken to already be in the canonical unit, so that
        # environment-variable overrides and plain floats keep working.
        quantity = (
            value if isinstance(value, u.Quantity) else u.Quantity(value, unit)
        )
        try:
            quantity.to(unit)
        except u.UnitConversionError as exc:
            msg = (
                f"expected a quantity in units convertible to {unit}, "
                f"got {quantity.unit}"
            )
            raise ValueError(msg) from exc
        return quantity

    return _validate


def _annotated(unit: u.UnitBase) -> Any:
    """Build the Annotated alias for a unit-checked Quantity field."""
    return Annotated[
        u.Quantity,
        BeforeValidator(_quantity_validator(unit)),
        PlainSerializer(str, return_type=str),
    ]


# --- FIELD TYPES --- #
# Use these as the annotation of any physical parameter in a config class.
Length = _annotated(LENGTH)
Mass = _annotated(MASS)
Time = _annotated(TIME)
Angle = _annotated(ANGLE)
Torque = _annotated(TORQUE)
AngularVelocity = _annotated(ANGULAR_VELOCITY)
Inertia = _annotated(INERTIA)
RotationalDamping = _annotated(ROTATIONAL_DAMPING)
RotationalStiffness = _annotated(ROTATIONAL_STIFFNESS)

# Config classes holding Quantity fields need this: pydantic has no core
# schema for Quantity, so it must be allowed as an arbitrary type.
QUANTITY_MODEL_CONFIG = SettingsConfigDict(arbitrary_types_allowed=True)
