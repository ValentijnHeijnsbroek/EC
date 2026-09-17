"""Physical parameters of the ARIEL modules (core, brick, hinge/servo).

This module is the *single source of truth* for module geometry, mass and
servo/joint dynamics. Everything under
``ariel.body_phenotypes.robogen_lite.modules`` reads from here.

Units
-----
    * Every physical parameter is an :class:`astropy.units.Quantity` declared
      in whatever unit is natural for it — millimetres for caliper
      measurements, grams for scale readings, kgf*cm for servo datasheets,
      degrees for angles. See :mod:`ariel.parameters.units`.
    * The ``*_HALF``/``SERVO_*`` properties do the conversion to MuJoCo's SI
      convention and return plain floats/arrays. Module builders use those, so
      no ``Quantity`` ever reaches the simulation loop.

Geometry convention
-------------------
    * ``*_SIZE`` values are the **full outer dimensions** of a module, ordered
      ``(x, y, z)`` = (width, length, height). ``y`` is the attachment axis: a
      module occupies ``y in [0, SIZE.y]`` with its body origin on the *back*
      face.
    * MuJoCo box geoms take **half-extents**, so every ``size=`` argument must
      be ``SIZE / 2``. Use the ``*_HALF`` properties for this — never pass a
      ``*_SIZE`` straight to ``add_geom``.

Notes
-----
    * Previously the constants named ``*_DIMENSIONS`` were documented as
      "(length, width, height) in meters" but were passed to MuJoCo as
      half-extents, so every module was silently **twice** its stated size.
      The ``*_SIZE`` values below are now measured from the physical modules;
      relative to the geometry that used to be simulated, the core and brick
      shrank by 25% and the hinge by 48% along its attachment axis.
    * Joint dynamics (``armature``, ``damping``, ``frictionloss``) were absent
      entirely, which left the servos as ideal frictionless pivots. See
      :class:`ServoJointConfig`.

Sources
-------
    * Geometry and masses: "Ariel modules measurements.xlsx" (sheet
      ``Weights``), measured on the physical modules.
    * Servo electrical/mechanical ratings: SER0019 datasheet.

Todo
----
    [ ] Calibrate ``ServoJointConfig`` against a real SER0019 step response.
    [ ] Re-check ``continuous_torque_fraction``: it was calibrated on the old
        oversized geometry, and shorter lever arms change the force a given
        torque produces at the foot.

"""

# Standard library
from pathlib import Path

# Third-party libraries
import numpy as np
from pydantic_settings import BaseSettings

# Local libraries
from ariel.parameters.units import (
    ANGLE,
    ANGULAR_VELOCITY,
    INERTIA,
    LENGTH,
    MASS,
    QUANTITY_MODEL_CONFIG,
    ROTATIONAL_DAMPING,
    ROTATIONAL_STIFFNESS,
    TORQUE,
    Angle,
    AngularVelocity,
    Inertia,
    Length,
    Mass,
    RotationalDamping,
    RotationalStiffness,
    Time,
    Torque,
    kgf,
    si,
    u,
)

# Global constants
# Global functions
# Warning Control
# Type Checking
# Type Aliases
type WeightType = float
type DimensionType = tuple[float, float, float]

# --- DATA SETUP --- #
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(exist_ok=True)

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)


class SER0019(BaseSettings):
    """DFRobot SER0019 servo, assumed to run at 6 V.

    References
    ----------
        [1] https://github.com/ci-group/ariel-models/blob/master/v2/servo_specs/SER0019_sevo.pdf
    """

    model_config = QUANTITY_MODEL_CONFIG

    # --- Datasheet --- #
    # Mechanical travel is 180 degrees total.
    MIN_ANGLE: Angle = -90 * u.deg
    MAX_ANGLE: Angle = 90 * u.deg
    # Time to sweep 60 degrees, no load.
    SPEED_60_DEG: Time = 0.18 * u.s
    # Datasheets quote this in kilogram-force centimetres.
    STALL_TORQUE: Torque = 13.5 * kgf * u.cm
    MAX_TORQUE_POWER: Mass = 15 * u.kg
    DIMENSIONS: Length = [54.5, 20, 47.5] * u.mm
    MASS: Mass = 65.562 * u.g  # measured, mean over sampled servos
    MIN_FREQUENCY: float = 50  # Hz
    MAX_FREQUENCY: float = 330  # Hz

    @property
    def stall_torque_nm(self) -> float:
        """Stall torque in N*m (the datasheet quotes kgf*cm)."""
        return si(self.STALL_TORQUE, TORQUE)  # 13.5 kgf*cm -> ~1.324 N*m

    @property
    def no_load_speed(self) -> u.Quantity:
        """No-load angular speed, as a quantity."""
        return (60 * u.deg / self.SPEED_60_DEG).to(ANGULAR_VELOCITY)

    @property
    def max_angular_speed(self) -> float:
        """No-load angular speed in rad/s."""
        return si(self.no_load_speed, ANGULAR_VELOCITY)  # ~5.82 rad/s

    @property
    def angle_range(self) -> tuple[float, float]:
        """Mechanical travel limits in radians."""
        return (si(self.MIN_ANGLE, ANGLE), si(self.MAX_ANGLE, ANGLE))


class ServoJointConfig(BaseSettings):
    """Dynamics of the simulated servo joint and its position controller.

    Why each field exists
    ---------------------
    ``armature``
        Reflected inertia of the servo rotor and gear train. A geared hobby
        servo has an output-referred inertia far larger than the plastic
        bracket bolted to it. With ``armature = 0`` the joint inertia is
        ~1e-5 kg*m^2, which makes the position servo enormously stiff relative
        to the mass it drives and is the main source of limb jitter.
    ``damping`` / ``frictionloss``
        Gearbox viscous drag and stiction. Without them an unpowered joint is
        a frictionless pivot that stores and returns energy for free.
    ``force_limit``
        Hard clamp on actuator output. Without it the ``-kv * qvel`` bias term
        is unbounded and the servo can deliver several times its stall torque
        whenever a limb is moving fast — free energy for the optimiser.
    ``kp`` / ``kv``
        Position/velocity gains of the affine-bias position actuator. Because
        the output is clamped to the stall torque, a large ``kp`` simply means
        "saturate until close to the setpoint", which is how a real servo
        behaves. The old ``kp = 1`` was too soft to hold a limb still, and the
        residual wobble at rest is exactly the observed limb jitter.

    Notes
    -----
        * ``damping = None`` derives the value from the datasheet as
          ``stall_torque / no_load_speed``. That ratio *is* the slope of a DC
          motor's torque-speed curve, so it caps how fast the servo can drive
          its joint at the speed it is actually rated for. Leaving it out let
          joints reach ~25 rad/s against a 5.8 rad/s rating.
        * ``kv`` is 0 by default: all velocity damping lives on the joint,
          where ``mjINT_IMPLICITFAST`` integrates it implicitly and is far more
          stable than the explicit ``-kv * qvel`` actuator bias.
        * These are physically-motivated defaults, not measured values. They
          should be fitted against a real servo step response.
    """

    model_config = QUANTITY_MODEL_CONFIG

    # Output-referred rotor + gearbox inertia.
    armature: Inertia = 1e-3 * u.kg * u.m**2
    # None -> derive from the datasheet torque-speed slope.
    damping: RotationalDamping | None = None
    # Dry friction, ~4% of stall torque. A plastic-geared hobby servo is close
    # to non-backdrivable; with this near zero, external impacts spin the
    # joints well past the speed the motor could ever reach.
    frictionloss: Torque = 0.05 * u.N * u.m
    kp: RotationalStiffness = 10.0 * u.N * u.m / u.rad
    kv: RotationalDamping = 0.0 * u.N * u.m * u.s / u.rad

    # Hard limits
    limit_joint_range: bool = True  # enforce the servo's mechanical travel

    # Fraction of the datasheet STALL torque the actuator may actually deliver.
    #
    # Stall is a momentary rating: a servo held there draws maximum current,
    # browns out its supply, overheats and strips its plastic gears. Letting
    # the simulation deliver it continuously hands the optimiser a servo that
    # is both indestructible and infinitely powered.
    #
    # Calibrated against the real robot rather than picked off a datasheet.
    # Sweeping 20-100% of stall on the iguana under both an open-loop sine and
    # an evolved gait, 50% is the largest value at which evolution cannot find
    # a gait that lifts the core off the ground (0.0 cm of lift vs +1.4 cm at
    # 55% and +4.1 cm at 100%), while still travelling 0.83 m in 10 s, inside
    # the 50-100 cm/10 s observed on hardware. Mean mechanical power is then
    # 9.1 W, under the 15.4 W these eight motors can physically produce
    # (stall * no_load_speed / 4 per servo); at 100% it was 29.7 W.
    #
    # Set to 1.0 to recover the raw datasheet behaviour.
    continuous_torque_fraction: float = 0.5

    # Explicit override. None -> continuous_torque_fraction * stall torque.
    force_limit: Torque | None = None

    # Constraint solver parameters for the joint limit (see mjModel.jnt_solref)
    limit_solref_timeconst: Time = 10 * u.ms
    limit_solref_dampratio: float = 1.0
    limit_solimp: tuple[float, float, float, float, float] = (
        0.9,
        0.95,
        0.001,
        0.5,
        2.0,
    )

    @property
    def armature_si(self) -> float:
        """Joint armature in kg*m^2."""
        return si(self.armature, INERTIA)

    @property
    def frictionloss_si(self) -> float:
        """Joint dry friction in N*m."""
        return si(self.frictionloss, TORQUE)

    @property
    def kp_si(self) -> float:
        """Actuator position gain in N*m/rad."""
        return si(self.kp, ROTATIONAL_STIFFNESS)

    @property
    def kv_si(self) -> float:
        """Actuator velocity gain in N*m*s/rad."""
        return si(self.kv, ROTATIONAL_DAMPING)

    @property
    def limit_solref(self) -> tuple[float, float]:
        """Joint-limit ``solref`` as MuJoCo wants it: (seconds, dampratio)."""
        return (si(self.limit_solref_timeconst, u.s), self.limit_solref_dampratio)


class ArielModulesConfig(BaseSettings):
    """Geometry and mass of every ARIEL module type."""

    model_config = QUANTITY_MODEL_CONFIG

    # --- Core Config --- #
    # Measured: core without battery 966.85 g, battery 396.50 g, so the
    # assembled core is 1363.35 g. Adding the camera module (blue blocks +
    # 2 screws + camera = 22.77 g) gives 1386.12 g; use that if the camera is
    # fitted. The previous 1.0 kg was a placeholder.
    CORE_MASS: Mass = 1363.347 * u.g
    CORE_SIZE: Length = [150, 150, 150] * u.mm  # full outer size, measured

    # Vertical offset of the core's FRONT/BACK/LEFT/RIGHT attachment sites,
    # measured from the face centre and expressed as a fraction of CORE_SIZE.z
    # so it scales with the module. The original code offset these by
    # -CORE_SIZE.y / 4 (it used the *y* constant to shift *z*), putting the
    # four lateral sites 5 cm below the face centre while TOP/BOTTOM stayed
    # centred — so a limb attached to a core sat lower than the same limb
    # attached to a brick. Set to 0.0 to make the core consistent with the
    # brick; kept at -0.25 here to preserve existing morphologies.
    CORE_SIDE_SITE_Z_RATIO: float = -0.25
    # ------------------------------ #

    # --- Brick Config --- #
    # Measured per colour: red 59.53 g, blue 61.13 g, green 56.47 g. The mean
    # across colours is used here; substitute a specific colour if a build
    # uses only one.
    BRICK_MASS: Mass = 59.043 * u.g
    BRICK_SIZE: Length = [75, 75, 75] * u.mm  # full outer size, measured
    # ------------------------------ #

    # --- Hinge Config --- #
    ACTUATOR: SER0019 = SER0019()
    JOINT: ServoJointConfig = ServoJointConfig()

    # Measured hinge assembly = 104.634 g, broken down as:
    #     servo                          65.562 g
    #     plastic, round hole (red)      11.484 g
    #     plastic, round knob (red)      16.488 g
    #     extras (8 big + 4 small screws, connector)  11.100 g
    #
    # The servo body and the bracket it bolts into are the stator; the
    # knob part turns with the output horn, so it is the rotor. The screws and
    # connector are assigned to the stator, which is where they mount — that
    # split is an assumption, the 104.634 g total is measured.
    STATOR_MASS: Mass = (65.562 + 11.484 + 11.100) * u.g  # 88.146 g
    ROTOR_MASS: Mass = 16.488 * u.g

    # Measured hinge envelope: 75 x 52 x 52 mm. The **75 mm dimension is the
    # attachment axis (y)** — the hinge is a link 75 mm long between its two
    # mating faces, with a 52 x 52 mm cross-section. The spreadsheet labels it
    # "Width", which is why it was first mapped to x; that made the hinge a
    # short wide slab attached by its side face instead of its back face.
    #
    # The SER0019 is 54.5 mm long, which cannot fit across a 52 mm module but
    # sits along a 75 mm one — independent confirmation of the orientation.
    #
    # The stator/rotor split is NOT in the spreadsheet; only the 75 mm total
    # is measured. The hardware hinge is symmetric about its rotation axis, so
    # the axis sits at the midpoint and each half is 37.5 mm. (An earlier
    # 55/20 split, inferred from the servo's own 54.5 mm length, put the axis
    # 55 mm from one face and 20 mm from the other and was visibly lopsided.)
    #
    # The *mass* split stays asymmetric — 88.1 g stator vs 16.5 g rotor —
    # because the servo is bolted to the stator half. That is measured; only
    # the geometry is symmetric.
    STATOR_SIZE: Length = [52, 37.5, 52] * u.mm
    ROTOR_SIZE: Length = [52, 37.5, 52] * u.mm

    # Visual-only shrink applied to the stator geom so it does not z-fight with
    # the rotor. Has no effect on collisions (the pair is excluded).
    STATOR_VISUAL_SHRINK: float = 0.99
    # ------------------------------ #

    # --- Derived half-extents (what MuJoCo actually wants, in metres) --- #
    @property
    def CORE_HALF(self) -> np.ndarray:  # noqa: N802
        """Core box half-extents in metres."""
        return si(self.CORE_SIZE / 2, LENGTH)

    @property
    def BRICK_HALF(self) -> np.ndarray:  # noqa: N802
        """Brick box half-extents in metres."""
        return si(self.BRICK_SIZE / 2, LENGTH)

    @property
    def STATOR_HALF(self) -> np.ndarray:  # noqa: N802
        """Hinge stator box half-extents in metres."""
        return si(self.STATOR_SIZE / 2, LENGTH)

    @property
    def ROTOR_HALF(self) -> np.ndarray:  # noqa: N802
        """Hinge rotor box half-extents in metres."""
        return si(self.ROTOR_SIZE / 2, LENGTH)

    # --- Derived masses (kg) --- #
    @property
    def CORE_MASS_SI(self) -> float:  # noqa: N802
        """Core mass in kg."""
        return si(self.CORE_MASS, MASS)

    @property
    def BRICK_MASS_SI(self) -> float:  # noqa: N802
        """Brick mass in kg."""
        return si(self.BRICK_MASS, MASS)

    @property
    def STATOR_MASS_SI(self) -> float:  # noqa: N802
        """Hinge stator mass in kg."""
        return si(self.STATOR_MASS, MASS)

    @property
    def ROTOR_MASS_SI(self) -> float:  # noqa: N802
        """Hinge rotor mass in kg."""
        return si(self.ROTOR_MASS, MASS)

    @property
    def HINGE_MASS(self) -> float:  # noqa: N802
        """Total mass of one hinge module (stator + rotor) in kg."""
        return si(self.STATOR_MASS + self.ROTOR_MASS, MASS)

    # --- Derived geometry --- #
    @property
    def CORE_SIDE_SITE_Z(self) -> float:  # noqa: N802
        """Absolute z offset of the core's lateral attachment sites (metres)."""
        return self.CORE_SIDE_SITE_Z_RATIO * si(self.CORE_SIZE[2], LENGTH)

    @property
    def HINGE_LENGTH(self) -> float:  # noqa: N802
        """Length of a hinge module along the attachment (y) axis, in metres."""
        return si(self.STATOR_SIZE[1] + self.ROTOR_SIZE[1], LENGTH)

    # --- Derived servo dynamics --- #
    @property
    def SERVO_FORCE_LIMIT(self) -> float:  # noqa: N802
        """Actuator force limit in N*m.

        Defaults to ``continuous_torque_fraction`` of the datasheet stall
        torque, since stall is a momentary rating that a real servo cannot
        sustain. See :class:`ServoJointConfig` for the calibration.
        """
        override = self.JOINT.force_limit
        if override is not None:
            return si(override, TORQUE)
        return (
            self.JOINT.continuous_torque_fraction
            * self.ACTUATOR.stall_torque_nm
        )

    @property
    def SERVO_DAMPING(self) -> float:  # noqa: N802
        """Joint damping in N*m*s/rad.

        Defaults to the slope of the servo's torque-speed curve,
        ``stall_torque / no_load_speed``, which makes the joint's terminal
        velocity under full torque equal to the servo's rated speed. Astropy
        checks that this ratio really does come out in N*m*s/rad.
        """
        override = self.JOINT.damping
        if override is not None:
            return si(override, ROTATIONAL_DAMPING)
        slope = self.ACTUATOR.STALL_TORQUE / self.ACTUATOR.no_load_speed
        return si(slope, ROTATIONAL_DAMPING)
