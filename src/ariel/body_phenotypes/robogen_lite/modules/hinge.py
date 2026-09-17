"""Hinge (servo) module.

Geometry, mass and joint dynamics all come from
:class:`ariel.parameters.ariel_modules.ArielModulesConfig` — do not re-declare
constants here.

Notes
-----
    * The wrapper ``hinge`` body used to be given ``mass=STATOR_MASS +
      ROTOR_MASS`` even though it carries no geom. Because the stator and rotor
      child bodies already declare those masses, every hinge weighed twice what
      it should, as a point mass with zero inertia. The wrapper is massless now.
    * The servo joint had no ``range``, ``armature``, ``damping`` or
      ``frictionloss``, and the actuator had no ``forcerange``. That made it an
      unbounded, frictionless, inertialess pivot that could deliver several
      times the servo's stall torque — the main lever for physics exploits.

Todo:
----
    [ ] ".rotate" as superclass method?
"""

# Third-party libraries
import mujoco
import numpy as np
import quaternion as qnp

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleType
from ariel.body_phenotypes.robogen_lite.modules.module import Module
from ariel.parameters.ariel_modules import ArielModulesConfig

# Global functions
ariel_modules_config = ArielModulesConfig()

# Type Aliases
type WeightType = float
type DimensionType = tuple[float, float, float]


class HingeModule(Module):
    """Hinge module specifications."""

    index: int | None = None
    module_type: ModuleType = ModuleType.HINGE

    def __init__(self, index: int) -> None:
        """Initialize the hinge module.

        Parameters
        ----------
        index
            The index of the hinge module being instantiated
        """
        # Set the index of the module
        self.index = index

        # Configuration
        cfg = ariel_modules_config
        joint_cfg = cfg.JOINT
        servo = cfg.ACTUATOR
        stator_half = cfg.STATOR_HALF
        rotor_half = cfg.ROTOR_HALF

        # Create the parent spec.
        spec = mujoco.MjSpec()

        # Angles in this spec are radians, matching MujocoConfig.degree.
        # MuJoCo's own default is degrees, and each element keeps the compiler
        # settings of the spec it was authored in even after being attached
        # into a world -- so leaving this at the default silently reinterpreted
        # every angle-valued field (e.g. joint range) as degrees.
        spec.compiler.degree = False

        # ========= Hinge =========
        # Massless wrapper: the stator and rotor below carry the whole mass.
        hinge_name = self.module_type.name.lower()
        hinge = spec.worldbody.add_body(
            name=hinge_name,
        )

        # ========= Stator =========
        stator_name = "stator"
        stator = hinge.add_body(
            name=stator_name,
            pos=[0, stator_half[1], 0],
        )
        stator.add_geom(
            name=stator_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            mass=cfg.STATOR_MASS_SI,
            # Shrunk purely to avoid z-fighting with the rotor; the pair is
            # excluded from collision below, so this changes no physics.
            size=np.array(stator_half) * cfg.STATOR_VISUAL_SHRINK,
            rgba=(223 / 255, 41 / 255, 53 / 255, 1),
        )

        # ========= Rotor =========
        rotor_name = "rotor"
        rotor = hinge.add_body(
            name=rotor_name,
            pos=[0, stator_half[1] * 2 + rotor_half[1], 0],
        )
        rotor.add_geom(
            name=rotor_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            mass=cfg.ROTOR_MASS_SI,
            size=rotor_half,
            rgba=(160 / 255, 24 / 255, 33 / 255, 1),
        )

        # ======== Attachment Points =========
        self.sites = {}
        self.sites[ModuleFaces.FRONT] = rotor.add_site(
            name=f"{hinge_name}-front",
            pos=[0, rotor_half[1], 0],
        )

        # ========= Servo =========
        # Robot actuators
        servo_axis = (0, 0, 1)

        servo_name = "servo"
        rotor.add_joint(
            name=servo_name,
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=servo_axis,
            pos=[0, -rotor_half[1], 0],
            # Mechanical travel of the servo. Without this the joint is
            # unbounded: a limb can wind past 180 degrees and drive itself
            # through the rest of the body.
            limited=joint_cfg.limit_joint_range,
            range=servo.angle_range,
            solref_limit=joint_cfg.limit_solref,
            solimp_limit=joint_cfg.limit_solimp,
            # Reflected gearbox inertia + gearbox drag. Without these the joint
            # is a frictionless pivot with ~1e-5 kg*m^2 of inertia, which
            # jitters under any position gain large enough to hold a limb up.
            armature=joint_cfg.armature_si,
            damping=cfg.SERVO_DAMPING,
            frictionloss=joint_cfg.frictionloss_si,
        )

        # Actuator parameters are defined over a range of 10...
        dynprm = np.zeros(10)
        gainprm = np.zeros(10)
        biasprm = np.zeros(10)

        # ... but only a few of the parameters are actually used
        # force = kp * ctrl - kp * qpos - kv * qvel
        gainprm[0] = joint_cfg.kp_si
        biasprm[:3] = [0, -joint_cfg.kp_si, -joint_cfg.kv_si]

        # Contact exclusion
        spec.add_exclude(
            bodyname1=stator_name,
            bodyname2=rotor_name,
        )

        # --- Actuator(s) --- #
        dyntype = mujoco.mjtDyn.mjDYN_NONE
        gaintype = mujoco.mjtGain.mjGAIN_FIXED
        biastype = mujoco.mjtBias.mjBIAS_AFFINE
        trntype = mujoco.mjtTrn.mjTRN_JOINT
        force_limit = cfg.SERVO_FORCE_LIMIT
        spec.add_actuator(
            name=servo_name,
            dyntype=dyntype,
            gaintype=gaintype,
            biastype=biastype,
            dynprm=dynprm,
            gainprm=gainprm,
            biasprm=biasprm,
            trntype=trntype,
            target=servo_name,
            ctrlrange=servo.angle_range,  # [-90, 90] degrees (range of 180)
            # The -kv * qvel bias term is unbounded, so without this clamp a
            # fast-moving joint can emit far more than the servo's stall
            # torque. Measured peaks were ~3x stall.
            forcelimited=True,
            forcerange=(-force_limit, force_limit),
        )

        # Save model specifications
        self.spec = spec
        self.body = hinge
        self.rotate(angle=0)  # Initialize with no rotation

    def rotate(
        self,
        angle: float,
    ) -> None:
        """
        Rotate the hinge module by a specified angle.

        Parameters
        ----------
        angle
            The angle in degrees to rotate the hinge.
        """
        # Convert angle to quaternion
        quat = qnp.from_euler_angles([
            np.deg2rad(180),
            -np.deg2rad(180 - angle),
            np.deg2rad(0),
        ])
        quat = np.roll(qnp.as_float_array(quat), shift=-1)

        # Set the quaternion for the hinge body
        self.body.quat = np.round(quat, decimals=3)
