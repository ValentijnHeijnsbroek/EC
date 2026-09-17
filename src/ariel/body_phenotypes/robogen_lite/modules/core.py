"""TODO(jmdm): description of script."""

# Third-party libraries
import mujoco
import numpy as np
import quaternion as qnp

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import (
    IDX_OF_CORE,
    ModuleFaces,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.modules.module import Module
from ariel.parameters.ariel_modules import ArielModulesConfig

# Global functions
ariel_modules_config = ArielModulesConfig()

# Type Aliases
type WeightType = float
type DimensionType = tuple[float, float, float]


class CoreModule(Module):
    """Core module specifications."""

    index: int | None = None
    module_type: ModuleType = ModuleType.CORE

    def __init__(self, index: int) -> None:
        """
        Initialize the core module.

        Parameters
        ----------
        index : int
            The index of the core module.

        Raises
        ------
        ValueError
            If the index is not the core module index.
        """
        # Check that the index is the core module index
        if index != IDX_OF_CORE:
            msg = f"Core module index must be {IDX_OF_CORE}, but got {index}."
            raise ValueError(msg)

        # Set the index
        self.index = IDX_OF_CORE

        # Configuration
        core_half = ariel_modules_config.CORE_HALF
        side_site_z = ariel_modules_config.CORE_SIDE_SITE_Z

        # Create the parent spec.
        spec = mujoco.MjSpec()

        # Angles in this spec are radians, matching MujocoConfig.degree.
        # MuJoCo's own default is degrees, and each element keeps the compiler
        # settings of the spec it was authored in even after being attached
        # into a world -- so leaving this at the default silently reinterpreted
        # every angle-valued field (e.g. joint range) as degrees.
        spec.compiler.degree = False

        # ========= Core =========
        core_name = self.module_type.name.lower()
        core = spec.worldbody.add_body(
            name=core_name,
        )
        core.add_geom(
            name=core_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            mass=ariel_modules_config.CORE_MASS_SI,
            size=core_half,
            # Body origin sits on the back face, so the geom centre is half a
            # length forward along y. (This read CORE_DIMENSIONS[0] before,
            # which only happened to be correct because the core is a cube.)
            pos=[0, core_half[1], 0],
            rgba=(253 / 255, 202 / 255, 64 / 255, 1),
        )

        core.add_camera(
            name=f"{core_name}_mycamera",
            pos=[0, 0, core_half[2] - 0.02],
            euler=np.deg2rad([-90, 0, 180]),
        )

        # ========= Attachment Points =========
        self.sites = {}
        shift = -1  # mujoco uses xyzw instead of wxyz
        self.sites[ModuleFaces.FRONT] = core.add_site(
            name=f"{core_name}-front",
            pos=[0, core_half[1] * 2, side_site_z],
            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(0),
                            np.deg2rad(180),
                            np.deg2rad(180),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),
        )
        self.sites[ModuleFaces.BACK] = core.add_site(
            name=f"{core_name}-back",
            pos=[0, 0, side_site_z],
            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(0),
                            np.deg2rad(0),
                            np.deg2rad(0),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),
        )
        self.sites[ModuleFaces.LEFT] = core.add_site(
            name=f"{core_name}-left",
            pos=[
                -core_half[0],
                core_half[1],
                side_site_z,
            ],
            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(90),
                            -np.deg2rad(90),
                            -np.deg2rad(90),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),
        )
        self.sites[ModuleFaces.RIGHT] = core.add_site(
            name=f"{core_name}-right",
            pos=[
                core_half[0],
                core_half[1],
                side_site_z,
            ],
            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(90),
                            np.deg2rad(90),
                            -np.deg2rad(90),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),
        )
        self.sites[ModuleFaces.TOP] = core.add_site(
            name=f"{core_name}-top",
            pos=[0, core_half[1], core_half[2]],
            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(0),
                            np.deg2rad(180),
                            np.deg2rad(90),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),
        )
        self.sites[ModuleFaces.BOTTOM] = core.add_site(
            name=f"{core_name}-bottom",
            pos=[0, core_half[1], -core_half[2]],
            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(0),
                            np.deg2rad(0),
                            -np.deg2rad(90),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),
        )

        # Save model specifications
        self.spec = spec

    def rotate(self, angle: float) -> None:
        """
        Rotate the core module by a specified angle.

        Parameters
        ----------
        angle : float
            The angle in radians to rotate the core.

        Raises
        ------
        AttributeError
            Core module does not support rotation.
        """
        if angle != 0:
            msg = f"Attempted to rotate the core module by: {angle}."
            msg += f"Core ({self.index}) module does not support rotation."
            raise AttributeError(msg)
