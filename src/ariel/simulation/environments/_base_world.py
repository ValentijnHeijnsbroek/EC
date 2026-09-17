"""
MuJoCo world: base class for MuJoCo world specifications.

References
----------
    [1] https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mju-euler2quat

Todo
----
    [ ] Document the class methods
"""

# Standard library
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import cast

# Third-party libraries
import mujoco as mj
import numpy as np

# Local libraries
from ariel import log
from ariel.parameters.ariel_types import Position, Rotation
from ariel.parameters.mujoco_params import MujocoConfig
from ariel.utils.mujoco_ops import euler_to_quat_conversion


class BaseWorld:
    """Base class for MuJoCo world specifications."""

    name: str = "base-world"

    spawns: int = 0
    spawn_prefix: str = "robot"
    default_spawn_position: Position = (0, 0, 0)  # x, y, z
    default_spawn_rotation: Rotation = (0, 0, 0)  # x, y, z

    is_precompiled: bool = False

    def __init__(
        self,
        name: str | None = None,
        mujoco_config: MujocoConfig | None = None,
        *,
        load_precompiled: bool = True,
    ) -> None:
        """
        Initialize the world specification.

        Parameters
        ----------
        name : str, optional
            Name of the world, by default None
        mujoco_config : MujocoConfig, optional
            Configuration parameters for MuJoCo, by default None
        load_precompiled : bool, optional
            Whether to load a precompiled XML file if available, by default True
        """

        # Use default mujoco config if none is provided
        self.mujoco_config = (
            MujocoConfig() if mujoco_config is None else mujoco_config
        )

        # Set world name
        if name is not None:
            self.name = name

        # Load precompiled XML if requested
        if load_precompiled is True:
            log.debug("Attempting to load precompiled XML...")
            self.is_precompiled = self.load_from_xml()
            if self.is_precompiled:
                log.debug("Precompiled XML loaded successfully.")
                # A cached XML predates the current config, so the solver and
                # compiler settings must be re-applied on top of it.
                self._apply_compiler_and_options(self.spec)
                return

        # Build and save specification
        self.spec: mj.MjSpec = self._init_spec()

    @abstractmethod
    def _expand_spec(self) -> None:
        """Expand the world specification with additional elements."""

    def _init_spec(self) -> mj.MjSpec:
        """Initialize the MuJoCo specification."""

        spec = mj.MjSpec()

        # Model name
        spec.modelname = self.name.replace("-", " ").title()

        # Copy during attach
        spec.copy_during_attach = True

        # --- Compiler + Option --- #
        self._apply_compiler_and_options(spec)

        # --- Visual --- #
        spec.visual.global_.offheight = self.mujoco_config.offheight
        spec.visual.global_.offwidth = self.mujoco_config.offwidth

        # Headlight: diffuse, ambient, specular
        spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
        spec.visual.headlight.ambient = [0.1, 0.1, 0.1]
        spec.visual.headlight.specular = [0.9, 0.9, 0.9]

        # RGBA and Global
        spec.visual.rgba.force = [1, 0, 0, 1]
        spec.visual.rgba.haze = [0.15, 0.25, 0.35, 1]
        spec.visual.global_.azimuth = 140
        spec.visual.global_.elevation = -20

        # Map and Scale
        spec.visual.map.force = 0.01
        spec.visual.scale.forcewidth = 0.3
        spec.visual.scale.contactwidth = 0.5
        spec.visual.scale.contactheight = 0.2

        # Quality
        spec.visual.quality.shadowsize = 8192

        # --- Assets ---
        # Skybox Texture
        spec.add_texture(
            type=mj.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mj.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[
                61 / 255,
                163 / 255,
                179 / 255,
            ],
            rgb2=[
                82 / 255,
                57 / 255,
                153 / 255,
            ],
            width=512,
            height=3072,
        )

        # Add a default light source
        spec.worldbody.add_light(
            name="light",
            pos=[0, 0, 1],
            castshadow=False,
            type=mj.mjtLightType.mjLIGHT_DIRECTIONAL,
        )

        # Add ortho camera and normal camera
        # mujoco >= 3.6.0: orthographic= replaced by proj= (mjtProjection)
        spec.worldbody.add_camera(
            name="ortho-cam",
            proj=mj.mjtProjection.mjPROJ_ORTHOGRAPHIC,
            pos=[-5, 0, 5],
            xyaxes=[0, -1, 0, 0.75, 0, 0.75],
            fovy=5,
        )

        spec.worldbody.add_camera(
            name="pretty-cam",
            proj=mj.mjtProjection.mjPROJ_PERSPECTIVE,
            pos=[-0.015, -3.003, 1.765],
            xyaxes=[1.000, -0.005, -0.000, 0.002, 0.507, 0.862],
            fovy=45,
        )
        return spec

    def _apply_compiler_and_options(self, spec: mj.MjSpec) -> None:
        """Write every :class:`MujocoConfig` compiler/solver field onto a spec.

        Kept separate from :meth:`_init_spec` so that it can also be applied to
        a spec restored from a precompiled XML.

        Parameters
        ----------
        spec : mj.MjSpec
            The specification to configure, modified in place.
        """
        cfg = self.mujoco_config

        # --- Option --- #
        spec.option.timestep = cfg.timestep_si
        spec.option.integrator = cfg.integrator
        spec.option.solver = cfg.solver
        spec.option.iterations = cfg.iterations
        spec.option.ls_iterations = cfg.ls_iterations
        spec.option.impratio = cfg.impratio
        spec.option.cone = cfg.cone

        # --- Compiler --- #
        spec.compiler.autolimits = cfg.autolimits
        spec.compiler.balanceinertia = cfg.balanceinertia
        spec.compiler.degree = cfg.degree

        # A contact time constant below 2 * timestep makes the constraint
        # solver unstable, which shows up as limbs vibrating or being flung.
        # These two settings live in different config fields, so it is easy to
        # change one and silently invalidate the other. Both are astropy
        # quantities, so the comparison is unit-safe even if one is written in
        # milliseconds and the other in seconds.
        min_timeconst = 2.0 * cfg.timestep
        if cfg.geom_solref_timeconst < min_timeconst:
            msg = (
                f"geom_solref_timeconst={cfg.geom_solref_timeconst} is below "
                f"2 * timestep ({min_timeconst}). Contacts will be unstable. "
                f"Either raise geom_solref_timeconst or lower the timestep."
            )
            log.warning(msg)

    def apply_contact_defaults(self, spawn_name: str | None = None) -> None:
        """Apply the configured contact parameters to geoms in the world.

        MuJoCo defaults are resolved per-spec at compile time, so defaults set
        on the world do *not* reach the geoms of an attached robot spec. This
        walks the merged spec instead.

        Parameters
        ----------
        spawn_name : str | None, optional
            If given, only geoms whose name starts with this prefix are treated
            as robot geoms for the purpose of self-collision filtering. All
            geoms receive the friction/solver defaults regardless.
        """
        cfg = self.mujoco_config

        for geom in self.spec.geoms:
            geom.solref = np.array(cfg.geom_solref)
            geom.solimp = np.array(cfg.geom_solimp)
            geom.friction = np.array(cfg.geom_friction)
            geom.condim = cfg.geom_condim

            # Robot geoms go on their own collision bit so that robot-robot and
            # robot-self contacts can be disabled without also disabling
            # contact with the terrain.
            #   bit 1 -> world / terrain, bit 2 -> robot
            if spawn_name is not None and geom.name.startswith(spawn_name):
                geom.contype = 2
                geom.conaffinity = 3 if cfg.enable_self_collision else 1

                # Self-contacts get a softer solref than geom_solref_timeconst.
                # See MujocoConfig.self_contact_solref_timeconst for why: the
                # stiff default can eject a robot on a single bad-depth
                # reading from the mesh<->mesh collider, mostly on
                # repeated-segment bodies (centipedes). Also softens
                # floor<->robot contacts via MuJoCo's solref averaging (5ms
                # mixes with 20ms to 12.5ms) - accepted tradeoff, see the
                # config docstring.
                geom.solref = np.array(cfg.self_contact_solref)

    def convert_boxes_to_meshes(self) -> int:
        """Replace every box geom with an equivalent 8-vertex convex mesh.

        Works around a defect in MuJoCo's analytic box<->box collider
        (``mjc_BoxBox``): for two boxes that share a parallel axis and overlap
        by less than ~0.05 mm it collapses the two-point contact manifold to a
        single point and reports a penetration depth orders of magnitude too
        deep. On an ARIEL body a true overlap of 0.018 mm was reported as
        76.458 mm; the solver ejected it with 5203 N*m of generalised force,
        injecting 39 J in one 2 ms step and launching the robot.

        Only the box<->box dispatch is affected, so making either geom a mesh
        is enough; this converts all of them. The mesh is the exact convex hull
        of the box, so collision geometry is unchanged and mass is preserved
        bit-for-bit. Body inertia moves by ~0.2% because MuJoCo integrates it
        over the mesh rather than using the closed-form box expression.

        Reproduced on MuJoCo 3.2.7 through 3.8.0 -- it is long-standing, not a
        regression, and neither ``margin`` nor the CCD flags avoid it.

        Returns
        -------
        int
            Number of geoms converted.
        """
        # One mesh asset per distinct box size, keyed by half-extents.
        meshes: dict[tuple[float, ...], str] = {}
        converted = 0

        for geom in self.spec.geoms:
            if geom.type != mj.mjtGeom.mjGEOM_BOX:
                continue

            half = tuple(round(float(v), 9) for v in geom.size[:3])
            name = meshes.get(half)
            if name is None:
                name = f"boxmesh_{len(meshes)}"
                vertices = [
                    s * half[axis]
                    for sx in (-1.0, 1.0)
                    for sy in (-1.0, 1.0)
                    for sz in (-1.0, 1.0)
                    for axis, s in enumerate((sx, sy, sz))
                ]
                mesh = self.spec.add_mesh()
                mesh.name = name
                mesh.uservert = vertices
                meshes[half] = name

            geom.type = mj.mjtGeom.mjGEOM_MESH
            geom.meshname = name
            converted += 1

        if converted:
            msg = (
                f"Converted {converted} box geoms to {len(meshes)} convex "
                f"meshes to avoid the MuJoCo box-box collider defect."
            )
            log.debug(msg)
        return converted

    def _find_lowest_position(
        self,
        spawn_name: str,
    ) -> float:
        """Find the lowest Z position of the spawned robot.

        Parameters
        ----------
        spawn_name : str
            Prefix name of the spawned robot to identify its geometries.

        Returns
        -------
            float
                The lowest Z position of the robot in the world."""
        # Generate model and data from a temporary copy of the spec
        model: mj.MjModel = cast("mj.MjModel", self.spec.compile())
        data = mj.MjData(model)

        # Step the simulation to ensure everything is stable
        mj.mj_forward(model, data)

        # Iterate over all geoms
        lowest_point = np.inf
        for i in range(model.ngeom):
            # Get the geometry
            geom = data.geom(i)
            bodyid = int(model.geom_bodyid[geom.id])
            parentid = int(model.body(bodyid).parentid[0])

            # Possible names
            name_of_geom = geom.name
            name_of_body = model.body(bodyid).name
            name_of_parent = model.body(parentid).name

            # If the geom does not belong to the spawned robot, skip it
            if (
                (spawn_name not in name_of_geom)
                and (spawn_name not in name_of_body)
                and (spawn_name not in name_of_parent)
            ):
                continue

            # Global position of the geometry (x, y, z)
            pos = data.geom_xpos[geom.id]

            # World rotation matrix (flat 9 values in row-major)
            r_mat = np.array(data.geom_xmat[geom.id]).reshape(3, 3)

            # Local half-sizes (sx, sy, sz)
            sx, sy, sz = model.geom_size[geom.id]  # box half extents

            # Generate 8 local corner offsets
            corners_local = np.array([
                [dx * sx, dy * sy, dz * sz]
                for dx in (-1, 1)
                for dy in (-1, 1)
                for dz in (-1, 1)
            ])

            # Transform corners: world_corner = pos + R @ local_corner
            corners_world = pos + corners_local @ r_mat.T  # (8,3)

            # Return the lowest Z value
            maybe_lowest_point = np.min(corners_world[:, 2])
            lowest_point = min(lowest_point, maybe_lowest_point)

        # Clear the temporary objects
        del model, data

        # Return the lowest position rounded to avoid floating point issues
        if lowest_point == np.inf:
            return 0.0
        return np.round(lowest_point, 6)

    def _find_contacts(self) -> set[tuple[str, str, float]]:
        # Generate model and data
        model = self.spec.compile()
        data = mj.MjData(model)

        # Step the simulation to ensure everything is stable
        mj.mj_forward(model, data)

        # Discover contacts between the world and the spawned robots
        contact_pairs = set()
        for contact in data.contact:
            geom1 = mj.mj_id2name(
                m=model,
                type=mj.mjtObj.mjOBJ_GEOM,
                id=contact.geom1,
            )
            geom2 = mj.mj_id2name(
                m=model,
                type=mj.mjtObj.mjOBJ_GEOM,
                id=contact.geom2,
            )
            contact_pairs |= {(geom1, geom2, contact.dist)}

        # Clear the temporary objects
        del model, data

        # Return the set of contact pairs
        return contact_pairs

    def _check_and_correct_spawn(
        self,
        spawn_site: mj.MjsBody,
        spawn_body: mj.MjsBody,
        spawn_name: str,
        base_point: float = 0.01,
        *,
        validate_no_collisions: bool = False,
    ) -> None:
        """
        Check and correct the spawn position to avoid collisions with the floor.


        Parameters
        ----------
        spawn_site : mj.MjsBody
            The site where the robot is spawned.
        spawn_body : mj.MjsBody
            The body of the spawned robot.
        spawn_name : str
            The prefix name of the spawned robot.
        base_point : float, optional
            Minimum distance above the lowest point of the robot, by default 0.01
        validate_no_collisions : bool, optional
            Whether to validate the spawn position for collisions, by default False
        """

        # Log the correction process
        msg = "-" * 60
        log.debug(msg)

        # Get the spawn position
        msg = f"Initial spawn position: {spawn_site.pos}"
        log.debug(msg)

        # Find lowest position of the robot
        lowest_position = self._find_lowest_position(spawn_name)
        msg = f"Lowest robot position: {lowest_position} m"
        log.debug(msg)

        # Adjust the spawn position to ensure the robot is above ground.
        # `lowest_position` is already a world-frame height, and `spawn_body`
        # is expressed relative to `spawn_site`, so the site height must NOT be
        # added again — doing so left the robot hovering at
        # `base_point + spawn_site.pos[2]` above the floor and free-falling
        # onto it at the start of every evaluation.
        diff_from_base = base_point - lowest_position
        spawn_body.pos[2] += diff_from_base
        msg = f"Adjusted spawn position: {spawn_body.pos}"
        log.debug(msg)

        # Validate the spawn position by checking for collisions
        if validate_no_collisions is True:
            contact_pairs = self._find_contacts()
            for contact in contact_pairs:
                # Unpack contact details
                geom1_name, geom2_name, dist = contact

                # If there is a collision with the floor, log a warning
                floor_name = self.mujoco_config.floor_name
                if floor_name in geom1_name or floor_name in geom2_name:
                    msg = "Spawn position causes collision with the floor!\n"
                    msg += f"--> '{geom1_name}', '{geom2_name}'\n"
                    msg += f"\t With distance: {dist}\n"
                    msg += " Please adjust the spawn position: \n"
                    msg += f"\t {spawn_site.pos=}"
                    log.warning(msg)
                else:
                    # Log other collisions as debug info
                    msg = "Spawn position causes collision!\n"
                    msg += f"--> '{geom1_name}', '{geom2_name}'\n"
                    msg += f"\t With distance: {dist}"
                    log.debug(msg)

    def spawn(
        self,
        robot_spec: mj.MjSpec,
        position: Position | None = None,
        rotation: Rotation | None = None,
        spawn_prefix: str | None = None,
        *,
        correct_collision_with_floor: bool = True,
        validate_no_collisions: bool = False,
        rotation_sequence: str = "XYZ",  # xyzXYZ, assume intrinsic
    ) -> mj.MjSpec:
        """
        Spawn a robot into the world at a specified position and orientation.

        Parameters
        ----------
        robot_spec : mj.MjSpec
            The MuJoCo specification of the robot to be spawned.
        position : Position, optional
            The (x, y, z) position to spawn the robot, by default None
        rotation : Rotation, optional
            The (x, y, z) Euler angles (in degrees) for the robot's orientation, by default None
        spawn_prefix : str, optional
            Prefix for naming the spawned robot, by default None
        correct_collision_with_floor : bool, optional
            Whether to adjust the spawn position to avoid collisions with the floor, by default True
        validate_no_collisions : bool, optional
            Whether to validate the spawn position for collisions after adjustment, by default False
        rotation_sequence : str, optional
            The sequence of axes for Euler to quaternion conversion, by default "XYZ"

        Returns
        -------
            mj.MjSpec
                The updated MuJoCo world specification with the spawned robot.
        """
        # Default spawn position
        if position is None:
            position = self.default_spawn_position
        else:
            position = deepcopy(position)

        # Default spawn orientation
        if rotation is None:
            rotation = self.default_spawn_rotation
        else:
            rotation = deepcopy(rotation)

        # If no prefix is given, use the default one
        if spawn_prefix is None:
            spawn_prefix = self.spawn_prefix

        # Convert rotation from Euler angles (degrees) to quaternion
        rotation_as_quat = euler_to_quat_conversion(
            rotation,
            rotation_sequence,
        )

        # Increment the spawn count
        self.spawns += 1

        # Create a spawn site at the specified position
        spawn_site = self.spec.worldbody.add_site(
            pos=np.array(position),
            quat=np.array(rotation_as_quat),
        )

        # Attach the robot body to the spawn site
        spawn_name = f"{spawn_prefix}{self.spawns}_"
        spawn_body = spawn_site.attach_body(
            body=robot_spec.worldbody,
            prefix=spawn_name,
        )

        # Apply contact/friction defaults and self-collision filtering to the
        # merged spec. Must happen before the spawn correction, which compiles
        # the spec to look for contacts.
        self.apply_contact_defaults(spawn_name=spawn_name)

        # Route box<->box pairs away from MuJoCo's faulty analytic collider.
        # Must also happen before the spawn correction, which compiles the spec
        # and reads contact distances back out of it.
        if self.mujoco_config.convert_boxes_to_meshes:
            self.convert_boxes_to_meshes()

        # Correct the spawn position if requested
        if correct_collision_with_floor is True:
            self._check_and_correct_spawn(
                spawn_site,
                spawn_body,
                spawn_name,
                validate_no_collisions=validate_no_collisions,
            )

        # Allow the robot to move freely
        spawn_body.add_freejoint()

        # Return a copy of the updated spec
        return self.spec

    def store_to_xml(self) -> None:
        # Derive save path
        this_script_path = Path(__file__)
        save_dir = this_script_path.parent / "pre_compiled"
        save_path = save_dir / f"{self.name}.xml"
        save_dir.mkdir(parents=True, exist_ok=True)
        xml = self.spec.to_xml()

        # Save file
        with save_path.open("w", encoding="utf-8") as f:
            f.write(xml)

    def load_from_xml(self) -> bool:
        # Derive XML path
        this_script_path = Path(__file__)
        xml_path = this_script_path.parent / "pre_compiled" / f"{self.name}.xml"

        # Check if file exists
        if not xml_path.exists():
            return False

        # Load the XML file
        self.spec = mj.MjSpec.from_file(str(xml_path))
        return True
