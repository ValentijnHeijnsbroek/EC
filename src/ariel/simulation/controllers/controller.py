"""TODO(jmdm): description of script."""

# Standard library
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Third-party libraries
import mujoco as mj
import numpy as np

# Local libraries
from ariel.utils.tracker import Tracker


@dataclass
class Controller:
    # Function that executes the actual control step
    controller_callback_function: Callable[..., Any]

    # How often to call the controller (for every simulation step)
    time_steps_per_ctrl_step: int = 50  # control frequency

    # How often to save the data
    time_steps_per_save: int = 500  # data-sampling frequency

    # How big a step to take towards the output fot the callback function
    alpha: float = 0.5

    # Maximum commanded angular change per second, in rad/s. A real servo
    # cannot teleport its setpoint, but a position actuator will happily chase
    # a step change with whatever torque it takes. `None` disables the limit.
    # Default is the SER0019 no-load speed (60 deg / 0.18 s).
    max_ctrl_rate: float | None = np.deg2rad(60.0) / 0.18

    # Optional tracker to save data during simulation
    tracker: Tracker = field(default_factory=Tracker)

    def set_control(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        *args: Any | None,
        **kwargs: dict[Any, Any] | None,
    ) -> None:
        """Sets the controller callback function and gives additional arguments.

        Parameters
        ----------
        model : mj.MjModel
            The Mujoco model of the simulation. This will be used to access
            model parameters, such as the number of actuators.
        data : mj.MjData
            The Mujoco data of the simulation. This will be used to access
            simulation variables, such as the current time and control values.
        *args : Any
            Additional arguments to pass to the controller callback function.
        **kwargs : dict[Any, Any]
            Additional keyword arguments to pass to the controller callback function.
        """

        # Calculate current time step
        time = data.time
        time_step = model.opt.timestep
        deduced_time_step = np.ceil(time / time_step)

        # Execute saving only at specific time-steps
        if (deduced_time_step % self.time_steps_per_save) == 0:
            self.tracker.update(data)

        # Execute control strategy only at specific time-steps
        if (deduced_time_step % self.time_steps_per_ctrl_step) == 0:
            # Save the old control values
            old_ctrl = data.ctrl.copy()

            # Execute the custom control function of the user
            output = np.array(
                self.controller_callback_function(
                    model,
                    data,
                    *args,
                    **kwargs,
                ),
            )

            # Calculate the new control values
            new_ctrl = (old_ctrl * (1 - self.alpha)) + (output * self.alpha)

            # Respect the servo's slew rate over the control interval, so a
            # step change in the setpoint cannot demand an impulsive torque.
            if self.max_ctrl_rate is not None:
                max_delta = self.max_ctrl_rate * (
                    self.time_steps_per_ctrl_step * time_step
                )
                new_ctrl = old_ctrl + np.clip(
                    new_ctrl - old_ctrl,
                    -max_delta,
                    max_delta,
                )

            # Ensure that the new control values are within the servo bounds.
            # Read the bounds off the model rather than hard-coding +-pi/2, so
            # that a change to the actuator config cannot silently desync.
            if model.nu and model.actuator_ctrllimited.all():
                lo = model.actuator_ctrlrange[:, 0]
                hi = model.actuator_ctrlrange[:, 1]
            else:
                lo, hi = -np.pi / 2, np.pi / 2
            data.ctrl = np.clip(new_ctrl, lo, hi)

            # Check if there are any NaN values in the control signal
            if np.any(np.isnan(data.ctrl)):
                msg = "NaN values detected in the control signal.\n"
                msg += f"{data.ctrl}"
                raise ValueError(msg)
