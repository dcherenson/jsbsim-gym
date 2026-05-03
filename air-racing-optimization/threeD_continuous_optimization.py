from twoD_continuous_optimization import (
    solve_2d_continuous_optimization,
    terrain_data_zoomed,
    terrain_resolution,
)
from get_task_info import course_info
import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools import units as u
from scipy import interpolate
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
from airplane import airplane  # See cessna152.py for details.
import pyvista as pv
from aerosandbox.numpy import integrate_discrete as nid
from time import perf_counter

total_start = perf_counter()
print("Preparing 2D warm start...", flush=True)
solution_quantities = solve_2d_continuous_optimization()
print("2D warm start ready.", flush=True)

N = solution_quantities["N"]

# F-16 parameters (aligned with this repo's JSBSim model):
# - wing area: aircraft/f16/f16.xml -> 300 ft^2
# - max afterburner thrust: aircraft/f16/Engines/F100-PW-229.xml -> 29000 lbf
F16_WING_AREA = 300 * u.foot ** 2
F16_MAX_THRUST_SL = 29000 * u.lbf
F16_MAX_ROLL_RATE_DPS = 100.0

initial_state = asb.DynamicsPointMass3DSpeedGammaTrack(
    mass_props=asb.MassProperties(mass=0),  # Placeholder
    x_e=course_info["north_start"],
    y_e=course_info["east_start"],
    z_e=-(course_info["start_terrain_elev_m"] + course_info["initial_altitude_agl_m"]),
    speed=800 * u.foot / u.sec,
    gamma=None,
    track=np.radians(course_info["initial_track_deg"]),
    alpha=None,
    beta=None,
    bank=0
)

final_state = asb.DynamicsPointMass3DSpeedGammaTrack(
    mass_props=asb.MassProperties(mass=0),  # Placeholder
    x_e=course_info["north_end"],
    y_e=course_info["east_end"],
    z_e=None,
    speed=None,
    gamma=None,
    track=None,
    alpha=None,
    beta=None,
    bank=None,
)

### Initialize the problem
print("Solving 3D problem...", flush=True)
opti = asb.Opti()

### Define time. Note that the horizon length is unknown.
duration = opti.variable(init_guess=solution_quantities["duration"], lower_bound=0)

time = np.linspace(0, duration, N)
dt = duration / (N - 1)

### Create a dynamics instance

dyn = asb.DynamicsPointMass3DSpeedGammaTrack(
    mass_props=asb.MassProperties(mass=12000),
    x_e=opti.variable(
        init_guess=solution_quantities["north"]
    ),
    y_e=opti.variable(
        init_guess=solution_quantities["east"]
    ),
    z_e=opti.variable(
        init_guess=-solution_quantities["assumed_altitude"] - 100 * u.foot
    ),
    speed=opti.variable(
        init_guess=solution_quantities["assumed_airspeed"],
        n_vars=N,
        lower_bound=10,
    ),
    gamma=opti.variable(
        init_guess=solution_quantities["gamma"],
        n_vars=N,
        lower_bound=np.radians(-45),
        upper_bound=np.radians(45)
    ),
    track=opti.variable(
        init_guess=solution_quantities["track"],
        n_vars=N,
    ),
    alpha=opti.variable(
        init_guess=1,
        n_vars=N,
        lower_bound=-15,
        upper_bound=15
    ),
    beta=np.zeros(N),
    bank=opti.variable(
        init_guess=solution_quantities["bank"],
        lower_bound=np.radians(-150),
        upper_bound=np.radians(150),
    )
)

for state in list(dyn.state.keys()) + ["alpha", "beta", "bank"]:
    if getattr(initial_state, state) is not None:
        print(f"Constraining initial state for '{state}'...")
        if getattr(initial_state, state) == 0:
            opti.subject_to(
                getattr(dyn, state)[0] == getattr(initial_state, state)
            )
        else:
            opti.subject_to(
                getattr(dyn, state)[0] / getattr(initial_state, state) == 1
            )

    if getattr(final_state, state) is not None:
        print(f"Constraining final   state for '{state}'...")
        if getattr(final_state, state) == 0:
            opti.subject_to(
                getattr(dyn, state)[-1] == getattr(final_state, state)
            )
        else:
            opti.subject_to(
                getattr(dyn, state)[-1] / getattr(final_state, state) == 1
            )

# Add some constraints on rate of change of inputs (alpha and bank angle)
pitch_rate = np.diff(dyn.alpha) / dt  # deg/sec
roll_rate = np.diff(np.degrees(dyn.bank)) / dt  # deg/sec
opti.subject_to([
    np.diff(dyn.alpha) / 10 < 1,
    np.diff(dyn.alpha) / 10 > -1,
    np.diff(np.degrees(dyn.bank)) / 90 < 1,
    np.diff(np.degrees(dyn.bank)) / 90 > -1,
    np.diff(np.degrees(dyn.track)) / 60 < 1,
    np.diff(np.degrees(dyn.track)) / 60 > -1,
    roll_rate / F16_MAX_ROLL_RATE_DPS < 1,
    roll_rate / F16_MAX_ROLL_RATE_DPS > -1,
])

### Add in forces
dyn.add_gravity_force(g=9.81)
#
# aero = asb.AeroBuildup(
#     airplane=airplane,
#     op_point=dyn.op_point,
#     model_size="xsmall",
#     include_wave_drag=False,
# ).run()

CL = (2 * np.pi * np.radians(dyn.alpha))  # Very crude model
lift = dyn.op_point.dynamic_pressure() * F16_WING_AREA * CL
accel_G = lift / dyn.mass_props.mass / 9.81
max_thrust = F16_MAX_THRUST_SL * dyn.op_point.atmosphere.density() / 1.225
drag = max_thrust * (dyn.speed / (1.0 * 343)) ** 2  # Very crude model, such that sea-level equilibrium at M1.0
drag *= 1 + 1 * (
    (CL - 0.05)
) ** 2
# Add extra drag from roll rate inputs
roll_rate_drag_multiplier = 4 * 0.2 * (  # You go 20% slower at max roll rate
        np.gradient(np.degrees(dyn.bank), time)  # degrees / sec
        / F16_MAX_ROLL_RATE_DPS
) ** 2
drag *= 1 + roll_rate_drag_multiplier

throttle = opti.variable(
    init_guess=1,
    n_vars=N,
    lower_bound=0,
    upper_bound=1,
)
# throttle = 1
thrust = throttle * max_thrust

dyn.add_force(
    thrust - drag, 0, -lift,
    axes="wind"
)

from terrain_model.precomputed_interpolated_model import get_elevation_interpolated_north_east

terrain_interp_start = perf_counter()
print("Building 3D terrain constraint...", flush=True)
terrain_altitude = get_elevation_interpolated_north_east(
    query_points_north=dyn.x_e,
    query_points_east=dyn.y_e,
    resolution=terrain_resolution,
    terrain_data=terrain_data_zoomed
)
print(
    f"3D terrain constraint ready in {perf_counter() - terrain_interp_start:.1f} s.",
    flush=True,
)
opti.subject_to([
    (dyn.y_e - 1) / 1e4 > terrain_data_zoomed["east_edges"][0] / 1e4,
    (dyn.y_e + 1) / 1e4 < terrain_data_zoomed["east_edges"][-1] / 1e4,
    (dyn.x_e - 1) / 1e4 > terrain_data_zoomed["north_edges"][0] / 1e4,
    (dyn.x_e + 1) / 1e4 < terrain_data_zoomed["north_edges"][-1] / 1e4,
])
altitude_agl = dyn.altitude - terrain_altitude

altitude_agl_limit = 100 * u.foot

opti.subject_to([
    altitude_agl / altitude_agl_limit > 1,
    nid.integrate_discrete_intervals(
        altitude_agl,
        multiply_by_dx=False,
        method="cubic"
    ) / altitude_agl_limit > 1,
])


# goal_direction = np.array([
#     terrain_data_zoomed["east_end"] - terrain_data_zoomed["east_start"],
#     terrain_data_zoomed["north_end"] - terrain_data_zoomed["north_start"]
# ])
# goal_direction /= np.linalg.norm(goal_direction)
#
# # opti.subject_to([
# #     np.dot(
# #         [np.sin(dyn.track), np.cos(dyn.track)],
# #         goal_direction,
# #         manual=True
# #     ) > np.cosd(75)  # Max deviation from goal direction
# # ])


### Add G-force constraints
# accel_G = -aero["F_w"][2] / dyn.mass_props.mass / 9.81
opti.subject_to([
    accel_G < 7,
    accel_G > -0.5
])

### Finalize the problem
dyn.constrain_derivatives(
    opti,
    time,
    method="trapz",
)  # Apply the dynamics constraints created up to this point

wiggliness = np.mean(
    nid.integrate_discrete_squared_curvature(
        dyn.alpha / 5,
        time,
    )
    + nid.integrate_discrete_squared_curvature(
        dyn.bank / np.radians(90),
        time,
    )
    + nid.integrate_discrete_squared_curvature(
        throttle / 1,
        time,
    )
)

opti.minimize(
    (np.maximum(duration, 0) / 240) ** 2
    + 15 * np.mean(dyn.altitude) / 440
    + 1e-3 * wiggliness
    # + 1e-3 * np.mean(dyn.bank ** 2)
)

solve_start = perf_counter()
print("Starting 3D IPOPT solve...", flush=True)
sol = opti.solve(
    behavior_on_failure="return_last",
    max_iter=1000000,
    options={
        "print_time": True,
        "ipopt.print_level": 5,
        "ipopt.hessian_approximation": "limited-memory",
        # "ipopt.mu_strategy": "monotone"
    }
)
print(
    f"3D IPOPT solve finished in {perf_counter() - solve_start:.1f} s.",
    flush=True,
)

dyn.other_fields = {
    "throttle": throttle,
    "thrust": thrust,
    "lift": lift,
    "drag": drag,
    "terrain_altitude": terrain_altitude,
    "altitude_agl": altitude_agl,
    "accel_G": accel_G,
    "wiggliness": wiggliness,
    "time": time,
}

dyn = sol(dyn)
print(
    f"Total 3D pipeline complete in {perf_counter() - total_start:.1f} s.",
    flush=True,
)

if __name__ == '__main__':

    dyn.save(
        "./final_results/dyn.asb"
    )

    fig, ax = plt.subplots(
        figsize=(16, 6)
    )
    plt.plot(
        solution_quantities["east"],
        solution_quantities["north"],
        ":",
        color="lime",
        linewidth=1,
        zorder=4
    )
    plt.plot(
        sol(dyn.y_e),
        sol(dyn.x_e),
        "-",
        color="red",
        linewidth=2,
        zorder=4
    )

    plt.imshow(
        terrain_data_zoomed["elev"],
        cmap='terrain',
        origin="lower",
        extent=(
            terrain_data_zoomed["east_edges"][0],
            terrain_data_zoomed["east_edges"][-1],
            terrain_data_zoomed["north_edges"][0],
            terrain_data_zoomed["north_edges"][-1],
        ),
        alpha=1,
        zorder=2
    )
    p.equal()
    p.show_plot(
        "3D Continuous Optimization",
        rotate_axis_labels=False,
        savefig=[
            # f"./figures/trajectory_{resolution}.svg",
        ]
    )

    plotter = dyn.draw(
        backend="pyvista",
        show=False,
        n_vehicles_to_draw=N // 200,
        scale_vehicle_model=800 / (N / 100),
    )

    grid = pv.RectilinearGrid(
        terrain_data_zoomed["north_edges"],
        terrain_data_zoomed["east_edges"],
    )
    grid["elev"] = terrain_data_zoomed["elev"].T.flatten()
    grid = grid.warp_by_scalar("elev", factor=-1)
    plotter.add_mesh(
        grid.extract_geometry(),
        scalars="elev",
        cmap='terrain',
        specular=0.5,
        specular_power=15,
        smooth_shading=True,
    )
    plotter.enable_terrain_style()
    plotter.show()
