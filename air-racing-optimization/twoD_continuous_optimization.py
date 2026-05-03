from pathfinding import terrain_data_zoomed, path as pathfinding_path
from get_task_info import course_info
import aerosandbox as asb
import aerosandbox.numpy as np
from scipy import interpolate, ndimage
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
from time import perf_counter
import os

N = 800


def _get_terrain_resolution():
    raw = os.getenv("AIR_RACING_TERRAIN_RESOLUTION")
    if raw:
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 2:
            raise ValueError(
                "AIR_RACING_TERRAIN_RESOLUTION must be 'north_res,east_res'."
            )
        north_res, east_res = [int(part) for part in parts]
        return north_res, east_res

    # Default to the cropped DEM's native resolution. Upsampling the crop to a
    # much denser grid makes CasADi's bspline build disproportionately slow.
    return tuple(int(v) for v in terrain_data_zoomed["elev"].shape)


terrain_resolution = _get_terrain_resolution()


def solve_2d_continuous_optimization():
    run_start = perf_counter()
    discrete_path = np.copy(pathfinding_path)
    native_terrain_resolution = tuple(int(v) for v in terrain_data_zoomed["elev"].shape)

    ds_path = (
        np.diff(discrete_path[:, 0]) ** 2 +
        np.diff(discrete_path[:, 1]) ** 2
    ) ** 0.5
    s_path = np.concatenate([
        [0],
        np.cumsum(ds_path)
    ])
    path_downsampled = interpolate.interp1d(
        s_path,
        discrete_path,
        kind="cubic",
        axis=0
    )(np.linspace(0, s_path[-1], N))

    path_downsampled = ndimage.gaussian_filter(
        path_downsampled,
        sigma=1,
        axes=[0],
    )

    path_downsampled_length = np.sum(
        np.sum(np.diff(path_downsampled, axis=0) ** 2, axis=1) ** 0.5
    )

    assumed_airspeed = 0.8 * 343

    print(
        (
            f"Solving 2D problem with N={N}, "
            f"terrain_resolution={terrain_resolution}, "
            f"native_crop_resolution={native_terrain_resolution}..."
        ),
        flush=True,
    )
    opti = asb.Opti()

    straight_line_distance = (
        (terrain_data_zoomed["east_start"] - terrain_data_zoomed["east_end"]) ** 2 +
        (terrain_data_zoomed["north_start"] - terrain_data_zoomed["north_end"]) ** 2
    ) ** 0.5

    east = opti.variable(
        init_guess=path_downsampled[:, 0],
    )
    north = opti.variable(
        init_guess=path_downsampled[:, 1],
    )

    ds = opti.variable(
        init_guess=path_downsampled_length / N,
        lower_bound=straight_line_distance / N * 0.9,
    )
    dt = ds / assumed_airspeed
    duration = dt * N

    opti.subject_to([
        east[0] == terrain_data_zoomed["east_start"],
        north[0] == terrain_data_zoomed["north_start"],
        east[-1] == terrain_data_zoomed["east_end"],
        north[-1] == terrain_data_zoomed["north_end"],
    ])
    opti.subject_to(
        (
            np.diff(east) ** 2 +
            np.diff(north) ** 2
        ) / ds ** 2 == 1
    )

    from terrain_model.precomputed_interpolated_model import get_elevation_interpolated_north_east

    terrain_interp_start = perf_counter()
    print("Building 2D terrain constraint...", flush=True)
    terrain_altitude = get_elevation_interpolated_north_east(
        query_points_north=north,
        query_points_east=east,
        resolution=terrain_resolution,
        terrain_data=terrain_data_zoomed,
    )
    print(
        f"2D terrain constraint ready in {perf_counter() - terrain_interp_start:.1f} s.",
        flush=True,
    )
    opti.subject_to([
        (east - 1) / 1e4 > terrain_data_zoomed["east_edges"][0] / 1e4,
        (east + 1) / 1e4 < terrain_data_zoomed["east_edges"][-1] / 1e4,
        (north - 1) / 1e4 > terrain_data_zoomed["north_edges"][0] / 1e4,
        (north + 1) / 1e4 < terrain_data_zoomed["north_edges"][-1] / 1e4,
    ])

    goal_direction = np.array([
        terrain_data_zoomed["east_end"] - terrain_data_zoomed["east_start"],
        terrain_data_zoomed["north_end"] - terrain_data_zoomed["north_start"]
    ])
    goal_direction /= np.linalg.norm(goal_direction)

    opti.subject_to([
        np.dot(
            [np.diff(east), np.diff(north)],
            goal_direction,
            manual=True
        ) / ds > np.cosd(75)  # Max deviation from goal direction
    ])

    allowed_G = 7
    allowable_discrete_track_change = (
        (allowed_G * 9.81) / assumed_airspeed * dt
    )

    de = np.diff(east)
    dn = np.diff(north)

    opti.subject_to(
        (de[1:] * de[:-1] + dn[1:] * dn[:-1]) / ds ** 2 > np.cos(allowable_discrete_track_change)
    )

    track = np.arctan2(
        np.diff(east),
        np.diff(north),
    )
    track_0 = course_info["initial_track_deg"]
    opti.subject_to([
        (np.diff(east)[0] * np.sind(track_0) + np.diff(north)[0] * np.cosd(track_0)) / ds > 0.99,
    ])
    from aerosandbox.numpy import integrate_discrete as nid

    wiggliness = np.mean(
        nid.integrate_discrete_squared_curvature(north)
        + nid.integrate_discrete_squared_curvature(east)
    )

    opti.minimize(
        (np.maximum(duration, 0) / 240) ** 4
        + 15 * np.mean(terrain_altitude / 440)
        + 1e-3 * wiggliness
    )

    solve_start = perf_counter()
    print("Starting 2D IPOPT solve...", flush=True)
    sol = opti.solve(
        behavior_on_failure="return_last",
        max_iter=1000,
        options={
            "print_time": True,
            "ipopt.print_level": 5,
        }
    )
    print(
        f"2D IPOPT solve finished in {perf_counter() - solve_start:.1f} s.",
        flush=True,
    )

    solution_quantities = {
        "N"               : N,
        "east"            : sol(east),
        "north"           : sol(north),
        "ds"              : sol(ds),  # Valid since all are constrained to be equal
        "dt"              : sol(dt),  # Valid since all are constrained to be equal
        "assumed_airspeed": assumed_airspeed,
        "duration"        : sol(duration),
        "terrain_altitude": sol(terrain_altitude),
        "discrete_path"   : discrete_path,
    }
    solution_quantities["path"] = np.stack(
        [
            solution_quantities["east"],
            solution_quantities["north"]
        ],
        axis=1
    )
    solution_quantities["assumed_altitude"] = ndimage.gaussian_filter(
        solution_quantities["terrain_altitude"],
        2,
        mode="nearest"
    )

    solution_quantities["track"] = np.arctan2(
        np.gradient(solution_quantities["east"]),
        np.gradient(solution_quantities["north"]),
    )

    solution_quantities["gamma"] = np.arctan2(
        np.gradient(solution_quantities["assumed_altitude"]),
        solution_quantities["ds"],
    )
    solution_quantities["load_factor_vertical"] = 1 + (
        solution_quantities["assumed_airspeed"]
        * np.gradient(solution_quantities["gamma"])
        / solution_quantities["dt"]
    ) / 9.81
    solution_quantities["load_factor_horizontal"] = (
        solution_quantities["assumed_airspeed"]
        * np.gradient(solution_quantities["track"])
        / solution_quantities["dt"]
    ) / 9.81
    solution_quantities["bank"] = np.arctan2(
        solution_quantities["load_factor_horizontal"],
        solution_quantities["load_factor_vertical"],
    )

    print(
        f"2D warm start complete in {perf_counter() - run_start:.1f} s.",
        flush=True,
    )

    return solution_quantities


if __name__ == '__main__':
    solution_quantities = solve_2d_continuous_optimization()

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )
    plt.plot(
        solution_quantities["path"][:, 0],
        solution_quantities["path"][:, 1],
        "-",
        color="red",
        linewidth=3,
        zorder=4
    )
    plt.plot(
        solution_quantities["discrete_path"][:, 0],
        solution_quantities["discrete_path"][:, 1],
        ":",
        color="lime",
        linewidth=1,
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

    from matplotlib import patheffects

    plt.annotate(
        "Start",
        xy=(terrain_data_zoomed["east_start"], terrain_data_zoomed["north_start"]),
        xytext=(0, 0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color="k",
        path_effects=[
            patheffects.withStroke(
                linewidth=3,
                foreground="w",
            )
        ]
    )
    plt.annotate(
        "Finish",
        xy=(terrain_data_zoomed["east_end"], terrain_data_zoomed["north_end"]),
        xytext=(0, 0),
        textcoords="offset points",
        ha="right",
        va="top",
        color="k",
        path_effects=[
            patheffects.withStroke(
                linewidth=3,
                foreground="w",
            )
        ]
    )


    p.equal()
    p.show_plot(
        "2D Continuous Optimization",
        rotate_axis_labels=False,
        savefig=[
            f"./figures/2D_trajectory.svg",
        ]
    )
