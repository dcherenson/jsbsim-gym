import gymnasium as gym
import numpy as np
import pygame as pg
from gymnasium.envs.registration import registry

from jsbsim_gym.canyon import DEMCanyon, ProceduralCanyon
from jsbsim_gym.data_collection_env import DataCollectionEnv
from jsbsim_gym.visualization.quaternion import Quaternion
from jsbsim_gym.visualization.rendering import Grid, RenderObject, Viewer, load_heightfield_mesh, load_mesh

M_TO_FT = 3.28084
FT_TO_M = 1.0 / M_TO_FT

OBS_P_N_FT = 0
OBS_P_E_FT = 1
OBS_H_FT = 2
OBS_U_FPS = 3
OBS_V_FPS = 4
OBS_W_FPS = 5
OBS_ALPHA = 6
OBS_BETA = 7
OBS_PHI = 8
OBS_THETA = 9
OBS_PSI = 10
OBS_P = 11
OBS_Q = 12
OBS_R = 13
OBS_CANYON_HALF_WIDTH_FT = 14
OBS_CANYON_WIDTH_GRAD = 15
OBS_LATERAL_NORM = 16
OBS_ALTITUDE_ERROR_FT = 17


class CanyonFlightEnv(DataCollectionEnv):
    """F-16 canyon-following environment with variable-width canyon geometry."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        canyon_mode="procedural",
        base_width_ft=550.0,
        amplitude_ft=220.0,
        width_freq=0.0012,
        dem_path=None,
        dem_bbox=None,
        dem_valley_rel_elev=0.30,
        dem_smoothing_window=31,
        dem_min_width_ft=120.0,
        dem_max_width_ft=5000.0,
        dem_fly_direction="south_to_north",
        dem_start_pixel=None,
        dem_start_heading_deg=None,
        dem_start_heading_mode="keep_initial",
        dem_render_mesh=True,
        dem_render_stride=4,
        dem_render_vertical_exaggeration=1.0,
        dem_render_half_width_ft=None,
        dem_render_padding_ft=6000.0,
        dem_use_proxy_canyon_bounds=False,
        wall_margin_ft=35.0,
        wall_visual_offset_ft=80.0,
        wall_radius_ft=10.0,
        wall_height_ft=500.0,
        target_altitude_ft=2800.0,
        entry_altitude_ft=None,
        min_altitude_ft=400.0,
        max_altitude_ft=10000.0,
        max_episode_steps=1200,
        terrain_collision_buffer_ft=20.0,
        wind_sigma=4.0,
        canyon_span_ft=14000.0,
        canyon_segment_spacing_ft=120.0,
        entry_speed_kts=450.0,
        entry_roll_deg=None,
        entry_pitch_deg=None,
        entry_alpha_deg=None,
        entry_beta_deg=None,
    ):
        super().__init__(render_mode=render_mode)

        self.canyon_mode = canyon_mode

        if canyon_mode == "procedural":
            self.canyon = ProceduralCanyon(
                base_width=base_width_ft,
                amplitude=amplitude_ft,
                freq=width_freq,
            )
        elif canyon_mode == "dem":
            if dem_path is None:
                raise ValueError("canyon_mode='dem' requires dem_path")
            if dem_bbox is None or len(dem_bbox) != 4:
                raise ValueError("canyon_mode='dem' requires dem_bbox=(south, north, west, east)")

            south, north, west, east = dem_bbox
            self.canyon = DEMCanyon(
                dem_path=dem_path,
                south=float(south),
                north=float(north),
                west=float(west),
                east=float(east),
                valley_rel_elev=dem_valley_rel_elev,
                smoothing_window=dem_smoothing_window,
                min_width_ft=dem_min_width_ft,
                max_width_ft=dem_max_width_ft,
                fly_direction=dem_fly_direction,
                dem_start_pixel=dem_start_pixel,
            )
        else:
            raise ValueError("canyon_mode must be 'procedural' or 'dem'")

        self.wall_margin_ft = wall_margin_ft
        self.wall_visual_offset_ft = wall_visual_offset_ft
        self.wall_radius_ft = wall_radius_ft
        self.wall_height_ft = wall_height_ft
        self.requested_dem_start_pixel = dem_start_pixel
        self.dem_start_pixel = dem_start_pixel
        self.dem_start_heading_mode = dem_start_heading_mode
        self.dem_start_heading_deg = dem_start_heading_deg
        self.dem_render_mesh = bool(dem_render_mesh)
        self.dem_render_stride = max(int(dem_render_stride), 1)
        self.dem_render_vertical_exaggeration = float(dem_render_vertical_exaggeration)
        self.dem_render_half_width_ft = dem_render_half_width_ft
        self.dem_render_padding_ft = float(dem_render_padding_ft)
        self.dem_use_proxy_canyon_bounds = bool(dem_use_proxy_canyon_bounds)

        self.dem_start_info = None
        self.dem_render_base_elev_ft = 0.0
        self.dem_start_elev_ft = 0.0
        if canyon_mode == "dem":
            self.dem_render_base_elev_ft = float(getattr(self.canyon, "surface_ref_m", 0.0) * M_TO_FT)
            if hasattr(self.canyon, "get_pixel_info"):
                if dem_start_pixel is not None:
                    px, py = dem_start_pixel
                    # Honor an explicitly requested DEM start pixel exactly.
                    self.dem_start_info = self.canyon.get_pixel_info(px, py)
                else:
                    px = self.canyon.cols - 1
                    py = self.canyon.rows - 1 if dem_fly_direction == "south_to_north" else 0
                    if hasattr(self.canyon, "get_centerline_pixel_info"):
                        self.dem_start_info = self.canyon.get_centerline_pixel_info(px, py)
                    else:
                        self.dem_start_info = self.canyon.get_pixel_info(px, py)
                self.dem_start_pixel = (
                    int(self.dem_start_info["pixel_x"]),
                    int(self.dem_start_info["pixel_y"]),
                )
                self.dem_start_elev_ft = float(self.dem_start_info["elevation_msl_ft"])

                if self.dem_start_heading_deg is None:
                    heading_mode = str(self.dem_start_heading_mode).lower()
                    if heading_mode == "toward_center":
                        target_local_north_ft = 0.5 * float(self.canyon.get_total_length_ft())
                        target_local_east_ft = 0.0
                        dn = target_local_north_ft - float(self.dem_start_info["local_north_ft"])
                        de = target_local_east_ft - float(self.dem_start_info["local_east_ft"])
                        if abs(dn) + abs(de) > 1e-6:
                            self.dem_start_heading_deg = float(np.degrees(np.arctan2(de, dn)))
                        else:
                            self.dem_start_heading_deg = 0.0
                    elif heading_mode == "follow_canyon" and hasattr(self.canyon, "get_heading_for_pixel"):
                        self.dem_start_heading_deg = float(
                            self.canyon.get_heading_for_pixel(
                                self.dem_start_pixel[0],
                                self.dem_start_pixel[1],
                            )
                        )

        altitude_ref_ft = self.dem_start_elev_ft if canyon_mode == "dem" else 0.0
        self.target_altitude_ft = altitude_ref_ft + target_altitude_ft
        self.entry_altitude_ft = altitude_ref_ft + (
            target_altitude_ft if entry_altitude_ft is None else entry_altitude_ft
        )
        self.min_altitude_ft = altitude_ref_ft + min_altitude_ft
        self.max_altitude_ft = altitude_ref_ft + max_altitude_ft
        self.max_episode_steps = max_episode_steps
        self.terrain_collision_buffer_ft = float(terrain_collision_buffer_ft)

        self.wind_sigma = wind_sigma

        if canyon_mode == "dem" and hasattr(self.canyon, "get_total_length_ft"):
            self.canyon_span_ft = min(float(canyon_span_ft), float(self.canyon.get_total_length_ft()))
        else:
            self.canyon_span_ft = canyon_span_ft
        self.canyon_segment_spacing_ft = canyon_segment_spacing_ft
        self.entry_speed_fps = entry_speed_kts * 1.68781
        self.entry_roll_deg = None if entry_roll_deg is None else float(entry_roll_deg)
        self.entry_pitch_deg = None if entry_pitch_deg is None else float(entry_pitch_deg)
        self.entry_alpha_deg = None if entry_alpha_deg is None else float(entry_alpha_deg)
        self.entry_beta_deg = None if entry_beta_deg is None else float(entry_beta_deg)
        self.canyon_segment_count = max(int(self.canyon_span_ft / canyon_segment_spacing_ft) + 1, 8)

        self.step_count = 0
        self.last_p_n_ft = 0.0
        self.start_p_n_ft = 0.0
        self.start_p_e_ft = 0.0
        self.last_distance_from_start_ft = 0.0
        self.canyon_anchor_north_ft = None
        self.canyon_center_east_ft = None
        self.terrain_anchor_north_ft = None
        self.terrain_origin_east_ft = None

        self.wall_left_objects = []
        self.wall_right_objects = []
        self.terrain = None
        self.f16 = None
        self.f16_visual_scale = None
        self.camera_pos = None
        self.camera_look_at = None
        self.hud_font = None
        self.hud_small_font = None
        self.hud_heading_cmd_deg = None
        self.hud_mode_labels = ("AUTO", "TRACK", "CMD")

        self.observation_space = gym.spaces.Box(
            low=np.full((18,), -np.inf, dtype=np.float32),
            high=np.full((18,), np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    def _build_obs(self, state_dict):
        half_width_ft = 0.5 * state_dict["canyon_width"]
        usable_half_ft = max(half_width_ft - self.wall_margin_ft, 1.0)
        lateral_error_ft = self._get_lateral_error_ft(state_dict)
        lateral_norm = lateral_error_ft / usable_half_ft
        altitude_error_ft = state_dict["h"] - self.target_altitude_ft

        return np.array(
            [
                state_dict["p_N"],
                state_dict["p_E"],
                state_dict["h"],
                state_dict["u"],
                state_dict["v"],
                state_dict["w"],
                state_dict["alpha"],
                state_dict["beta"],
                state_dict["phi"],
                state_dict["theta"],
                state_dict["psi"],
                state_dict["p"],
                state_dict["q"],
                state_dict["r"],
                half_width_ft,
                state_dict["canyon_width_grad"],
                lateral_norm,
                altitude_error_ft,
            ],
            dtype=np.float32,
        )

    def _build_info(self, state_dict):
        half_width_ft = 0.5 * state_dict["canyon_width"]
        usable_half_ft = max(half_width_ft - self.wall_margin_ft, 1.0)
        lateral_error_ft = self._get_lateral_error_ft(state_dict)
        lateral_norm = lateral_error_ft / usable_half_ft
        terrain_elevation_msl_ft, terrain_clearance_ft = self._get_terrain_elevation_ft_and_clearance(state_dict)

        return {
            "canyon_width_ft": float(state_dict["canyon_width"]),
            "canyon_half_width_ft": float(half_width_ft),
            "canyon_width_grad": float(state_dict["canyon_width_grad"]),
            "lateral_error_ft": float(lateral_error_ft),
            "lateral_error_norm": float(lateral_norm),
            "altitude_error_ft": float(state_dict["h"] - self.target_altitude_ft),
            "terrain_elevation_msl_ft": float(terrain_elevation_msl_ft),
            "terrain_clearance_ft": float(terrain_clearance_ft),
        }

    def _get_terrain_elevation_ft_and_clearance(self, state_dict):
        terrain_elevation_msl_ft = 0.0
        if self.canyon_mode == "dem" and hasattr(self.canyon, "get_elevation_msl_ft_from_latlon"):
            lat_deg = float(self.simulation.get_property_value("position/lat-gc-deg"))
            lon_deg = float(self.simulation.get_property_value("position/long-gc-deg"))
            terrain_elevation_msl_ft = float(self.canyon.get_elevation_msl_ft_from_latlon(lat_deg, lon_deg))

        terrain_clearance_ft = float(state_dict["h"] - terrain_elevation_msl_ft)
        return terrain_elevation_msl_ft, terrain_clearance_ft

    def _get_lateral_error_ft(self, state_dict):
        if (
            self.canyon_mode == "dem"
            and hasattr(self.canyon, "get_local_from_latlon")
            and hasattr(self.canyon, "north_samples_ft")
            and hasattr(self.canyon, "center_east_samples_ft")
        ):
            try:
                lat_deg = float(self.simulation.get_property_value("position/lat-gc-deg"))
                lon_deg = float(self.simulation.get_property_value("position/long-gc-deg"))
                local_north_ft, local_east_ft = self.canyon.get_local_from_latlon(lat_deg, lon_deg)
                center_east_ft = float(
                    np.interp(local_north_ft, self.canyon.north_samples_ft, self.canyon.center_east_samples_ft)
                )
                return float(local_east_ft - center_east_ft)
            except Exception:
                pass
        return float(state_dict["p_E"] - (self.canyon_center_east_ft or 0.0))

    def reset(self, seed=None, options=None):
        # Match JSBSim initial conditions to this canyon scenario.
        if self.canyon_mode == "dem" and self.dem_start_info is not None:
            self.simulation.set_property_value('ic/lat-gc-deg', float(self.dem_start_info["lat_deg"]))
            self.simulation.set_property_value('ic/long-gc-deg', float(self.dem_start_info["lon_deg"]))
            if self.dem_start_heading_deg is not None:
                self.simulation.set_property_value('ic/psi-true-deg', float(self.dem_start_heading_deg))
        if self.entry_roll_deg is not None:
            self.simulation.set_property_value('ic/phi-deg', float(self.entry_roll_deg))
        if self.entry_pitch_deg is not None:
            self.simulation.set_property_value('ic/theta-deg', float(self.entry_pitch_deg))
        if self.entry_alpha_deg is not None:
            self.simulation.set_property_value('ic/alpha-deg', float(self.entry_alpha_deg))
        if self.entry_beta_deg is not None:
            self.simulation.set_property_value('ic/beta-deg', float(self.entry_beta_deg))
        self.simulation.set_property_value('ic/h-sl-ft', float(self.entry_altitude_ft))
        self.simulation.set_property_value('ic/vt-fps', float(self.entry_speed_fps))
        super().reset(seed=seed, options=options)
        self.step_count = 0
        self.camera_pos = None
        self.camera_look_at = None
        self.hud_heading_cmd_deg = None
        self.hud_mode_labels = ("AUTO", "TRACK", "CMD")

        state = self.get_full_state_dict()
        self.last_p_n_ft = state["p_N"]
        self.start_p_n_ft = state["p_N"]
        self.start_p_e_ft = state["p_E"]
        self.last_distance_from_start_ft = 0.0

        if self.canyon_mode == "dem" and self.dem_start_info is not None:
            self.canyon_anchor_north_ft = state["p_N"] - float(self.dem_start_info["local_north_ft"])
            self.canyon_center_east_ft = state["p_E"]
            self.terrain_anchor_north_ft = self.canyon_anchor_north_ft
            self.terrain_origin_east_ft = state["p_E"] + float(self.dem_start_info["local_east_ft"])
        else:
            self.canyon_anchor_north_ft = state["p_N"]
            self.canyon_center_east_ft = state["p_E"]
            self.terrain_anchor_north_ft = self.canyon_anchor_north_ft
            self.terrain_origin_east_ft = self.canyon_center_east_ft

        if hasattr(self.canyon, "set_anchor_north"):
            self.canyon.set_anchor_north(self.canyon_anchor_north_ft)

        return self._build_obs(state), self._build_info(state)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        _, state_next, ground_done = self.step_collect(action)
        self.step_count += 1

        half_width_ft = 0.5 * state_next["canyon_width"]
        usable_half_ft = max(half_width_ft - self.wall_margin_ft, 1.0)
        lateral_error_ft = self._get_lateral_error_ft(state_next)
        _, terrain_clearance_ft = self._get_terrain_elevation_ft_and_clearance(state_next)

        out_of_canyon = np.abs(lateral_error_ft) > usable_half_ft
        if self.canyon_mode == "dem" and self.dem_render_mesh and not self.dem_use_proxy_canyon_bounds:
            out_of_canyon = False
        out_of_altitude = (state_next["h"] < self.min_altitude_ft) or (state_next["h"] > self.max_altitude_ft)
        terrain_collision = terrain_clearance_ft <= self.terrain_collision_buffer_ft

        terminated = bool(ground_done or terrain_collision or out_of_canyon or out_of_altitude)
        truncated = bool(self.step_count >= self.max_episode_steps and not terminated)

        progress_ft = state_next["p_N"] - self.last_p_n_ft
        self.last_p_n_ft = state_next["p_N"]

        distance_from_start_ft = float(
            np.hypot(state_next["p_N"] - self.start_p_n_ft, state_next["p_E"] - self.start_p_e_ft)
        )
        progress_from_start_ft = distance_from_start_ft - self.last_distance_from_start_ft
        self.last_distance_from_start_ft = distance_from_start_ft

        progress_reward = np.clip(progress_ft / 120.0, -5.0, 5.0)
        center_penalty = np.abs(lateral_error_ft) / max(usable_half_ft, 1.0)
        altitude_penalty = np.abs(state_next["h"] - self.target_altitude_ft) / max(self.target_altitude_ft, 1.0)

        reward = progress_reward - 0.7 * center_penalty - 0.2 * altitude_penalty
        if terrain_collision:
            reward -= 150.0
        elif terminated:
            reward -= 100.0

        info = self._build_info(state_next)
        info["progress_ft"] = float(progress_ft)
        info["distance_from_start_ft"] = float(distance_from_start_ft)
        info["progress_from_start_ft"] = float(progress_from_start_ft)
        info["terrain_collision"] = bool(terrain_collision)
        if terrain_collision:
            info["terrain_clearance_ft"] = float(terrain_clearance_ft)
            info["terrain_collision_buffer_ft"] = float(self.terrain_collision_buffer_ft)
        if terrain_collision:
            info["termination_reason"] = "terrain_collision"
        elif out_of_canyon:
            info["termination_reason"] = "hit_canyon_wall"
        elif out_of_altitude:
            info["termination_reason"] = "altitude_out_of_bounds"
        elif ground_done:
            info["termination_reason"] = "ground_collision"
        elif truncated:
            info["termination_reason"] = "time_limit"
        else:
            info["termination_reason"] = "running"

        return self._build_obs(state_next), float(reward), terminated, truncated, info

    def _initialize_viewer(self, scale):
        self.viewer = Viewer(1280, 720, headless=(self.render_mode == "rgb_array"))

        f16_mesh = load_mesh(self.viewer.ctx, self.viewer.prog, "f16.obj")
        self.f16 = RenderObject(f16_mesh)
        # JSBSim state positions are converted with `scale`, so the aircraft
        # mesh should use the same world scale to keep physical proportions.
        self.f16_visual_scale = 3.0*scale
        self.f16.transform.scale = self.f16_visual_scale
        self.f16.color = 0.0, 0.0, 0.4

        self.terrain = None

        if (
            self.canyon_mode == "dem"
            and self.dem_render_mesh
            and hasattr(self.canyon, "get_surface_grid")
        ):
            self.terrain = self._build_dem_terrain_object(scale)
            self.viewer.objects.append(self.terrain)
        else:
            wall_mesh = load_mesh(self.viewer.ctx, self.viewer.unlit, "cylinder.obj")
            wall_radius_m = self.wall_radius_ft * FT_TO_M
            wall_half_height_m = 0.5 * self.wall_height_ft * FT_TO_M
            wall_scale = np.array(
                [wall_radius_m * scale, wall_half_height_m * scale, wall_radius_m * scale],
                dtype=np.float32,
            )

            self.wall_left_objects = []
            self.wall_right_objects = []

            for _ in range(self.canyon_segment_count):
                left = RenderObject(wall_mesh)
                right = RenderObject(wall_mesh)

                left.transform.scale = wall_scale.copy()
                right.transform.scale = wall_scale.copy()

                left.color = 0.62, 0.44, 0.28
                right.color = 0.62, 0.44, 0.28

                self.wall_left_objects.append(left)
                self.wall_right_objects.append(right)

                self.viewer.objects.append(left)
                self.viewer.objects.append(right)

        self.viewer.objects.append(self.f16)
        self.viewer.objects.append(Grid(self.viewer.ctx, self.viewer.unlit, 25, 1.0))

    def _build_dem_terrain_object(self, scale):
        elevation_m, north_ft, east_ft = self.canyon.get_surface_grid()

        if self.dem_render_half_width_ft is not None:
            half_width_ft = float(self.dem_render_half_width_ft)
            if half_width_ft > 0.0:
                col_mask = np.abs(east_ft) <= half_width_ft
                if np.count_nonzero(col_mask) >= 3:
                    elevation_m = elevation_m[:, col_mask]
                    east_ft = east_ft[col_mask]

        if self.dem_render_padding_ft > 0.0 and elevation_m.shape[0] > 1 and elevation_m.shape[1] > 1:
            row_spacing_ft = float(np.mean(np.diff(north_ft)))
            col_spacing_ft = float(np.mean(np.diff(east_ft)))

            pad_rows = max(int(round(self.dem_render_padding_ft / max(row_spacing_ft, 1e-6))), 0)
            pad_cols = max(int(round(self.dem_render_padding_ft / max(col_spacing_ft, 1e-6))), 0)

            if pad_rows > 0 or pad_cols > 0:
                elevation_m = np.pad(elevation_m, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode="reflect")

                if pad_rows > 0:
                    north_prefix = north_ft[0] - row_spacing_ft * np.arange(pad_rows, 0, -1, dtype=np.float32)
                    north_suffix = north_ft[-1] + row_spacing_ft * np.arange(1, pad_rows + 1, dtype=np.float32)
                    north_ft = np.concatenate([north_prefix, north_ft, north_suffix]).astype(np.float32)

                if pad_cols > 0:
                    east_prefix = east_ft[0] - col_spacing_ft * np.arange(pad_cols, 0, -1, dtype=np.float32)
                    east_suffix = east_ft[-1] + col_spacing_ft * np.arange(1, pad_cols + 1, dtype=np.float32)
                    east_ft = np.concatenate([east_prefix, east_ft, east_suffix]).astype(np.float32)

        stride = self.dem_render_stride
        elevation_m = elevation_m[::stride, ::stride]
        north_ft = north_ft[::stride]
        east_ft = east_ft[::stride]

        # Keep DEM east axis aligned with the existing aircraft/world convention
        # where positive east maps to negative render x.
        x_coords_m = -east_ft.astype(np.float32) * FT_TO_M
        z_coords_m = north_ft.astype(np.float32) * FT_TO_M
        heights_m = elevation_m.astype(np.float32) * self.dem_render_vertical_exaggeration

        terrain_mesh = load_heightfield_mesh(
            self.viewer.ctx,
            self.viewer.terrain_prog,
            x_coords=x_coords_m,
            heights=heights_m,
            z_coords=z_coords_m,
            vertex_colors=self._terrain_colormap(heights_m),
        )
        terrain = RenderObject(terrain_mesh)
        terrain.transform.scale = scale
        terrain.color = 1.0, 1.0, 1.0
        return terrain

    def _terrain_colormap(self, heights_m):
        h_min = float(np.min(heights_m))
        h_max = float(np.max(heights_m))
        if h_max - h_min <= 1e-6:
            h_norm = np.zeros_like(heights_m, dtype=np.float32)
        else:
            h_norm = ((heights_m - h_min) / (h_max - h_min)).astype(np.float32)

        stops = np.array([0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.0], dtype=np.float32)
        colors = np.array(
            [
                [0.16, 0.30, 0.14],
                [0.26, 0.44, 0.20],
                [0.43, 0.55, 0.28],
                [0.65, 0.63, 0.40],
                [0.70, 0.55, 0.38],
                [0.56, 0.44, 0.34],
                [0.90, 0.89, 0.85],
            ],
            dtype=np.float32,
        )

        out = np.empty(h_norm.shape + (3,), dtype=np.float32)
        for ch in range(3):
            out[..., ch] = np.interp(h_norm, stops, colors[:, ch])
        return out

    def _update_canyon_wall_positions(self, scale):
        if self.canyon_anchor_north_ft is None:
            self.canyon_anchor_north_ft = float(self.state[0] * M_TO_FT)
        if self.canyon_center_east_ft is None:
            self.canyon_center_east_ft = float(self.state[1] * M_TO_FT)

        if self.terrain is not None:
            if self.canyon_mode == "dem" and hasattr(self.canyon, "get_local_from_latlon"):
                # DEM mesh vertices are already in DEM-local meters.
                self.terrain.transform.z = 0.0
                self.terrain.transform.x = 0.0
                self.terrain.transform.y = 0.0
            else:
                anchor_north_ft = (
                    self.terrain_anchor_north_ft
                    if self.terrain_anchor_north_ft is not None
                    else self.canyon_anchor_north_ft
                )
                origin_east_ft = (
                    self.terrain_origin_east_ft
                    if self.terrain_origin_east_ft is not None
                    else self.canyon_center_east_ft
                )
                self.terrain.transform.z = anchor_north_ft * FT_TO_M * scale
                self.terrain.transform.x = -origin_east_ft * FT_TO_M * scale
                self.terrain.transform.y = 0.0

        if not self.wall_left_objects or not self.wall_right_objects:
            return

        # Canyon geometry is fixed in the world frame. The aircraft starts at
        # this anchored entrance and flies through the static canyon.
        start_north_ft = self.canyon_anchor_north_ft
        wall_radius_scaled = self.wall_radius_ft * FT_TO_M * scale
        base_half_height_scaled = 0.5 * self.wall_height_ft * FT_TO_M * scale
        min_half_height_scaled = 0.5 * base_half_height_scaled
        max_half_height_scaled = 3.0 * base_half_height_scaled

        for idx in range(self.canyon_segment_count):
            north_ft = start_north_ft + idx * self.canyon_segment_spacing_ft

            if self.canyon_mode == "dem" and hasattr(self.canyon, "get_wall_profile"):
                left_half_ft, right_half_ft, wall_height_ft = self.canyon.get_wall_profile(north_ft)
                left_half_ft = max(left_half_ft + self.wall_visual_offset_ft, self.wall_radius_ft + 5.0)
                right_half_ft = max(right_half_ft + self.wall_visual_offset_ft, self.wall_radius_ft + 5.0)
                wall_half_height_scaled = np.clip(
                    0.5 * wall_height_ft * FT_TO_M * scale,
                    min_half_height_scaled,
                    max_half_height_scaled,
                )
            else:
                width_ft, _ = self.canyon.get_geometry(north_ft)
                half_width_ft = max(width_ft * 0.5 + self.wall_visual_offset_ft, self.wall_radius_ft + 5.0)
                left_half_ft = half_width_ft
                right_half_ft = half_width_ft
                wall_half_height_scaled = base_half_height_scaled

            north_m = north_ft * FT_TO_M
            center_east_ft = self.canyon_center_east_ft
            left_east_m = (center_east_ft - left_half_ft) * FT_TO_M
            right_east_m = (center_east_ft + right_half_ft) * FT_TO_M

            left = self.wall_left_objects[idx]
            right = self.wall_right_objects[idx]

            left.transform.scale[0] = wall_radius_scaled
            left.transform.scale[1] = wall_half_height_scaled
            left.transform.scale[2] = wall_radius_scaled

            right.transform.scale[0] = wall_radius_scaled
            right.transform.scale[1] = wall_half_height_scaled
            right.transform.scale[2] = wall_radius_scaled

            left.transform.z = north_m * scale
            left.transform.x = -left_east_m * scale
            left.transform.y = wall_half_height_scaled

            right.transform.z = north_m * scale
            right.transform.x = -right_east_m * scale
            right.transform.y = wall_half_height_scaled

    def _get_hud_fonts(self):
        if (
            self.hud_font is not None
            and self.hud_small_font is not None
            and getattr(self, "hud_title_font", None) is not None
            and getattr(self, "hud_micro_font", None) is not None
            and getattr(self, "hud_tiny_font", None) is not None
            and getattr(self, "hud_mono_font", None) is not None
        ):
            return

        if not pg.get_init():
            pg.init()
        if not pg.font.get_init():
            pg.font.init()

        self.hud_font = pg.font.Font(None, 30)
        self.hud_small_font = pg.font.Font(None, 22)
        self.hud_title_font = pg.font.Font(None, 36)
        self.hud_micro_font = pg.font.Font(None, 18)
        self.hud_tiny_font = pg.font.Font(None, 14)
        self.hud_mono_font = pg.font.Font(None, 28)

    def set_hud_commands(self, heading_cmd_deg=None, mode_labels=None):
        if heading_cmd_deg is None or not np.isfinite(heading_cmd_deg):
            self.hud_heading_cmd_deg = None
        else:
            self.hud_heading_cmd_deg = float(heading_cmd_deg) % 360.0
        if mode_labels is not None:
            labels = [str(label).upper()[:12] for label in mode_labels if str(label).strip()]
            if labels:
                while len(labels) < 3:
                    labels.append("")
                self.hud_mode_labels = tuple(labels[:3])

    def _draw_hud(self, frame):
        if frame is None:
            return frame

        self._get_hud_fonts()

        surface = pg.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        width, height = surface.get_size()

        state_dict = self.get_full_state_dict()
        heading_deg = (float(np.degrees(state_dict["psi"])) + 360.0) % 360.0
        heading_cmd_deg = self.hud_heading_cmd_deg
        vt_fps = float(state_dict["V"])
        airspeed_kt = vt_fps / 1.687809857
        altitude_ref_ft = self.dem_start_elev_ft if self.canyon_mode == "dem" else 0.0
        altitude_ft = float(state_dict["h"] - altitude_ref_ft + 200)
        if not np.isfinite(airspeed_kt):
            airspeed_kt = 0.0
        if not np.isfinite(altitude_ft):
            altitude_ft = 0.0
        if heading_cmd_deg is not None and not np.isfinite(heading_cmd_deg):
            heading_cmd_deg = None

        font_big = self.hud_title_font
        font_main = self.hud_font
        font_small = self.hud_small_font

        panel_edge = (206, 206, 194)
        text_white = (236, 236, 226)
        text_green = (22, 235, 90)
        text_magenta = (242, 132, 242)
        text_purple = (206, 138, 255)

        def _textured_panel(rect, base_rgb, edge_rgb, alpha=255, radius=10):
            pg.draw.rect(surface, base_rgb, rect, border_radius=radius)
            pg.draw.rect(surface, edge_rgb, rect, 2, border_radius=radius)

        pfd_rect = pg.Rect(int(width * 0.03), int(height * 0.02), int(width * 0.94), int(height * 0.95))

        # Left airspeed tape with textured grey background and PFD-style labels.
        left_panel = pg.Rect(pfd_rect.x + 10, pfd_rect.y + 92, 170, pfd_rect.height - 212)
        _textured_panel(left_panel, (88, 84, 86), panel_edge, alpha=236, radius=0)
        left_track = pg.Rect(left_panel.x + 10, left_panel.y + 18, left_panel.width - 20, left_panel.height - 74)
        pg.draw.rect(surface, (70, 66, 68), left_track)

        speed_center_y = left_track.centery
        px_per_kt = 2.4
        min_v = int(np.floor((airspeed_kt - 120.0) / 5.0) * 5)
        max_v = int(np.ceil((airspeed_kt + 120.0) / 5.0) * 5)
        for speed_tick in range(min_v, max_v + 5, 5):
            y_tick = int(round(speed_center_y + (airspeed_kt - speed_tick) * px_per_kt))
            if y_tick < left_track.y + 2 or y_tick > left_track.bottom - 2:
                continue
            is_major = (speed_tick % 20) == 0
            tick_len = 32 if is_major else 15
            pg.draw.line(surface, text_white, (left_track.x + 6, y_tick), (left_track.x + 6 + tick_len, y_tick), 2 if is_major else 1)
            if is_major:
                spd_lbl = font_main.render(f"{speed_tick:03d}", True, text_white)
                surface.blit(spd_lbl, (left_track.x + 45, y_tick - spd_lbl.get_height() // 2))

        spd_bug_txt = font_big.render(f"{int(round(airspeed_kt)):03d}", True, text_white)
        kts_txt = font_small.render("KTS", True, text_white)
        speed_bug_w = max(108, spd_bug_txt.get_width() + kts_txt.get_width() + 24)
        speed_bug = pg.Rect(left_track.x - 8, speed_center_y - 30, speed_bug_w, 60)
        pg.draw.polygon(
            surface,
            (234, 226, 210),
            [(speed_bug.right + 10, speed_center_y), (speed_bug.right, speed_center_y - 12), (speed_bug.right, speed_center_y + 12)],
        )
        _textured_panel(speed_bug, (10, 12, 20), (236, 236, 216), alpha=250, radius=2)
        spd_x = speed_bug.x + 8
        spd_y = speed_bug.centery - spd_bug_txt.get_height() // 2
        surface.blit(spd_bug_txt, (spd_x, spd_y))
        kts_x = spd_x + spd_bug_txt.get_width() + 4
        kts_y = speed_bug.centery - kts_txt.get_height() // 2
        surface.blit(kts_txt, (kts_x, kts_y))

        # Right integrated altitude panel.
        right_panel = pg.Rect(pfd_rect.right - 182, pfd_rect.y + 92, 170, pfd_rect.height - 212)
        _textured_panel(right_panel, (90, 86, 92), panel_edge, alpha=236, radius=0)
        alt_track = pg.Rect(right_panel.x + 8, right_panel.y + 18, right_panel.width - 16, right_panel.height - 74)
        pg.draw.rect(surface, (72, 68, 70), alt_track)

        alt_center_y = alt_track.centery
        alt_window_half_ft = 800.0
        alt_px_per_ft = (0.5 * max(alt_track.height - 8, 1)) / alt_window_half_ft
        min_alt_tick = int(np.floor((altitude_ft - (alt_window_half_ft + 120.0)) / 20.0) * 20)
        max_alt_tick = int(np.ceil((altitude_ft + (alt_window_half_ft + 120.0)) / 20.0) * 20)
        for alt_tick in range(min_alt_tick, max_alt_tick + 20, 20):
            y_tick = int(round(alt_center_y + (altitude_ft - alt_tick) * alt_px_per_ft))
            if y_tick < alt_track.y + 2 or y_tick > alt_track.bottom - 2:
                continue
            major = (alt_tick % 100) == 0
            tick_len = 26 if major else 12
            pg.draw.line(surface, text_white, (alt_track.x + 4, y_tick), (alt_track.x + 4 + tick_len, y_tick), 2 if major else 1)
            if major:
                alt_lbl = font_main.render(f"{int(alt_tick):4d}", True, text_white)
                surface.blit(alt_lbl, (alt_track.x + 36, y_tick - alt_lbl.get_height() // 2))

        pg.draw.line(surface, (240, 240, 230), (alt_track.x + 1, alt_center_y), (alt_track.right - 1, alt_center_y), 2)
        alt_bug = pg.Rect(alt_track.x + 20, alt_center_y - 32, 98, 64)
        _textured_panel(alt_bug, (8, 10, 18), (245, 245, 220), alpha=248, radius=2)
        alt_bug_txt = font_big.render(f"{int(round(altitude_ft)):04d}FT", True, (160, 210, 255))
        surface.blit(alt_bug_txt, (alt_bug.centerx - alt_bug_txt.get_width() // 2, alt_bug.centery - alt_bug_txt.get_height() // 2))
        return np.transpose(pg.surfarray.array3d(surface), (1, 0, 2)).copy()

    def render(self, mode="human"):
        scale = 1e-3

        if self.viewer is None:
            self._initialize_viewer(scale)

        if self.canyon_mode == "dem" and hasattr(self.canyon, "get_local_from_latlon"):
            lat_deg = float(self.simulation.get_property_value("position/lat-gc-deg"))
            lon_deg = float(self.simulation.get_property_value("position/long-gc-deg"))
            local_north_ft, local_east_ft = self.canyon.get_local_from_latlon(lat_deg, lon_deg)
            x = local_north_ft * FT_TO_M * scale
            y = local_east_ft * FT_TO_M * scale
        else:
            x = self.state[0] * scale
            y = self.state[1] * scale

        if self.canyon_mode == "dem":
            z = (self.state[2] - self.dem_render_base_elev_ft * FT_TO_M) * scale
        else:
            z = self.state[2] * scale

        self.f16.transform.z = x
        self.f16.transform.x = -y
        self.f16.transform.y = z

        rot = Quaternion.from_euler(*self.state[9:])
        rot = Quaternion(rot.w, -rot.y, -rot.z, rot.x)
        self.f16.transform.rotation = rot

        self._update_canyon_wall_positions(scale)

        rot_mat = self.f16.transform.rotation.mat()
        forward = rot_mat.dot(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        up = rot_mat.dot(np.array([0.0, 1.0, 0.0], dtype=np.float32))

        f_norm = np.linalg.norm(forward)
        if f_norm > 1e-6:
            forward = forward / f_norm
        else:
            forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Keep chase camera stably behind the aircraft while still responding to pitch.
        forward_flat = forward.copy()
        forward_flat[1] = 0.0
        ff_norm = np.linalg.norm(forward_flat)
        if ff_norm > 1e-6:
            forward_flat = forward_flat / ff_norm
            chase_forward = 0.75 * forward_flat + 0.25 * forward
            cf_norm = np.linalg.norm(chase_forward)
            if cf_norm > 1e-6:
                chase_forward = chase_forward / cf_norm
            else:
                chase_forward = forward
        else:
            chase_forward = forward

        chase_distance = 100.0 * scale
        chase_height = 95.0 * scale
        look_ahead = 185.0 * scale
        look_up = 28.0 * scale

        desired_camera_pos = (
            self.f16.transform.position
            - chase_distance * chase_forward
            + np.array([0.0, chase_height, 0.0], dtype=np.float32)
        )
        desired_look_at = self.f16.transform.position + look_ahead * chase_forward + look_up * up

        if self.camera_pos is None or self.camera_look_at is None:
            self.camera_pos = desired_camera_pos.copy()
            self.camera_look_at = desired_look_at.copy()
        else:
            smooth = 0.14
            self.camera_pos = (1.0 - smooth) * self.camera_pos + smooth * desired_camera_pos
            self.camera_look_at = (1.0 - smooth) * self.camera_look_at + smooth * desired_look_at

        ray = self.camera_look_at - self.camera_pos
        ray_x, ray_y, ray_z = ray
        yaw = np.arctan2(ray_x, ray_z)
        pitch = np.arctan2(ray_y, np.sqrt(ray_x * ray_x + ray_z * ray_z))

        self.viewer.set_view(*self.camera_pos, Quaternion.from_euler(-pitch, yaw, 0.0, mode=1))
        self.viewer.render()

        if self.render_mode == "rgb_array":
            frame = self.viewer.get_frame()
            return self._draw_hud(frame)

        return None


def wrap_canyon_flight(render_mode=None, **kwargs):
    return CanyonFlightEnv(render_mode=render_mode, **kwargs)


if "JSBSimCanyon-v0" not in registry:
    gym.register(
        id="JSBSimCanyon-v0",
        entry_point="jsbsim_gym.canyon_env:wrap_canyon_flight",
        max_episode_steps=1200,
    )
