import numpy as np


class SimpleCanyonController:
    """Simple centerline-following guidance and control law for canyon flight.

    The controller uses a lightweight guidance layer to estimate cross-track and
    heading error to the canyon centerline, then applies damped attitude/speed
    control to produce [roll, pitch, yaw, throttle] commands.
    """

    def __init__(
        self,
        env,
        target_speed_fps=320.0,
        target_clearance_ft=700.0,
        lookahead_rows=40,
        terrain_lookahead_ft=(600.0, 1200.0, 1800.0, 2400.0),
        dt=1.0 / 30.0,
        use_dem_centerline=False,
    ):
        self.env = env.unwrapped
        self.canyon = self.env.canyon
        self.target_speed_fps = float(target_speed_fps)
        self.target_clearance_ft = float(target_clearance_ft)
        self.lookahead_rows = int(max(1, lookahead_rows))
        self.terrain_lookahead_ft = tuple(float(x) for x in terrain_lookahead_ft)
        self.dt = float(max(dt, 1e-3))
        self.use_dem_centerline = bool(use_dem_centerline)

        self.wall_margin_ft = float(getattr(self.env, "wall_margin_ft", 0.0))
        self.target_altitude_ft = float(getattr(self.env, "target_altitude_ft", 0.0))

        self._center_east_ft = None
        self._reference_heading_rad = None
        self._altitude_error_integral = 0.0
        self._prev_lateral_error_ft = 0.0
        self._prev_terrain_clearance_ft = self.target_clearance_ft

        self.last_guidance = {}

    @staticmethod
    def _wrap_angle_rad(angle_rad):
        return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))

    def _has_dem_centerline(self):
        return (
            hasattr(self.canyon, "get_local_from_latlon")
            and hasattr(self.canyon, "get_centerline_heading_deg")
            and hasattr(self.canyon, "north_samples_ft")
            and hasattr(self.canyon, "center_east_samples_ft")
            and hasattr(self.canyon, "width_samples_ft")
        )

    def reset(self, state_dict):
        self._altitude_error_integral = 0.0
        self._center_east_ft = float(
            getattr(self.env, "canyon_center_east_ft", state_dict["p_E"])
        )
        self._reference_heading_rad = float(state_dict["psi"])
        self._prev_lateral_error_ft = float(state_dict["p_E"] - self._center_east_ft)
        self._prev_terrain_clearance_ft = self._terrain_clearance_ft(state_dict)

    def _guidance_from_dem(self, state_dict):
        lat_deg = float(self.env.simulation.get_property_value("position/lat-gc-deg"))
        lon_deg = float(self.env.simulation.get_property_value("position/long-gc-deg"))

        local_north_ft, local_east_ft = self.canyon.get_local_from_latlon(lat_deg, lon_deg)
        center_east_ft = float(
            np.interp(
                local_north_ft,
                self.canyon.north_samples_ft,
                self.canyon.center_east_samples_ft,
            )
        )
        width_ft = float(
            np.interp(
                local_north_ft,
                self.canyon.north_samples_ft,
                self.canyon.width_samples_ft,
            )
        )

        row_spacing_ft = max(float(getattr(self.canyon, "row_spacing_ft", 1.0)), 1.0)
        row_ordered = local_north_ft / row_spacing_ft
        center_heading_deg = float(
            self.canyon.get_centerline_heading_deg(
                row_ordered=row_ordered,
                lookahead_rows=self.lookahead_rows,
            )
        )
        desired_heading_rad = np.deg2rad(center_heading_deg)

        lateral_error_ft = float(local_east_ft - center_east_ft)
        heading_error_rad = self._wrap_angle_rad(float(state_dict["psi"]) - desired_heading_rad)

        return {
            "lateral_error_ft": lateral_error_ft,
            "heading_error_rad": heading_error_rad,
            "centerline_heading_deg": center_heading_deg,
            "canyon_width_ft": width_ft,
            "local_north_ft": float(local_north_ft),
        }

    def _guidance_fallback(self, state_dict):
        if self._center_east_ft is None:
            self._center_east_ft = float(state_dict["p_E"])
        if self._reference_heading_rad is None:
            self._reference_heading_rad = float(state_dict["psi"])

        lateral_error_ft = float(state_dict["p_E"] - self._center_east_ft)
        heading_error_rad = self._wrap_angle_rad(float(state_dict["psi"]) - self._reference_heading_rad)

        return {
            "lateral_error_ft": lateral_error_ft,
            "heading_error_rad": heading_error_rad,
            "centerline_heading_deg": float(np.rad2deg(self._reference_heading_rad)),
            "canyon_width_ft": float(state_dict.get("canyon_width", 600.0)),
            "local_north_ft": float(state_dict["p_N"]),
        }

    def _compute_guidance(self, state_dict):
        if self.use_dem_centerline and self._has_dem_centerline():
            try:
                dem_guidance = self._guidance_from_dem(state_dict)
                width_ft = float(dem_guidance.get("canyon_width_ft", state_dict.get("canyon_width", 1.0)))
                lateral_ft = float(dem_guidance.get("lateral_error_ft", 0.0))
                if not np.isfinite(lateral_ft) or abs(lateral_ft) > 4.0 * max(width_ft, 1.0):
                    return self._guidance_fallback(state_dict)
                return dem_guidance
            except Exception:
                return self._guidance_fallback(state_dict)
        return self._guidance_fallback(state_dict)

    def _terrain_clearance_ft(self, state_dict):
        if hasattr(self.canyon, "get_elevation_msl_ft_from_latlon"):
            try:
                lat_deg = float(self.env.simulation.get_property_value("position/lat-gc-deg"))
                lon_deg = float(self.env.simulation.get_property_value("position/long-gc-deg"))
                terrain_elev_ft = float(self.canyon.get_elevation_msl_ft_from_latlon(lat_deg, lon_deg))
                return float(state_dict["h"] - terrain_elev_ft)
            except Exception:
                pass
        return float(state_dict["h"])

    def _dem_elevation_msl_ft_from_local(self, local_north_ft, local_east_ft):
        if not (
            hasattr(self.canyon, "ordered_dem_msl_m")
            and hasattr(self.canyon, "north_samples_ft")
            and hasattr(self.canyon, "east_samples_ft")
        ):
            return None

        north_axis = np.asarray(self.canyon.north_samples_ft, dtype=np.float32)
        east_axis = np.asarray(self.canyon.east_samples_ft, dtype=np.float32)
        dem = np.asarray(self.canyon.ordered_dem_msl_m, dtype=np.float32)
        rows, cols = dem.shape

        north_ft = float(np.clip(local_north_ft, float(north_axis[0]), float(north_axis[-1])))
        east_ft = float(np.clip(local_east_ft, float(east_axis[0]), float(east_axis[-1])))

        row_ordered = float(np.interp(north_ft, north_axis, np.arange(rows, dtype=np.float32)))
        col = float(np.interp(east_ft, east_axis, np.arange(cols, dtype=np.float32)))

        r0 = int(np.floor(row_ordered))
        c0 = int(np.floor(col))
        r1 = min(r0 + 1, rows - 1)
        c1 = min(c0 + 1, cols - 1)

        dr = float(row_ordered - r0)
        dc = float(col - c0)

        z00 = float(dem[r0, c0])
        z01 = float(dem[r0, c1])
        z10 = float(dem[r1, c0])
        z11 = float(dem[r1, c1])

        z0 = (1.0 - dc) * z00 + dc * z01
        z1 = (1.0 - dc) * z10 + dc * z11
        z_msl_m = (1.0 - dr) * z0 + dr * z1
        return float(z_msl_m * 3.28084)

    def _predict_min_clearance_ahead_ft(self, state_dict, guidance):
        current_clearance_ft = self._terrain_clearance_ft(state_dict)

        if not (
            hasattr(self.canyon, "get_local_from_latlon")
            and hasattr(self.canyon, "north_samples_ft")
            and hasattr(self.canyon, "center_east_samples_ft")
        ):
            return current_clearance_ft

        try:
            lat_deg = float(self.env.simulation.get_property_value("position/lat-gc-deg"))
            lon_deg = float(self.env.simulation.get_property_value("position/long-gc-deg"))
            local_north_ft, local_east_ft = self.canyon.get_local_from_latlon(lat_deg, lon_deg)

            north_axis = np.asarray(self.canyon.north_samples_ft, dtype=np.float32)
            center_axis = np.asarray(self.canyon.center_east_samples_ft, dtype=np.float32)
            aircraft_h_msl_ft = float(state_dict["h"])

            min_clearance_ft = current_clearance_ft
            for delta_north_ft in self.terrain_lookahead_ft:
                sample_north_ft = float(local_north_ft + delta_north_ft)
                sample_center_east_ft = float(
                    np.interp(sample_north_ft, north_axis, center_axis)
                )
                elev_msl_ft = self._dem_elevation_msl_ft_from_local(
                    sample_north_ft,
                    sample_center_east_ft,
                )
                if elev_msl_ft is None:
                    continue
                sample_clearance_ft = aircraft_h_msl_ft - elev_msl_ft
                min_clearance_ft = min(min_clearance_ft, sample_clearance_ft)

            return float(min_clearance_ft)
        except Exception:
            return current_clearance_ft

    def get_action(self, state_dict):
        guidance = self._compute_guidance(state_dict)

        lateral_error_ft = float(guidance["lateral_error_ft"])
        lateral_error_rate_fps = (lateral_error_ft - self._prev_lateral_error_ft) / self.dt
        self._prev_lateral_error_ft = lateral_error_ft

        heading_error_rad = float(guidance["heading_error_rad"])
        canyon_width_ft = float(guidance["canyon_width_ft"])

        half_width_ft = 0.5 * canyon_width_ft
        usable_half_ft = max(half_width_ft - self.wall_margin_ft, 80.0)
        lateral_norm = np.clip(lateral_error_ft / usable_half_ft, -2.5, 2.5)

        phi = float(state_dict["phi"])
        theta = float(state_dict["theta"])
        p_rate = float(state_dict["p"])
        q_rate = float(state_dict["q"])
        r_rate = float(state_dict["r"])
        beta = float(state_dict.get("beta", 0.0))

        terrain_clearance_ft = self._terrain_clearance_ft(state_dict)
        clearance_rate_fps = (terrain_clearance_ft - self._prev_terrain_clearance_ft) / self.dt
        self._prev_terrain_clearance_ft = terrain_clearance_ft
        clearance_error_ft = float(terrain_clearance_ft - self.target_clearance_ft)
        min_ahead_clearance_ft = self._predict_min_clearance_ahead_ft(state_dict, guidance)
        ahead_clearance_error_ft = float(min_ahead_clearance_ft - self.target_clearance_ft)
        self._altitude_error_integral += clearance_error_ft * self.dt
        self._altitude_error_integral = float(np.clip(self._altitude_error_integral, -3000.0, 3000.0))

        speed_fps = float(
            np.sqrt(
                float(state_dict["u"]) ** 2
                + float(state_dict["v"]) ** 2
                + float(state_dict["w"]) ** 2
            )
        )

        # Lateral line-following around proxy canyon centerline p_E ~= center_east.
        roll_des_rad = np.clip(
            -0.0030 * lateral_error_ft
            -0.0005 * lateral_error_rate_fps
            -0.70 * heading_error_rad,
            -0.75,
            0.75,
        )
        roll_cmd = np.clip(
            2.80 * (roll_des_rad - phi) - 0.35 * p_rate,
            -1.0,
            1.0,
        )

        # Altitude hold with terrain lookahead feedforward.
        desired_theta_rad = np.clip(
            -0.0022 * clearance_error_ft
            - 0.0014 * ahead_clearance_error_ft
            - 0.000020 * self._altitude_error_integral
            - 0.0030 * clearance_rate_fps,
            -0.10,
            0.55,
        )
        theta_error_rad = desired_theta_rad - theta
        pitch_cmd = np.clip(
            -4.2 * theta_error_rad + 0.75 * q_rate,
            -1.0,
            1.0,
        )

        # Simple yaw damping with a small heading-error trim.
        yaw_cmd = np.clip(
            -0.28 * r_rate - 0.10 * beta + 0.08 * (roll_des_rad - phi),
            -0.35,
            0.35,
        )

        # Speed hold around a cruise trim throttle.
        throttle_cmd = np.clip(
            0.26
            + 0.0014 * (self.target_speed_fps - speed_fps)
            - 0.0007 * np.clip(-ahead_clearance_error_ft, 0.0, None),
            0.0,
            0.80,
        )

        margin_ft = usable_half_ft - abs(lateral_error_ft)
        self.last_guidance = {
            **guidance,
            "lateral_norm": float(lateral_norm),
            "lateral_error_rate_fps": float(lateral_error_rate_fps),
            "altitude_error_ft": float(state_dict["h"] - self.target_altitude_ft),
            "terrain_clearance_ft": float(terrain_clearance_ft),
            "clearance_error_ft": float(clearance_error_ft),
            "ahead_min_clearance_ft": float(min_ahead_clearance_ft),
            "ahead_clearance_error_ft": float(ahead_clearance_error_ft),
            "clearance_rate_fps": float(clearance_rate_fps),
            "speed_fps": float(speed_fps),
            "heading_error_deg": float(np.degrees(heading_error_rad)),
            "roll_des_deg": float(np.degrees(roll_des_rad)),
            "margin_to_wall_ft": float(margin_ft),
            "roll_cmd": float(roll_cmd),
            "pitch_cmd": float(pitch_cmd),
            "yaw_cmd": float(yaw_cmd),
            "throttle_cmd": float(throttle_cmd),
        }

        return np.array([roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd], dtype=np.float32)
