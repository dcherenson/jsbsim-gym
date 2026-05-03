import numpy as np

class PIDTrajectoryController:
    """Wrapper that implements the get_action() interface used by run_scenario.py"""
    def __init__(self, env, reference_trajectory, config=None):
        self.env = env
        
        # Build gains (can be overridden by config)
        if config is None:
            config = {}
            
        guidance_gains = config.get('guidance', {
            'kp_xtrk': 0.05, 'kd_xtrk': 0.01,
            'kp_z': 0.01, 'kd_z': 0.005,
            'bank_alpha_gain': 0.35,
            'bank_error_deadband_rad': np.deg2rad(2.0),
            'bank_correction_turn_threshold_rad': np.deg2rad(10.0),
            'bank_alpha_boost_max_rad': np.deg2rad(8.0),
        })
        
        autopilot_gains = config.get('autopilot', {
            'alpha_p': 3.0, 'alpha_i': 1.0, 
            'q_p': 3.0, 'q_d': 0.0,
            'phi_p': 2.5,
            'p_p': 0.5, 'p_d': 0.0,
            'beta_p': 3.0, 'beta_i': 1.5,
            'washout_tau': 1.0, 'r_highpass_gain': 0.5,
            'v_p': 0.01, 'v_i': 0.002
        })
        
        # Convert the nominal reference dict into the format SpatialGuidance expects
        north_ft = reference_trajectory['north_ft']
        east_ft = reference_trajectory['east_ft']
        alt_ft = reference_trajectory['altitude_ft']
        
        self.nominal_trajectory_data = {
            'enu': np.column_stack((east_ft, north_ft, alt_ft)),
            'heading_rad': reference_trajectory['heading_rad'],
            'phi_opt': reference_trajectory['phi_rad'],
            'alpha_opt': reference_trajectory['alpha_rad'],
            'v_opt': reference_trajectory['speed_fps']
        }
        
        # No LLA conversions needed since we use local ENU directly
        self.guidance = SpatialGuidance(guidance_gains, self.nominal_trajectory_data)
        # Override lla_to_enu to just return local feet directly since we now pass it local feet
        self.guidance.lla_to_enu = lambda current_e, current_n, current_u, *_: np.array([current_e, current_n, current_u])
        
        self.autopilot = F16Autopilot(autopilot_gains)

    def reset(self, initial_state=None):
        pass # Reset integrals etc if needed

    def get_action(self, obs=None):
        dt = 1.0 / 30.0 # Typical JSBSim Gym dt
        
        # We need JSBSim properties; running env.unwrapped.get_full_state_dict() gets most of them
        state = self.env.unwrapped.get_full_state_dict()
        
        # For our fake ENU, we pass east, north, altitude directly
        # state['p_E'] and state['p_N'] are already computed as flat coordinates in feet in some cases,
        # but the JSBSim Canyon env has `get_local_from_latlon` that was used to create the nominal trajectory ENU.
        lat_rad = self.env.unwrapped.simulation.get_property_value("position/lat-gc-rad")
        lon_rad = self.env.unwrapped.simulation.get_property_value("position/long-gc-rad")
        alt = self.env.unwrapped.simulation.get_property_value("position/h-sl-ft")
        
        # Get altitude relative to the canyon entrance
        altitude_ref_ft = float(getattr(self.env.unwrapped, "dem_start_elev_ft", 0.0))
        h_rel = alt - altitude_ref_ft
        
        north_ft, east_ft = self.env.unwrapped.canyon.get_local_from_latlon(np.degrees(lat_rad), np.degrees(lon_rad))
        current_enu_ft = (east_ft, north_ft, h_rel)
        
        V = self.env.unwrapped.simulation.get_property_value("velocities/vt-fps")
        p = self.env.unwrapped.simulation.get_property_value("velocities/p-rad_sec")
        q = self.env.unwrapped.simulation.get_property_value("velocities/q-rad_sec")
        r = self.env.unwrapped.simulation.get_property_value("velocities/r-rad_sec")
        
        alpha = self.env.unwrapped.simulation.get_property_value("aero/alpha-rad")
        beta = self.env.unwrapped.simulation.get_property_value("aero/beta-rad")
        
        phi = self.env.unwrapped.simulation.get_property_value("attitude/phi-rad")
        
        phi_cmd, alpha_cmd, v_opt = self.guidance.update(
            current_enu_ft,
            (0, 0, 0),
            dt,
            current_phi=phi,
        )
        
        cmds = (phi_cmd, alpha_cmd, v_opt)
        states = (alpha, beta, phi, p, q, r, V)
        
        elevator, aileron, rudder, throttle = self.autopilot.update(cmds, states, dt)
        
        self.last_diagnostics = {}
        self.last_diagnostics.update(self.guidance.last_diagnostics)
        self.last_diagnostics.update(self.autopilot.last_diagnostics)
        
        # JSBSim fcs/elevator-cmd-norm > 0 is trailing edge down (pitch down).
        return np.array([aileron, elevator, 0*rudder, throttle], dtype=np.float32)

class DiscretePID:
    """Discrete-time PID controller with anti-windup."""
    def __init__(self, kp, ki, kd, output_min=-1.0, output_max=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, setpoint, process_variable, dt):
        if dt <= 0:
            return 0.0

        error = setpoint - process_variable
        
        # Proportional
        p_term = self.kp * error
        
        # Derivative (backward difference)
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative
        
        # Temporary output without integral
        temp_out = p_term + self.integral + d_term
        
        # Anti-windup (conditional integration)
        # Only integrate if we are not saturated or if the integral will reduce saturation
        if not (temp_out >= self.output_max and error > 0) and not (temp_out <= self.output_min and error < 0):
             # Forward Euler integration
             self.integral += self.ki * error * dt

        # Final output
        output = p_term + self.integral + d_term
        output = np.clip(output, self.output_min, self.output_max)
        
        self.prev_error = error
        return output

class WashoutFilter:
    """High-pass washout filter for yaw rate (Dutch roll damping)."""
    def __init__(self, time_constant):
        self.tau = time_constant
        self.x = 0.0
        self.prev_input = 0.0

    def update(self, current_input, dt):
        if self.tau == 0:
            return current_input
        # Tustin or simple backward diff depending on discretisation
        # Simple high pass: y(k) = (tau/(tau+dt))*y(k-1) + (tau/(tau+dt))*(u(k)-u(k-1))
        alpha = self.tau / (self.tau + dt)
        filtered = alpha * self.x + alpha * (current_input - self.prev_input)
        self.x = filtered
        self.prev_input = current_input
        return filtered

class SpatialGuidance:
    """Translates 3D positional errors into aerodynamic corrections using lookahead and nz-scaling."""
    def __init__(self, gains, nominal_trajectory_data):
        self.kp_xtrk = gains.get('kp_xtrk', 0.05)
        self.kd_xtrk = gains.get('kd_xtrk', 0.01)
        self.kp_z = gains.get('kp_z', 0.01)
        self.kd_z = gains.get('kd_z', 0.005)
        
        # New lookahead parameter (distance in feet to anticipate turns)
        self.lookahead_dist = gains.get('lookahead_dist_ft', 1000.0)
        
        self.trajectory_data = nominal_trajectory_data
        
        self.path_enu = nominal_trajectory_data['enu'] # (N, 3) East, North, Up
        self.path_heading = nominal_trajectory_data['heading_rad'] # (N,)
        self.alpha_opt = nominal_trajectory_data['alpha_opt'] # (N,)
        self.phi_opt = nominal_trajectory_data['phi_opt'] # (N,)
        self.v_opt = nominal_trajectory_data['v_opt'] # (N,)
        
        self.prev_e_xtrk = 0.0
        self.prev_e_z = 0.0

    def lla_to_enu(self, lat, lon, alt, ref_lat, ref_lon, ref_alt):
        a = 6378137.0
        f = 1 / 298.257223563
        e2 = 2*f - f**2
        
        d_lat = np.radians(lat - ref_lat)
        d_lon = np.radians(lon - ref_lon)
        ref_lat_rad = np.radians(ref_lat)
        
        N = a / np.sqrt(1 - e2 * np.sin(ref_lat_rad)**2)
        M = a * (1 - e2) / (1 - e2 * np.sin(ref_lat_rad)**2)**1.5
        
        east = d_lon * N * np.cos(ref_lat_rad)
        north = d_lat * M
        up = alt - ref_alt
        
        return np.array([east, north, up])

    def find_closest_point(self, current_enu):
        distances = np.linalg.norm(self.path_enu - current_enu, axis=1)
        return np.argmin(distances)
        
    def get_lookahead_index(self, closest_idx):
        """Finds the index of the point lookahead_dist ahead on the path."""
        accumulated_dist = 0.0
        curr_idx = closest_idx
        max_idx = len(self.path_enu) - 1
        
        while accumulated_dist < self.lookahead_dist and curr_idx < max_idx:
            step_dist = np.linalg.norm(self.path_enu[curr_idx + 1] - self.path_enu[curr_idx])
            accumulated_dist += step_dist
            curr_idx += 1
            
        return curr_idx

    def update(self, current_lla, ref_lla, dt, current_phi=None):
        current_enu = self.lla_to_enu(*current_lla, *ref_lla)
        
        idx = self.find_closest_point(current_enu)
        target_idx = self.get_lookahead_index(idx)
        
        closest_p = self.path_enu[idx]
        trk_hdg = self.path_heading[idx]
        
        # Calculate errors relative to the closest point
        d_east = current_enu[0] - closest_p[0]
        d_north = current_enu[1] - closest_p[1]
        
        e_xtrk = d_north * (-np.sin(trk_hdg)) + d_east * (np.cos(trk_hdg))
        e_z = closest_p[2] - current_enu[2] 
        
        if dt > 0:
            e_xtrk_dot = (e_xtrk - self.prev_e_xtrk) / dt
            e_z_dot = (e_z - self.prev_e_z) / dt
        else:
            e_xtrk_dot = 0.0
            e_z_dot = 0.0
            
        self.prev_e_xtrk = e_xtrk
        self.prev_e_z = e_z
        
        # Roll guidance: error correction + lookahead feedforward
        delta_phi = self.kp_xtrk * e_xtrk + self.kd_xtrk * e_xtrk_dot
        phi_cmd = self.phi_opt[idx] #+ delta_phi
        
        # Pitch guidance: Load factor (nz) based altitude correction
        delta_nz = self.kp_z * e_z + self.kd_z * e_z_dot
        
        if current_phi is None:
            current_phi = 0.0
            
        # Limit phi in the denominator to ~80 degrees to prevent singularity
        phi_for_lift = np.clip(current_phi, -1.4, 1.4)
        
        # Calculate commanded normal load factor: nz = (1g + correction) / cos(phi)
        nz_cmd = (1.0 + delta_nz) / np.cos(phi_for_lift)
        nz_cmd = 1.0
        
        # Scale the nominal 1g lookahead alpha by the commanded load factor
        alpha_cmd = self.alpha_opt[idx] * nz_cmd
        
        v_opt_val = self.v_opt[idx]
        
        self.last_diagnostics = {
            'e_xtrk': float(e_xtrk),
            'e_z': float(e_z),
            'e_xtrk_dot': float(e_xtrk_dot),
            'e_z_dot': float(e_z_dot),
            'closest_idx': int(idx),
            'target_idx': int(target_idx),
            'delta_phi': float(delta_phi),
            'delta_nz': float(delta_nz),
            'nz_cmd': float(nz_cmd),
            'phi_cmd': float(phi_cmd),
            'alpha_cmd': float(alpha_cmd),
            'v_opt_val': float(v_opt_val)
        }
        
        return phi_cmd, alpha_cmd, v_opt_val
class F16Autopilot:
    """Translates guidance commands into actuator deflections."""
    def __init__(self, gains):
        # Pitch Axis (Elevator): outer PI on alpha, inner PD on q
        self.alpha_pi = DiscretePID(gains['alpha_p'], gains['alpha_i'], 0.0, -10.0, 10.0)
        self.q_pd = DiscretePID(gains['q_p'], 0.0, gains['q_d'], -1.0, 1.0)
        
        # Roll Axis (Aileron): outer P on phi, inner PD on p
        self.phi_p = DiscretePID(gains['phi_p'], 0.0, 0.0, -10.0, 10.0)
        self.p_pd = DiscretePID(gains['p_p'], 0.0, gains['p_d'], -1.0, 1.0)
        
        # Yaw Axis (Rudder - Turn Coordination): PI on beta
        self.beta_pi = DiscretePID(gains['beta_p'], gains['beta_i'], 0.0, -1.0, 1.0)
        self.yaw_washout = WashoutFilter(time_constant=gains.get('washout_tau', 1.0))
        self.r_gain = gains.get('r_highpass_gain', 1.0)
        
        # Speed Axis (Throttle): PI on V
        self.v_pi = DiscretePID(gains['v_p'], gains['v_i'], 0.0, 0.0, 1.0)

    def update(self, cmds, states, dt):
        """
        cmds: (phi_cmd, alpha_cmd, v_opt)
        states: (alpha, beta, phi, p, q, r, V)
        """
        phi_cmd, alpha_cmd, v_opt = cmds
        alpha, beta, phi, p, q, r, V = states
        # alpha_cmd = 0.2
        # phi_cmd = -2.0
        # Pitch Axis
        q_cmd = self.alpha_pi.update(alpha_cmd, alpha, dt)
        elevator = -self.q_pd.update(q_cmd, q, dt)
        
        # Roll Axis
        p_cmd = self.phi_p.update(phi_cmd, phi, dt)
        aileron = self.p_pd.update(p_cmd, p, dt)
        
        # Yaw Axis
        rudder_beta = self.beta_pi.update(0.0, beta, dt)
        r_highpass = self.yaw_washout.update(r, dt)
        rudder = np.clip(rudder_beta + self.r_gain * r_highpass, -1.0, 1.0)
        
        # Speed Axis
        throttle = self.v_pi.update(v_opt, V, dt)
        
        self.last_diagnostics = {
            'q_cmd': float(q_cmd),
            'e_alpha': float(alpha_cmd - alpha),
            'e_q': float(q_cmd - q),
            'elevator_cmd': float(elevator),
            'p_cmd': float(p_cmd),
            'e_phi': float(phi_cmd - phi),
            'e_p': float(p_cmd - p),
            'aileron_cmd': float(aileron),
            'rudder_beta': float(rudder_beta),
            'r_highpass': float(r_highpass),
            'rudder_cmd': float(rudder),
            'e_V': float(v_opt - V),
            'throttle_cmd': float(throttle)
        }
        
        return elevator, aileron, rudder, throttle

class CascadedControllerWrapper:
    """Wrapper to execute the control sequence."""
    def __init__(self, config):
        """
        config should contain:
        - 'guidance_gains': Dict of gains for SpatialGuidance
        - 'autopilot_gains': Dict of gains for F16Autopilot
        - 'nominal_trajectory_data': Dict of path data
        - 'ref_lla': Canyon entrance LLA (lat, lon, alt)
        """
        self.guidance = SpatialGuidance(config['guidance_gains'], config['nominal_trajectory_data'])
        self.autopilot = F16Autopilot(config['autopilot_gains'])
        self.ref_lla = config['ref_lla']

    def update(self, observation, dt=0.05):
        """
        Execute control sequence in strict order.
        observation: dictionary from jsbsim_gym containing required JSBSim properties
        """
        # 1. Extract ENU position and aerodynamic states
        lat = observation['position/lat-geod-deg']
        lon = observation['position/long-gc-deg']
        alt = observation['position/h-sl-ft']
        
        V = observation['velocities/vt-fps']
        p = observation['velocities/p-rad_sec']
        q = observation['velocities/q-rad_sec']
        r = observation['velocities/r-rad_sec']
        
        alpha = observation['aero/alpha-rad']
        beta = observation['aero/beta-rad']
        
        phi = observation['attitude/roll-rad']
        
        current_lla = (lat, lon, alt)
        
        # 2. Call SpatialGuidance
        phi_cmd, alpha_cmd, v_opt = self.guidance.update(
            current_lla,
            self.ref_lla,
            dt,
            current_phi=phi,
        )
        
        # 3. Pass to Autopilot
        cmds = (phi_cmd, alpha_cmd, v_opt)
        states = (alpha, beta, phi, p, q, r, V)
        
        # 4 & 5. Execute AutoPilot loops
        elevator, aileron, rudder, throttle = self.autopilot.update(cmds, states, dt)
        
        # 6. Return action mapping to fcs commands
        return np.array([aileron, -elevator, rudder, throttle])
