import gymnasium as gym
import numpy as np
from jsbsim_gym.env import JSBSimEnv, RADIUS
from jsbsim_gym.canyon import ProceduralCanyon

RADIUS_FT = RADIUS * 3.28084  # Radius of earth in feet

class DataCollectionEnv(JSBSimEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.canyon = ProceduralCanyon()
        
        # O-U Process state for Wind
        self.wind_state = np.zeros(3)
        self.wind_theta = 0.5   # OU mean reversion speed
        self.wind_sigma = 15.0  # Base volatility for gusts
        self.dt = 1.0 / 120.0   # JSBSim default tick rate
        self.time_sec = 0.0

    def reset(self, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        self.wind_state = np.zeros(3)
        self.time_sec = 0.0
        return out
        
    def get_full_state_dict(self):
        """
        Extracts all explicitly requested telemetry directly from JSBSim.
        """
        lat_rad = self.simulation.get_property_value("position/lat-gc-rad")
        long_rad = self.simulation.get_property_value("position/long-gc-rad")
        
        p_N = lat_rad * RADIUS_FT
        p_E = long_rad * RADIUS_FT
        h = self.simulation.get_property_value("position/h-sl-ft")
        
        # Using body velocities u, v, w or just total velocity V.
        # The prompt asked for: "Velocities: velocities/u-fps, velocities/v-fps, velocities/w-fps"
        # Wait, the prompt says: "Velocities: velocities/u-fps, velocities/v-fps, velocities/w-fps (or body equivalents)."
        # In the dataframe definition it says "Columns explicitly name: p_N, p_E, h, V, alpha, beta..."
        # Wait, if they say 'V', do they mean 'vt-fps' (Total Velocity)? Let's put both u, v, w and V. 
        # I will include u, v, w to be safe, but map V to vt-fps.
        u = self.simulation.get_property_value("velocities/u-fps")
        v = self.simulation.get_property_value("velocities/v-fps")
        w = self.simulation.get_property_value("velocities/w-fps")
        V = self.simulation.get_property_value("velocities/vt-fps")
        
        alpha = self.simulation.get_property_value("aero/alpha-rad")
        beta = self.simulation.get_property_value("aero/beta-rad")
        
        phi = self.simulation.get_property_value("attitude/phi-rad")
        theta = self.simulation.get_property_value("attitude/theta-rad")
        psi = self.simulation.get_property_value("attitude/psi-rad")
        
        p = self.simulation.get_property_value("velocities/p-rad_sec")
        q = self.simulation.get_property_value("velocities/q-rad_sec")
        r = self.simulation.get_property_value("velocities/r-rad_sec")
        
        qbar = self.simulation.get_property_value("aero/qbar-psf")
        alpha_dot = self.simulation.get_property_value("aero/alphadot-rad_sec")
        
        width, grad = self.canyon.get_geometry(p_N)
        
        wind_n = self.simulation.get_property_value("atm/wind-north-fps")
        wind_e = self.simulation.get_property_value("atm/wind-east-fps")
        wind_d = self.simulation.get_property_value("atm/wind-down-fps")
        
        return {
            'p_N': p_N, 'p_E': p_E, 'h': h, 
            'V': V, 'u': u, 'v': v, 'w': w,
            'alpha': alpha, 'beta': beta,
            'phi': phi, 'theta': theta, 'psi': psi,
            'p': p, 'q': q, 'r': r,
            'qbar': qbar, 'alpha_dot': alpha_dot,
            'canyon_width': width, 'canyon_width_grad': grad,
            'wind_u': wind_n, 'wind_v': wind_e, 'wind_w': wind_d
        }

    def step_collect(self, action):
        """
        Advances the simulation and extracts canonical telemetry.
        Action array: [roll_cmd, pitch_cmd, yaw_cmd, throttle]
        """
        roll_cmd, pitch_cmd, yaw_cmd, throttle = action
        
        state_t = self.get_full_state_dict()
        
        self.simulation.set_property_value("fcs/aileron-cmd-norm", roll_cmd)
        self.simulation.set_property_value("fcs/elevator-cmd-norm", pitch_cmd)
        self.simulation.set_property_value("fcs/rudder-cmd-norm", yaw_cmd)
        self.simulation.set_property_value("fcs/throttle-cmd-norm", throttle)
        
        for _ in range(self.down_sample):
            self.simulation.set_property_value("propulsion/tank/contents-lbs", 1000)
            self.simulation.set_property_value("propulsion/tank[1]/contents-lbs", 1000)
            self.simulation.set_property_value("gear/gear-cmd-norm", 0.0)
            self.simulation.set_property_value("gear/gear-pos-norm", 0.0)
            
            # Get real-time p_N
            curr_lat = self.simulation.get_property_value("position/lat-gc-rad")
            curr_p_N = curr_lat * RADIUS_FT
            
            W_c, _ = self.canyon.get_geometry(curr_p_N)
            
            # Amplitude inversely proportional to W_c
            scale = 1000.0 / max(W_c, 50.0)
            
            # O-U step
            noise = np.random.randn(3)
            self.wind_state += -self.wind_theta * self.wind_state * self.dt + self.wind_sigma * scale * np.sqrt(self.dt) * noise
            
            self.simulation.set_property_value('atm/wind-north-fps', self.wind_state[0])
            self.simulation.set_property_value('atm/wind-east-fps', self.wind_state[1])
            self.simulation.set_property_value('atm/wind-down-fps', self.wind_state[2])
            
            self.simulation.run()
            self.time_sec += self.dt
            
        self._get_state()
        state_t_plus_1 = self.get_full_state_dict()
        
        done = False
        if state_t_plus_1['h'] < 10:
            done = True
            
        # Extract actions into state_t for easy row tracking
        state_t['delta_t'] = throttle
        state_t['delta_e'] = pitch_cmd
        state_t['delta_a'] = roll_cmd
        state_t['delta_r'] = yaw_cmd
        
        return state_t, state_t_plus_1, done
