import numpy as np

class PersistentExcitationController:
    def __init__(
        self,
        target_h=1000.0,
        target_p_e=0.0,
        speed_schedule_fps=None,
        aoa_schedule_deg=None,
        schedule_segment_sec=12.0,
        seed=0,
    ):
        self.target_h = target_h
        self.target_p_e = target_p_e
        self.schedule_segment_sec = float(max(schedule_segment_sec, 1.0))
        self.rng = np.random.default_rng(seed)
        self.speed_schedule_fps = np.asarray(
            speed_schedule_fps if speed_schedule_fps is not None else [320.0, 420.0, 520.0, 620.0, 740.0, 860.0],
            dtype=np.float64,
        )
        self.aoa_schedule_deg = np.asarray(
            aoa_schedule_deg if aoa_schedule_deg is not None else [-2.0, 2.0, 6.0, 10.0, 14.0, 18.0],
            dtype=np.float64,
        )
        if self.speed_schedule_fps.size == 0:
            raise ValueError("speed_schedule_fps must not be empty")
        if self.aoa_schedule_deg.size == 0:
            raise ValueError("aoa_schedule_deg must not be empty")

        self._speed_error_int = 0.0
        self._aoa_error_int = 0.0
        self._last_targets = {
            "target_speed_fps": float(self.speed_schedule_fps[0]),
            "target_aoa_deg": float(self.aoa_schedule_deg[0]),
        }
        
        # Mutually prime frequencies for persistent excitation
        self.frequencies = {
            'throttle': [0.11, 0.31, 0.73],
            'elevator': [0.13, 0.37, 0.79],
            'aileron':  [0.17, 0.41, 0.83],
            'rudder':   [0.19, 0.43, 0.89]
        }
        
        # Amplitudes for the perturbations
        self.amplitudes = {
            'throttle': 0.12,
            'elevator': 0.44,
            'aileron': 0.2,
            'rudder': 0.1
        }

    @property
    def last_targets(self):
        return dict(self._last_targets)

    def _scheduled_targets(self, time_sec):
        idx = int(max(time_sec, 0.0) // self.schedule_segment_sec)
        speed = float(self.speed_schedule_fps[idx % self.speed_schedule_fps.size])
        aoa = float(self.aoa_schedule_deg[idx % self.aoa_schedule_deg.size])
        self._last_targets = {
            "target_speed_fps": speed,
            "target_aoa_deg": aoa,
        }
        return speed, aoa

    def get_action(self, state_dict, time_sec):
        """
        Computes baseline control and superimposes multisine perturbations.
        
        Args:
            state_dict (dict): Dictionary containing system states.
            time_sec (float): Current simulation time.
            
        Returns:
            list: [roll_cmd, pitch_cmd, yaw_cmd, throttle]
        """
        h = state_dict['h']
        p_E = state_dict['p_E']
        theta = state_dict['theta']
        phi = state_dict['phi']
        q = state_dict['q']

        speed_fps = float(np.sqrt(state_dict['u'] ** 2 + state_dict['v'] ** 2 + state_dict['w'] ** 2))
        aoa_deg = float(np.degrees(state_dict['alpha']))
        target_speed_fps, target_aoa_deg = self._scheduled_targets(time_sec)

        # Track speed and AoA envelopes to increase model coverage across the
        # nonlinear flight envelope while maintaining safe canyon flight.
        speed_error = target_speed_fps - speed_fps
        aoa_error = target_aoa_deg - aoa_deg
        self._speed_error_int = float(np.clip(self._speed_error_int + speed_error * (1.0 / 30.0), -800.0, 800.0))
        self._aoa_error_int = float(np.clip(self._aoa_error_int + aoa_error * (1.0 / 30.0), -400.0, 400.0))
        
        # Simple baseline controller to keep aircraft alive and roughly centered
        # 1. Altitude and AoA envelope control
        h_error = self.target_h - h
        segment_phase = (max(time_sec, 0.0) % self.schedule_segment_sec) / self.schedule_segment_sec
        if target_aoa_deg >= 14.0:
            pitch_program_bias = -0.50 if segment_phase < 0.70 else -0.28
        elif target_aoa_deg >= 10.0:
            pitch_program_bias = -0.38 if segment_phase < 0.65 else -0.20
        elif target_aoa_deg >= 6.0:
            pitch_program_bias = -0.24 if segment_phase < 0.60 else -0.10
        elif target_aoa_deg <= 2.0:
            pitch_program_bias = 0.24 if segment_phase < 0.55 else 0.10
        else:
            pitch_program_bias = 0.0
        baseline_elevator = np.clip(
            -h_error * 0.0010
            - 0.13 * aoa_error
            - 0.006 * self._aoa_error_int
            + theta * 0.30
            + q * 0.10
            + pitch_program_bias,
            -0.95,
            0.95,
        )
        
        # 2. Heading / lateral position control
        pE_error = self.target_p_e - p_E
        baseline_aileron = np.clip(pE_error * 0.001 - phi * 0.8, -0.5, 0.5)
        
        baseline_rudder = 0.0
        baseline_throttle = np.clip(
            0.56
            + 0.0018 * speed_error
            + 0.00035 * self._speed_error_int
            + 0.0020 * max(aoa_error, 0.0)
            - 0.030 * max(target_aoa_deg - 8.0, 0.0),
            0.08,
            1.00,
        )
        
        # 3. Superimpose multisine
        def multisine(t, freqs, amp):
            return amp * sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
            
        delta_t = multisine(time_sec, self.frequencies['throttle'], self.amplitudes['throttle'])
        delta_e = multisine(time_sec, self.frequencies['elevator'], self.amplitudes['elevator'])
        delta_a = multisine(time_sec, self.frequencies['aileron'], self.amplitudes['aileron'])
        delta_r = multisine(time_sec, self.frequencies['rudder'], self.amplitudes['rudder'])
        
        throttle = np.clip(baseline_throttle + delta_t, 0.0, 1.0)
        pitch_cmd = np.clip(baseline_elevator + delta_e, -1.0, 1.0)
        roll_cmd = np.clip(baseline_aileron + delta_a, -1.0, 1.0)
        yaw_cmd = np.clip(baseline_rudder + delta_r, -1.0, 1.0)
        
        return [roll_cmd, pitch_cmd, yaw_cmd, throttle]
