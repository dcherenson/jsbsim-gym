import numpy as np

class PersistentExcitationController:
    def __init__(self, target_h=1000.0, target_p_e=0.0):
        self.target_h = target_h
        self.target_p_e = target_p_e
        
        # Mutually prime frequencies for persistent excitation
        self.frequencies = {
            'throttle': [0.11, 0.31, 0.73],
            'elevator': [0.13, 0.37, 0.79],
            'aileron':  [0.17, 0.41, 0.83],
            'rudder':   [0.19, 0.43, 0.89]
        }
        
        # Amplitudes for the perturbations
        self.amplitudes = {
            'throttle': 0.1,
            'elevator': 0.2,
            'aileron': 0.2,
            'rudder': 0.1
        }

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
        
        # Simple baseline controller to keep aircraft alive and roughly centered
        # 1. Altitude control
        h_error = self.target_h - h
        baseline_elevator = np.clip(-h_error * 0.002 + theta * 0.8, -0.5, 0.5)
        
        # 2. Heading / lateral position control
        pE_error = self.target_p_e - p_E
        baseline_aileron = np.clip(pE_error * 0.001 - phi * 0.8, -0.5, 0.5)
        
        baseline_rudder = 0.0
        baseline_throttle = 0.6
        
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
