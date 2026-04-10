import numpy as np

class ProceduralCanyon:
    def __init__(self, base_width=1000.0, amplitude=400.0, freq=0.0005):
        """
        Procedural canyon width parameterization.
        
        Args:
            base_width (float): Baseline width of the canyon in feet.
            amplitude (float): Amplitude of sinusoidal variation in feet.
            freq (float): Spatial frequency in rad/ft.
        """
        self.base_width = base_width
        self.amplitude = amplitude
        self.freq = freq

    def get_geometry(self, p_N):
        """
        Computes local width and spatial gradient.
        
        Args:
            p_N (float): Inertial North coordinate in feet.
            
        Returns:
            width (float): W_c(p_N) in feet.
            grad (float): dW_c/dp_N
        """
        width = self.base_width + self.amplitude * np.sin(self.freq * p_N)
        grad = self.amplitude * self.freq * np.cos(self.freq * p_N)
        return width, grad
