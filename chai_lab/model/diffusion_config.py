class DiffusionConfig:
    BASE_S_churn: float = 80
    BASE_S_tmax: float = 80.0
    BASE_S_noise: float = 1.003  # add this to increase noise and results diversity in general 
    BASE_sigma_data: float = 16.0  # time-dependent noise factor, affect the power of each denoise, checkout: inference_noise_schedule.get_schedule
    
    # PDB guided configuration (for cases with initial coordinates)
    GUIDED_S_churn: float = 20
    GUIDED_S_tmax: float = 10.0
    GUIDED_S_noise: float = 1.001
    GUIDED_sigma_data: float = 4.0  
    
    # Runtime configuration (dynamically set based on whether PDB initialization is used)
    S_churn: float = BASE_S_churn
    S_tmin: float = 4e-4
    S_tmax: float = BASE_S_tmax
    S_noise: float = BASE_S_noise
    sigma_data: float = BASE_sigma_data
    second_order: bool = True  # EDM 2nd order correction, always set to True
    
    @classmethod
    def set_guided_mode(cls, enable_guided: bool = True):
        """Set whether to use guided diffusion mode"""
        if enable_guided:
            cls.S_churn = cls.GUIDED_S_churn
            cls.S_tmax = cls.GUIDED_S_tmax
            cls.S_noise = cls.GUIDED_S_noise
            cls.sigma_data = cls.GUIDED_sigma_data
        else:
            cls.S_churn = cls.BASE_S_churn
            cls.S_tmax = cls.BASE_S_tmax
            cls.S_noise = cls.BASE_S_noise
            cls.sigma_data = cls.BASE_sigma_data
