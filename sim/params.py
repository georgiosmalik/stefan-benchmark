#---------------------------------
# Parameters of the Stefan problem
#---------------------------------

# Parametry ulohy: geometricke na generovani site, fyzikalni pro formulaci problemu,

#------------------------------------------------
# Material and physical parameters of the problem
#------------------------------------------------
# Physical parameters:
L_m = 335e3               # Latent heat of melting
rho = 1000.
rho_l = 1000.            # Density of water (Kowal value: 999.840281167108)
rho_s = 917.             # Density of ice
cp_l = 4182.              # Specific heat of water
cp_s = 2116.              # Specific heat of ice
cp_m = (cp_l+cp_s)/2                   # Specific heat of the mushy region
k_l = 0.6               # Thermal conductivity of water
k_s = 2.26                # Thermal conductivity of ice
mu_l = 0.00103               # Dynamic viscosity of water
mu_s = 1e5                # Dynamic viscosity of ice
nu_l = mu_l/rho_l            # Kinematic viscosity of water
nu_s = mu_s/rho_s            # Kinematic viscosity of ice
kappa_l = k_l/(rho_l*cp_l)    # Heat diffusivity of water
kappa_s = k_s/(rho_s*cp_s)    # Heat diffusivity of ice
kappa = kappa_s/kappa_l             # Heat diffusivity ratio
alpha_l = 2.5e-4          # Thermal expansivity of water
alpha_s = 0.0             # Thermal expansivity of ice
g = 9.81                  # Gravitational acceleration
#------------------------------------------------
# Problem formulation parameters:
# Geometric:
# 2d,3d
R1 = 0.1
R2 = 1.
meshres = 20
# Physical:
theta_0 = 373.                      # heating temperature for 1D
theta_0_3D = 5.                     # heating power for 3D
theta_m = 273.                      # melting temperature
theta_inf = 263.                    # initial temperature
delta_theta = theta_0 - theta_m     # temperature difference
K = 1.                              # boundary heat flow for Stefan 2D
theta_left = 283.                   # Heating temperature [K]
theta_right = 263.                  # Freezing temperature [K]
#------------------------------------------------
# Dimensionless parameters:
beta = rho_s/rho_l*L_m/cp_l/delta_theta
                                    # Inverse Stefan number
St_l = cp_l*(theta_0 - theta_m)/L_m
                                    # Stefan number for liquid region
St_s = cp_s*(theta_m - theta_inf)/L_m
                                    # Stefan number for liquid region
#================================================
