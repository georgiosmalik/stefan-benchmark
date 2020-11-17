#---------------------------------
# Parameters of the Stefan problem
#---------------------------------

# Parametry ulohy: geometricke na generovani site, fyzikalni pro formulaci problemu,
# Ukoly:
# 1. prepsat veliciny, aby odpovidaly znaceni v clanku
# 2. vymazat nepouzivane veliciny

#------------------------------------------------
# Material and physical parameters of the problem
#------------------------------------------------
# Physical parameters:
L_m = 335e3                 # Latent heat of melting [J/kg]
rho = 1000.                 # Density [kg*m**(-3)]
cp_l = 4182.                # Specific heat of water
cp_s = 2116.                # Specific heat of ice
cp_m = (cp_l+cp_s)/2        # Specific heat of the mushy region
k_l = 0.6                   # Thermal conductivity of water [W/m/K]
k_s = 2.26                  # Thermal conductivity of ice [W/m/K]
kappa_l = k_l/(rho*cp_l)    # Heat diffusivity of water
kappa_s = k_s/(rho*cp_s)    # Heat diffusivity of ice
#------------------------------------------------
# Problem formulation parameters:
# Geometric:
# 2d,3d
R1 = 0.1
R2 = 1.
meshres = {1:1000,2:220,3:0.002}      # 2d: 220 results in h_max=0.009
# Physical:
q_0 = 2e5                           # source heat flux in origin (value for 1d)
theta_0 = 373.                      # heating temperature for 1D
theta_m = 273.                      # melting temperature
theta_i = 263.                      # initial temperature
#================================================
