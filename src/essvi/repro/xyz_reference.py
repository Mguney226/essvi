"""Golden numbers from the methodology §10 synthetic worked example ("XYZ").

These validate the SSVI math independently of real data. The worked example uses the SIMPLIFIED
power law phi = eta/theta^gamma for hand arithmetic (production carries the (1+theta)^(1-gamma)
factor, within ~0.5%).
"""

# Stage 4 fitted parameters
THETA_1 = 0.007646
THETA_2 = 0.024137
RHO = -0.32682
ETA = 0.6858
GAMMA = 0.4992

# phi (simplified form) the methodology prints in `phi`
PHI_1 = 7.8124
PHI_2 = 4.4011

# Stage 5 Gatheral-Jacquier butterfly quantities (simplified phi)
#   slice 1:  theta*phi*(1+|rho|) = 0.0793 ; theta*phi^2*(1+|rho|) = 0.6192
#   slice 2:  theta*phi*(1+|rho|) = 0.1409 ; theta*phi^2*(1+|rho|) = 0.6203
GJ_1A, GJ_1B = 0.0793, 0.6192
GJ_2A, GJ_2B = 0.1409, 0.6203

# Forward-ATM vols sigma = sqrt(theta/t), t1=0.104110, t2=0.276712
T_1, T_2 = 0.104110, 0.276712
SIGMA_ATM_1, SIGMA_ATM_2 = 0.2710, 0.2953
