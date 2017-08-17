# ======================================
# Simple standard functions

from math import *

def Step( x_step ):
    return lambda x: 1.0 if x > x_step else 0.0

# Rect function by specifying boundaries
def Rect( x_l, x_r ):
    return lambda x: 1.0/(x_r-x_l) if ( x>=x_l and x<=x_r ) else 0.0

def Gauss( mu, sigma ):
    return lambda x: 1.0/(sqrt(2*pi)*sigma) * exp( -0.5 * ( (x-mu)/abs(sigma) )**2 )

# Rect function by giving central value and left and right error, integral normalized to 1.0
def RectAsym( x_c, e_l, e_r ):
    x_l = x_c - abs(e_l); x_r = x_c + abs(e_r)
    return lambda x: 1.0/(x_r-x_l) if ( x>=x_l and x<=x_r ) else 0.0

# Asymmetric Gaussian, integral normalized to 1.0
def GaussAsym( mu, sigma_l, sigma_r ):
    sigma_l = abs(sigma_l); sigma_r = abs(sigma_r)
    return lambda x: 1.0/(sqrt(2*pi)*sigma_r) * exp(-0.5 *((x-mu)/abs(sigma_r))**2) if x > mu else 1.0/(sqrt(2*pi)*sigma_l) * exp(-0.5 *((x-mu)/sigma_l)**2)
