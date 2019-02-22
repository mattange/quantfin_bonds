# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 10:22:45 2019

@author: mangelon
"""
import numpy as np
import pandas as pd
from scipy.optimize import newton


def price(yield_to_mat, redemption, rate, freq, maturity_dt, settlement_dt, 
          dcc="30/360 US", eom=False):
    """
    Returns the price given the yield to maturity maturity date, the face value at maturity,
    the interest rate paid with frequency freq
    
    """
    
    # calculate the number of payments on the basis of 365 days years
    freq_lbl = str(int(12/freq)) + 'M'
    Ncoupons = int(np.floor((maturity_dt - settlement_dt) / np.timedelta64(365,'D') * freq) + 1)
    dt = pd.date_range(end=maturity_dt, periods=Ncoupons+1, freq=freq_lbl)
    dtws = dt.insert(1, pd.to_datetime(settlement_dt))
    cf = coupon_factor(dt, dcc, eom)
    DSCcf = coupon_factor(dtws[1:3],dcc)
    Ecf = coupon_factor(dt[0:2],dcc)
    Acf = Ecf - DSCcf
    k = np.arange(1, Ncoupons+1)
    
    if np.isscalar(yield_to_mat):
        xx = 100 * rate*cf / (1 + yield_to_mat*cf)**(k-1 + DSCcf/Ecf)
        p = redemption / (1 + yield_to_mat/freq)**(Ncoupons-1 + DSCcf/Ecf) \
            + np.sum(xx) - 100 * rate/freq * Acf/Ecf
        p = np.asscalar(p)
    else:
        p = np.empty_like(yield_to_mat)
        for ix in range(yield_to_mat.size):
            xx = 100 * rate*cf / (1 + yield_to_mat[ix]*cf)**(k-1 + DSCcf/Ecf)
            p[ix] = redemption / (1 + yield_to_mat[ix]/freq)**(Ncoupons-1 + DSCcf/Ecf) \
                + np.sum(xx) - 100 * rate/freq * Acf/Ecf
    return p


def ytm(p, redemption, rate, freq, maturity_dt, settlement_dt, ytm_guess=None, 
        dcc="30/360 US", eom=False):
    """
    Returns the yield to maturity.
    
    """
    if ytm_guess is None:
        ytm_guess = rate
    if np.isscalar(p):
        def func(y):
            return p - price(y, redemption, rate, freq, maturity_dt, settlement_dt, dcc, eom)
        
        yy = newton(func, ytm_guess)
    else:
        yy = np.empty_like(p)
        def func(y, ix): 
            return p[ix] - price(y, redemption, rate, freq, maturity_dt, settlement_dt, dcc, eom)
        for ix in range(p.size):
            yy[ix] = newton(func, ytm_guess, args=(ix,))
    return yy


def dv01(x, redemption, rate, freq, maturity_dt, settlement_dt, 
         dcc="30/360 US", eom=False, x_is_yield=True, step=0.5):
    """
    Returns the numerically computed local DV01 for 1 bp move in the yield at the given price.
    Calculations done symmetrically at +/- bps step on the yield given or at +/- step of the price 
    given.
    
    """
       
    if x_is_yield is False:
        yy1 = ytm(x + step, redemption, rate, freq, maturity_dt, settlement_dt, dcc, eom)
        yy2 = ytm(x - step, redemption, rate, freq, maturity_dt, settlement_dt, dcc, eom)
        res = 2*step / (yy1 - yy2) / 10000
    else:
        p1 = price(x + step/10000, redemption, rate, freq, maturity_dt, settlement_dt, dcc, eom)
        p2 = price(x - step/10000, redemption, rate, freq, maturity_dt, settlement_dt, dcc, eom)
        res = (p1 - p2) / 2 / step
    return res
    
    
def coupon_factor(dt, dcc, eom=False):
    """
    Returns the coupon factors of periods between dates according to the daycount convention.
    
    """
    eom_date = dt.is_month_end
    Y = dt.year.values
    Y2 = Y[1:]
    Y1 = Y[:-1]
    M = dt.month.values
    M2 = M[1:]
    M1 = M[:-1]
    D = dt.day.values
    D2 = D[1:]
    D1 = D[:-1]
    if dcc == "30/360 US":
        # to be verified, not exactly accurate 100%
        D2[(eom & (M1==2 & eom_date[:-1])) & (M2==2 & eom_date[1:])] = 30
        D1[eom & (M1==2 & eom_date[:-1])] = 30
        D2[(D2==31) & ((D1==30) | (D1==31))] = 30
        D1[D1==31] = 30
        cf = (360*(Y2-Y1) + 30*(M2-M1) + (D2-D1)) / 360
    else:
        raise NotImplementedError()
        
    return cf
