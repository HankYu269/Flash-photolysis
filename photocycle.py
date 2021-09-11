import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def one_phase_fit (cal_time, cal_abs):

    # a = Y0, tau = time constant, p = plateau
    def exp_func(x, a, tau, p):
        return (a-p)*np.exp(-(x/tau))+p
    
    # Curve fitting
    popt, pcov = curve_fit(exp_func, cal_time, cal_abs)
    fit_abs = [exp_func(i, *popt) for i in cal_time]
    hftime = np.log(2) * popt[1]
    print("tau: {} s".format(round(popt[1], 3)))
    print("hftime: {} s".format(round(hftime, 3)))
    return cal_time, fit_abs


def volt_to_abs (df, option):

    df = np.array(df)
    v0 = df[0:999, 1]
    volt = df[0:, 1]
    v0avg = np.mean(v0)
    abs = []
    for i in volt:
        abs.append(-(np.log10(i/v0avg)))
    
    # Find the extremum
    time = df[0:, 0]
    if option == 'min':
        index = abs.index(min(abs))
    elif option == 'max':
        index = abs.index(max(abs))
    cri_abs = abs[index]
    cri_time = time[index]
    
    cal_time, cal_abs = [], []
    for i in range(len(time[0:])):
        if (i >= index):
            cal_time.append(time[i])
            
    for i in range(len(abs[0:])):
        if (i >= index):
            cal_abs.append(abs[i])
    
    # One-phase fitting
    fit_time, fit_abs = one_phase_fit(cal_time, cal_abs)
    print("critical value: {} at {} s".format(round(cri_abs, 3), cri_time))
    
    # Plotting
    plt.plot(time, abs, 'k')
    plt.plot(fit_time, fit_abs, 'r')
    plt.show()


if __name__ == '__main__':
    import sys
    import os
    input_path = sys.argv[1]
    option = sys.argv[2]
    df = pd.read_csv(input_path, skiprows=15, header=None, index_col=None)
    volt_to_abs(df, option)
    input_path = os.path.splitext(input_path)[0]
    #fig.savefig(input_path+".png", bbox_inches='tight')
