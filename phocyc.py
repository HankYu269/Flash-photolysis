import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

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
    text = "τ: {} s".format(round(popt[1], 3))
    return cal_time, fit_abs, text

def volt_to_abs (df, option):

    df = np.array(df)
    v0 = df[0:999, 1]
    volt = df[0:, 1]
    v0avg = np.mean(v0)
    abs = []
    for i in volt:
        abs.append(-(np.log10(i/v0avg)))
    
    # Find global extrema
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
    
    # 2D Savitzky-Golay filter
    sg_abs = savgol_filter(abs, 11, 2)

    # One-phase fitting
    fit_time, fit_abs, text = one_phase_fit(cal_time, cal_abs)
    print("critical value: {} at {} s".format(round(cri_abs, 3), cri_time))
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    font = {'family': 'arial', 'size': 20, 'weight': 'semibold'}
    plt.plot(time, sg_abs, 'gray', label='Raw data')
    plt.plot(fit_time, fit_abs, 'r', label='Fitting curve')
    plt.xlabel('Time (s)', fontdict=font, labelpad=13)
    plt.ylabel('Δ Abs. (AU)', fontdict=font, labelpad=20)
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(input_path[0:10], y=1.03, fontfamily='arial', fontsize=25, fontweight='semibold')
    ax.legend(loc='best', fontsize='xx-large', framealpha=0)
    ax.set_aspect(0.65/ax.get_data_ratio(), adjustable='box')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.75)
    props = dict(alpha=0.5, boxstyle='round', facecolor='white')
    ax.text(0.75, 0.25, text, bbox=props, fontsize=21, transform=ax.transAxes)
    ax.tick_params(width=1.75)
    plt.show()
    return fig


if __name__ == '__main__':
    import sys
    import os
    input_path = sys.argv[1]
    option = sys.argv[2]
    df = pd.read_csv(input_path, skiprows=15, header=None, index_col=None)
    fig = volt_to_abs(df, option)
    input_path = os.path.splitext(input_path)[0]
    #fig.savefig(input_path+"_1.png", bbox_inches='tight')
