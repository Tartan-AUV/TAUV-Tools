import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import math


class Stats(Enum):
    PWM = "PWM (µs)"
    RPM = "RPM"
    CURRENT = "Current (A)"
    VOLTAGE = "Voltage (V)"
    POWER = "Power (W)"
    FORCE = "Force (Kg f)"
    EFFICIENCY = "Efficiency (g/W)"


class ThrustCurveGenerator:
    def __init__(self, data_fpath):
        self.test_voltages = [10, 12, 14, 16, 18, 20]
        self.sheet_names = [f"{v} V" for v in self.test_voltages]
        dataframe = pd.read_excel(data_fpath, sheet_name=self.sheet_names)

        self.data = dict()

        for v in self.test_voltages:
            sheet = f"{v} V"
            curr_df = dataframe[sheet]
            stats = {}

            for col in curr_df.keys():
                paren_idx = col.find("(")
                if paren_idx >= 0:
                    enum_name = col[:paren_idx].strip().upper()
                else:
                    enum_name = col.strip().upper()
                
                if Stats[enum_name] in self.data:
                    self.data[Stats[enum_name]][v] = dataframe[sheet][col].to_numpy()
                else:
                    self.data[Stats[enum_name]] = {v: dataframe[sheet][col].to_numpy()}
        
        # Number of values less than 1500 -- where thrusters go from reverse to forward direction
        self.cutoff_point = (self.data[Stats.PWM][10] < 1500).sum()
        
        self.fwd_data = dict()
        self.rev_data = dict()
        
        for k in self.data.keys():
            self.fwd_data[k] = {v: self.data[k][v][self.cutoff_point:] for v in self.test_voltages}
            self.rev_data[k] = {v: self.data[k][v][:self.cutoff_point] for v in self.test_voltages}

    def quadratic_roots(self, a, b, c):
        solns = [
            (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a),
            (-b - math.sqrt(b ** 2 - 4 * a * c)) / (2 * a),
        ]

        return solns
    
    def bivar_thrust_model(self, fwd_coeffs, rev_coeffs):
        def inv(thrust_des, v):
            if thrust_des < 0:
                a1, b1, a2, b2, c = rev_coeffs
                selector = min
            else:
                a1, b1, a2, b2, c = fwd_coeffs
                selector = max
                
            new_c = a2 * v ** 2 + b2 * v + c - thrust_des
            return selector(self.quadratic_roots(a1, b1, new_c))

        def fwd(pwm, v):
            if pwm >= self.data[Stats.PWM][10][self.cutoff_point]:
                a1, b1, a2, b2, c = fwd_coeffs
            else:
                a1, b1, a2, b2, c = rev_coeffs
            return a1 * pwm ** 2 + b1 * pwm + a2 * v ** 2 + b2 * v + c

        return inv, fwd

    def linterp_thrust_model(self, fwd_polys, rev_polys):
        def select_bounds(v):
            if v < 10 or v > 20:
                raise LookupError

            if 10 <= v < 12:
                lower, upper = 10, 12
            elif 12 <= v < 14:
                lower, upper = 12, 14
            elif 14 <= v < 16:
                lower, upper = 14, 16
            elif 16 <= v < 18:
                lower, upper = 16, 18
            else:
                lower, upper = 18, 20

            return lower, upper

        def inv(thrust_des, v):
            try:
                lower, upper = select_bounds(v)
            except LookupError:
                print("Voltage outside operating range")
                return
            scale = (v - lower) / 2
            
            if thrust_des < 0:
                polys = rev_polys
                selector = min
            else:
                polys = fwd_polys
                selector = max
                
            (a, b, c) = (1 - scale) * polys[lower] + scale * polys[upper]
            return selector(self.quadratic_roots(a, b, c - thrust_des))

        def fwd(pwm, v):
            try:
                lower, upper = select_bounds(v)
            except LookupError:
                print("Voltage outside operating range")
                return
            scale = (v - lower) / 2

            if pwm >= self.data[Stats.PWM][10][self.cutoff_point]:
                polys = fwd_polys
            else:
                polys = rev_polys

            (a, b, c) = (1 - scale) * polys[lower] + scale * polys[upper]
            return a * pwm ** 2 + b * pwm + c

        return inv, fwd
    
    def find_bivar_coeffs(self):
        def solve_dir(pwms, kgfs):
            data_pts = []
            b_pts = []

            for v in self.test_voltages:
                for pwm, kgf in zip(pwms[v], kgfs[v]):
                    data_pts.append([
                        pwm ** 2, pwm, v ** 2, v, 1
                    ])
                    b_pts.append(kgf)

            A_matrix = np.array(data_pts)
            b_vec = np.array(b_pts)

            coeffs, _, _, _ = np.linalg.lstsq(A_matrix, b_vec)
            return coeffs

        pwms_fwd, kgfs_fwd = self.fwd_data[Stats.PWM], self.fwd_data[Stats.FORCE]
        pwms_rev, kgfs_rev = self.rev_data[Stats.PWM], self.rev_data[Stats.FORCE]
        
        coeffs_fwd = solve_dir(pwms_fwd, kgfs_fwd)
        coeffs_rev = solve_dir(pwms_rev, kgfs_rev)

        return coeffs_fwd, coeffs_rev

    def find_linterp_polys(self):
        def solve_dir(pwms, kgfs):
            polynomials = {}

            for k in self.test_voltages:
                k_pwms, k_kgfs = pwms[k], kgfs[k]
                coeffs = np.polyfit(x=k_pwms, y=k_kgfs, deg=2)
                polynomials[k] = coeffs

            return polynomials
        
        pwms_fwd, kgfs_fwd = self.fwd_data[Stats.PWM], self.fwd_data[Stats.FORCE]
        pwms_rev, kgfs_rev = self.rev_data[Stats.PWM], self.rev_data[Stats.FORCE]
        
        polys_fwd = solve_dir(pwms_fwd, kgfs_fwd)
        polys_rev = solve_dir(pwms_rev, kgfs_rev)

        return polys_fwd, polys_rev
