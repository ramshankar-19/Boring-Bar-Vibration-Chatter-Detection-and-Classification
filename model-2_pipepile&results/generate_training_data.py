#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Boring Bar Data Generator for Shuffle-BiLSTM Training
--------------------------------------------------------------
Generates diverse dataset across multiple cutting parameters
to create robust training data for chatter detection ML model.

Based on Liu et al. (2023) experimental setup:
- 192 cutting experiments
- Varying: cutting speed, feed rate, cutting depth, overhang
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import os

# ===================== CONFIGURATION =====================

class BoringBarConfig:
    """Configuration for boring bar and cutting parameters"""
    
    # Sampling parameters
    fs = 40000              # Sampling frequency [Hz]
    duration_per_cut = 3.0  # Duration per cutting experiment [s]
    
    # Boring bar material (AISI 4340)
    E = 205e9               # Young's modulus [Pa]
    rho = 7850              # Density [kg/m³]
    poisson = 0.29          # Poisson's ratio
    
    # Experimental ranges (based on Liu et al. Table 1)
    cutting_speeds = [100, 200, 300, 400]      # m/min
    feed_rates = [0.1, 0.2, 0.3, 0.4]          # mm/rev
    cutting_depths = [0.1, 0.2, 0.3, 0.4]      # mm
    overhangs = [420, 480, 540]                 # mm
    
    # Bar diameter (fixed)
    diameter = 60  # mm
    
    # Damping ratio range
    damping_ratios = [0.01, 0.0167, 0.02]  # Structural damping


# ===================== BORING BAR DYNAMICS =====================

class BoringBarModel:
    """Physics-based boring bar vibration model"""
    
    def __init__(self, overhang_mm, diameter_mm, damping_ratio):
        self.L = overhang_mm / 1000.0  # Convert to meters
        self.D = diameter_mm / 1000.0
        self.zeta = damping_ratio
        
        # Calculate natural frequency (Timoshenko beam - first mode)
        self.calculate_natural_frequency()
        
        # Effective modal mass
        self.m_eff = self.calculate_modal_mass()
        
    def calculate_natural_frequency(self):
        """Calculate first bending mode natural frequency"""
        # For cantilevered beam, first mode coefficient
        lambda_1 = 1.875  # First mode eigenvalue
        
        # Second moment of area (hollow circular section)
        I = np.pi * (self.D**4) / 64
        
        # Cross-sectional area
        A = np.pi * (self.D**2) / 4
        
        # Natural frequency (Euler-Bernoulli approximation)
        omega_n = (lambda_1**2 / self.L**2) * np.sqrt(
            (BoringBarConfig.E * I) / (BoringBarConfig.rho * A)
        )
        
        self.omega_n = omega_n  # rad/s
        self.f_n = omega_n / (2 * np.pi)  # Hz
        
    def calculate_modal_mass(self):
        """Calculate effective modal mass at tip"""
        A = np.pi * (self.D**2) / 4
        m_total = BoringBarConfig.rho * A * self.L
        
        # For first mode of cantilever beam
        m_eff = 0.25 * m_total
        return m_eff
    
    def cutting_force(self, t, v_cut, f_feed, ap, state='stable'):
        """
        Generate cutting force time series
        
        Parameters:
        - t: time array
        - v_cut: cutting speed [m/min]
        - f_feed: feed rate [mm/rev]
        - ap: depth of cut [mm]
        - state: 'stable', 'transition', or 'severe'
        """
        # Convert to SI units
        v_ms = v_cut / 60.0  # m/s
        
        # Spindle speed (RPM) - assuming workpiece diameter ~180mm
        workpiece_dia = 0.18  # meters
        spindle_rpm = (v_ms * 60) / (np.pi * workpiece_dia)
        f_spindle = spindle_rpm / 60.0  # Hz
        omega_spindle = 2 * np.pi * f_spindle
        
        # Mean cutting force (empirical formula)
        # F = Kc * ap * feed_rate
        Kc = 2000e6  # Cutting force coefficient [N/m²] for steel
        F_mean = Kc * (ap / 1000.0) * (f_feed / 1000.0)
        
        # Base spindle forcing
        F_base = F_mean * (1 + 0.1 * np.sin(omega_spindle * t))
        
        if state == 'stable':
            # Small random variations
            F = F_base + 0.02 * F_mean * np.random.randn(len(t))
            
        elif state == 'transition':
            # Chatter frequency slightly offset from natural frequency
            f_chatter = self.f_n + np.random.uniform(-15, 15)
            omega_chatter = 2 * np.pi * f_chatter
            
            # Growing chatter amplitude
            growth = 1.0
            envelope = np.tanh(growth * t / BoringBarConfig.duration_per_cut)
            F_chatter = 0.2 * F_mean * np.sin(omega_chatter * t) * envelope
            
            F = F_base + F_chatter + 0.05 * F_mean * np.random.randn(len(t))
            
        elif state == 'severe':
            # Strong chatter at natural frequency
            omega_chatter = 2 * np.pi * self.f_n
            
            # Exponential growth (regenerative mechanism)
            growth = 2.0
            envelope = np.exp(growth * t / BoringBarConfig.duration_per_cut)
            envelope = np.clip(envelope, 1, 4)
            
            # Fundamental + harmonics
            F_chatter = (
                0.5 * F_mean * np.sin(omega_chatter * t) +
                0.25 * F_mean * np.sin(2 * omega_chatter * t) +
                0.15 * F_mean * np.sin(3 * omega_chatter * t)
            ) * envelope
            
            F = F_base + F_chatter + 0.1 * F_mean * np.random.randn(len(t))
        
        return F
    
    def simulate_acceleration(self, t, v_cut, f_feed, ap, state='stable'):
        """Simulate axial acceleration response"""
        
        # Generate cutting force
        F = self.cutting_force(t, v_cut, f_feed, ap, state)
        
        # ODE: m*a + c*v + k*x = F
        # x'' + 2*zeta*omega_n*x' + omega_n^2*x = F/m
        
        def equation(y, t_val):
            x, v = y
            F_t = np.interp(t_val, t, F)
            a = (F_t / self.m_eff) - 2 * self.zeta * self.omega_n * v - self.omega_n**2 * x
            return [v, a]
        
        # Solve ODE
        y0 = [0.0, 0.0]
        solution = odeint(equation, y0, t)
        
        # Extract displacement
        x = solution[:, 0]
        
        # Calculate acceleration
        dt = t[1] - t[0]
        a_x = np.gradient(np.gradient(x, dt), dt)
        
        # Add sensor noise (based on state)
        if state == 'stable':
            noise = 0.5
        elif state == 'transition':
            noise = 1.0
        else:  # severe
            noise = 2.0
        
        a_x += noise * np.random.randn(len(a_x))
        
        # Sensor limits (±50g)
        g = 9.81
        a_x = np.clip(a_x, -50*g, 50*g)
        
        return a_x


# ===================== DATA GENERATION MANAGER =====================

class DatasetGenerator:
    """Manages generation of complete training dataset"""
    
    def __init__(self, output_dir='boring_bar_dataset'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.experiment_count = 0
        self.metadata = []
        
        print(f"Dataset will be saved to: {self.output_dir.absolute()}")
        
    def generate_single_experiment(self, overhang, diameter, damping, 
                                   v_cut, f_feed, ap, state):
        """Generate one cutting experiment"""
        
        # Create boring bar model
        model = BoringBarModel(overhang, diameter, damping)
        
        # Time array
        t = np.arange(0, BoringBarConfig.duration_per_cut, 
                      1/BoringBarConfig.fs)
        
        # Simulate acceleration
        a_x = model.simulate_acceleration(t, v_cut, f_feed, ap, state)
        
        # Create experiment ID
        exp_id = f"exp_{self.experiment_count:04d}"
        
        # Store metadata
        metadata = {
            'exp_id': exp_id,
            'overhang_mm': overhang,
            'diameter_mm': diameter,
            'damping_ratio': damping,
            'cutting_speed_m_min': v_cut,
            'feed_rate_mm_rev': f_feed,
            'cutting_depth_mm': ap,
            'state': state,
            'natural_freq_hz': model.f_n,
            'samples': len(a_x),
            'duration_s': BoringBarConfig.duration_per_cut
        }
        
        self.metadata.append(metadata)
        self.experiment_count += 1
        
        return t, a_x, metadata
    
    def generate_full_dataset(self, experiments_per_state=64):
        """
        Generate complete dataset with balanced classes
        
        Target: 192 experiments (as in Liu et al. paper)
        - 64 stable
        - 64 transition
        - 64 severe
        """
        
        print("\n" + "="*70)
        print("GENERATING FULL BORING BAR DATASET")
        print("="*70)
        
        all_data = []
        
        for state in ['stable', 'transition', 'severe']:
            print(f"\nGenerating {experiments_per_state} '{state}' experiments...")
            
            for i in range(experiments_per_state):
                # Randomly sample parameters
                overhang = np.random.choice(BoringBarConfig.overhangs)
                damping = np.random.choice(BoringBarConfig.damping_ratios)
                v_cut = np.random.choice(BoringBarConfig.cutting_speeds)
                f_feed = np.random.choice(BoringBarConfig.feed_rates)
                ap = np.random.choice(BoringBarConfig.cutting_depths)
                
                # Generate experiment
                t, a_x, metadata = self.generate_single_experiment(
                    overhang=overhang,
                    diameter=BoringBarConfig.diameter,
                    damping=damping,
                    v_cut=v_cut,
                    f_feed=f_feed,
                    ap=ap,
                    state=state
                )
                
                # Store data
                df = pd.DataFrame({
                    'time_s': t,
                    'accel_x_m_s2': a_x,
                    'exp_id': metadata['exp_id'],
                    'state': state
                })
                
                all_data.append(df)
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"  Completed {i+1}/{experiments_per_state}")
        
        # Combine all data
        df_all = pd.concat(all_data, ignore_index=True)
        
        # Save combined dataset
        output_file = self.output_dir / 'boring_bar_full_dataset.csv'
        df_all.to_csv(output_file, index=False)
        print(f"\n✓ Saved full dataset: {output_file}")
        print(f"  Total samples: {len(df_all):,}")
        
        # Save metadata
        metadata_file = self.output_dir / 'experiment_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"✓ Saved metadata: {metadata_file}")
        
        # Save separate files by state
        for state in ['stable', 'transition', 'severe']:
            df_state = df_all[df_all['state'] == state]
            state_file = self.output_dir / f'boring_bar_{state}.csv'
            df_state.to_csv(state_file, index=False)
            print(f"✓ Saved {state} data: {state_file} ({len(df_state):,} samples)")
        
        # Generate summary statistics
        self.generate_summary_report(df_all)
        
        return df_all
    
    def generate_summary_report(self, df):
        """Generate summary statistics and visualizations"""
        
        print("\n" + "="*70)
        print("DATASET SUMMARY")
        print("="*70)
        
        # Class distribution
        print("\nClass Distribution:")
        class_counts = df.groupby('state')['exp_id'].nunique()
        for state, count in class_counts.items():
            print(f"  {state.capitalize()}: {count} experiments")
        
        # Acceleration statistics by state
        print("\nAcceleration Statistics by State:")
        for state in ['stable', 'transition', 'severe']:
            data = df[df['state'] == state]['accel_x_m_s2']
            print(f"\n  {state.capitalize()}:")
            print(f"    Mean: {data.mean():.3f} m/s²")
            print(f"    Std:  {data.std():.3f} m/s²")
            print(f"    Min:  {data.min():.3f} m/s²")
            print(f"    Max:  {data.max():.3f} m/s²")
        
        # Save visualization
        self.plot_sample_data(df)
        
        print("\n" + "="*70)
    
    def plot_sample_data(self, df, samples_per_state=3):
        """Plot sample data from each state"""
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        for idx, state in enumerate(['stable', 'transition', 'severe']):
            # Get sample experiments
            exp_ids = df[df['state'] == state]['exp_id'].unique()[:samples_per_state]
            
            for exp_id in exp_ids:
                data = df[df['exp_id'] == exp_id]
                t = data['time_s'].values
                a_x = data['accel_x_m_s2'].values
                
                # Time domain
                axes[idx, 0].plot(t, a_x, alpha=0.6, linewidth=0.5)
                
                # Frequency domain
                f, Pxx = signal.welch(a_x, fs=BoringBarConfig.fs, nperseg=4096)
                axes[idx, 1].semilogy(f, Pxx, alpha=0.6)
            
            # Formatting
            axes[idx, 0].set_ylabel('Acceleration [m/s²]')
            axes[idx, 0].set_title(f'{state.capitalize()} - Time Domain')
            axes[idx, 0].grid(True, alpha=0.3)
            
            axes[idx, 1].set_ylabel('PSD [m²/s⁴/Hz]')
            axes[idx, 1].set_title(f'{state.capitalize()} - Frequency Domain')
            axes[idx, 1].set_xlim([0, 500])
            axes[idx, 1].grid(True, alpha=0.3)
        
        axes[2, 0].set_xlabel('Time [s]')
        axes[2, 1].set_xlabel('Frequency [Hz]')
        
        plt.tight_layout()
        
        plot_file = self.output_dir / 'sample_data_visualization.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization: {plot_file}")
        plt.close()


# ===================== MAIN EXECUTION =====================

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("BORING BAR TRAINING DATA GENERATOR FOR SHUFFLE-BiLSTM")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Sampling frequency: {BoringBarConfig.fs/1000:.0f} kHz")
    print(f"  Duration per experiment: {BoringBarConfig.duration_per_cut} s")
    print(f"  Samples per experiment: {int(BoringBarConfig.fs * BoringBarConfig.duration_per_cut):,}")
    print(f"\nParameter Ranges:")
    print(f"  Cutting speeds: {BoringBarConfig.cutting_speeds} m/min")
    print(f"  Feed rates: {BoringBarConfig.feed_rates} mm/rev")
    print(f"  Cutting depths: {BoringBarConfig.cutting_depths} mm")
    print(f"  Overhangs: {BoringBarConfig.overhangs} mm")
    
    # Create generator
    generator = DatasetGenerator(output_dir='boring_bar_dataset')
    
    # Generate dataset (192 total experiments = 64 per state)
    df_all = generator.generate_full_dataset(experiments_per_state=64)
    
    print("\n" + "="*70)
    print("✓ DATA GENERATION COMPLETE!")
    print("="*70)
    print(f"\nTotal experiments: {len(df_all['exp_id'].unique())}")
    print(f"Total samples: {len(df_all):,}")
    print(f"Dataset size: ~{len(df_all) * 8 / 1e6:.1f} MB")
    print(f"\nOutput directory: {generator.output_dir.absolute()}")
    
    print("\nNext steps:")
    print("  1. Review generated data in 'boring_bar_dataset' folder")
    print("  2. Apply SPWVD transformation (use step3_spwvd_transform.py)")
    print("  3. Train Shuffle-BiLSTM model")
    

if __name__ == "__main__":
    main()