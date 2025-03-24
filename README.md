# EXP.NO.8-Simulation-of-QPSK

8.Simulation of QPSK

# AIM
To analyse the modulation of QPSK Signal

# SOFTWARE REQUIRED
Personal Computer

Python

# ALGORITHMS
1. Initialize Parameters: Define symbols, period (T), and sampling frequency (fs).
2. Generate Bits: Create a random binary sequence.  
3. Map to Phases: Group bits in pairs → 00, 01, 10, 11 mapped to 4 phases.  
4. Create QPSK Signal: Modulate sine and cosine waves with mapped phases.  
5. Generate Time Vector: Create a time vector for the entire signal.  
6. Plot Signal: Visualize in-phase, quadrature, and resultant waveforms.  
7. Analyze Performance: Optionally add noise or compute BER.  
8. Display Results: Show waveforms and analysis plots.  

# PROGRAM
import numpy as np

import matplotlib.pyplot as plt

num_symbols = 10 

T = 1.0 # Symbol period

fs = 100.0 # Sampling frequency

t = np.arange(0, T, 1/fs) # Time vector for one symbol

bits = np.random.randint(0, 2, num_symbols * 2) # Two bits per QPSK symbol 

symbols = 2 * bits[0::2] + bits[1::2] # Map bits to QPSK symbols

qpsk_signal = np.array([]) 

symbol_times = []

symbol_phases = {0: 0, 1: np.pi/2, 2: np.pi, 3: 3*np.pi/2}

for i, symbol in enumerate(symbols): 

    phase = symbol_phases[symbol] 
    
    symbol_time = i * T
    
    qpsk_segment = np.cos(2 * np.pi * t / T + phase) + 1j * np.sin(2 * np.pi * t / T + phase) 
    
    qpsk_signal = np.concatenate((qpsk_signal, qpsk_segment))
    
    symbol_times.append(symbol_time)
    
t_total = np.arange(0, num_symbols * T, 1/fs)

plt.figure(figsize=(14, 12))

plt.subplot(3, 1, 1)

plt.plot(t_total, np.real(qpsk_signal), label='In-phase') 

for i, symbol_time in enumerate(symbol_times):

    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)
    
    plt.text(symbol_time + T/4, 0, f'{symbols[i]:02b}', fontsize=12, color='blue') 
    
plt.title('QPSK Signal - In-phase Component with Symbols')

plt.xlabel('Time') 

plt.ylabel('Amplitude') 

plt.grid(True) 

plt.legend()

plt.subplot(3, 1, 2)

plt.plot(t_total, np.imag(qpsk_signal), label='Quadrature', color='orange') 

for i, symbol_time in enumerate(symbol_times):

    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5) 
    
    plt.text(symbol_time + T/4, 0, f'{symbols[i]:02b}', fontsize=12, color='blue')
    
plt.title('QPSK Signal - Quadrature Component with Symbols') 

plt.xlabel('Time')

plt.ylabel('Amplitude') 

plt.grid(True)

plt.legend()

plt.subplot(3, 1, 3)

plt.plot(t_total, np.real(qpsk_signal), label='Resultant QPSK Waveform', color='green')

for i, symbol_time in enumerate(symbol_times):

    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5) 
    
    plt.text(symbol_time + T/4, 0, f'{symbols[i]:02b}', fontsize=12, color='blue')
    
plt.title('Resultant QPSK Waveform') 

plt.xlabel('Time') 

plt.ylabel('Amplitude')

plt.grid(True) 

plt.legend()

plt.tight_layout() 

plt.show()

# OUTPUT
![QPSK](https://github.com/user-attachments/assets/af3cbabe-2a97-46df-9caf-31ff70fa1f40)


 
# RESULT
The QPSK modulation simulation was successfully completed using Python.

The In-phase and Quadrature components were generated and visualized.

The resultant QPSK waveform accurately reflected phase shifts for each symbol.

Symbol transitions and phase mapping were clearly observable in the plots.
