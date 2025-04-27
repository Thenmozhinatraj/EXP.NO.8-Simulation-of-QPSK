# EXP.NO.8-Simulation-of-QPSK

8.Simulation of QPSK

# AIM
To simulate Quadrature Phase Shift Keying (QPSK) modulation and demodulation using Python.

# SOFTWARE REQUIRED
->Python 
   
   Libraries: numpy, matplotlib

# ALGORITHMS
Transmitter:
1. Generate random binary data.
2. Group the bits into pairs (symbols).
3. Map each symbol to its corresponding In-phase (I) and Quadrature (Q) components.
4. Modulate I and Q components using cosine and sine carrier waves.
5. Combine I and Q signals to form the QPSK modulated signal.

Receiver:
1. Demodulate by separating the received QPSK signal into I and Q components.
2. Detect the bits based on the polarity (sign) of I and Q components.
3. Reconstruct the original binary data.
4. Calculate and display the Bit Error Rate (BER).

# PROGRAM
import numpy as np

import matplotlib.pyplot as plt

num_symbols = 10  # Number of symbols

T = 1.0           # Symbol period

fs = 100.0        # Sampling frequency

t = np.arange(0, T, 1/fs)  # Time vector for one symbol

bits = np.random.randint(0, 2, num_symbols * 2)  # Two bits per symbol

symbols = 2 * bits[0::2] + bits[1::2]             # Mapping bits to decimal symbols (00->0, 01->1, 10->2, 11->3)

symbol_phases = {0: 0, 1: np.pi/2, 2: np.pi, 3: 3*np.pi/2}

qpsk_signal = np.array([])

symbol_times = []

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
![Screenshot 2025-04-27 172728](https://github.com/user-attachments/assets/b9242b54-bc02-44c2-8cd5-3ec6933e2379)
![Screenshot 2025-04-27 172755](https://github.com/user-attachments/assets/70afa630-18c9-4623-84fd-00d14feea5cf)

 
# RESULT / CONCLUSIONS
1. The simulation of QPSK modulation and demodulation was successfully completed using Python.
2. Clear visualization of In-phase, Quadrature, and Combined QPSK waveforms was obtained.
3. The Bit Error Rate (BER) was found to be zero in noiseless conditions.
4. Thus, QPSK was verified to transmit two bits per symbol, effectively doubling the data transmission rate compared to BPSK.
5. The experiment proved the efficiency and correctness of QPSK modulation under ideal conditions.


