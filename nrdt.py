import time as ti
import soundfile as sf
import numpy as np


def read_signal(wfile):
    data, samplerate = sf.read(wfile)
    N = np.shape(data)[0]
    data = np.reshape(data.T, (1, N))
    # iesirea este in formatul acceptat de functiile RDT
    return data


def NRDT(signal, w, flag, channels):
    # Signal is an NP.ARRAY - format [1,N]
    # channells is a list of delays
    #
    # ----------- channels = 1 2 5 ... 9

    signal = signal.astype('float32')  # may help
    Nsamples = np.size(signal, 1)
    delmax = w / 4  # delay should be no more than w/4 (w usually is a power of 2)
    res = np.where(channels <= delmax)
    # print(res)
    channels = channels[res]  # remove chanells not satisfyiong this condition.
    m = np.shape(channels)[0]

    spectrograms = Nsamples // w  # The number of spectrograms computed
    Samples = spectrograms * w  # The number of samples used to compute the spectrograms.The other samples are discarded
    matrix = np.reshape(signal[0, 0:Samples],
                        (spectrograms, w))  # each line is one to be submited for computation of spectrogram

    spectrum = np.zeros((m, spectrograms))
    for i in range(0, spectrograms):
        values = matrix[i, :]  # the whole line
        for k in range(0, m):
            delay = channels[k]  # delays
            t = np.array(range(delay, w - delay - 1))
            difus = np.abs(values[t - delay] + values[t + delay] - 2 * values[t])
            if flag == 0:
                spectrum[k, i] = np.mean(difus) / 4
            elif flag == 1:
                spectrum[k, i] = np.mean(
                    difus / (np.abs(values[t - delay]) + np.abs(values[t + delay]) + 2 * np.abs(values[t]) + 1e-12)) / 4
    return spectrum


def get_features_nrdt(filename, M, w, flag, prag, chan):
    # Implements "spectral image (F3org)" using RDT applied on  M  segments, W window size .
    # Gives errors if the number of windows per segment is smaller than  1
    # Needs tuning of M, w.
    # chan - a list of delays for the "spectral" channels
    # prag - it is usually taken 0 (in special cases larger)
    # flag - 0 (normal) / 1 (scaled RDT)
    # =========================================================================
    signal = read_signal(filename)
    print(signal.shape)
    # signal [1,N] este scalaat -1,1
    delmax = w / 4  # ne asiguram ca delay-ul maxim nu depaseste w/4 (w este de regula putere a lui 2)
    res = np.where(chan <= delmax)
    # print(res)
    channels = chan[res]
    m = np.shape(chan)[0]

    # print('Threshold for sample removal', prag )
    # print('Full length of original signal is : ',np.size(signal))
    t1 = ti.time()
    res = np.where(np.abs(signal) >= prag)
    semnal = signal[0, res[1]]
    semnal = np.reshape(semnal.T, (1, np.shape(semnal)[0]))

    Features = np.zeros((M * m))
    Feat_spec = np.zeros((M, m))
    Npsgm = np.shape(semnal)[1] // M  # The number of samples per each segment
    print('Nsegm=', Npsgm, 'Windows per each segment: ', Npsgm // w)
    try:
        for isgm in range(0, M):  # Calculate the RDT on each segment of the signal
            ssegment = np.reshape(semnal[0, isgm * Npsgm:(isgm + 1) * Npsgm], (1, Npsgm))
            spectrum = NRDT(ssegment, w, flag, chan)
            mediumRDT = sum(
                np.transpose(spectrum))  # The medium spectrogram is the sum on columns of the transposed spectrum matrix
            Features[isgm * m:(isgm + 1) * m] = mediumRDT  # The feature vector for the signal to be recognized
            Feat_spec[isgm, :] = mediumRDT.T
    except Exception:
        return (None, "Number of samples is too low")
    t2 = ti.time()

    return Features, Feat_spec