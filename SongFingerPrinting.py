import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from IPython.display import Audio
import collections
import numpy as np
from microphone import record_audio
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.filters import minimum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.morphology import iterate_structure
import librosa
from collections import defaultdict
from numba import njit

class SongFingerPrinting():
    def __init__(self):
        self.database = collections.defaultdict(list)
        self.songnames = []
        self.fs = 0

    def findpeaks(self, spec):
        """
        Finds the local peaks of a given spectrogram

        Parameters
        ----------
        spec : Spectrogram or np.ndarray
            a spectrogram of the given sound data

        Returns
        -------
        bins : np.ndarray
            the coordinates of the local maximums within the spectrogram
        """

        if type(spec) != np.ndarray:
            songname, fs = librosa.load(spec, sr=44100, mono=True)
            self.fs = fs
            spec, freqs, times = mlab.specgram(songname, NFFT = 4096, Fs = fs, window=mlab.window_hanning, noverlap=(4096//2))

        ys, xs = np.histogram(spec.flatten(), bins=spec.size//2, normed=True)
        dx = xs[-1] - xs[-2]
        cdf = np.cumsum(ys)*dx  # this gives you the cumulative distribution of amplitudes
        cutoff = xs[np.searchsorted(cdf, 0.77)]

        foreground = (spec >= cutoff)

        struct = generate_binary_structure(2,1)
        neighborhood = iterate_structure(struct, 20)
        local = spec == maximum_filter(spec, footprint=neighborhood)

        intersection = local & foreground
        bins = np.argwhere(intersection)
        return bins
    
    def addtodb(self, name, artist, bins):

        """
        Adds a song to the database

        Parameters
        ----------
        name : String
            The name of the song
            
        artist: String
            The artist of the song
            
        bins: np.ndarray
            the coordinates of the local maximum in a spectograph

        Returns
        -------
        None
        """

        self.songnames.append(name)
        for index, (f1, t1) in enumerate(bins):
            fanout = bins[index+1:index+21]

            for f2, t2 in fanout:
                x = (f1, f2, t2-t1)
                self.database[x].append((name, artist, t1)) # if we have a key for this fp, we add the song,artist, and time 

    def findprob(self, song):
        """
        Finds the frequency of each matching fingerprint within an excerpt

        Parameters
        ----------
        song : List
            all the songname-frequency-count tuples (e.g. song[0] could be ("in the night", 2000, 3)

        Returns
        -------
        probs : String
            represents the % chance of a given song (in the format of __a number__% chance of being ___song name____).
        """
        songcounts = {}
        count = 0
        probs = ""

        for asong in song:
            if asong[0][0] in songcounts:
                songcounts[asong[0][0]] += asong[1]
            else:
                songcounts[asong[0][0]] = asong[1]
            count+= asong[1]
            

            
        for i in range(len(self.songnames)):
            try:
                print(song)
                print(songcounts)
                probs += str(np.round(songcounts[self.songnames[i]] / count, decimals = 3) * 100)
                probs += "% chance of being "
                probs += self.songnames[i]
                probs += ". "
            except:
                pass
        
        if count < 10:
            return "not enough frequencies detected, but here is the best guess: " + probs

        return probs

    def match_song(self, excerpt): #note to self: excerpt = [(t1,f1),(t2,f2), etc]
        """
        Matches the given song excerpt with the song the excerpt came from

        Parameters
        ----------
        excerpt : List
            contains tuples which represent a given time (t sub n) at which the corresponding frequency occurred (f sub n)

        Returns
        -------
        self.findprob(...) : String
            represents the % chance of a given song (in the format of __a number__% chance of being ___song name____).
        """
        templist = []
        for index, (fe1, te1) in enumerate(excerpt):
            fanout = excerpt[index+1:index+21]

            for fe2, te2 in fanout:
                x = (fe1, fe2, te2-te1)

                values = self.database.get(x)

                if values is None:
                    continue
                else:
                    for avalue in values:
                        templist.append((avalue[0], avalue[1], avalue[2] - te1, 2))

        return self.findprob(collections.Counter(templist).most_common())

    def make_excerpt(self, time):
        """
        Converts microphone input into a spectrograph and then into a list containing the coordinates of the local peaks.

        Parameters
        ----------
        time : int
            how long the excerpt will be

        Returns
        -------
        self.findpeaks(...) : Spectrograph
            the coordinates of the local maximums within the spectrogram
        """
        recording = record_audio(time)
        excerpt = np.hstack([np.fromstring(i, np.int16) for i in recording[0]])

        excerpt_spec, freqs_spec, times_spec = mlab.specgram(excerpt, NFFT=4096, Fs=self.fs,
                                                          window=mlab.window_hanning,
                                                          noverlap=(4096 // 2))

        return self.findpeaks(excerpt_spec)

    def example_load(self):
        """
        Loads a select bunch of songs into the database
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        self.addtodb("what u need", "the weeknd", self.findpeaks(r"/Users/ji-macbook15/Desktop/moo/2.mp3"))
        self.addtodb("im the one", "dj khaled ft. justin bieber, quavo, chance the rapper, lil wayne", self.findpeaks(r"/Users/ji-macbook15/Desktop/moo/1.mp3"))
        self.addtodb("in the night", "the weeknd", self.findpeaks(r"/Users/ji-macbook15/Desktop/moo/3.mp3"))
        self.addtodb("work from home", "5h", self.findpeaks(r"/Users/ji-macbook15/Desktop/moo/4.mp3"))
        self.addtodb("night changes", "1d", self.findpeaks(r"/Users/ji-macbook15/Desktop/moo/5.mp3"))
        self.addtodb("blank space", "taylor swift", self.findpeaks(r"/Users/ji-macbook15/Desktop/moo/6.mp3"))
