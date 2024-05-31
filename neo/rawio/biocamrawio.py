"""
Class for reading data from a 3-brain Biocam system.

See:
https://www.3brain.com/products/single-well/biocam-x

Authors: Alessio Buccino, Robert Wolff
"""

from .baserawio import (
    BaseRawIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
)

import numpy as np
import json


class BiocamRawIO(BaseRawIO):
    """
    Class for reading data from a Biocam h5 file.

    Parameters
    ----------
    filename: str, default: ''
        The *.h5 file to be read

    Examples
    --------
        >>> import neo.rawio
        >>> r = neo.rawio.BiocamRawIO(filename='biocam.h5')
        >>> r.parse_header()
        >>> print(r)
        >>> raw_chunk = r.get_analogsignal_chunk(block_index=0,
                                                 seg_index=0,
                                                 i_start=0,
                                                 i_stop=1024,
                                                 channel_names=channel_names)
        >>> float_chunk = r.rescale_signal_raw_to_float(raw_chunk,
                                                        dtype='float64',
                                                        channel_indexes=[0, 3, 6])
    """

    extensions = ["h5", "brw"]
    rawmode = "one-file"

    def __init__(self, filename=""):
        BaseRawIO.__init__(self)
        self.filename = filename

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        self._header_dict = open_biocam_file_header(self.filename)
        self._num_channels = self._header_dict["num_channels"]
        self._num_frames = self._header_dict["num_frames"]
        self._sampling_rate = self._header_dict["sampling_rate"]
        self._filehandle = self._header_dict["file_handle"]
        self._read_function = self._header_dict["read_function"]
        self._channels = self._header_dict["channels"]
        gain = self._header_dict["gain"]
        offset = self._header_dict["offset"]

        signal_streams = np.array([("Signals", "0")], dtype=_signal_stream_dtype)

        sig_channels = []
        for c, chan in enumerate(self._channels):
            ch_name = f"ch{chan[0]}-{chan[1]}"
            chan_id = str(c + 1)
            sr = self._sampling_rate  # Hz
            dtype = "uint16"
            units = "uV"
            gain = gain
            offset = offset
            stream_id = "0"
            sig_channels.append((ch_name, chan_id, sr, dtype, units, gain, offset, stream_id))
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)

        # No events
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        self.header = {}
        self.header["nb_block"] = 1
        self.header["nb_segment"] = [1]
        self.header["signal_streams"] = signal_streams
        self.header["signal_channels"] = sig_channels
        self.header["spike_channels"] = spike_channels
        self.header["event_channels"] = event_channels

        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        all_starts = [[0.0]]
        return all_starts[block_index][seg_index]

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self._num_frames / self._sampling_rate
        all_stops = [[t_stop]]
        return all_stops[block_index][seg_index]

    def _get_signal_size(self, block_index, seg_index, stream_index):
        if stream_index != 0:
            raise ValueError("`stream_index` must be 0")
        return self._num_frames

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        if stream_index != 0:
            raise ValueError("`stream_index must be 0")
        return self._segment_t_start(block_index, seg_index)

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, stream_index, channel_indexes):
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._num_frames
        if channel_indexes is None:
            channel_indexes = slice(None)
        data = self._read_function(self._filehandle, i_start, i_stop, self._num_channels)
        return data[:, channel_indexes]


def open_biocam_file_header(filename):
    """Open a Biocam hdf5 file, read and return the recording info, pick the correct method to access raw data,
    and return this to the caller."""
    import h5py

    rf = h5py.File(filename, "r")

    if "3BRecInfo" in rf.keys():  # brw v3.x
        # Read recording variables
        rec_vars = rf.require_group("3BRecInfo/3BRecVars/")
        bit_depth = rec_vars["BitDepth"][0]
        max_uv = rec_vars["MaxVolt"][0]
        min_uv = rec_vars["MinVolt"][0]
        num_frames = rec_vars["NRecFrames"][0]
        sampling_rate = rec_vars["SamplingRate"][0]
        signal_inv = rec_vars["SignalInversion"][0]

        # Get the actual number of channels used in the recording
        file_format = rf["3BData"].attrs.get("Version", None)
        format_100 = False
        if file_format == 100:
            num_channels = len(rf["3BData/Raw"][0])
            format_100 = True
        elif file_format in (101, 102) or file_format is None:
            num_channels = int(rf["3BData/Raw"].shape[0] / num_frames)
        else:
            raise Exception("Unknown data file format.")

        # # get channels
        channels = rf["3BRecInfo/3BMeaStreams/Raw/Chs"][:]

        # determine correct function to read data
        if format_100:
            if signal_inv == 1:
                read_function = readHDF5t_100
            elif signal_inv == -1:
                read_function = readHDF5t_100_i
            else:
                raise Exception("Unknown signal inversion")
        else:
            if signal_inv == 1:
                read_function = readHDF5t_101
            elif signal_inv == -1:
                read_function = readHDF5t_101_i
            else:
                raise Exception("Unknown signal inversion")

        gain = (max_uv - min_uv) / (2**bit_depth)
        offset = min_uv

        return dict(
            file_handle=rf,
            num_frames=num_frames,
            sampling_rate=sampling_rate,
            num_channels=num_channels,
            channels=channels,
            file_format=file_format,
            signal_inv=signal_inv,
            read_function=read_function,
            gain=gain,
            offset=offset,
        )
    else:  # brw v4.x
        # Read recording variables
        experiment_settings = json.JSONDecoder().decode(rf["ExperimentSettings"][0].decode())
        max_uv = experiment_settings["ValueConverter"]["MaxAnalogValue"]
        min_uv = experiment_settings["ValueConverter"]["MinAnalogValue"]
        max_digital = experiment_settings["ValueConverter"]["MaxDigitalValue"]
        min_digital = experiment_settings["ValueConverter"]["MinDigitalValue"]
        scale_factor = experiment_settings["ValueConverter"]["ScaleFactor"]
        sampling_rate = experiment_settings["TimeConverter"]["FrameRate"]

        if 'EventsBasedRawRanges' in experiment_settings['DataSettings']:
            for key in rf:
                if key[:5] == "Well_":
                    num_channels = len(rf[key]["StoredChIdxs"])
                    num_frames = np.array(rf['TOC'])[-1][-1]
                    break
            read_function = readHDF5t_brw4_event_based
        
        else:
            for key in rf:
                if key[:5] == "Well_":
                    num_channels = len(rf[key]["StoredChIdxs"])
                    if len(rf[key]["Raw"]) % num_channels:
                        raise RuntimeError(f"Length of raw data array is not multiple of channel number in {key}")
                    num_frames = len(rf[key]["Raw"]) // num_channels
                    break
            read_function = readHDF5t_brw4

        try:
            num_channels_x = num_channels_y = int(np.sqrt(num_channels))
        except NameError:
            raise RuntimeError("No Well found in the file")
        if num_channels_x * num_channels_y != num_channels:
            raise RuntimeError(f"Cannot determine structure of the MEA plate with {num_channels} channels")
        channels = 1 + np.concatenate(np.transpose(np.meshgrid(range(num_channels_x), range(num_channels_y))))

        gain = scale_factor * (max_uv - min_uv) / (max_digital - min_digital)
        offset = min_uv
        

        return dict(
            file_handle=rf,
            num_frames=num_frames,
            sampling_rate=sampling_rate,
            num_channels=num_channels,
            channels=channels,
            read_function=read_function,
            gain=gain,
            offset=offset,
        )

def GenerateSyntheticNoise_arr(file, wellID, startFrame, endFrame, numFrames, n_channels):
    # collect the TOCs
    toc = np.array(file['TOC'])
    noiseToc = np.array(file[wellID + '/NoiseTOC'])
    # from the given start position in frames, localize the corresponding noise positions
    # using the TOC
    tocStartIdx = np.searchsorted(toc[:, 1], startFrame)
    noiseStartPosition = noiseToc[tocStartIdx]
    noiseEndPosition = noiseStartPosition
    for i in range(tocStartIdx + 1, len(noiseToc)):
        nextPosition = noiseToc[i]
        if nextPosition > noiseStartPosition:
            noiseEndPosition = nextPosition
            break
    if noiseEndPosition == noiseStartPosition:
        for i in range(tocStartIdx - 1, 0, -1):
            previousPosition = noiseToc[i]
            if previousPosition < noiseStartPosition:
                noiseEndPosition = noiseStartPosition
                noiseStartPosition = previousPosition
                break

    # obtain the noise info at the start position
    noiseChIdxs = file[wellID + '/NoiseChIdxs'][noiseStartPosition:noiseEndPosition]
    noiseMean = file[wellID + '/NoiseMean'][noiseStartPosition:noiseEndPosition]
    noiseStdDev = file[wellID + '/NoiseStdDev'][noiseStartPosition:noiseEndPosition]
    noiseLength = noiseEndPosition - noiseStartPosition

    noiseInfo = {}
    meanCollection = []
    stdDevCollection = []

    for i in range(1, noiseLength):
        noiseInfo[noiseChIdxs[i]] = [noiseMean[i], noiseStdDev[i]]
        meanCollection.append(noiseMean[i])
        stdDevCollection.append(noiseStdDev[i])

    # calculate the median mean and standard deviation of all channels to be used for
    # invalid channels
    dataMean = np.median(meanCollection)
    dataStdDev = np.median(stdDevCollection)

    arr = np.zeros((numFrames, n_channels))

    for chIdx in range(n_channels):
        if chIdx in noiseInfo:
            arr[:,chIdx] = np.array(np.random.normal(noiseInfo[chIdx][0], noiseInfo[chIdx][1],numFrames), dtype=np.int16)
        else:
            arr[:,chIdx] = np.array(np.random.normal(dataMean, dataStdDev, numFrames),dtype=np.int16)

    return arr

def DecodeEventBasedRawData_arr(rf, wellID, t0, t1, nch):

    toc = np.array(rf['TOC']) # Main table of contents
    eventsToc = np.array(rf[wellID + '/EventsBasedSparseRawTOC']) # Events table of contents

    # Only select the chunks in the desired range
    tocStartIdx = np.searchsorted(toc[:, 1], t0)
    tocEndIdx = min(np.searchsorted(toc[:, 1], t1, side='right') + 1, len(toc) - 1)
    eventsStartPosition = eventsToc[tocStartIdx]
    eventsEndPosition = eventsToc[tocEndIdx]

    numFrames = t1 - t0

    binaryData = rf[wellID + '/EventsBasedSparseRaw'][eventsStartPosition:eventsEndPosition]
    binaryDataLength = len(binaryData)

    synth = True

    if synth:
        arr = GenerateSyntheticNoise_arr(rf, wellID, t0, t1, numFrames, nch)
    
    else:
        arr = np.zeros((numFrames, nch))

    pos = 0
    while pos < binaryDataLength:
        chIdx = int.from_bytes(binaryData[pos:pos + 4], byteorder='little', signed=True)
        pos += 4
        chDataLength = int.from_bytes(binaryData[pos:pos + 4], byteorder='little', signed=True)
        pos += 4
        chDataPos = pos
        while pos < chDataPos + chDataLength:
            fromInclusive = int.from_bytes(binaryData[pos:pos + 8], byteorder='little', signed=True)
            pos += 8
            toExclusive = int.from_bytes(binaryData[pos:pos + 8], byteorder='little', signed=True)
            pos += 8
            rangeDataPos = pos
            for j in range(fromInclusive, toExclusive):
                if j >= t0 + numFrames: 
                    break
                if j >= t0:
                    arr[j - t0,chIdx] = int.from_bytes(
                    binaryData[rangeDataPos:rangeDataPos + 2], byteorder='little', signed=True)
                rangeDataPos += 2
            pos += (toExclusive - fromInclusive) * 2

    return arr

def readHDF5t_100(rf, t0, t1, nch):
    return rf["3BData/Raw"][t0:t1]


def readHDF5t_100_i(rf, t0, t1, nch):
    return 4096 - rf["3BData/Raw"][t0:t1]


def readHDF5t_101(rf, t0, t1, nch):
    return rf["3BData/Raw"][nch * t0 : nch * t1].reshape((t1 - t0, nch), order="C")


def readHDF5t_101_i(rf, t0, t1, nch):
    return 4096 - rf["3BData/Raw"][nch * t0 : nch * t1].reshape((t1 - t0, nch), order="C")


def readHDF5t_brw4(rf, t0, t1, nch):
    for key in rf:
        if key[:5] == "Well_":
            return rf[key]["Raw"][nch * t0 : nch * t1].reshape((t1 - t0, nch), order="C")
        
def readHDF5t_brw4_event_based(rf, t0, t1, nch):
    for key in rf:
        if key[:5] == "Well_":
            return DecodeEventBasedRawData_arr(rf, key, t0, t1, nch)
