
import asyncio
import queue
import wave

import av
from loguru import logger
import numpy as np
import pyaudio

from flitter.clock import system_clock
from flitter.cache import SharedCache
from flitter.render import Renderer


PA_FORMATS = {
    'flt': pyaudio.paFloat32,
    'fltp': pyaudio.paFloat32,
    's16': pyaudio.paInt16,
    's16p': pyaudio.paInt16,
    's32': pyaudio.paInt32,
}


class WavPlayer:
    BUFFERS = 3
    BUFFER_SIZE = 0.05

    def __init__(self, filename):
        self._filename = filename
        self._volume = 1
        self._start = None
        self._loop = False
        self._run_task = None
        self._queue = queue.Queue(maxsize=self.BUFFERS+1)

    async def start(self, PA, device_info):
        self._run_task = asyncio.create_task(self.run(PA, device_info))
        self._event_loop = asyncio.get_event_loop()

    async def update(self, frame_time, volume, position, loop):
        self._volume = volume
        self._start = frame_time - position
        self._loop = loop

    async def run(self, PA, device_info):
        try:
            path = SharedCache[self._filename]
            wavfile = wave.open(str(path._path), 'rb')
            width = wavfile.getsampwidth()
            nchannels = wavfile.getnchannels()
            sample_rate = wavfile.getframerate()
            nframes = wavfile.getnframes()
            format = {1: pyaudio.paInt8, 2: pyaudio.paInt16, 4: pyaudio.paFloat32}[width]
            dtype = {1: 'int8', 2: 'int16', 4: 'float32'}[width]
        except Exception:
            logger.exception("Cannot open audio file: {}", self._filename)
            return
        logger.debug("Starting audio player for: {}", self._filename)
        output_stream = None
        read_position = 0
        buffer_size = int(sample_rate * self.BUFFER_SIZE)
        try:
            while True:
                position = int((system_clock() - self._start) * sample_rate)
                if self._loop:
                    position %= nframes
                elif position < 0 or position > nframes:
                    if output_stream is not None:
                        output_stream.close()
                        output_stream = None
                        wavfile.rewind()
                        read_position = 0
                        self._queue = queue.Queue(maxsize=self.BUFFERS+1)
                    await asyncio.sleep(self.BUFFER_SIZE)
                    continue
                if position < read_position - buffer_size:
                    logger.debug("Rewind input stream for: {}", self._filename)
                    wavfile.rewind()
                    read_position = 0
                if output_stream is not None and not output_stream.is_active():
                    logger.debug("Time-out on output stream for: {}", self._filename)
                    output_stream.close()
                    output_stream = None
                    wavfile.rewind()
                    read_position = 0
                    self._queue = queue.Queue(maxsize=self.BUFFERS+1)
                while position >= read_position:
                    data = wavfile.readframes(buffer_size)
                    read_position += len(data) // (nchannels * width)
                    if read_position > position - buffer_size * self.BUFFERS:
                        if self._volume < 1:
                            n = len(data) // nchannels // width
                            frame_array = np.ndarray(shape=(n, nchannels), buffer=data, dtype=dtype)
                            frame_array = (frame_array * self._volume).astype(dtype)
                            data = frame_array.tobytes()
                        try:
                            await asyncio.to_thread(self._queue.put, data, timeout=self.BUFFER_SIZE*self.BUFFERS)
                        except queue.Full:
                            logger.debug("Time-out on output stream for: {}", self._filename)
                            output_stream.close()
                            output_stream = None
                            wavfile.rewind()
                            read_position = 0
                            self._queue = queue.Queue(maxsize=self.BUFFERS+1)
                if output_stream is None:
                    logger.debug("Starting output stream for: {}", self._filename)
                    output_stream = PA.open(channels=nchannels, format=format, rate=sample_rate, frames_per_buffer=buffer_size,
                                            output=True, output_device_index=device_info['index'], stream_callback=self._callback)
                await asyncio.sleep(self.BUFFER_SIZE / 3)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Unexpected error playing audio: {}", self._filename)
        logger.debug("Stopping audio player for: {}", self._filename)
        if output_stream is not None:
            while output_stream.is_active():
                await asyncio.sleep(self.BUFFER_SIZE)
            output_stream.close()
        wavfile.close()

    def _callback(self, data, frame_count, time_info, status_flags):
        try:
            return self._queue.get(timeout=self.BUFFER_SIZE / 2), pyaudio.paContinue
        except queue.Empty:
            return None, pyaudio.paComplete

    async def stop(self):
        if self._run_task is not None:
            self._run_task.cancel()
            await self._run_task


class AvPlayer:
    def __init__(self, filename):
        self._filename = filename
        self._volume = 1
        self._start = None
        self._loop = False
        self._run_task = None
        self._data = None
        self._written = False

    async def start(self, PA, device_info):
        self._run_task = asyncio.create_task(self.run(PA, device_info))
        self._event_loop = asyncio.get_event_loop()

    async def update(self, frame_time, volume, position, loop):
        self._volume = volume
        self._start = frame_time - position
        self._loop = loop

    async def run(self, PA, device_info):
        try:
            path = SharedCache[self._filename]
            container = av.open(path._path)
            input_stream = container.streams.audio[0]
            context = input_stream.codec_context
            format = PA_FORMATS[context.format.name]
        except (FileNotFoundError, av.InvalidDataError, IndexError):
            logger.exception("Cannot open audio file: {}", self._filename)
            return
        logger.debug("Starting audio player for: {}", self._filename)
        start = input_stream.start_time or 0
        duration = input_stream.duration
        time_base = input_stream.time_base
        decoder = None
        frame = None
        frame_start = None
        output_stream = None
        try:
            while True:
                if decoder is None:
                    if output_stream is not None:
                        output_stream.close()
                        output_stream = None
                    self._data = None
                    await asyncio.to_thread(container.seek, 0, stream=input_stream)
                    decoder = await asyncio.to_thread(container.decode, streams=(input_stream.index,))
                    frame = None
                position = int((system_clock() - self._start) / time_base)
                if self._loop:
                    timestamp = start + (position % duration)
                else:
                    timestamp = start + position
                if timestamp < start or timestamp > start + duration:
                    if output_stream is not None:
                        output_stream.close()
                        output_stream = None
                    await asyncio.sleep(0.1)
                    continue
                if frame is None or self._written and timestamp > frame_start:
                    try:
                        frame = next(decoder)
                    except (StopIteration):
                        pass
                    else:
                        frame_array = frame.to_ndarray()
                        if len(frame_array.shape) == 2 and frame_array.shape[0] < frame_array.shape[1]:
                            frame_array = frame_array.transpose()
                        if self._volume < 1:
                            frame_array = (frame_array * self._volume).astype(frame_array.dtype)
                        data = frame_array.tobytes()
                if frame is not None:
                    frame_start = frame.pts
                    frame_duration = (frame.samples / time_base) / frame.sample_rate
                    frame_end = frame.pts + frame_duration
                    if frame_end < timestamp:
                        frame = None
                        continue
                    elif timestamp < frame_start - frame_duration:
                        logger.trace("Rewind input stream for: {}", self._filename)
                        decoder = None
                        continue
                    if data is not self._data:
                        self._data = data
                        self._written = False
                    if output_stream is not None and not output_stream.is_active():
                        logger.trace("Time-out on output stream for: {}", self._filename)
                        output_stream.close()
                        output_stream = None
                    if output_stream is None:
                        logger.trace("Starting output stream for: {}", self._filename)
                        output_stream = PA.open(channels=context.channels, format=format, rate=context.sample_rate, frames_per_buffer=frame.samples,
                                                output=True, output_device_index=device_info['index'], stream_callback=self._callback)
                    await asyncio.sleep(float(frame_duration * time_base / 3))
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Unexpected error playing audio: {}", self._filename)
        logger.debug("Stoppping audio player for: {}", self._filename)
        self._data = None
        if output_stream is not None:
            while output_stream.is_active():
                await asyncio.sleep(0.1)
            output_stream.close()
        container.close()

    def _callback(self, data, frame_count, time_info, status_flags):
        if self._data is not None:
            self._written = True
            return self._data, pyaudio.paContinue
        return None, pyaudio.paComplete

    async def stop(self):
        if self._run_task is not None:
            self._run_task.cancel()
            await self._run_task


class AudioRenderer(Renderer):
    PA = None

    def __init__(self, **kwargs):
        if AudioRenderer.PA is None:
            logger.debug("Initialising PyAudio/PortAudio system")
            AudioRenderer.PA = pyaudio.PyAudio()
        self._device = False
        self._device_info = None
        self._players = {}

    async def purge(self):
        for player in self._players.values():
            await player.stop()
        self._players = {}

    async def destroy(self):
        await self.purge()
        logger.debug("Done with PyAudio/PortAudio system")

    async def update(self, engine, node, time=None, **kwargs):
        device = node.get('device', 1, str)
        if device != self._device:
            await self.purge()
            if device:
                for i in range(self.PA.get_device_count()):
                    info = self.PA.get_device_info_by_index(i)
                    if info['name'] == device:
                        logger.debug('Using audio device: {}', device)
                        self._device_info = info
                        break
                else:
                    self._device_info = None
            else:
                self._device_info = self.PA.get_default_output_device_info()
                logger.debug('Using default audio device: {}', self._device_info['name'])
            self._device = device
        if self._device_info is not None:
            players = self._players
            self._players = {}
            for child in node.children:
                if child.kind == 'sample':
                    filename = child.get('filename', 1, str)
                    volume = child.get('volume', 1, float, 1)
                    position = child.get('position', 1, float, 0)
                    loop = child.get('loop', 1, bool, False)
                    if filename:
                        if filename in players:
                            player = players.pop(filename)
                        elif filename.endswith('.wav'):
                            player = WavPlayer(filename)
                            await player.start(self.PA, self._device_info)
                        else:
                            player = AvPlayer(filename)
                            await player.start(self.PA, self._device_info)
                        await player.update(time, volume, position, loop)
                        self._players[filename] = player
            for player in players.values():
                await player.stop()
