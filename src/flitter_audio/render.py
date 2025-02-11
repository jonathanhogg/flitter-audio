
import asyncio

import av
from loguru import logger
import numpy as np
import pyaudio

from flitter.clock import system_clock
from flitter.cache import SharedCache
from flitter.render import Renderer


PA_FORMATS = {
    np.dtype('float32'): pyaudio.paFloat32,
    np.dtype('int16'): pyaudio.paInt16,
    np.dtype('int32'): pyaudio.paInt32,
}


class AvPlayer:
    def __init__(self, filename):
        self._filename = filename
        self._volume = 1
        self._start = None
        self._loop = False
        self._channels = None
        self._run_task = None
        self._resync = False

    async def start(self, PA, device_info):
        self._run_task = asyncio.create_task(self.run(PA, device_info))
        self._event_loop = asyncio.get_event_loop()

    async def update(self, frame_time, volume, position, loop, channels):
        self._volume = volume
        start = frame_time - position
        if self._start is None or abs(start - self._start) > 0.1:
            self._start = start
            self._resync = True
        self._loop = loop
        if channels != self._channels:
            self._channels = channels
            self._resync = True

    async def run(self, PA, device_info):
        try:
            path = SharedCache[self._filename]
            container = av.open(path._path)
            input_stream = container.streams.audio[0]
            context = input_stream.codec_context
        except (FileNotFoundError, av.InvalidDataError, IndexError, KeyError):
            logger.exception("Cannot open audio file: {}", self._filename)
            return
        logger.debug("Starting audio player for: {}", self._filename)
        input_nchannels = context.channels
        sample_rate = context.sample_rate
        start = input_stream.start_time or 0
        duration = input_stream.duration
        end = start + duration
        time_base = input_stream.time_base
        decoder = None
        output_stream = None
        output_channel_order = None
        frame = None
        max_channels = device_info['maxOutputChannels']
        try:
            while True:
                if self._resync:
                    logger.trace("Resync stream: {}", self._filename)
                    if decoder is not None:
                        decoder.close()
                        decoder = None
                    if self._channels:
                        new_output_channel_order = []
                        for ch in self._channels:
                            if 1 <= ch <= input_nchannels:
                                new_output_channel_order.append(ch)
                            else:
                                new_output_channel_order.append(0)
                    else:
                        new_output_channel_order = list(range(1, input_nchannels+1))
                    nchannels = len(new_output_channel_order)
                    if nchannels > max_channels:
                        logger.warning("Device {} does not support {} channels; truncating", device_info['name'], nchannels)
                        new_output_channel_order = new_output_channel_order[:max_channels]
                    if new_output_channel_order != output_channel_order:
                        if output_stream is not None:
                            logger.trace("Channel order changed; closing output stream for: {}", self._filename)
                            await asyncio.to_thread(output_stream.stop_stream)
                            output_stream.close()
                            output_stream = None
                        output_channel_order = new_output_channel_order
                        logger.trace("Channel order {} for: {}", ','.join(str(ch) for ch in output_channel_order), self._filename)
                position = int((system_clock() - self._start) / time_base)
                if self._loop:
                    timestamp = start + (position % duration)
                else:
                    timestamp = start + position
                    if timestamp < start or timestamp > end:
                        if output_stream is not None:
                            logger.trace("Position out of range; closing output stream for: {}", self._filename)
                            await asyncio.to_thread(output_stream.stop_stream)
                            output_stream.close()
                            output_stream = None
                        if decoder is not None:
                            decoder.close()
                            decoder = None
                        await asyncio.sleep(0.05)
                        continue
                if decoder is None:
                    self._resync = False
                    if self._loop:
                        remaining = float((end - timestamp) * time_base)
                        if remaining < 0.2:
                            await asyncio.sleep(remaining)
                            continue
                    if timestamp < 0.1 / time_base:
                        logger.trace("Rewind input stream: {}", self._filename)
                        container.seek(0)
                    else:
                        logger.trace("Seek input stream to {:.1f}s: {}", timestamp * time_base, self._filename)
                        container.seek(timestamp, stream=input_stream)
                    decoder = container.decode(streams=(input_stream.index,))
                    frame = None
                if frame is None:
                    try:
                        frame = next(decoder)
                    except StopIteration:
                        logger.trace("Hit end of input stream: {}", self._filename)
                        if self._loop:
                            if decoder is not None:
                                decoder.close()
                                decoder = None
                        else:
                            if output_stream is not None:
                                logger.trace("Closing output stream for: {}", self._filename)
                                await asyncio.to_thread(output_stream.stop_stream)
                                output_stream.close()
                                output_stream = None
                            await asyncio.sleep(0.1)
                    continue
                frame_end = frame.pts + frame.samples / time_base / sample_rate
                if frame_end < timestamp:
                    frame = None
                    continue
                frame_array = frame.to_ndarray()
                if len(frame_array.shape) == 2 and frame_array.shape[0] == input_nchannels:
                    frame_array = frame_array.T
                elif len(frame_array.shape) != 2 or frame_array.shape[1] != input_nchannels:
                    frame_array = frame_array.reshape((-1, input_nchannels))
                nframes = frame_array.shape[0]
                output_array = np.zeros((nframes, len(output_channel_order)), dtype=frame_array.dtype)
                for outch, inch in enumerate(output_channel_order):
                    if inch:
                        if self._volume < 1:
                            output_array[:, outch] = frame_array[:, inch-1] * self._volume
                        else:
                            output_array[:, outch] = frame_array[:, inch-1]
                if output_stream is None:
                    logger.trace("Opening output stream for: {}", self._filename)
                    output_stream = PA.open(channels=len(output_channel_order), format=PA_FORMATS[output_array.dtype],
                                            rate=sample_rate, frames_per_buffer=sample_rate//10,
                                            output=True, output_device_index=device_info['index'])
                while output_stream.get_write_available() < output_array.shape[0]:
                    await asyncio.sleep(0.01)
                output_stream.write(output_array.tobytes())
                frame = None
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Unexpected error playing audio: {}", self._filename)
        logger.debug("Stopping audio player for: {}", self._filename)
        if output_stream is not None:
            output_stream.stop_stream()
            output_stream.close()
        if decoder is not None:
            decoder.close()
            decoder = None
        container.close()

    async def stop(self):
        if self._run_task is not None:
            self._run_task.cancel()
            await self._run_task


class AudioRenderer(Renderer):
    PA = None
    Renderers = []

    def __init__(self, **kwargs):
        if AudioRenderer.PA is None:
            logger.debug("Initialising PyAudio/PortAudio system")
            AudioRenderer.PA = pyaudio.PyAudio()
        self._device = False
        self._device_info = None
        self._players = {}
        AudioRenderer.Renderers.append(self)

    async def purge(self):
        for player in self._players.values():
            await player.stop()
        self._players = {}

    async def destroy(self):
        await self.purge()
        AudioRenderer.Renderers.remove(self)
        if not AudioRenderer.Renderers:
            logger.debug("Stopping PyAudio/PortAudio system")
            AudioRenderer.PA.terminate()
            AudioRenderer.PA = None

    async def update(self, engine, node, time=None, **kwargs):
        device = node.get('device', 1, str)
        if device != self._device:
            await self.purge()
            if device:
                for i in range(self.PA.get_device_count()):
                    info = self.PA.get_device_info_by_index(i)
                    if device.lower() in info['name'].lower():
                        self._device_info = info
                        logger.debug('Using audio device: {}', self._device_info['name'])
                        break
                else:
                    logger.warning("Unable to find audio device matching: {}", device)
                    self._device_info = self.PA.get_default_output_device_info()
                    logger.debug('Using default audio device: {}', self._device_info['name'])
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
                    audio_id = child.get('id', 1, str, filename)
                    volume = child.get('volume', 1, float, 1)
                    position = child.get('position', 1, float, 0)
                    loop = child.get('loop', 1, bool, False)
                    channels = child.get('channels', 0, int)
                    if filename:
                        if audio_id in players:
                            player = players.pop(audio_id)
                            await player.update(time, volume, position, loop, channels)
                        else:
                            player = AvPlayer(filename)
                            await player.update(time, volume, position, loop, channels)
                            await player.start(self.PA, self._device_info)
                        self._players[audio_id] = player
            for player in players.values():
                await player.stop()
