#!/bin/sh
VIDEO_SIZE=506x253
ffmpeg -f rawvideo -pixel_format rgb32 -video_size $VIDEO_SIZE -i output.raw -f gif output.gif
