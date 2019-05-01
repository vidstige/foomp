#!/bin/sh
VIDEO_SIZE=506x253
ffplay -v warning -loop 0 -f rawvideo -pixel_format rgb32 -video_size $VIDEO_SIZE -i -
