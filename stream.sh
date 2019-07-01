#!/bin/sh
#VIDEO_SIZE=506x253
VIDEO_SIZE=1024x640
ffplay -v warning -loop 0 -f rawvideo -pixel_format rgb32 -framerate 30 -video_size $VIDEO_SIZE -i -
