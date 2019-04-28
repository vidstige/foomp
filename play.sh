#!/bin/sh
ffplay -v warning -loop 0 -f rawvideo -pixel_format rgb32 -video_size 320x200 -i output.raw
