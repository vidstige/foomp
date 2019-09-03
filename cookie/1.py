import sys
w, h = 506, 253
frame = b'\xff\x66\x55\xff' * (w * h)
for i in range(1000):
    sys.stdout.buffer.write(frame)
