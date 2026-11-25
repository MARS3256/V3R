"""Local RTMP server that loops a video, viewable from a browser.

Starts MediaMTX (tools/mediamtx.exe): RTMP ingest on rtmp://127.0.0.1:1935, WebRTC/HLS egress.
Pushes the video to rtmp://127.0.0.1:1935/live/drone on a loop (-stream_loop -1 -re, real time).

Watch in a browser:  http://127.0.0.1:8889/live/drone   (WebRTC, low latency)
            or:  http://127.0.0.1:8888/live/drone      (HLS)

usage: python tests/serve_rtmp.py [video] [path]
"""
import os
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIAMTX = os.path.join(ROOT, "tools", "mediamtx.exe")
VIDEO = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "samples", "drone1.mp4")
PATH = sys.argv[2] if len(sys.argv) > 2 else "live/drone"
RTMP_URL = f"rtmp://127.0.0.1:1935/{PATH}"
BROWSER_WEBRTC = f"http://127.0.0.1:8889/{PATH}"
BROWSER_HLS = f"http://127.0.0.1:8888/{PATH}"


def wait_port(port, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.25)
    return False


def main():
    import imageio_ffmpeg

    if not os.path.isfile(MEDIAMTX):
        raise FileNotFoundError(f"MediaMTX not found at {MEDIAMTX}")
    if not os.path.isfile(VIDEO):
        raise FileNotFoundError(f"video not found at {VIDEO}")

    server = subprocess.Popen([MEDIAMTX], cwd=os.path.dirname(MEDIAMTX))
    if not wait_port(1935):
        server.terminate()
        raise SystemExit("MediaMTX RTMP port 1935 did not open")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # re-encode without B-frames: MediaMTX WebRTC (the browser page) rejects H.264 with B-frames
    enc = ["-an", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-bf", "0", "-g", "50"]
    push = subprocess.Popen([ffmpeg, "-loglevel", "error", "-re", "-stream_loop", "-1", "-i", VIDEO,
                             *enc, "-f", "flv", RTMP_URL])
    print(f"server pid {server.pid}, ffmpeg pid {push.pid}")
    print(f"looping {VIDEO} -> {RTMP_URL}")
    print(f"watch (WebRTC): {BROWSER_WEBRTC}")
    print(f"watch (HLS):    {BROWSER_HLS}")
    print("Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1)
            if push.poll() is not None:
                print("ffmpeg push exited; restarting...")
                push = subprocess.Popen([ffmpeg, "-loglevel", "error", "-re", "-stream_loop", "-1", "-i", VIDEO,
                                         *enc, "-f", "flv", RTMP_URL])
    except KeyboardInterrupt:
        pass
    finally:
        push.terminate()
        server.terminate()


if __name__ == "__main__":
    main()