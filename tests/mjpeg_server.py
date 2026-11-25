"""Serve the sample video as a live MJPEG HTTP stream for stream-mode tests.

Run with:
  .venv-worker/bin/python tests/mjpeg_server.py
Then open http://127.0.0.1:8768 in V3R or a browser.
"""

from __future__ import annotations

import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2

ROOT = "/home/mars/Desktop/Study/cap/V3R"
VIDEO = f"{ROOT}/samples/drone1.mp4"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        capture = cv2.VideoCapture(VIDEO)
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = capture.read()
                    if not ok:
                        break
                ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
                time.sleep(1.0 / fps)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            capture.release()


class Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    Server(("127.0.0.1", 8768), Handler).serve_forever()
