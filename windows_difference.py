# -*- coding: utf-8 -*-
"""
Foreground-window screenshot diff highlighter with global hotkeys on Windows.
This version adds BLINKING border-only (no fill) polygons with configurable
high-contrast or rainbow colors to make differences easier to see.

Hotkeys (configurable at top):
- Ctrl+Shift+D: Toggle overlay (show/clear).
- Ctrl+Shift+B: Toggle debug preview windows (left/right halves).
- Ctrl+Shift+Q: Quit app.

Requirements:
    pip install pillow opencv-python numpy pywin32 pynput
"""

import ctypes
import os
import sys
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageGrab, ImageTk
import tkinter as tk

# Windows APIs
import win32con
import win32gui

# Global hotkeys
from pynput import keyboard


# ===========================
# Configuration Parameters
# (All user-tunable settings)
# ===========================

# --- Hotkeys ---
HOTKEY_TOGGLE_OVERLAY: str = "<ctrl>+<shift>+d"   # Toggle overlay show/clear
HOTKEY_DEBUG_TOGGLE: str = "<ctrl>+<shift>+b"     # Toggle debug preview windows
HOTKEY_QUIT: str = "<ctrl>+<shift>+q"             # Quit application

# --- Target rectangle mode ---
# One of: "client" (client area), "frame" (DWM extended frame bounds), "window" (GetWindowRect)
# "frame" usually aligns better with what you see on screen.
RECT_MODE: str = "window"

# --- Diff thresholds & cleanup ---
DIFF_TOLERANCE: int = 15          # Threshold on absdiff grayscale (0-255). Higher = fewer differences.
MIN_CONTOUR_AREA: int = 120       # Minimal contour area to keep.
MORPH_KERNEL_SIZE: int = 3        # Morphology kernel size for noise cleanup.
APPROX_POLY_EPSILON_RATIO: float = 0.01  # Polygon approximation epsilon ratio (fraction of perimeter).

# --- Window size guard (avoid tiny accidental captures) ---
MIN_WINDOW_WIDTH: int = 100
MIN_WINDOW_HEIGHT: int = 80

# --- Overlay visualization (border only; do not fill) ---
BORDER_THICKNESS: int = 3         # Thickness of the polygon outline.
DRAW_FILL: bool = False           # Keep False to only draw border/skeleton.

# --- Blink/animation settings ---
# BLINK_MODE:
#   "two_color" -> alternate between BLINK_COLORS_BGR
#   "rainbow"   -> cycle through RAINBOW_COLORS_BGR
#   "off"       -> draw once (no blinking)
BLINK_MODE: str = "rainbow"

# Interval in milliseconds between frames (lower = faster blink).
BLINK_INTERVAL_MS: int = 80

# High-contrast pair for "two_color" mode (BGR in OpenCV)
# Default: Yellow <-> Magenta which pops well on most backgrounds.
BLINK_COLORS_BGR: List[Tuple[int, int, int]] = [
    (0, 255, 255),   # Yellow
    (255, 0, 255),   # Magenta
]

# Rainbow palette for "rainbow" mode (BGR in OpenCV)
RAINBOW_COLORS_BGR: List[Tuple[int, int, int]] = [
    (0, 0, 255),     # Red
    (0, 127, 255),   # Orange
    (0, 255, 255),   # Yellow
    (0, 255, 0),     # Green
    (255, 127, 0),   # Sky blue
    (255, 0, 0),     # Blue
    (255, 0, 127),   # Purple-ish
]

# --- Transparent overlay background key ---
# Pure green works well as a colorkey on Windows.
TRANSPARENT_KEY_RGB: Tuple[int, int, int] = (0, 254, 0)
TRANSPARENT_KEY_HEX: str = "#00FE00"


# ===========================
# DPI Awareness
# ===========================

def set_dpi_awareness() -> None:
    """Enable Per-Monitor v2 DPI awareness if available; fallback to system DPI aware."""
    try:
        # -4 is DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


# ===========================
# Utilities (Rects & Capture)
# ===========================

def _get_system_metric(index: int) -> int:
    """Wrapper for user32.GetSystemMetrics."""
    return ctypes.windll.user32.GetSystemMetrics(index)


def _clamp_to_virtual_screen(
    left: int, top: int, right: int, bottom: int
) -> Tuple[int, int, int, int]:
    """Clamp a rect to the virtual screen to avoid negative or out-of-bounds values."""
    sm_x = _get_system_metric(win32con.SM_XVIRTUALSCREEN)
    sm_y = _get_system_metric(win32con.SM_YVIRTUALSCREEN)
    sm_w = _get_system_metric(win32con.SM_CXVIRTUALSCREEN)
    sm_h = _get_system_metric(win32con.SM_CYVIRTUALSCREEN)
    x1 = max(left, sm_x)
    y1 = max(top, sm_y)
    x2 = min(right, sm_x + sm_w)
    y2 = min(bottom, sm_y + sm_h)
    return x1, y1, x2, y2


def _get_frame_bounds_rect(hwnd: int) -> Tuple[int, int, int, int]:
    """
    Use DwmGetWindowAttribute(DWMWA_EXTENDED_FRAME_BOUNDS=9) to get frame bounds in screen coords.
    Fallback to GetWindowRect if DWM API not available.
    """
    DWMWA_EXTENDED_FRAME_BOUNDS = 9

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    rect = RECT()
    try:
        dwmapi = ctypes.WinDLL("dwmapi")
        res = dwmapi.DwmGetWindowAttribute(
            ctypes.c_void_p(hwnd),
            ctypes.c_int(DWMWA_EXTENDED_FRAME_BOUNDS),
            ctypes.byref(rect),
            ctypes.sizeof(rect),
        )
        if res == 0:
            return rect.left, rect.top, rect.right, rect.bottom
    except Exception:
        pass

    # Fallback
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    return left, top, right, bottom


def get_foreground_rect(mode: str = RECT_MODE) -> Tuple[int, int, int, int]:
    """
    Get the rectangle of the current foreground window in screen coordinates.
    Modes:
      - "client": client area via GetClientRect + ClientToScreen
      - "frame" : extended frame bounds via DwmGetWindowAttribute
      - "window": GetWindowRect
    Rect is clamped to virtual screen.
    """
    hwnd = win32gui.GetForegroundWindow()
    if not hwnd:
        raise RuntimeError("No foreground window detected.")

    if mode == "client":
        left_c, top_c, right_c, bottom_c = win32gui.GetClientRect(hwnd)
        pt_lt = win32gui.ClientToScreen(hwnd, (left_c, top_c))
        pt_rb = win32gui.ClientToScreen(hwnd, (right_c, bottom_c))
        left, top = pt_lt
        right, bottom = pt_rb
    elif mode == "frame":
        left, top, right, bottom = _get_frame_bounds_rect(hwnd)
    elif mode == "window":
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    else:
        raise ValueError(f"Unknown RECT_MODE: {mode}")

    left, top, right, bottom = _clamp_to_virtual_screen(left, top, right, bottom)
    if right - left < MIN_WINDOW_WIDTH or bottom - top < MIN_WINDOW_HEIGHT:
        raise RuntimeError(f"Rect too small: {(left, top, right, bottom)} (mode={mode})")
    return left, top, right, bottom


def capture_window_image(rect: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Capture an image of the given screen-rect using PIL.ImageGrab.
    Returns a numpy array (BGR) suitable for OpenCV.
    """
    left, top, right, bottom = rect
    img = ImageGrab.grab(bbox=(left, top, right, bottom))
    img_rgb = np.array(img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


# ===========================
# Diff & Polygons
# ===========================

def compute_diff_polygons(img_bgr: np.ndarray) -> List[np.ndarray]:
    """
    Split the image into left/right halves, compute absolute difference,
    threshold, clean noise, find contours, and approximate polygons.
    Returns a list of polygons (ndarray shape (N, 1, 2), dtype int32) in left-half coordinates.
    """
    h, w = img_bgr.shape[:2]
    mid = w // 2

    left_half = img_bgr[:, :mid]
    right_half = img_bgr[:, w - mid:]  # ensure equal widths

    diff = cv2.absdiff(left_half, right_half)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, DIFF_TOLERANCE, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons: List[np.ndarray] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue
        peri = cv2.arcLength(cnt, True)
        epsilon = APPROX_POLY_EPSILON_RATIO * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        polygons.append(approx.astype(np.int32))

    return polygons


def make_overlay_image(
    size: Tuple[int, int],
    polygons: List[np.ndarray],
    left_half_offset_x: int = 0,
    border_color_bgr: Tuple[int, int, int] = (0, 255, 255),
    fill: bool = DRAW_FILL,
) -> Image.Image:
    """
    Create a PIL Image with transparent background key and draw polygons.
    Polygons are in left-half coordinates; apply horizontal offset if needed.
    Only border is drawn by default; set fill=True to also fill polygons (not recommended here).
    """
    w, h = size
    overlay_bgr = np.zeros((h, w, 3), dtype=np.uint8)

    # Fill with transparent key (in BGR order)
    overlay_bgr[:, :] = (
        TRANSPARENT_KEY_RGB[2],
        TRANSPARENT_KEY_RGB[1],
        TRANSPARENT_KEY_RGB[0],
    )

    for poly in polygons:
        poly_full = poly.copy()
        poly_full[:, 0, 0] = poly_full[:, 0, 0] + left_half_offset_x

        if fill:
            # Normally we do not fill for "skeleton/border-only" visualization.
            cv2.fillPoly(overlay_bgr, [poly_full], color=border_color_bgr)

        cv2.polylines(
            overlay_bgr,
            [poly_full],
            isClosed=True,
            color=border_color_bgr,
            thickness=BORDER_THICKNESS,
        )

    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay_rgb)


# ===========================
# Tk Windows
# ===========================

class OverlayWindow:
    """
    A topmost, borderless Toplevel aligned to the target rect with a transparent color key.
    Supports blinking by updating the image periodically.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.window: Optional[tk.Toplevel] = None
        self.label: Optional[tk.Label] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self.visible: bool = False

        # Animation state
        self._animating: bool = False
        self._tick_job: Optional[str] = None
        self._frame_idx: int = 0
        self._polygons: List[np.ndarray] = []
        self._size: Tuple[int, int] = (1, 1)
        self._rect: Tuple[int, int, int, int] = (0, 0, 1, 1)

    def _ensure_created(self) -> None:
        if self.window is not None:
            return

        self.window = tk.Toplevel(self.root)
        self.window.withdraw()
        self.window.attributes("-topmost", True)
        self.window.overrideredirect(True)
        try:
            self.window.wm_attributes("-transparentcolor", TRANSPARENT_KEY_HEX)
        except tk.TclError:
            # Fallback: semi-transparent whole window (colorkey not supported).
            self.window.attributes("-alpha", 0.8)

        self.label = tk.Label(self.window, bg=TRANSPARENT_KEY_HEX)
        self.label.pack(fill=tk.BOTH, expand=True)

    def _apply_frame(self, rect: Tuple[int, int, int, int], overlay_img: Image.Image) -> None:
        self._ensure_created()
        left, top, right, bottom = rect
        w = max(1, right - left)
        h = max(1, bottom - top)

        self.window.geometry(f"{w}x{h}+{left}+{top}")
        self._photo = ImageTk.PhotoImage(overlay_img)

        assert self.label is not None
        self.label.configure(image=self._photo)
        self.window.deiconify()
        self.visible = True

    def _next_color(self) -> Tuple[int, int, int]:
        """Pick next border color based on BLINK_MODE and frame index."""
        if BLINK_MODE == "off":
            # Use first color from high-contrast set.
            return BLINK_COLORS_BGR[0]

        if BLINK_MODE == "rainbow":
            palette = RAINBOW_COLORS_BGR
        else:
            palette = BLINK_COLORS_BGR

        if not palette:
            # Safety: fallback to yellow if palette is empty.
            return (0, 255, 255)

        return palette[self._frame_idx % len(palette)]

    def _tick(self) -> None:
        """Render one frame and schedule the next if animating."""
        if not self._animating:
            return

        border_color = self._next_color()
        overlay_img = make_overlay_image(
            size=self._size,
            polygons=self._polygons,
            left_half_offset_x=0,
            border_color_bgr=border_color,
            fill=DRAW_FILL,
        )
        self._apply_frame(self._rect, overlay_img)

        self._frame_idx += 1
        if BLINK_MODE != "off":
            # Schedule next frame
            self._tick_job = self.root.after(BLINK_INTERVAL_MS, self._tick)

    def start_animation(
        self,
        rect: Tuple[int, int, int, int],
        size: Tuple[int, int],
        polygons: List[np.ndarray],
    ) -> None:
        """Start (or restart) blinking animation with given geometry and polygons."""
        self._ensure_created()

        # Cancel any pending job first
        self.stop_animation()

        self._rect = rect
        self._size = size
        self._polygons = polygons
        self._frame_idx = 0
        self._animating = True

        # Draw first frame immediately
        self._tick()

    def stop_animation(self) -> None:
        """Stop blinking and cancel scheduled jobs."""
        self._animating = False
        if self._tick_job is not None:
            try:
                self.root.after_cancel(self._tick_job)
            except Exception:
                pass
            self._tick_job = None

    def clear(self) -> None:
        """Clear the overlay (also stops animation)."""
        self.stop_animation()
        if self.window is None or self.label is None:
            self.visible = False
            return
        self.label.configure(image="")
        self._photo = None
        self.window.withdraw()
        self.visible = False


class DebugPreviewWindows:
    """
    Two normal Toplevel windows to show left/right captured halves for debugging.
    Lazily created and guarded to avoid TclError.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.left_win: Optional[tk.Toplevel] = None
        self.right_win: Optional[tk.Toplevel] = None
        self.left_label: Optional[tk.Label] = None
        self.right_label: Optional[tk.Label] = None
        self._left_photo: Optional[ImageTk.PhotoImage] = None
        self._right_photo: Optional[ImageTk.PhotoImage] = None
        self.visible: bool = False

    def _ensure_created(self) -> None:
        if self.left_win is not None and self.right_win is not None:
            return

        self.left_win = tk.Toplevel(self.root)
        self.left_win.title("Left Half (Debug)")
        self.left_win.withdraw()
        self.left_label = tk.Label(self.left_win)
        self.left_label.pack(fill=tk.BOTH, expand=True)

        self.right_win = tk.Toplevel(self.root)
        self.right_win.title("Right Half (Debug)")
        self.right_win.withdraw()
        self.right_label = tk.Label(self.right_win)
        self.right_label.pack(fill=tk.BOTH, expand=True)

        # Place side by side initially
        self.left_win.geometry("640x480+100+100")
        self.right_win.geometry("640x480+760+100")

    def update_images(self, left_img_bgr: np.ndarray, right_img_bgr: np.ndarray) -> None:
        if not self.visible:
            return

        self._ensure_created()

        assert self.left_label is not None and self.right_label is not None

        left_rgb = cv2.cvtColor(left_img_bgr, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_img_bgr, cv2.COLOR_BGR2RGB)
        left_pil = Image.fromarray(left_rgb)
        right_pil = Image.fromarray(right_rgb)

        self._left_photo = ImageTk.PhotoImage(left_pil)
        self._right_photo = ImageTk.PhotoImage(right_pil)

        self.left_label.configure(image=self._left_photo)
        self.right_label.configure(image=self._right_photo)

    def show(self) -> None:
        self._ensure_created()
        assert self.left_win is not None and self.right_win is not None
        self.left_win.deiconify()
        self.right_win.deiconify()
        self.visible = True

    def hide(self) -> None:
        if self.left_win is None or self.right_win is None:
            self.visible = False
            return
        self.left_win.withdraw()
        self.right_win.withdraw()
        self.visible = False


# ===========================
# Coordinator
# ===========================

class DiffHighlighterApp:
    """
    Coordinates hotkeys, window capture, diff computation, overlay display, and debug previews.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.overlay = OverlayWindow(root)
        self.debug_views = DebugPreviewWindows(root)
        self._lock = threading.Lock()

        # Cache last halves for debug rendering
        self.last_left_img_bgr: Optional[np.ndarray] = None
        self.last_right_img_bgr: Optional[np.ndarray] = None

    def toggle_overlay(self) -> None:
        """
        Hotkey handler: if overlay visible -> clear; else capture and show (with blinking).
        """
        with self._lock:
            if self.overlay.visible:
                self._schedule(self.overlay.clear)
                return

            try:
                rect = get_foreground_rect(RECT_MODE)
                img_bgr = capture_window_image(rect)

                # Cache halves for debug
                h, w = img_bgr.shape[:2]
                mid = w // 2
                self.last_left_img_bgr = img_bgr[:, :mid]
                self.last_right_img_bgr = img_bgr[:, w - mid:]

                # Compute polygons and start animated overlay
                polys = compute_diff_polygons(img_bgr)
                self._schedule(self.overlay.start_animation, rect, (w, h), polys)

                # If debug is visible, update immediately
                if (
                    self.debug_views.visible
                    and self.last_left_img_bgr is not None
                    and self.last_right_img_bgr is not None
                ):
                    self._schedule(
                        self.debug_views.update_images,
                        self.last_left_img_bgr,
                        self.last_right_img_bgr,
                    )
            except Exception as exc:
                print(f"[ERROR] toggle_overlay failed: {exc}", file=sys.stderr)

    def toggle_debug(self) -> None:
        """
        Hotkey handler: show/hide debug preview windows.
        If turning on and cached halves exist, render immediately.
        """
        with self._lock:
            if self.debug_views.visible:
                self._schedule(self.debug_views.hide)
            else:
                self._schedule(self.debug_views.show)
                if self.last_left_img_bgr is not None and self.last_right_img_bgr is not None:
                    self._schedule(
                        self.debug_views.update_images,
                        self.last_left_img_bgr,
                        self.last_right_img_bgr,
                    )

    def quit_app(self) -> None:
        """Hotkey handler: quit application."""
        with self._lock:
            def _quit():
                try:
                    self.root.quit()
                except Exception:
                    pass
                os._exit(0)

            self._schedule(_quit)

    def _schedule(self, func, *args, **kwargs) -> None:
        """Schedule UI updates on Tk main thread using after(0, ...)."""
        self.root.after(0, lambda: func(*args, **kwargs))


def start_hotkeys(app: DiffHighlighterApp):
    """Register global hotkeys. Runs on a background thread."""
    hotkeys = keyboard.GlobalHotKeys({
        HOTKEY_TOGGLE_OVERLAY: app.toggle_overlay,
        HOTKEY_DEBUG_TOGGLE: app.toggle_debug,
        HOTKEY_QUIT: app.quit_app,
    })
    hotkeys.start()
    print(
        "Hotkeys: toggle_overlay={}, debug_toggle={}, quit={}".format(
            HOTKEY_TOGGLE_OVERLAY, HOTKEY_DEBUG_TOGGLE, HOTKEY_QUIT
        )
    )
    return hotkeys


# ===========================
# Main Entry
# ===========================

def main() -> None:
    set_dpi_awareness()
    root = tk.Tk()
    root.withdraw()  # We do not show the root; only Toplevels are used.

    app = DiffHighlighterApp(root)
    start_hotkeys(app)

    # Run Tk main loop on the main thread
    root.mainloop()


if __name__ == "__main__":
    main()
