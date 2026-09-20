"""
Fortis Desktop launcher: starts the FastAPI server in a background thread
and opens it in a native pywebview window. Falls back to the default
browser when pywebview is not installed.
"""
import argparse
import atexit
import os
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.app import create_server, PORT  # noqa: E402

_server = None
_server_thread = None


def _shutdown_server():
    global _server, _server_thread
    if _server is not None:
        _server.should_exit = True
    if _server_thread is not None:
        _server_thread.join(timeout=3.0)


atexit.register(_shutdown_server)


def _setup_windows_dpi():
    if sys.platform != "win32":
        return
    try:
        import ctypes
        shcore = ctypes.windll.shcore
        shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                import ctypes
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


def _set_app_user_model_id():
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("com.fortis.advisor")
    except Exception:
        pass


def _port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _generate_ico_from_png(png_path: str, ico_path: str) -> bool:
    try:
        from PIL import Image
        img = Image.open(png_path)
        img.save(ico_path, format="ICO", sizes=[(32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
        return True
    except Exception:
        return False


def _get_icon_path(static_dir: str) -> str | None:
    ico_path = os.path.join(static_dir, "logo.ico")
    png_path = os.path.join(static_dir, "logo.png")
    if os.path.exists(ico_path):
        return ico_path
    if os.path.exists(png_path):
        try:
            if _generate_ico_from_png(png_path, ico_path):
                return ico_path
        except Exception:
            pass
        return png_path
    return None


def _show_startup_error(message: str) -> None:
    html_content = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Fortis Startup Error</title>
<style>
body {{ font-family: system-ui, -apple-system, sans-serif; padding: 40px; background: #1a1a2e; color: #eee; }}
h1 {{ color: #e74c3c; }}
pre {{ background: #0f0f1a; padding: 15px; border-radius: 6px; overflow-x: auto; }}
</style>
</head>
<body>
<h1>Fortis Server Failed to Start</h1>
<p>The local server could not be reached. Please check the console for details or re-run the setup.</p>
<pre>{message}</pre>
</body>
</html>"""
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".html", prefix="fortis_error_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(html_content)
        import webview
        webview.create_window(
            "Fortis — Startup Error",
            f"file:///{tmp_path}",
            width=640,
            height=480,
            resizable=False,
        )
    except Exception:
        pass


def _inject_external_link_handler(window) -> None:
    js_code = """
(function() {
    if (window.__fortisExternalHandlerInstalled) return;
    window.__fortisExternalHandlerInstalled = true;
    document.addEventListener('click', function(e) {
        var target = e.target;
        while (target && target.tagName !== 'A') {
            target = target.parentElement;
        }
        if (!target || target.tagName !== 'A') return;
        var href = target.getAttribute('href');
        if (!href) return;
        if (href.startsWith('http://') || href.startsWith('https://')) {
            e.preventDefault();
            fetch('/open-external', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({url: href})
            }).catch(function() {});
        }
    }, true);
})();
"""
    try:
        window.evaluate_js(js_code)
    except Exception:
        pass


def main() -> None:
    global _server, _server_thread

    parser = argparse.ArgumentParser(description="Fortis Desktop")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--headless", action="store_true", help="Run server only, no window")
    args = parser.parse_args()

    _setup_windows_dpi()
    _set_app_user_model_id()

    app_url = f"http://127.0.0.1:{args.port}"
    if _port_in_use(args.port):
        print(f"Port {args.port} is already in use — another Fortis instance is likely running.", file=sys.stderr)
        print("Close the existing window first (or start with --port <other> to run a second copy).", file=sys.stderr)
        sys.exit(1)

    _server = create_server(port=args.port)
    _server_thread = threading.Thread(target=_server.run, daemon=True)
    _server_thread.start()

    health_ok = False
    health_error = None
    for _ in range(50):
        try:
            urllib.request.urlopen(app_url + "/health", timeout=1)
            health_ok = True
            break
        except Exception as e:
            health_error = str(e)
            time.sleep(0.2)

    if args.headless:
        if not health_ok:
            print(f"Fortis server failed to start at {app_url}: {health_error}", file=sys.stderr)
            return
        print(f"Fortis server running at {app_url}")
        try:
            while _server_thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        return

    if not health_ok:
        print(f"Fortis server failed to start at {app_url}: {health_error}", file=sys.stderr)
        _show_startup_error(f"Health check failed after retries. Last error: {health_error}")
        import webview
        webview.start()
        _shutdown_server()
        return

    try:
        import webview  # pywebview
        static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "server", "static")
        icon = _get_icon_path(static_dir)
        window = webview.create_window(
            "Fortis — Cybersecurity Advisor",
            app_url,
            width=1280,
            height=860,
            min_size=(960, 640),
        )

        def on_loaded():
            _inject_external_link_handler(window)

        window.events.loaded += on_loaded
        start_kwargs = {"icon": icon} if icon and icon.endswith(".ico") else {}
        webview.start(**start_kwargs)
        _shutdown_server()
    except ImportError:
        print("pywebview not installed — opening in the default browser instead.")
        webbrowser.open(app_url)
        try:
            while _server_thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    except Exception as e:
        print(f"Failed to launch window: {e}", file=sys.stderr)
        print("Opening in default browser instead.", file=sys.stderr)
        webbrowser.open(app_url)
        try:
            while _server_thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
