import sys
import threading
import time
import webview
from config.config import config

# Attempt to import the Flask app instance and monitoring loop.
# Please ensure your main application file is named 'app.py' or 'main.py'
# and the Flask instance is named 'app'.
try:
    from app import app, monitoring_loop
except ImportError:
    try:
        from main import app, monitoring_loop
    except ImportError:
        print("Error: Could not import 'app' or 'monitoring_loop'. Please ensure your main Flask file is named 'app.py' or 'main.py'.")
        sys.exit(1)

def start_server():
    """Starts the Flask server in a separate thread."""
    # We disable the reloader because it doesn't work well in a separate thread
    app.run(host=config.HOST, port=config.PORT, debug=False, use_reloader=False)

if __name__ == '__main__':
    # Start the monitoring loop in a background thread
    # This handles the camera and AI detection
    monitor_t = threading.Thread(target=monitoring_loop)
    monitor_t.daemon = True
    monitor_t.start()

    # Start Flask in a background thread
    server_t = threading.Thread(target=start_server)
    server_t.daemon = True
    server_t.start()

    # Give the server a second to initialize
    time.sleep(1)

    # Create the native desktop window
    webview.create_window(
        config.WINDOW_TITLE,
        f'http://{config.HOST}:{config.PORT}',
        width=config.FRAME_WIDTH,
        height=config.FRAME_HEIGHT
    )

    # Start the GUI loop
    webview.start()