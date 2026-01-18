from flask import Flask, request, jsonify
from routes.dashboard_routes import dashboard_bp
from routes.api_routes import api_bp
from utils.helpers import setup_logging
import threading
from config.config import config

app = Flask(__name__)
app.register_blueprint(dashboard_bp)
app.register_blueprint(api_bp, url_prefix='/api')

setup_logging()

@app.route('/set_camera', methods=['POST'])
def set_camera():
    return jsonify({'error': 'camera selection disabled; using camera 1'}), 403

if __name__ == '__main__':
    # The reloader can cause issues in a multi-threaded app, especially with libraries
    # that manage their own event loops or resources (like a camera thread).
    # Disabling it prevents crashes on file changes during development.
    app.run(debug=True, use_reloader=False)
