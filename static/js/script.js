// Auto refresh dashboard and logs every 10 seconds
setInterval(function() {
    if (window.location.pathname === '/' || window.location.pathname === '/logs') {
        location.reload();
    }
}, 10000);

// remove or neutralize camera change handler
const cameraSelect = document.getElementById('camera-select');
if (cameraSelect) {
  cameraSelect.remove(); // remove selector from DOM if still present
}
const cameraInfo = document.getElementById('camera-info');
if (cameraInfo) {
  cameraInfo.textContent = 'Camera: 1 (fixed)';
}
