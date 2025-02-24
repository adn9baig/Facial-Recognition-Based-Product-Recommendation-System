const video = document.getElementById('video');
const captureButton = document.getElementById('capture-face');
const loginForm = document.getElementById('login-form');
const faceDataInput = document.getElementById('face-data');

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    });

captureButton.addEventListener('click', () => {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/png');
    faceDataInput.value = dataURL;
    alert('Face captured.');
});