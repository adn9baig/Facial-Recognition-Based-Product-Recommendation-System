const video = document.getElementById('video');
const captureButton = document.getElementById('capture-face');
const registrationForm = document.getElementById('registration-form');
const faceDataInput = document.getElementById('face-data');

// Load face-api.js models
Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('/models')
]).then(startVideo);

function startVideo() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        });
}

captureButton.addEventListener('click', async () => {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Detect faces in the captured frame
    const faceDetection = await faceapi.detectSingleFace(canvas);

    if (faceDetection) {
        const dataURL = canvas.toDataURL('image/png');
        faceDataInput.value = dataURL;
        alert('Face captured.');
    } else {
        alert('No face detected.');
    }
});
