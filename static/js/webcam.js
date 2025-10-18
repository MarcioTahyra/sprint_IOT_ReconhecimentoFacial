function showNotification(message, isSuccess = false) {
    const notificationArea = document.getElementById('notification-area');
    if (!notificationArea) {
        console.log("Notificação:", message);
        return;
    }

    notificationArea.textContent = message;
    notificationArea.style.display = 'block';
    notificationArea.style.backgroundColor = isSuccess ? '#0a6400' : '#8b0000';

    setTimeout(() => {
        notificationArea.style.display = 'none';
        notificationArea.textContent = '';
    }, 5000);
}

function initWebcam(formId, videoId, canvasId, dataInputId, btnId, apiUrl, redirectUrl) {
    const video = document.getElementById(videoId);
    const canvas = document.getElementById(canvasId);
    const context = canvas.getContext('2d');
    const imageDataInput = document.getElementById(dataInputId);
    const form = document.getElementById(formId);
    const submitBtn = document.getElementById(btnId);
    let cameraStarted = false;

    if (!document.getElementById('notification-area')) {
        const notificationDiv = document.createElement('div');
        notificationDiv.id = 'notification-area';
        notificationDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px;
            border-radius: 5px;
            color: white;
            z-index: 1000;
            display: none;
            font-weight: bold;
        `;
        document.body.appendChild(notificationDiv);
    }

    function startCamera() {
        if (cameraStarted) return;

        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    video.srcObject = stream;
                    cameraStarted = true;
                    video.style.display = 'block';
                    submitBtn.textContent = (apiUrl.includes('login') ? 'ENTRAR' : 'CADASTRAR');
                    submitBtn.dataset.action = 'submit';
                })
                .catch(err => {
                    console.error("Erro ao acessar a câmera: ", err);
                    showNotification("ERRO: Não foi possível acessar a câmera. Verifique as permissões.");
                    submitBtn.disabled = true;
                });
        } else {
            showNotification("Seu navegador não suporta a API de Webcam.");
            submitBtn.disabled = true;
        }
    }

    submitBtn.dataset.action = 'start-camera';

    form.addEventListener('submit', function(e) {
        e.preventDefault();

        if (submitBtn.dataset.action === 'start-camera') {
            const username = form.querySelector('input[name="username"]').value.trim();
            const password = form.querySelector('input[name="password"]').value.trim();

            if (!username || !password) {
                 showNotification("Preencha Usuário e Senha antes de iniciar a câmera.");
                 return;
            }

            startCamera();
            return;
        }

        if (video.videoWidth === 0) {
            showNotification("Aguarde a câmera iniciar ou verifique as permissões.");
            return;
        }

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        const imageData = canvas.toDataURL('image/jpeg', 0.8);

        imageDataInput.value = imageData;

        submitBtn.disabled = true;
        submitBtn.textContent = 'Processando...';

        fetch(apiUrl, {
            method: 'POST',
            body: new FormData(form)
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errorData => {
                    throw new Error(errorData.message || `Erro do servidor: ${response.status}`);
                }).catch(() => {
                    throw new Error(`Erro do servidor. Código: ${response.status}`);
                });
            }
            return response.json();
        })
        .then(data => {
            showNotification(data.message, data.success);
            submitBtn.disabled = false;
            submitBtn.textContent = (apiUrl.includes('login') ? 'ENTRAR' : 'CADASTRAR');

            if (data.success) {
                setTimeout(() => {
                    window.location.href = redirectUrl;
                }, 1000);
            }
        })
        .catch(error => {
            console.error('Erro de comunicação:', error);
            const errorMessage = error.message.includes('Erro do servidor') ? error.message : 'Erro de comunicação com o servidor. Verifique se o Flask está rodando.';
            showNotification(errorMessage);
            submitBtn.disabled = false;
            submitBtn.textContent = (apiUrl.includes('login') ? 'ENTRAR' : 'CADASTRAR');
        });
    });
}