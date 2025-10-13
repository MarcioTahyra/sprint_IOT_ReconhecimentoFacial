function initWebcam(formId, videoId, canvasId, dataInputId, btnId, apiUrl, redirectUrl) {
    const video = document.getElementById(videoId);
    const canvas = document.getElementById(canvasId);
    const context = canvas.getContext('2d');
    const imageDataInput = document.getElementById(dataInputId);
    const form = document.getElementById(formId);
    const submitBtn = document.getElementById(btnId);

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
            })
            .catch(err => {
                console.error("Erro ao acessar a câmera: ", err);
                alert("ERRO: Não foi possível acessar a câmera. Verifique as permissões.");
                submitBtn.disabled = true; // Desabilita o botão se a câmera falhar
            });
    } else {
        alert("Seu navegador não suporta a API de Webcam.");
        submitBtn.disabled = true;
    }

    form.addEventListener('submit', function(e) {
        e.preventDefault();

        if (video.videoWidth === 0) {
            alert("Aguarde a câmera iniciar ou verifique as permissões.");
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
        .then(response => response.json())
        .then(data => {
            alert(data.message);
            submitBtn.disabled = false;
            submitBtn.textContent = (apiUrl.includes('login') ? 'Entrar' : 'Cadastrar');

            if (data.success) {
                window.location.href = redirectUrl;
            }
        })
        .catch(error => {
            console.error('Erro de rede:', error);
            alert('Erro de comunicação com o servidor. Verifique se o Flask está rodando.');
            submitBtn.disabled = false;
            submitBtn.textContent = (apiUrl.includes('login') ? 'Entrar' : 'Cadastrar');
        });
    });
}