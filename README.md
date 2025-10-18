# Reconhecimento Facial para Autenticação Bancária

## Integrantes
- Gustavo Kenzo - 98481
- Vinícius Almeida Bernardino de Souza - 97888
- Márcio Hitoshi Tahyra - 552511

## Objetivo
Este projeto é um sistema de autenticação para aplicativos bancários que combina:
- **Login por senha** (tradicional)
- **Reconhecimento facial** (biometria)

A proposta é aumentar a segurança, dificultando acesso indevido em caso de vazamento de senhas.

---

## Execução
### Pré-requisitos
- Python 3.8+  
- Webcam funcional  
- Modelos Dlib necessários:
  - `shape_predictor_5_face_landmarks.dat`
  - `dlib_face_recognition_resnet_model_v1.dat`

### Instalação
```bash
python -m pip intall cmake dlib-bin opencv-python pyserial flask Pillow
```

### Inicialização
```bash
python app.py
```

## Parâmetros principais

- THRESH = 0.6 → limiar de distância entre vetores faciais.
- Tempo de captura = 3s → rosto deve estar visível por 3 segundos antes do registro/validação.


# Integração e Fluxo da Autenticação Facial

O sistema combina o **frontend (HTML/JavaScript)** com o **backend (Python/Flask)** e a biblioteca **Dlib**.

---

## 1. Frontend (Webcam Capture)

O arquivo `webcam.js` é responsável por:

- Solicitar acesso à câmera via `navigator.mediaDevices.getUserMedia`.
- Ao clicar no botão **submit** do formulário (Login ou Cadastro), ele desenha o frame atual da `<video>` em um `<canvas>` escondido.
- O conteúdo do canvas é convertido para uma string **Base64 (image/jpeg)**.
- Essa string Base64 é inserida no campo `<input type="hidden" name="image_data">`.
- O formulário envia `username`, `password` e `image_data` para o servidor Flask.

---

## 2. Backend (Flask e Dlib)

### Fluxo de Cadastro (`/api/cadastro`)

1. **Validação:**  
   O servidor Flask em `app.py` recebe a requisição e verifica se o usuário já existe.

2. **Processamento Biométrico:**  
   O Base64 (`image_data`) é enviado para a função `processar_frame_para_embedding` em `dlib_utils.py`.  
   Essa função:
   - Decodifica o Base64 para um objeto de imagem.  
   - Detecta o rosto e calcula o vetor facial (**embedding**) de 128 dimensões.

3. **Armazenamento:**  
   O Flask chama `salvar_usuario`, que armazena a senha e o vetor facial nos arquivos `users.pkl` e `db.pkl`, respectivamente.

---

### Fluxo de Login (`/api/login`)

1. **Autenticação Tradicional:**  
   O Flask verifica se o `username` existe e se a `password` enviada está correta.

2. **Processamento Biométrico:**  
   Se a senha for válida, o Base64 da imagem é processado para obter um novo vetor facial.

3. **Reconhecimento Facial (Comparação):**  
   O Flask chama `reconhecer(username, vec)`, que:
   - Compara o vetor recém-capturado (`vec`) com o vetor armazenado (`db[nome]`).
   - Calcula a **distância euclidiana** (`np.linalg.norm`).  
     Se a distância for menor ou igual ao **THRESH**, a autenticação facial é aprovada.

4. **Acesso:**  
   Se **senha** e **reconhecimento facial** forem bem-sucedidos, o usuário é logado via `session` e redirecionado para `/dashboard`.

---


## Nota Ética

O reconhecimento facial envolve dados biométricos sensíveis, que são regulados por legislações como a LGPD.
Este projeto tem fins exclusivamente educacionais/prototipagem e não deve ser usado em produção sem:

- Consentimento explícito dos usuários.
- Armazenamento seguro (criptografia, anonimização).
- Auditoria de vieses (garantir que funcione para diferentes etnias, idades e gêneros).
- Conformidade com as leis de proteção de dados.

  ## Vídeo

