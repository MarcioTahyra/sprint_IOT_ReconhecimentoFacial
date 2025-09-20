# Reconhecimento Facial para Autenticação Bancária

## Objetivo
Este projeto é um **protótipo** de sistema de autenticação para aplicativos bancários que combina:
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
python -m pip intall cmake dlib-bin opencv-python pyserial
```

## Parâmetros principais

- THRESH = 0.6 → limiar de distância entre vetores faciais.
- Tempo de captura = 3s → rosto deve estar visível por 3 segundos antes do registro/validação.

## Nota Ética

O reconhecimento facial envolve dados biométricos sensíveis, que são regulados por legislações como a LGPD.
Este projeto tem fins exclusivamente educacionais/prototipagem e não deve ser usado em produção sem:

- Consentimento explícito dos usuários.
- Armazenamento seguro (criptografia, anonimização).
- Auditoria de vieses (garantir que funcione para diferentes etnias, idades e gêneros).
- Conformidade com as leis de proteção de dados.

  ## link do vídeos explicativo
  https://youtu.be/aI_Wlgz2OXw
