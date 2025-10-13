from flask import Flask, render_template, request, redirect, url_for, jsonify
from dlib_utils import processar_frame_para_embedding, reconhecer, salvar_usuario, users

app = Flask(__name__)
app.secret_key = '123123'

@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/cadastro')
def cadastro():
    return render_template('cadastro.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/api/login', methods=['POST'])
def api_login():

    username = request.form.get('username')
    password = request.form.get('password')
    img_data = request.form.get('image_data')

    if not username or not password or not img_data:
        return jsonify({'success': False, 'message': 'Dados incompletos.'}), 400

    if username not in users:
        return jsonify({'success': False, 'message': 'Usuário não encontrado.'})
    if users[username] != password:
        return jsonify({'success': False, 'message': 'Senha incorreta.'})

    vec = processar_frame_para_embedding(img_data)

    if vec is None:
        return jsonify({'success': False, 'message': 'Rosto não detectado. Posicione-se melhor.'})

    if reconhecer(username, vec):
        # Em um app real, iniciaria uma sessão Flask
        print(f"[ACESSO LIBERADO] {username}")
        return jsonify({'success': True, 'message': f'Bem-vindo, {username}!'})
    else:
        return jsonify({'success': False, 'message': 'Autenticação facial falhou. Rosto não reconhecido.'})


@app.route('/api/cadastro', methods=['POST'])
def api_cadastro():

    username = request.form.get('username')
    password = request.form.get('password')
    img_data = request.form.get('image_data')

    if not username or not password or not img_data:
        return jsonify({'success': False, 'message': 'Dados incompletos.'}), 400

    if username in users:
        return jsonify({'success': False, 'message': 'Usuário já existe.'})

    vec = processar_frame_para_embedding(img_data)

    if vec is None:
        return jsonify({'success': False, 'message': 'Rosto não detectado. Não foi possível cadastrar.'})

    salvar_usuario(username, password, vec)

    return jsonify({'success': True, 'message': f"Usuário '{username}' cadastrado com sucesso! Faca login."})


if __name__ == '__main__':
    app.run(debug=True)