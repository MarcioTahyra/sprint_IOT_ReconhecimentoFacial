from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import random

from dlib_utils import processar_frame_para_embedding, reconhecer, salvar_usuario, users

app = Flask(__name__)
app.secret_key = 'sua_chave_secreta_aqui'

CARTEIRAS = {
    'Conservador': {
        'alocacao': {'Renda Fixa': '80%', 'Fundos': '20%', 'Renda Variável': '0%'},
        'ativos': [
            {'nome': 'Tesouro Selic 2029', 'percentual': '40%', 'risco': 'Baixo'},
            {'nome': 'CDB Banco W (Pós-Fixado)', 'percentual': '40%', 'risco': 'Baixo'},
            {'nome': 'Fundo Renda Fixa DI', 'percentual': '20%', 'risco': 'Baixo'}
        ],
        'retorno': '~10.0% a.a.'
    },
    'Moderado': {
        'alocacao': {'Renda Fixa': '50%', 'Fundos': '30%', 'Renda Variável': '20%'},
        'ativos': [
            {'nome': 'Tesouro Selic 2029', 'percentual': '30%', 'risco': 'Baixo'},
            {'nome': 'LCI Banco X (Pré-Fixado)', 'percentual': '20%', 'risco': 'Médio'},
            {'nome': 'Fundo Multimercado Alpha', 'percentual': '15%', 'risco': 'Médio'},
            {'nome': 'Fundo de Crédito Privado Beta', 'percentual': '15%', 'risco': 'Médio'},
            {'nome': 'ETF BOVA11', 'percentual': '10%', 'risco': 'Alto'},
            {'nome': 'Ação Empresa Y', 'percentual': '10%', 'risco': 'Alto'}
        ],
        'retorno': '~12.5% a.a.'
    },
    'Agressivo': {
        'alocacao': {'Renda Fixa': '10%', 'Fundos': '30%', 'Renda Variável': '60%'},
        'ativos': [
            {'nome': 'Tesouro IPCA+ 2045', 'percentual': '10%', 'risco': 'Médio'},
            {'nome': 'Fundo de Ações Indexado', 'percentual': '20%', 'risco': 'Alto'},
            {'nome': 'Fundo Multimercado Gamma', 'percentual': '10%', 'risco': 'Alto'},
            {'nome': 'Ação Tech Z', 'percentual': '30%', 'risco': 'Muito Alto'},
            {'nome': 'BDRs Internacionais', 'percentual': '20%', 'risco': 'Alto'},
            {'nome': 'Fundo Imobiliário (FII)', 'percentual': '10%', 'risco': 'Médio'}
        ],
        'retorno': '~15.0%+ a.a.'
    }
}


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
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login'))

    username = session.get('username', 'Usuário')

    perfil_escolhido = random.choice(list(CARTEIRAS.keys()))

    dados_carteira = CARTEIRAS[perfil_escolhido]

    return render_template('dashboard.html',
                           username=username,
                           perfil=perfil_escolhido,
                           alocacao=dados_carteira['alocacao'],
                           ativos=dados_carteira['ativos'],
                           retorno=dados_carteira['retorno'])


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
        session['username'] = username
        session['logged_in'] = True
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
        return jsonify({'success': False, 'message': 'Rosto não detectado. Posicione-se melhor.'})

    salvar_usuario(username, password, vec)

    return jsonify({'success': True, 'message': 'Usuário cadastrado com sucesso!'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)