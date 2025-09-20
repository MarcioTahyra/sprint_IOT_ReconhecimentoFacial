import cv2
import dlib
import numpy as np
import os
import pickle
import tkinter as tk
from tkinter import messagebox

# Usuários cadastrados:
# -yan     / 123
# -fallen  / 123

PREDICTOR = "shape_predictor_5_face_landmarks.dat"
RECOG = "dlib_face_recognition_resnet_model_v1.dat"
DB_FILE = "db.pkl"
USERS_FILE = "users.pkl"

# THRESH (0~1): Parâmetro que define a tolerância aceita entre face detectada e armazenada
THRESH = 0.5


BG_COLOR = "#2c3e50"
FG_COLOR = "#ecf0f1"
BTN_COLOR = "#3498db"
BTN_HOVER_COLOR = "#2980b9"
ENTRY_BG = "#34495e"
FONT_TITLE = ("Calibri", 24, "bold")
FONT_LABEL = ("Calibri", 12)
FONT_BUTTON = ("Calibri", 14, "bold")

try:
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(PREDICTOR)
    rec = dlib.face_recognition_model_v1(RECOG)
except RuntimeError as e:
    print(f"Erro ao carregar os modelos Dlib: {e}")
    exit()

db = pickle.load(open(DB_FILE, "rb")) if os.path.exists(DB_FILE) else {}
users = pickle.load(open(USERS_FILE, "rb")) if os.path.exists(USERS_FILE) else {}

def get_embedding(img, rect):
    shape = sp(img, rect)
    chip = dlib.get_face_chip(img, shape)
    return np.array(rec.compute_face_descriptor(chip), dtype=np.float32)

def reconhecer(nome, vec):
    if nome not in db:
        return False
    dist = np.linalg.norm(vec - db[nome])
    return dist <= THRESH


def salvar_usuario(nome, senha, vec):
    users[nome] = senha
    db[nome] = vec
    with open(USERS_FILE, "wb") as f:
        pickle.dump(users, f)
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)


def capturar_rosto():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Erro de Câmera", "Não foi possível acessar a câmera.")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    vec = None
    tempo_ini = 0
    capture_success = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rects = detector(rgb, 1)

        if rects:
            rect = rects[0]
            vec = get_embedding(rgb, rect)
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Rosto detectado!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if tempo_ini == 0:
                tempo_ini = cv2.getTickCount()

            tempo_pass = (cv2.getTickCount() - tempo_ini) / cv2.getTickFrequency()
            remaining_time = 3 - int(tempo_pass)

            cv2.putText(frame, f"Capturando em {remaining_time}...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)

            if tempo_pass >= 3:
                capture_success = True
                break
        else:
            tempo_ini = 0
            cv2.putText(frame, "Posicione seu rosto no centro", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Captura de Rosto - Aguarde 3 segundos", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return vec if capture_success else None


def tela_cadastro():
    janela = tk.Tk()
    janela.title("Cadastrar")
    janela.geometry("360x640")
    janela.resizable(False, False)
    janela.configure(bg=BG_COLOR)

    def registrar():
        nome = entry_nome.get().strip()
        senha = entry_senha.get().strip()

        if not nome or not senha:
            messagebox.showwarning("Campos vazios", "Preencha todos os campos.")
            return
        if nome in users:
            messagebox.showerror("Erro", "Usuário já existe.")
            return

        janela.withdraw()
        messagebox.showinfo("Captura de Rosto",
                            "A seguir, vamos capturar seu rosto.\nPosicione-se em frente à câmera e aguarde 3 segundos.")
        vec = capturar_rosto()

        if vec is not None:
            salvar_usuario(nome, senha, vec)
            messagebox.showinfo("Sucesso", f"Usuário '{nome}' cadastrado com sucesso!")
            janela.destroy()
            tela_inicial()
        else:
            messagebox.showerror("Erro de Captura", "Não foi possível capturar o rosto. Tente novamente.")
            janela.deiconify()

    def on_enter(e):
        e.widget['background'] = BTN_HOVER_COLOR

    def on_leave(e):
        e.widget['background'] = BTN_COLOR

    frame = tk.Frame(janela, bg=BG_COLOR)
    frame.pack(expand=True, padx=40)

    tk.Label(frame, text="Criar Conta", font=FONT_TITLE, bg=BG_COLOR, fg=FG_COLOR).pack(pady=(0, 40))

    tk.Label(frame, text="Nome de usuário:", font=FONT_LABEL, bg=BG_COLOR, fg=FG_COLOR).pack(anchor='w')
    entry_nome = tk.Entry(frame, font=FONT_LABEL, bg=ENTRY_BG, fg=FG_COLOR, relief='flat', insertbackground=FG_COLOR)
    entry_nome.pack(pady=(5, 20), ipady=8, fill='x')

    tk.Label(frame, text="Senha:", font=FONT_LABEL, bg=BG_COLOR, fg=FG_COLOR).pack(anchor='w')
    entry_senha = tk.Entry(frame, font=FONT_LABEL, bg=ENTRY_BG, fg=FG_COLOR, show="*", relief='flat',
                           insertbackground=FG_COLOR)
    entry_senha.pack(pady=(5, 40), ipady=8, fill='x')

    btn_registrar = tk.Button(frame, text="Cadastrar", font=FONT_BUTTON, bg=BTN_COLOR, fg=FG_COLOR,
                              relief='flat', command=registrar, activebackground=BTN_HOVER_COLOR,
                              activeforeground=FG_COLOR)
    btn_registrar.pack(pady=10, ipady=10, fill='x')
    btn_registrar.bind("<Enter>", on_enter)
    btn_registrar.bind("<Leave>", on_leave)

    btn_voltar = tk.Button(frame, text="Voltar", font=FONT_LABEL, bg=BG_COLOR, fg=FG_COLOR,
                           relief='flat', activebackground=BG_COLOR, activeforeground=FG_COLOR,
                           command=lambda: [janela.destroy(), tela_inicial()])
    btn_voltar.pack(pady=20)

    janela.mainloop()


def tela_login():
    janela = tk.Tk()
    janela.title("Login")
    janela.geometry("360x640")
    janela.resizable(False, False)
    janela.configure(bg=BG_COLOR)

    def autenticar():
        nome = entry_nome.get().strip()
        senha = entry_senha.get().strip()

        if not nome or not senha:
            messagebox.showwarning("Campos vazios", "Preencha todos os campos.")
            return
        if nome not in users:
            messagebox.showerror("Erro", "Usuário não encontrado.")
            return
        if senha != users[nome]:
            messagebox.showerror("Erro", "Senha incorreta.")
            return

        janela.withdraw()
        messagebox.showinfo("Verificação Facial",
                            "Vamos verificar seu rosto.\nPosicione-se em frente à câmera e aguarde 3 segundos.")
        vec = capturar_rosto()

        if vec is None:
            messagebox.showerror("Erro de Captura", "Não foi possível capturar o rosto. Tente novamente.")
            janela.deiconify()
            return

        if reconhecer(nome, vec):
            messagebox.showinfo("Acesso Liberado", f"Bem-vindo, {nome}!")
            print(f"[ACESSO LIBERADO] {nome}")
            janela.destroy()
        else:
            messagebox.showerror("Acesso Negado", "Rosto não reconhecido.")
            janela.deiconify()

    def on_enter(e):
        e.widget['background'] = BTN_HOVER_COLOR

    def on_leave(e):
        e.widget['background'] = BTN_COLOR

    frame = tk.Frame(janela, bg=BG_COLOR)
    frame.pack(expand=True, padx=40)

    tk.Label(frame, text="Login", font=FONT_TITLE, bg=BG_COLOR, fg=FG_COLOR).pack(pady=(0, 40))

    tk.Label(frame, text="Nome de usuário:", font=FONT_LABEL, bg=BG_COLOR, fg=FG_COLOR).pack(anchor='w')
    entry_nome = tk.Entry(frame, font=FONT_LABEL, bg=ENTRY_BG, fg=FG_COLOR, relief='flat', insertbackground=FG_COLOR)
    entry_nome.pack(pady=(5, 20), ipady=8, fill='x')

    tk.Label(frame, text="Senha:", font=FONT_LABEL, bg=BG_COLOR, fg=FG_COLOR).pack(anchor='w')
    entry_senha = tk.Entry(frame, font=FONT_LABEL, bg=ENTRY_BG, fg=FG_COLOR, show="*", relief='flat',
                           insertbackground=FG_COLOR)
    entry_senha.pack(pady=(5, 40), ipady=8, fill='x')

    btn_entrar = tk.Button(frame, text="Entrar", font=FONT_BUTTON, bg=BTN_COLOR, fg=FG_COLOR,
                           relief='flat', command=autenticar, activebackground=BTN_HOVER_COLOR,
                           activeforeground=FG_COLOR)
    btn_entrar.pack(pady=10, ipady=10, fill='x')
    btn_entrar.bind("<Enter>", on_enter)
    btn_entrar.bind("<Leave>", on_leave)

    btn_voltar = tk.Button(frame, text="Voltar", font=FONT_LABEL, bg=BG_COLOR, fg=FG_COLOR,
                           relief='flat', activebackground=BG_COLOR, activeforeground=FG_COLOR,
                           command=lambda: [janela.destroy(), tela_inicial()])
    btn_voltar.pack(pady=20)

    janela.mainloop()


def tela_inicial():
    root = tk.Tk()
    root.title("Reconhecimento Facial")
    root.geometry("360x640")
    root.resizable(False, False)
    root.configure(bg=BG_COLOR)

    def on_enter(e): e.widget['background'] = BTN_HOVER_COLOR

    def on_leave(e): e.widget['background'] = BTN_COLOR

    main_frame = tk.Frame(root, bg=BG_COLOR)
    main_frame.pack(expand=True, padx=40)

    tk.Label(main_frame, text="Bem-vindo", font=FONT_TITLE, bg=BG_COLOR, fg=FG_COLOR).pack(pady=(0, 60))

    btn_login = tk.Button(main_frame, text="Login", font=FONT_BUTTON, bg=BTN_COLOR, fg=FG_COLOR,
                          relief='flat', command=lambda: [root.destroy(), tela_login()],
                          activebackground=BTN_HOVER_COLOR, activeforeground=FG_COLOR)
    btn_login.pack(pady=10, ipady=10, fill='x')
    btn_login.bind("<Enter>", on_enter)
    btn_login.bind("<Leave>", on_leave)

    btn_cadastro = tk.Button(main_frame, text="Cadastrar", font=FONT_BUTTON, bg=BTN_COLOR, fg=FG_COLOR,
                             relief='flat', command=lambda: [root.destroy(), tela_cadastro()],
                             activebackground=BTN_HOVER_COLOR, activeforeground=FG_COLOR)
    btn_cadastro.pack(pady=10, ipady=10, fill='x')
    btn_cadastro.bind("<Enter>", on_enter)
    btn_cadastro.bind("<Leave>", on_leave)

    btn_sair = tk.Button(main_frame, text="Sair", font=FONT_LABEL, bg=BG_COLOR, fg=FG_COLOR,
                         relief='flat', activebackground=BG_COLOR, activeforeground=FG_COLOR, command=root.destroy)
    btn_sair.pack(pady=(40, 0))

    root.mainloop()

if __name__ == "__main__":
    tela_inicial()