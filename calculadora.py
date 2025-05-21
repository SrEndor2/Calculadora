import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sympy as sp

# Métodos numéricos implementados

def falsa_posicion(f, a, b, tol=1e-6, max_iter=100):
    """Método de Falsa Posición para encontrar raíces de una función."""
    if f(a) * f(b) >= 0:
        raise ValueError("El intervalo [a, b] debe contener una raíz.")
    c = a
    for i in range(max_iter):
        c = b - f(b) * (b - a) / (f(b) - f(a))
        if abs(f(c)) < tol:
            return c
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
    return c

def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    """Método de Newton-Raphson para encontrar raíces de una función."""
    x = x0
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        dfx = df(x)
        if dfx == 0:
            raise ValueError("La derivada es cero, no se puede continuar.")
        x = x - fx / dfx
    return x

def biseccion(f, a, b, tol=1e-6, max_iter=100):
    """Método de Bisección para encontrar raíces de una función."""
    if f(a) * f(b) >= 0:
        raise ValueError("El intervalo [a, b] debe contener una raíz.")
    for i in range(max_iter):
        c = (a + b) / 2
        if abs(f(c)) < tol:
            return c
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

def gauss_seidel(A, b, x0, tol=1e-6, max_iter=100):
    """Método de Gauss-Seidel para resolver sistemas de ecuaciones lineales."""
    n = len(b)
    x = x0.copy()
    historia = [x.copy()]
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], x_old[i+1:])) / A[i, i]
        historia.append(x.copy())
        if np.linalg.norm(x - x_old) < tol:
            return x, np.array(historia)
    return x, np.array(historia)

def interpolacion(x, y, x_new):
    """Interpolación polinomial de Lagrange. Devuelve y_new y la fórmula simbólica."""
    n = len(x)
    y_new = np.zeros_like(x_new)
    x_sym = sp.symbols('x')
    P = 0
    for i in range(n):
        p = np.ones_like(x_new)
        L = 1
        for j in range(n):
            if i != j:
                p *= (x_new - x[j]) / (x[i] - x[j])
                L *= (x_sym - x[j]) / (x[i] - x[j])
        y_new += y[i] * p
        P += y[i] * L
    return y_new, sp.simplify(P)

def runge_kutta(f, t0, y0, h, n):
    """Método de Runge-Kutta de cuarto orden para EDOs."""
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)
    t[0] = t0
    y[0] = y0
    for i in range(n):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h/2, y[i] + k1/2)
        k3 = h * f(t[i] + h/2, y[i] + k2/2)
        k4 = h * f(t[i] + h, y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
        t[i + 1] = t[i] + h
    return t, y

def secante(f, x0, x1, tol=1e-6, max_iter=100):
    """Método de la Secante para encontrar raíces de una función."""
    for i in range(max_iter):
        if abs(f(x1) - f(x0)) < 1e-14:
            raise ValueError("Diferencia muy pequeña entre f(x1) y f(x0), posible división por cero.")
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(f(x2)) < tol:
            return x2
        x0, x1 = x1, x2
    return x2

class RaicesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Calculadora de Métodos Numéricos")
        self.root.geometry("480x520")
        self.root.configure(bg="#f4f6fb")

        # Título destacado
        self.title_label = tk.Label(root, text="Calculadora de Métodos Numéricos", font=("Arial", 18, "bold"), bg="#3a7ca5", fg="white", pady=12, bd=0, relief="flat")
        self.title_label.pack(fill=tk.X, pady=(0, 10))

        # Frame fijo para el menú de selección de método
        self.top_frame = tk.Frame(root, bg="#f4f6fb")
        self.top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        tk.Label(self.top_frame, text="Método:", font=("Arial", 12), bg="#f4f6fb").pack(side=tk.LEFT, padx=5)
        self.metodo_var = tk.StringVar()
        self.metodo_combo = ttk.Combobox(self.top_frame, textvariable=self.metodo_var, state="readonly", width=25, font=("Arial", 11))
        self.metodo_combo['values'] = ("Falsa Posición", "Secante", "Newton-Raphson", "Bisección", "Gauss-Seidel", "Interpolación", "Runge-Kutta")
        self.metodo_combo.current(0)
        self.metodo_combo.pack(side=tk.LEFT, padx=5)
        self.metodo_combo.bind("<<ComboboxSelected>>", self.actualizar_campos)

        # Estilo para el menú desplegable
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TCombobox', fieldbackground='#e9eef6', background='#e9eef6', bordercolor='#3a7ca5', borderwidth=1, relief="flat")

        # Frame para los campos dinámicos de entrada
        self.center_frame = tk.Frame(root, bg="#f4f6fb", bd=2, relief="groove")
        self.center_frame.pack(expand=True, pady=10, padx=20, fill=tk.BOTH)

        # Frame fijo para el botón de calcular
        self.bottom_frame = tk.Frame(root, bg="#f4f6fb")
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=15)
        self.calc_btn = tk.Button(self.bottom_frame, text="Calcular", command=self.calcular, font=("Arial", 13, "bold"), bg="#3a7ca5", fg="white", activebackground="#28516b", activeforeground="white", bd=0, relief="ridge", padx=20, pady=8, cursor="hand2")
        self.calc_btn.pack(ipady=2)

        # Diccionarios para los campos y etiquetas
        self.labels = {}
        self.entries = {}
        campos = [
            ("f(x):", "funcion", "x**3 - 2*x - 5"),
            ("Tolerancia:", "tol", "0.0001"),
            ("Máx. Iteraciones:", "iter", "50"),
            ("a:", "a", "1"),
            ("b:", "b", "3"),
            ("x0:", "x0", "2"),
            ("x1 (Secante):", "x1s", "2.5"),
            ("Matriz A (Gauss-Seidel, filas separadas por ;):", "A", "4,1;1,3"),
            ("Vector b (Gauss-Seidel):", "bvec", "1,2"),
            ("x0 (Gauss-Seidel):", "x0gs", "0,0"),
            ("X (Interpolación):", "xinterp", "0,1,2,3"),
            ("Y (Interpolación):", "yinterp", "0,1,4,9"),
            ("f(t, y) (Runge-Kutta):", "frk", "t + y"),
            ("t0 (Runge-Kutta):", "t0", "0"),
            ("y0 (Runge-Kutta):", "y0", "1"),
            ("h (Runge-Kutta):", "h", "0.1"),
            ("n (Runge-Kutta):", "n", "10"),
        ]
        self.campos = campos
        for i, (label, key, default) in enumerate(campos):
            self.labels[key] = tk.Label(self.center_frame, text=label, font=("Arial", 12), bg="#f4f6fb")
            self.entries[key] = tk.Entry(self.center_frame, width=25, font=("Arial", 12), bg="#e9eef6", relief="groove", bd=2, highlightthickness=0)
            self.entries[key].insert(0, default)

        self.actualizar_campos()

    def actualizar_campos(self, event=None):
        """Muestra solo los campos necesarios según el método seleccionado."""
        metodo = self.metodo_var.get()
        # Oculta todos los campos
        for key in self.labels:
            self.labels[key].grid_remove()
            self.entries[key].grid_remove()
        # Muestra solo los necesarios
        row = 0
        if metodo in ["Falsa Posición", "Bisección"]:
            for key in ["funcion", "tol", "iter", "a", "b"]:
                self.labels[key].grid(row=row, column=0, sticky="e", padx=8, pady=6)
                self.entries[key].grid(row=row, column=1, padx=8, pady=6)
                row += 1
        elif metodo == "Secante":
            for key in ["funcion", "tol", "iter", "x0", "x1s"]:
                self.labels[key].grid(row=row, column=0, sticky="e", padx=8, pady=6)
                self.entries[key].grid(row=row, column=1, padx=8, pady=6)
                row += 1
        elif metodo == "Newton-Raphson":
            for key in ["funcion", "tol", "iter", "x0"]:
                self.labels[key].grid(row=row, column=0, sticky="e", padx=8, pady=6)
                self.entries[key].grid(row=row, column=1, padx=8, pady=6)
                row += 1
        elif metodo == "Gauss-Seidel":
            for key in ["A", "bvec", "x0gs", "tol", "iter"]:
                self.labels[key].grid(row=row, column=0, sticky="e", padx=8, pady=6)
                self.entries[key].grid(row=row, column=1, padx=8, pady=6)
                row += 1
        elif metodo == "Interpolación":
            for key in ["xinterp", "yinterp"]:
                self.labels[key].grid(row=row, column=0, sticky="e", padx=8, pady=6)
                self.entries[key].grid(row=row, column=1, padx=8, pady=6)
                row += 1
        elif metodo == "Runge-Kutta":
            for key in ["frk", "t0", "y0", "h", "n"]:
                self.labels[key].grid(row=row, column=0, sticky="e", padx=8, pady=6)
                self.entries[key].grid(row=row, column=1, padx=8, pady=6)
                row += 1

    def calcular(self):
        """Ejecuta el método seleccionado y muestra el resultado y/o gráfica."""
        metodo = self.metodo_var.get()
        try:
            if metodo == "Falsa Posición":
                f_str = self.entries["funcion"].get()
                f = lambda x: eval(f_str)
                a = float(self.entries["a"].get())
                b = float(self.entries["b"].get())
                tol = float(self.entries["tol"].get())
                max_iter = int(self.entries["iter"].get())
                raiz = falsa_posicion(f, a, b, tol, max_iter)
                self.graficar_funcion(f, a, b, raiz, f_str)
                messagebox.showinfo("Resultado", f"Raíz ≈ {raiz}")
            elif metodo == "Secante":
                f_str = self.entries["funcion"].get()
                f = lambda x: eval(f_str)
                x0 = float(self.entries["x0"].get())
                x1 = float(self.entries["x1s"].get())
                tol = float(self.entries["tol"].get())
                max_iter = int(self.entries["iter"].get())
                raiz = secante(f, x0, x1, tol, max_iter)
                self.graficar_funcion(f, x0-5, x1+5, raiz, f_str)
                messagebox.showinfo("Resultado", f"Raíz ≈ {raiz}")
            elif metodo == "Bisección":
                f_str = self.entries["funcion"].get()
                f = lambda x: eval(f_str)
                a = float(self.entries["a"].get())
                b = float(self.entries["b"].get())
                tol = float(self.entries["tol"].get())
                max_iter = int(self.entries["iter"].get())
                raiz = biseccion(f, a, b, tol, max_iter)
                self.graficar_funcion(f, a, b, raiz, f_str)
                messagebox.showinfo("Resultado", f"Raíz ≈ {raiz}")
            elif metodo == "Newton-Raphson":
                f_str = self.entries["funcion"].get()
                x0 = float(self.entries["x0"].get())
                tol = float(self.entries["tol"].get())
                max_iter = int(self.entries["iter"].get())
                # Derivada automática con sympy
                x = sp.symbols('x')
                f_expr = sp.sympify(f_str)
                df_expr = sp.diff(f_expr, x)
                f_lambda = sp.lambdify(x, f_expr, 'numpy')
                df_lambda = sp.lambdify(x, df_expr, 'numpy')
                raiz = newton_raphson(f_lambda, df_lambda, x0, tol, max_iter)
                self.graficar_funcion(f_lambda, x0-5, x0+5, raiz, f_str)
                messagebox.showinfo("Resultado", f"Raíz ≈ {raiz}")
            elif metodo == "Gauss-Seidel":
                A = np.array([list(map(float, row.split(','))) for row in self.entries["A"].get().split(';')])
                bvec = np.array(list(map(float, self.entries["bvec"].get().split(','))))
                x0gs = np.array(list(map(float, self.entries["x0gs"].get().split(','))))
                tol = float(self.entries["tol"].get())
                max_iter = int(self.entries["iter"].get())
                sol, historia = gauss_seidel(A, bvec, x0gs, tol, max_iter)
                self.graficar_gauss_seidel(historia)
                messagebox.showinfo("Resultado", f"Solución: {sol}")
            elif metodo == "Interpolación":
                x = np.array(list(map(float, self.entries["xinterp"].get().split(','))))
                y = np.array(list(map(float, self.entries["yinterp"].get().split(','))))
                x_new = np.linspace(min(x), max(x), 100)
                y_new, formula = interpolacion(x, y, x_new)
                plt.figure()
                plt.plot(x, y, 'o', label='Puntos')
                plt.plot(x_new, y_new, label='Interpolación')
                plt.title('Interpolación de Lagrange')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.legend()
                plt.grid(True)
                plt.show()
                messagebox.showinfo("Polinomio de Lagrange", f"P(x) = {formula}")
            elif metodo == "Runge-Kutta":
                f_str = self.entries["frk"].get()
                f = lambda t, y: eval(f_str)
                t0 = float(self.entries["t0"].get())
                y0 = float(self.entries["y0"].get())
                h = float(self.entries["h"].get())
                n = int(self.entries["n"].get())
                t, y = runge_kutta(f, t0, y0, h, n)
                plt.figure()
                plt.plot(t, y, marker='o')
                plt.title('Runge-Kutta')
                plt.xlabel('t')
                plt.ylabel('y')
                plt.grid(True)
                plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def graficar_funcion(self, f, a, b, raiz, f_str):
        """Grafica la función y resalta la raíz encontrada."""
        x = np.linspace(a-2, b+2, 400)
        y = np.vectorize(f)(x)
        plt.figure()
        plt.plot(x, y, label=f"f(x) = {f_str}")
        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(0, color='gray', linestyle='--')
        plt.scatter([raiz], [f(raiz)], color='red', label=f"Raíz ≈ {raiz:.5f}")
        plt.title("Gráfica de la función")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def graficar_gauss_seidel(self, historia):
        """Grafica la convergencia de las variables en Gauss-Seidel."""
        plt.figure()
        for i in range(historia.shape[1]):
            plt.plot(historia[:, i], marker='o', label=f"x_{i+1}")
        plt.xlabel('Iteración')
        plt.ylabel('Valor de x')
        plt.title('Convergencia de Gauss-Seidel')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Inicia la aplicación
    root = tk.Tk()
    app = RaicesApp(root)
    root.mainloop() 