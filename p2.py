import numpy as np
import matplotlib.pyplot as plt

B = 0.5 * np.array([[3.0, 1.0], 
                    [1.0, 3.0]])

def g(y):
    y1, y2 = y[0], y[1]
    return (y1**2 + y2 - 11)**2 + (y1 + y2**2 - 7)**2

def grad_g(y):
    y1, y2 = y[0], y[1]
    df_dy1 = 4 * y1 * (y1**2 + y2 - 11) + 2 * (y1 + y2**2 - 7)
    df_dy2 = 2 * (y1**2 + y2 - 11) + 4 * y2 * (y1 + y2**2 - 7)
    return np.array([df_dy1, df_dy2])

def hess_g(y):
    y1, y2 = y[0], y[1]
    d2f_dy1_dy1 = 12 * y1**2 + 4 * y2 - 42
    d2f_dy2_dy2 = 12 * y2**2 + 4 * y1 - 26
    d2f_dy1_dy2 = 4 * y1 + 4 * y2
    return np.array([[d2f_dy1_dy1, d2f_dy1_dy2],
                     [d2f_dy1_dy2, d2f_dy2_dy2]])

def f(x):
    return g(B @ x)

def grad_f(x):
    return B.T @ grad_g(B @ x)

def hess_f(x):
    return B.T @ hess_g(B @ x) @ B


def armijo_linesearch(f, x, d, gk, alpha0=1.0, beta=0.5, c=1e-4):
    alpha = alpha0
    f_x = f(x)
    directional_derivative = np.dot(gk, d) 
    
    for _ in range(100): 
        x_temp = x + alpha * d
        if f(x_temp) <= f_x + c * alpha * directional_derivative:
            return alpha
        alpha *= beta
        
    return alpha

def make_positive_definite(H, tau=1e-5):
    evals = np.linalg.eigvals(H)
    min_eval = np.min(evals)
    if min_eval <= 0:
        H = H + (abs(min_eval) + tau) * np.eye(H.shape[0])
    return H


def gradient_descent(f, grad_f, x_init, max_iter=500, tol=1e-3):
    x = np.array(x_init, dtype=float)
    history_f = []
    history_grad_norm = []
    
    for k in range(max_iter):
        gk = grad_f(x)
        norm_gk = np.linalg.norm(gk)
        
        history_f.append(f(x))
        history_grad_norm.append(norm_gk)
        
        if norm_gk < tol:
            break
            
        d = -gk
        alpha = armijo_linesearch(f, x, d, gk, alpha0=0.5)
        x = x + alpha * d
        
    return x, history_f, history_grad_norm

def newtons_method(f, grad_f, hess_f, x_init, max_iter=500, tol=1e-3):
    x = np.array(x_init, dtype=float)
    history_f = []
    history_grad_norm = []
    
    for k in range(max_iter):
        gk = grad_f(x)
        norm_gk = np.linalg.norm(gk)
        
        history_f.append(f(x))
        history_grad_norm.append(norm_gk)
        
        if norm_gk < tol:
            break
            
        Hk = hess_f(x)
        Hk_pd = make_positive_definite(Hk)
        d = np.linalg.solve(Hk_pd, -gk)
        alpha = armijo_linesearch(f, x, d, gk, alpha0=1.0)
        x = x + alpha * d
        
    return x, history_f, history_grad_norm

def coordinate_descent(f, grad_f, hess_f, x_init, max_iter=500, tol=1e-3):
    x = np.array(x_init, dtype=float)
    history_f = []
    history_grad_norm = []
    n_dims = len(x)
    
    for k in range(max_iter):
        gk_full = grad_f(x)
        norm_gk = np.linalg.norm(gk_full)
        
        history_f.append(f(x))
        history_grad_norm.append(norm_gk)
        
        if norm_gk < tol:
            break
            
        for i in range(n_dims):
            g_curr = grad_f(x)
            H_curr = hess_f(x)
            
            g_i = g_curr[i]
            H_ii = H_curr[i, i]
            
            if H_ii <= 0:
                H_ii = abs(H_ii) + 1e-5
                
            d_i = -g_i / H_ii
            
            d_full = np.zeros(n_dims)
            d_full[i] = d_i
            
            alpha = armijo_linesearch(f, x, d_full, g_curr, alpha0=1.0)
            x[i] = x[i] + alpha * d_i
            
    return x, history_f, history_grad_norm


print("Testing different initializations (Newton's Method):")
initial_points = [[4.0, 4.0], [-4.0, 4.0], [-4.0, -4.0], [4.0, -4.0]]

for x0 in initial_points:
    x_opt, _, _ = newtons_method(f, grad_f, hess_f, x0)
    print(f"Start: {x0} ---> Converged to minimum at: [{x_opt[0]:.3f}, {x_opt[1]:.3f}]")

print("-" * 50)

np.random.seed(0) 
x_start = np.random.randn(2)

x_gd, history_f_gd, history_g_gd = gradient_descent(f, grad_f, x_start)
x_nw, history_f_nw, history_g_nw = newtons_method(f, grad_f, hess_f, x_start)
x_cd, history_f_cd, history_g_cd = coordinate_descent(f, grad_f, hess_f, x_start)

f_min = 0.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.semilogy([val - f_min for val in history_f_gd], label='Gradient Descent', color='blue')
ax1.semilogy([val - f_min for val in history_f_nw], label="Newton's Method", color='red')
ax1.semilogy([val - f_min for val in history_f_cd], label='Coordinate Descent', color='green')
ax1.set_title('Distance from Optimum ($f(x^{(k)}) - f_{min}$)')
ax1.set_xlabel('Iteration ($k$)')
ax1.set_ylabel('Objective Value Error (Log Scale)')
ax1.legend()
ax1.grid(True, which="both", ls="--")

ax2.semilogy(history_g_gd, label='Gradient Descent', color='blue')
ax2.semilogy(history_g_nw, label="Newton's Method", color='red')
ax2.semilogy(history_g_cd, label='Coordinate Descent', color='green')
ax2.set_title('Gradient Norm ($||\\nabla f(x^{(k)})||$)')
ax2.set_xlabel('Iteration ($k$)')
ax2.set_ylabel('Norm (Log Scale)')
ax2.legend()
ax2.grid(True, which="both", ls="--")

plt.tight_layout()
plt.show()

p = np.array([3.0, -2.0])

# ==========================================
# הגדרת הפונקציות החדשות עם הרגולריזציה
# ==========================================
def h(x, lam):
    return f(x) + lam * np.linalg.norm(x - p)**2

def grad_h(x, lam):
    return grad_f(x) + 2 * lam * (x - p)

def hess_h(x, lam):
    # הנגזרת השנייה של איבר הרגולריזציה היא 2*lambda כפול מטריצת היחידה
    return hess_f(x) + 2 * lam * np.eye(len(x))


print("\nTesting Regularization with different Lambda values:")
np.random.seed(0) 
x_start = np.random.randn(2)
for lam in [10, 50, 100]:
    print(f"\n--- Lambda = {lam} ---")
    
    # Gradient Descent
    x_gd, _, _ = gradient_descent(lambda x: h(x, lam), lambda x: grad_h(x, lam), x_start)
    print(f"Gradient Descent:   Converged to [{x_gd[0]:.4f}, {x_gd[1]:.4f}]")
    
    # Newton's Method
    x_nw, _, _ = newtons_method(lambda x: h(x, lam), lambda x: grad_h(x, lam), lambda x: hess_h(x, lam), x_start)
    print(f"Newton's Method:    Converged to [{x_nw[0]:.4f}, {x_nw[1]:.4f}]")
    
    # Coordinate Descent
    x_cd, _, _ = coordinate_descent(lambda x: h(x, lam), lambda x: grad_h(x, lam), lambda x: hess_h(x, lam), x_start)
    print(f"Coordinate Descent: Converged to [{x_cd[0]:.4f}, {x_cd[1]:.4f}]")