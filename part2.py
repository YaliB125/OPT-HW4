import numpy as np
import matplotlib.pyplot as plt

B = 0.5 * np.array([[3, 1], 
                    [1, 3]])

def g(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def grad_g(x):
    df_dx1 = 4 * x[0] * (x[0]**2 + x[1] - 11) + 2 * (x[0] + x[1]**2 - 7)
    df_dx2 = 2 * (x[0]**2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1]**2 - 7)
    return np.array([df_dx1, df_dx2])

def hessian_g(x):
    d2f_dx1dx1 = 12 * x[0]**2 + 4 * x[1] - 42
    d2f_dx2dx2 = 4 * x[0] + 12 * x[1]**2 - 26
    d2f_dx1dx2 = 4 * x[0] + 4 * x[1]
    return np.array([[d2f_dx1dx1, d2f_dx1dx2], 
                     [d2f_dx1dx2, d2f_dx2dx2]])

def f(x):
    return g(B @ x)

def grad_f(x):
    return B.T @ grad_g(B @ x)

def hessian_f(x):
    return B.T @ hessian_g(B @ x) @ B


def armijo_linesearch(func, x, d, grad, alpha0=1.0, beta=0.5, c=1e-4, max_iter=20):
    alpha = alpha0
    for _ in range(max_iter):
        if func(x + alpha * d) <= func(x) + c * alpha * np.dot(grad, d):
            break
        alpha *= beta
    return alpha


def gradient_descent(x0, max_iters=200, tol=1e-6):
    x = np.array(x0, dtype=float)
    history_x = [x.copy()]
    history_f = [f(x)]
    history_grad_norm = [np.linalg.norm(grad_f(x))]
    
    for _ in range(max_iters):
        grad = grad_f(x)
        if np.linalg.norm(grad) < tol:
            break
        d = -grad
        alpha = armijo_linesearch(f, x, d, grad)
        x = x + alpha * d
        
        history_x.append(x.copy())
        history_f.append(f(x))
        history_grad_norm.append(np.linalg.norm(grad_f(x)))
        
    return x, np.array(history_f), np.array(history_grad_norm)


def newtons_method(x0, max_iters=200, tol=1e-6):
    x = np.array(x0, dtype=float)
    history_x = [x.copy()]
    history_f = [f(x)]
    history_grad_norm = [np.linalg.norm(grad_f(x))]
    
    for _ in range(max_iters):
        grad = grad_f(x)
        if np.linalg.norm(grad) < tol:
            break
            
        H = hessian_f(x)
        mu = max(0, abs(H[0, 1]) - H[0, 0] + 1e-3, abs(H[1, 0]) - H[1, 1] + 1e-3)
        H_sdd = H + mu * np.eye(H.shape[0])
        d = np.linalg.solve(H_sdd, -grad)
        
        alpha = armijo_linesearch(f, x, d, grad)
        x = x + alpha * d
        
        history_x.append(x.copy())
        history_f.append(f(x))
        history_grad_norm.append(np.linalg.norm(grad_f(x)))
        
    return x, np.array(history_f), np.array(history_grad_norm)


def coordinate_descent(x0, max_iters=200, tol=1e-6):
    x = np.array(x0, dtype=float)
    history_x = [x.copy()]
    history_f = [f(x)]
    history_grad_norm = [np.linalg.norm(grad_f(x))]
    
    for _ in range(max_iters):
        x_prev = x.copy()
        
        for i in range(2):
            grad = grad_f(x)
            H = hessian_f(x)
            
            d = np.zeros(2)
            hessian_val = max(abs(H[i, i]), 1e-5) 
            d[i] = -grad[i] / hessian_val
            
            alpha = armijo_linesearch(f, x, d, grad)
            x = x + alpha * d
            
        history_x.append(x.copy())
        history_f.append(f(x))
        history_grad_norm.append(np.linalg.norm(grad_f(x)))
        
        if np.linalg.norm(x - x_prev) < tol:
            break
            
    return x, np.array(history_f), np.array(history_grad_norm)




print(" SECTION 3 ")
x0 = np.random.randn(2)
print(f"Random Landing Point: {x0.round(4)}")

x_gd, f_gd, grad_gd = gradient_descent(x0)
x_nt, f_nt, grad_nt = newtons_method(x0)
x_cd, f_cd, grad_cd = coordinate_descent(x0)

print(f"Reached Minimum (Newton): {x_nt.round(4)}")

f_min = min(f_gd[-1], f_nt[-1], f_cd[-1]) 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.semilogy(f_gd - f_min + 1e-12, label='GD'); ax1.semilogy(f_nt - f_min + 1e-12, label='Newton'); ax1.semilogy(f_cd - f_min + 1e-12, label='CD')
ax1.set_title("Distance from Minimum"); ax1.legend()
ax2.semilogy(grad_gd, label='GD'); ax2.semilogy(grad_nt, label='Newton'); ax2.semilogy(grad_cd, label='CD')
ax2.set_title("Norm of Gradient"); ax2.legend()
plt.show()



p = np.array([3.0, -2.0]) 

def f_reg(x, lmbda):
    return f(x) + lmbda * np.linalg.norm(x - p)**2

def grad_f_reg(x, lmbda):
    return grad_f(x) + 2 * lmbda * (x - p)

def hessian_f_reg(x, lmbda):
    return hessian_f(x) + 2 * lmbda * np.eye(2)
 

def newtons_method_custom(x0, lmbda, max_iters=200, tol=1e-6):
    x = np.array(x0, dtype=float)
    
    for _ in range(max_iters):
        grad = grad_f_reg(x, lmbda)
        if np.linalg.norm(grad) < tol:
            break
            
        H = hessian_f_reg(x, lmbda)
        mu = max(0, abs(H[0, 1]) - H[0, 0] + 1e-3, abs(H[1, 0]) - H[1, 1] + 1e-3)
        H_sdd = H + mu * np.eye(H.shape[0])
        d = np.linalg.solve(H_sdd, -grad)
        
        func_for_search = lambda val: f_reg(val, lmbda)
        alpha = armijo_linesearch(func_for_search, x, d, grad)
        
        x = x + alpha * d
        
    return x


print("\n  SECTION 4 ")
x_start_far = np.array([-5.0, -5.0]) 
lambdas = [10, 50, 100, 500, 1000]

print(f"Landing far away at: {x_start_far}")
print(f"Target point (p):    {p}\n")

for lmbda in lambdas:
    x_res = newtons_method_custom(x_start_far, lmbda)
    dist = np.linalg.norm(x_res - p)
    print(f"Lambda = {lmbda:4d} | Found Min at: [{x_res[0]:8.5f}, {x_res[1]:8.5f}] | Distance from p: {dist:.5f}")


print("\n SECTION 5")
lmbda_large = 100

x_stage1 = newtons_method_custom(x_start_far, lmbda=lmbda_large)
print(f"Stage 1 (Regularization): Reached {x_stage1.round(5)}")

x_final, _, _ = newtons_method(x_stage1)
print(f"Stage 2 (Exact Minimum):  Reached {x_final.round(5)}")


