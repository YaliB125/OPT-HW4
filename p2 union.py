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


def armijo_linesearch(f ,grad_f, x, d, beta, max_iter):
    """
    d is descent direction
    f is the function 
    grand_f is the gradient of f
    x is the current solution
    """

    c = 1e-4
    alpha = 1

    for _ in range(max_iter):
        if f(x + alpha*d) <= f(x) + c* alpha * np.dot(grad_f(x), d):
            return alpha
        
        alpha = beta * alpha

    return alpha


def gradient_descent(f, grad_f, x0, max_iter, epsilon):
    gradient_norm_first_sol = np.linalg.norm(grad_f(x0))

    gradient_norm_solutions = []
    f_values = []
    iterations = []

    x = x0

    for k in range(max_iter):
        iterations.append(k)
        f_values.append(f(x))

        g = grad_f(x)
        d = -g

        gradient_norm_curr_sol = np.linalg.norm(g)         
        gradient_norm_solutions.append(gradient_norm_curr_sol)

        if gradient_norm_curr_sol <= epsilon * (gradient_norm_first_sol + 1e-12):
            """ 
            not doing "if gradinet_norm_curr_sol / gradinet_norm_first_sol < epsilon" 
            to aviod possible divition in zero
            """
            return x, gradient_norm_solutions, f_values, iterations
        
        armijo_alpha = armijo_linesearch(f, grad_f, x, d, beta=0.5, max_iter=20)
        x = x + armijo_alpha * d
    
    return x, gradient_norm_solutions, f_values, iterations

def newton(f, grad_f, hessian_f, x0, max_iter, epsilon):
    gradient_norm_first_sol = np.linalg.norm(grad_f(x0))

    gradient_norm_solutions = []
    f_values = []
    iterations = []

    x = x0

    for k in range(max_iter):
        iterations.append(k)
        f_values.append(f(x))

        g = grad_f(x)
        h = hessian_f(x)

        # d = np.linalg.solve(h, -g)

        d = np.zeros_like(x)

        a = 1e-6

        while True:
            h_mod = h + a * np.eye(h.shape[0])

            try:
                d = np.linalg.solve(h_mod, -g)

                if np.dot(g, d) < 0:
                    break

            except np.linalg.LinAlgError:
                pass

            a *= 10

        gradient_norm_curr_sol = np.linalg.norm(g)         
        gradient_norm_solutions.append(gradient_norm_curr_sol)

        if gradient_norm_curr_sol <= epsilon * (gradient_norm_first_sol + 1e-12):
            """ 
            not doing "if gradinet_norm_curr_sol / gradinet_norm_first_sol < epsilon" 
            to aviod possible divition in zero
            """
            return x, gradient_norm_solutions, f_values, iterations
        
        armijo_alpha = armijo_linesearch(f, grad_f, x, d, beta=0.5, max_iter=20)
        x = x + armijo_alpha * d
    
    return x, gradient_norm_solutions, f_values, iterations


def coordinate_descent(f, grad_f, hessian_f, x0, max_iter, epsilon):
    gradient_norm_first_sol = np.linalg.norm(grad_f(x0))

    gradient_norm_solutions = []
    f_values = []
    iterations = []

    x = x0

    for k in range(max_iter):
        iterations.append(k)
        f_values.append(f(x))

        gradient_norm_curr_sol = np.linalg.norm(grad_f(x)) 
        gradient_norm_solutions.append(gradient_norm_curr_sol)

        if gradient_norm_curr_sol <= epsilon * (gradient_norm_first_sol + 1e-12):
            """ 
            not doing "if gradinet_norm_curr_sol / gradinet_norm_first_sol < epsilon" 
            to aviod possible divition in zero
            """
            return x, gradient_norm_solutions, f_values, iterations

        for i in range(x.shape[0]):
            g = grad_f(x)
            h = hessian_f(x)

            hii = h[i, i]

            d = np.zeros_like(x)

            if hii > 1e-8:
                d[i] = -g[i] / hii
            else:
                d[i] = -g[i]

            armijo_alpha = armijo_linesearch(f, grad_f, x, d, beta=0.5, max_iter=20)

            x[i] = x[i] + armijo_alpha * d[i]
        
    return x, gradient_norm_solutions, f_values, iterations



def plot_comparison_graphs(gd_data, newton_data, cd_data):
    plt.figure(figsize=(10, 5))
    plt.semilogy(gd_data['iters'], gd_data['grads'], label="Gradient Descent", color='blue')
    plt.semilogy(newton_data['iters'], newton_data['grads'], label="Newton's Method", color='orange')
    plt.semilogy(cd_data['iters'], cd_data['grads'], label="Coordinate Descent", color='green')
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"$\|\nabla f(x^{(k)})\|_2$")
    plt.title("Convergence of Gradient Norm - Comparison")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.semilogy(gd_data['iters'], gd_data['f_vals'], label="Gradient Descent", color='blue')
    plt.semilogy(newton_data['iters'], newton_data['f_vals'], label="Newton's Method", color='orange')
    plt.semilogy(cd_data['iters'], cd_data['f_vals'], label="Coordinate Descent", color='green')
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"$f(x^{(k)}) - f_{min}$ (Log Scale)")
    plt.title("Objective Function Value - Comparison")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()


def part3():
    x0 = np.random.randn(2)
    max_iter = 1000
    atol = 1e-6

    print("SECTION 3")
    print("Start Point:", np.round(x0, 4))

    gd_sol, gd_grads, gd_f, gd_iters = gradient_descent(f, grad_f, x0, max_iter, atol)
    nt_sol, nt_grads, nt_f, nt_iters = newton(f, grad_f, hessian_f, x0, max_iter, atol)
    cd_sol, cd_grads, cd_f, cd_iters = coordinate_descent(f, grad_f, hessian_f, x0, max_iter, atol)

    gd_data = {'iters': gd_iters, 'grads': gd_grads, 'f_vals': gd_f}
    newton_data = {'iters': nt_iters, 'grads': nt_grads, 'f_vals': nt_f}
    cd_data = {'iters': cd_iters, 'grads': cd_grads, 'f_vals': cd_f}

    plot_comparison_graphs(gd_data, newton_data, cd_data)

    print(f"Gradient Descent Solution: {np.round(gd_sol, 5)}")
    print(f"Newton Solution:           {np.round(nt_sol, 5)}")
    print(f"Coordinate Descent Solution: {np.round(cd_sol, 5)}")

part3()


p = np.array([3.0, -2.0]) 

def f_reg(x, lmbda):
    return f(x) + lmbda * np.linalg.norm(x - p)**2

def grad_f_reg(x, lmbda):
    return grad_f(x) + 2 * lmbda * (x - p)

def hessian_f_reg(x, lmbda):
    return hessian_f(x) + 2 * lmbda * np.eye(2)
 

print("\n  SECTION 4 ")
x_start_far = np.array([-5.0, -5.0]) 
lambdas = [10, 50, 100, 500, 1000]
max_iter = 1000
atol = 1e-6

print(f"Landing far away at: {x_start_far}")
print(f"Target point (p):    {p}\n")

for lmbda in lambdas:
    def f_curr(x): return f_reg(x, lmbda)
    def grad_curr(x): return grad_f_reg(x, lmbda)
    def hess_curr(x): return hessian_f_reg(x, lmbda)
    

    x_res, _, _, _ = newton(f_curr, grad_curr, hess_curr, x_start_far, max_iter, atol)
    
    dist = np.linalg.norm(x_res - p)
    print(f"Lambda = {lmbda:4d} | Found Min at: [{x_res[0]:8.5f}, {x_res[1]:8.5f}] | Distance from p: {dist:.5f}")


print("\n SECTION 5")
lmbda_large = 100

def f_curr_s1(x): return f_reg(x, lmbda_large)
def grad_curr_s1(x): return grad_f_reg(x, lmbda_large)
def hess_curr_s1(x): return hessian_f_reg(x, lmbda_large)

x_stage1, _, _, _ = newton(f_curr_s1, grad_curr_s1, hess_curr_s1, x_start_far, max_iter, atol)
print(f"Stage 1 (Regularization): Reached {x_stage1.round(5)}")
x_final, _, _, _ = newton(f, grad_f, hessian_f, x_stage1, max_iter, atol)
print(f"Stage 2 (Exact Minimum):  Reached {x_final.round(5)}")

