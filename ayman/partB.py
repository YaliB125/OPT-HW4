import numpy as np
import matplotlib.pyplot as plt

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

def gradient_descent_graphs(f, grad_f, x0, max_iter, atol):
    sol, gradient_norms, f_values, iterations = gradient_descent(f, grad_f, x0, max_iter, atol)

    plt.semilogy(np.array(iterations), np.array(gradient_norms))
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"$\|\nabla f(x^{(k)})\|_2$")
    plt.title("Convergence of Gradient Norm - GD Method")
    plt.grid(True)
    plt.show()

    plt.plot(np.array(iterations), np.array(f_values))
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"$f(x^{(k)})$")
    plt.yscale("log")
    plt.title("Value of $f$ - GD Method")
    plt.grid(True)
    plt.show()

    return sol

def newton_graphs(f, grad_f, hessian_f, x0, max_iter, atol):
    sol, gradient_norms, f_values, iterations = newton(f, grad_f, hessian_f, x0, max_iter, atol)

    plt.semilogy(np.array(iterations), np.array(gradient_norms))
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"$\|\nabla f(x^{(k)})\|_2$")
    plt.title("Convergence of Gradient Norm - Newton Method")
    plt.grid(True)
    plt.show()

    plt.plot(np.array(iterations), np.array(f_values))
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"$f(x^{(k)})$")
    plt.yscale("log")
    plt.title("Value of $f$ - Newton Method")
    plt.grid(True)
    plt.show()

    return sol

def coordinate_descent_graphs(f, grad_f, hessian_f, x0, max_iter, atol):
    sol, gradient_norms, f_values, iterations = coordinate_descent(f, grad_f, hessian_f, x0, max_iter, atol)

    plt.semilogy(np.array(iterations), np.array(gradient_norms))
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"$\|\nabla f(x^{(k)})\|_2$")
    plt.title("Convergence of Gradient Norm - CD Method")
    plt.grid(True)
    plt.show()

    plt.plot(np.array(iterations), np.array(f_values))
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"$f(x^{(k)})$")
    plt.yscale("log")
    plt.title("Value of $f$ - CD Method")
    plt.grid(True)
    plt.show()

    return sol


def part3():
    B = np.array([
        [3, 1],
        [1, 3]
    ])

    B = 0.5 * B

    def g(x):
        x1, x2 = x
        return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2
    
    def grad_g(x):
        x1, x2 = x

        return np.array([
            4*x1 * (x1**2 + x2 - 11) + 2 * (x1 + x2**2 - 7),
            2 * (x1**2 + x2 - 11) + 4*x2 * (x1 + x2**2 - 7)
        ])
    
    def hessian_g(x):
        x1, x2 = x

        return np.array([
            [12*x1**2 + 4*x2 - 42, 4*x1 + 4*x2],
            [4*x1 + 4*x2, 12*x2**2 + 4*x1 - 26]
        ])

    def f(x):
        return g(B @ x)
    
    def grad_f(x):
        return B.T @ grad_g(B @ x)
    
    def hessian_f(x):
        return B.T @ hessian_g(B @ x) @ B
    
    x0 = np.random.randn(2)
    max_iter = 1000
    atol = 1e-6

    print("Start Point:", np.round(x0, 4))

    """
    the minimum of g(x) is 0 and because B is invertable (det(B) = 0)
    we get that the minimum of f(x) = g(Bx) is also 0.
    so the graph of f(x^(k)) - f_min is just the graph of f(x^(k))
    """

    GD_sol = gradient_descent_graphs(f, grad_f, x0, max_iter, atol)
    Newton_sol = newton_graphs(f, grad_f, hessian_f, x0, max_iter, atol)
    CD_sol = coordinate_descent_graphs(f, grad_f, hessian_f, x0, max_iter, atol)

    print(f"Gradient Descent Solution: {GD_sol}")
    print(f"Newton Solution: {Newton_sol}")
    print(f"Coordinate Descent Solution: {CD_sol}")

"""
first start point:
Start Point: [ 0.1206 -2.1619]
Gradient Descent Solution: [-2.01368628 -1.5175619 ]
Newton Solution: [-2.01368619 -1.51756193]
Coordinate Descent Solution: [-2.01368597 -1.51756203]

second start point:
Start Point: [0.8397 0.97  ]
Gradient Descent Solution: [1.74999881 0.75000178]
Newton Solution: [1.75 0.75]
Coordinate Descent Solution: [1.75000109 0.74999885]

and we see that for different start points the methods go to a different local min
"""

part3()