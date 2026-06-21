import numpy as np
import matplotlib.pyplot as plt

def armijo_linesearch(f ,grad_f, x, d, alpha0, beta, max_iter):
    """
    d is descent direction
    f is the function 
    grand_f is the gradient of f
    x is the current solution
    """

    c = 1e-4
    alpha = alpha0

    fx = f(x)
    gx = grad_f(x)

    for _ in range(max_iter):
        if f(x + alpha*d) <= fx + c* alpha * np.dot(gx, d):
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
        
        armijo_alpha = armijo_linesearch(f, grad_f, x, d, alpha0=0.5, beta=0.5, max_iter=20)
        x = x + armijo_alpha * d
    
    return x, gradient_norm_solutions, f_values, iterations

def gauss_newton(f, grad_f, JTJ, x0, max_iter, epsilon, regularization=0):
    """
    regularization is used here to have Levenberg-Marquardt,
    and if regularization=0 so we have normal Gauss Newton
    """
    gradient_norm_first_sol = np.linalg.norm(grad_f(x0))

    gradient_norm_solutions = []
    f_values = []
    iterations = []

    x = x0
    gama = regularization

    for k in range(max_iter):
        iterations.append(k)
        f_values.append(f(x))

        g = grad_f(x)
        JTJ_x = JTJ(x)

        d = np.linalg.solve(JTJ_x + gama*np.eye(JTJ_x.shape[1]), -g)

        gradient_norm_curr_sol = np.linalg.norm(g)         
        gradient_norm_solutions.append(gradient_norm_curr_sol)

        if gradient_norm_curr_sol <= epsilon * (gradient_norm_first_sol + 1e-12):
            """ 
            not doing "if gradinet_norm_curr_sol / gradinet_norm_first_sol < epsilon" 
            to aviod possible divition in zero
            """
            return x, gradient_norm_solutions, f_values, iterations
        
        armijo_alpha = armijo_linesearch(f, grad_f, x, d, alpha0=1, beta=0.5, max_iter=20)
        x = x + armijo_alpha * d
    
    return x, gradient_norm_solutions, f_values, iterations

def gradient_descent_graphs(f, grad_f, x0, max_iter, atol):
    sol, gradient_norms, f_values, iterations = gradient_descent(f, grad_f, x0, max_iter, atol)

    plt.semilogy(np.array(iterations), np.array(gradient_norms))
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"$\|\nabla F(x^{(k)})\|_2$")
    plt.title("Convergence of Gradient Norm - GD Method")
    plt.grid(True)
    plt.show()

    plt.plot(np.array(iterations), np.array(f_values))
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"$F(x^{(k)})$")
    plt.yscale("log")
    plt.title("Value of $F$ - GD Method")
    plt.grid(True)
    plt.show()

    return sol

def gauss_newton_graphs(f, grad_f, JTJ, x0, max_iter, atol, regularization=0):
    sol, gradient_norms, f_values, iterations = gauss_newton(f, grad_f, JTJ, x0, max_iter, atol, regularization)

    plt.semilogy(np.array(iterations), np.array(gradient_norms))
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"$\|\nabla F(x^{(k)})\|_2$")
    plt.title("Convergence of Gradient Norm - Gauss-Newton/Levenberg-Marquardt  Method")
    plt.grid(True)
    plt.show()

    plt.plot(np.array(iterations), np.array(f_values))
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"$F(x^{(k)})$")
    plt.yscale("log")
    plt.title("Value of $f$ - Gauss-Newton/Levenberg-Marquardt Method")
    plt.grid(True)
    plt.show()

    return sol

def part5():
    y_observe = []

    with open("Covid-19-USA.txt", "r") as file:
        for line in file:
            y_observe.append(float(line.strip()))
        
    y_observe = np.array(y_observe) / 1e6

    def gaussian_model(theta1, theta2, theta3, x):
        return theta1 * np.exp(-theta2 * (x - theta3)**2)

    def f(theta):
        theta1, theta2, theta3 = theta
        x = np.arange(1, 100)

        return gaussian_model(theta1, theta2, theta3, x)

    def F(theta):
        r = f(theta) - y_observe

        return 0.5 * np.dot(r, r)

    def grad_fi(theta1, theta2, theta3, x):
        return np.array(
            [
                np.exp(-theta2 * (x - theta3)**2),
                -theta1 * (x - theta3)**2 * np.exp(-theta2 * (x - theta3)**2),
                2 * theta1 * theta2 * (x - theta3) * np.exp(-theta2 * (x - theta3)**2)
            ]
        )

    def jacobian_f(theta):
        theta1, theta2, theta3 = theta
        x = np.arange(1, 100)

        J = grad_fi(theta1, theta2, theta3, x)

        return J.T

    def grad_F(theta):
        return jacobian_f(theta).T @ (f(theta) - y_observe)

    def JTJ(theta):
        J = jacobian_f(theta)

        return J.T @ J

    max_iter = 500
    atol = 1e-3    
    theta0 = np.array([1, 0.001, 110])
    levenberg_marquardt_regularization = 1e-4

    GD_sol = gradient_descent_graphs(F, grad_F, theta0, max_iter, atol)
    Gauss_Newton_sol = gauss_newton_graphs(F, grad_F, JTJ, theta0, max_iter, atol, levenberg_marquardt_regularization)

    print(f"GD_sol: {GD_sol}")
    print(f"Gauss_Newton_sol: {Gauss_Newton_sol}")

def part6():
    y_observe = []

    with open("Covid-19-USA.txt", "r") as file:
        for line in file:
            y_observe.append(float(line.strip()))
        
    y_observe = np.array(y_observe) / 1e6

    def gaussian_model(theta1, theta2, theta3, x):
        return theta1 * np.exp(-0.001*theta2 * (x - 110*theta3)**2)

    def f(theta):
        theta1, theta2, theta3 = theta
        x = np.arange(1, 100)

        return gaussian_model(theta1, theta2, theta3, x)

    def F(theta):
        r = f(theta) - y_observe

        return 0.5 * np.dot(r, r)

    def grad_fi(theta1, theta2, theta3, x):
        return np.array(
            [
                np.exp(-0.001*theta2 * (x - 110*theta3)**2),
                -0.001 * theta1 * (x - 110*theta3)**2 * np.exp(-0.001*theta2 * (x - 110*theta3)**2),
                2 * 110 * theta1 * 0.001*theta2 * (x - 110*theta3) * np.exp(-0.001*theta2 * (x - 110*theta3)**2)
            ]
        )

    def jacobian_f(theta):
        theta1, theta2, theta3 = theta
        x = np.arange(1, 100)

        J = grad_fi(theta1, theta2, theta3, x)

        return J.T

    def grad_F(theta):
        return jacobian_f(theta).T @ (f(theta) - y_observe)

    def JTJ(theta):
        J = jacobian_f(theta)

        return J.T @ J

    max_iter = 500
    atol = 1e-3    
    theta0 = np.array([1, 1, 1])
    levenberg_marquardt_regularization = 1e-4

    GD_sol = gradient_descent_graphs(F, grad_F, theta0, max_iter, atol)
    Gauss_Newton_sol = gauss_newton_graphs(F, grad_F, JTJ, theta0, max_iter, atol, levenberg_marquardt_regularization)

    print(f"GD_sol: {GD_sol}")
    print(f"Gauss_Newton_sol: {Gauss_Newton_sol}")

def compare_methods(histories, title, ylabel):
    plt.figure()

    for label, iterations, values in histories:
        plt.semilogy(iterations, values, label=label)

    plt.xlabel(r"Iteration $k$")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    y_observe = []

    with open("Covid-19-USA.txt", "r") as file:
        for line in file:
            y_observe.append(float(line.strip()))
        
    y_observe = np.array(y_observe) / 1e6

    def gaussian_model_original(theta1, theta2, theta3, x):
        return theta1 * np.exp(-theta2 * (x - theta3)**2)

    def f_original(theta):
        theta1, theta2, theta3 = theta
        x = np.arange(1, 100)

        return gaussian_model_original(theta1, theta2, theta3, x)

    def F_original(theta):
        r = f_original(theta) - y_observe

        return 0.5 * np.dot(r, r)

    def grad_fi_original(theta1, theta2, theta3, x):
        return np.array(
            [
                np.exp(-theta2 * (x - theta3)**2),
                -theta1 * (x - theta3)**2 * np.exp(-theta2 * (x - theta3)**2),
                2 * theta1 * theta2 * (x - theta3) * np.exp(-theta2 * (x - theta3)**2)
            ]
        )

    def jacobian_f_original(theta):
        theta1, theta2, theta3 = theta
        x = np.arange(1, 100)

        J = grad_fi_original(theta1, theta2, theta3, x)

        return J.T

    def grad_F_original(theta):
        return jacobian_f_original(theta).T @ (f_original(theta) - y_observe)

    def JTJ_original(theta):
        J = jacobian_f_original(theta)

        return J.T @ J

    def gaussian_model_scaled(theta1, theta2, theta3, x):
        return theta1 * np.exp(-0.001*theta2 * (x - 110*theta3)**2)

    def f_scaled(theta):
        theta1, theta2, theta3 = theta
        x = np.arange(1, 100)

        return gaussian_model_scaled(theta1, theta2, theta3, x)

    def F_scaled(theta):
        r = f_scaled(theta) - y_observe

        return 0.5 * np.dot(r, r)

    def grad_fi_scaled(theta1, theta2, theta3, x):
        return np.array(
            [
                np.exp(-0.001*theta2 * (x - 110*theta3)**2),
                -0.001 * theta1 * (x - 110*theta3)**2 * np.exp(-0.001*theta2 * (x - 110*theta3)**2),
                2 * 110 * theta1 * 0.001*theta2 * (x - 110*theta3) * np.exp(-0.001*theta2 * (x - 110*theta3)**2)
            ]
        )

    def jacobian_f_scaled(theta):
        theta1, theta2, theta3 = theta
        x = np.arange(1, 100)

        J = grad_fi_scaled(theta1, theta2, theta3, x)

        return J.T

    def grad_F_scaled(theta):
        return jacobian_f_scaled(theta).T @ (f_scaled(theta) - y_observe)

    def JTJ_scaled(theta):
        J = jacobian_f_scaled(theta)

        return J.T @ J


    max_iter = 500
    atol = 1e-3    
    theta0_original = np.array([1, 0.001, 110])
    theta0_scaled = np.array([1, 1, 1])
    levenberg_marquardt_regularization = 1e-4

    _, GD_gradient_norm_original, GD_F_value_original, GD_iterations_original = gradient_descent(F_original, grad_F_original, theta0_original, max_iter, atol)
    _, GD_gradient_norm_scaled, GD_F_value_scaled, GD_iterations_scaled = gradient_descent(F_scaled, grad_F_scaled, theta0_scaled, max_iter, atol)
    _, GN_gradient_norm_original, GN_F_value_original, GN_iterations_original = gauss_newton(F_original, grad_F_original, JTJ_original, theta0_original, max_iter, atol, levenberg_marquardt_regularization)
    _, GN_gradient_norm_scaled, GN_F_value_scaled, GN_iterations_scaled = gauss_newton(F_scaled, grad_F_scaled, JTJ_scaled, theta0_scaled, max_iter, atol, levenberg_marquardt_regularization)

    compare_methods(
        [
            ("GD", GD_iterations_original, GD_gradient_norm_original),
            ("GN", GN_iterations_original, GN_gradient_norm_original)
        ],
        "Original Model: Convergence Of Gradient Norm",
        r"$\|\nabla F(x^{(k)})\|_2$"
    )

    compare_methods(
        [
            ("GD", GD_iterations_original, GD_F_value_original),
            ("GN", GN_iterations_original, GN_F_value_original)
        ],
        "Original Model: F Value",
        r"$F(x^{(k)})$"
    )

    compare_methods(
        [
            ("GD", GD_iterations_scaled, GD_gradient_norm_scaled),
            ("GN", GN_iterations_scaled, GN_gradient_norm_scaled)
        ],
        "Scaled Model: Convergence Of Gradient Norm",
        r"$\|\nabla F(x^{(k)})\|_2$"
    )

    compare_methods(
        [
            ("GD", GD_iterations_scaled, GD_F_value_scaled),
            ("GN", GN_iterations_scaled, GN_F_value_scaled)
        ],
        "Scaled Model: F Value",
        r"$F(x^{(k)})$"
    )

    compare_methods(
        [
            ("GD original", GD_iterations_original, GD_gradient_norm_original),
            ("GD scaled", GD_iterations_scaled, GD_gradient_norm_scaled)
        ],
        "GD - Gradient Norm: Original vs Scaled",
        r"$\|\nabla F(x^{(k)})\|_2$"
    )

    compare_methods(
        [
            ("GD original", GD_iterations_original, GD_F_value_original),
            ("GD scaled", GD_iterations_scaled, GD_F_value_scaled)
        ],
        "GD - F Value: Original vs Scaled",
        r"$F(x^{(k)})$"
    )

    compare_methods(
        [
            ("GN original", GN_iterations_original, GN_gradient_norm_original),
            ("GN scaled", GN_iterations_scaled, GN_gradient_norm_scaled)
        ],
        "GN - Gradient Norm: Original vs Scaled",
        r"$\|\nabla F(x^{(k)})\|_2$"
    )

    compare_methods(
        [
            ("GN original", GN_iterations_original, GN_F_value_original),
            ("GN scaled", GN_iterations_scaled, GN_F_value_scaled)
        ],
        "GN - F Value: Original vs Scaled",
        r"$F(x^{(k)})$"
    )


main()
