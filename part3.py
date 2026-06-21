import numpy as np
import matplotlib.pyplot as plt


file_name = 'covid-19-USA.txt'
data = np.loadtxt(file_name)

y_obs = data / 1_000_000.0
x = np.arange(1, len(y_obs) + 1)



def model(theta, x):
    return theta[0] * np.exp(-theta[1] * (x - theta[2])**2)

def objective(theta, x, y_obs):
    return 0.5 * np.sum((model(theta, x) - y_obs)**2)

def jacobian(theta, x):
    J = np.zeros((len(x), 3))
    f_val = model(theta, x)
    
    J[:, 0] = f_val / theta[0]
    J[:, 1] = -f_val * (x - theta[2])**2
    J[:, 2] = 2 * theta[1] * (x - theta[2]) * f_val
    return J

def gradient(theta, x, y_obs):
    J = jacobian(theta, x)
    return J.T @ (model(theta, x) - y_obs)

#==================================================================================
def scaled_model(theta, x):
    return theta[0] * np.exp(-0.001 * theta[1] * (x - 110 * theta[2])**2)

def scaled_objective(theta, x, y_obs):
    return 0.5 * np.sum((scaled_model(theta, x) - y_obs)**2)

def scaled_jacobian(theta, x):
    J = np.zeros((len(x), 3))
    f_val = scaled_model(theta, x)
    J[:, 0] = f_val / theta[0]
    J[:, 1] = f_val * (-0.001 * (x - 110 * theta[2])**2)
    J[:, 2] = f_val * (0.001 * 2 * theta[1] * 110 * (x - 110 * theta[2]))
    return J

def scaled_gradient(theta, x, y_obs):
    J = scaled_jacobian(theta, x)
    return J.T @ (scaled_model(theta, x) - y_obs)
#==================================================================================
def armijo_linesearch(obj_func, theta, d, grad, x, y_obs, alpha0=1.0, beta=0.5, c=1e-4, max_iter=20):
    alpha = alpha0
    for _ in range(max_iter):
        if obj_func(theta + alpha * d, x, y_obs) <= obj_func(theta, x, y_obs) + c * alpha * np.dot(grad, d):
            break
        alpha *= beta
    return alpha


def steepest_descent(theta0, x, y_obs, obj_func, grad_func, max_iters=500, tol=1e-3, alpha0=0.1):
    theta = np.array(theta0, dtype=float)
    history_F = [obj_func(theta, x, y_obs)]
    history_grad_norm = [np.linalg.norm(grad_func(theta, x, y_obs))]
    
    for _ in range(max_iters):
        grad = grad_func(theta, x, y_obs)
        
        if np.linalg.norm(grad) < tol:
            break
            
        d = -grad
        alpha = armijo_linesearch(obj_func, theta, d, grad, x, y_obs, alpha0=alpha0)
        theta = theta + alpha * d
        
        history_F.append(obj_func(theta, x, y_obs))
        history_grad_norm.append(np.linalg.norm(grad_func(theta, x, y_obs)))
        
    return theta, np.array(history_F), np.array(history_grad_norm)


def levenberg_marquardt(theta0, x, y_obs, obj_func, grad_func, jac_func, max_iters=500, tol=1e-3, mu=1e-4):
    theta = np.array(theta0, dtype=float)
    history_F = [obj_func(theta, x, y_obs)]
    history_grad_norm = [np.linalg.norm(grad_func(theta, x, y_obs))]
    
    for _ in range(max_iters):
        grad = grad_func(theta, x, y_obs)
        
        if np.linalg.norm(grad) < tol:
            break
            
        J = jac_func(theta, x)
        
        A = J.T @ J + mu * np.eye(3)
        b = -grad
        d = np.linalg.solve(A, b)
        
        alpha = armijo_linesearch(obj_func, theta, d, grad, x, y_obs, alpha0=1.0)
        theta = theta + alpha * d
        
        history_F.append(obj_func(theta, x, y_obs))
        history_grad_norm.append(np.linalg.norm(grad_func(theta, x, y_obs)))
        
    return theta, np.array(history_F), np.array(history_grad_norm)

theta_start_unscaled = [1.0, 0.001, 110.0]
theta_start_scaled = [1.0, 1.0, 1.0]

theta_sd, f_sd, g_sd = steepest_descent(theta_start_unscaled, x, y_obs, objective, gradient, alpha0=0.1)
theta_lm, f_lm, g_lm = levenberg_marquardt(theta_start_unscaled, x, y_obs, objective, gradient, jacobian, mu=1e-4)

theta_sd_scaled, f_sd_scaled, g_sd_scaled = steepest_descent(theta_start_scaled, x, y_obs, scaled_objective, scaled_gradient, alpha0=1.0)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(f_sd, label='Steepest Descent')
axs[0].plot(f_lm, label='Levenberg-Marquardt')
axs[0].set_title("Objective Function Convergence")
axs[0].set_xlabel("Iterations")
axs[0].set_ylabel("F(theta)")
axs[0].legend()

axs[1].semilogy(g_sd, label='Steepest Descent')
axs[1].semilogy(g_lm, label='Levenberg-Marquardt')
axs[1].set_title("Gradient Norm Convergence")
axs[1].set_xlabel("Iterations")
axs[1].set_ylabel("||Grad F(theta)|| (Log Scale)")
axs[1].legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.semilogy(g_sd, label='SD Original')
plt.semilogy(g_sd_scaled, label='SD Scaled')
plt.title("Convergence Comparison: Original vs Scaled SD")
plt.xlabel("Iterations")
plt.ylabel("Gradient Norm")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()