from flask import Flask, request, render_template, send_file
import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Define a function to run PINN model
def run_pinn(E, L, P, epochs):
    # Define the PDE
    def pde(x, u):
        d2u_dx2 = dde.grad.hessian(u, x)
        return E * d2u_dx2

    # Boundary conditions
    def bc_left(x, u):
        return u

    def bc_right(x, u):
        du_dx = dde.grad.jacobian(u, x)
        return E * du_dx - P

    # Geometry
    geom = dde.geometry.Interval(0, L)

    # Neural network
    net = dde.maps.FNN([1] + [20] * 3 + [1], "tanh", "Glorot uniform")

    # PDE problem
    data = dde.data.PDE(
        geom,
        pde,
        [dde.icbc.DirichletBC(geom, bc_left, lambda x: x == 0),
         dde.icbc.OperatorBC(geom, bc_right, lambda x: x == L)],
        num_domain=20,
        num_boundary=2
    )

    # Train the model
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    model.train(epochs=int(epochs))

    # Generate test points
    x_test = np.linspace(0, L, 100).reshape(-1, 1)
    u_pred = model.predict(x_test)  # Displacement

    # Compute strain (du/dx)
    du_dx_pred = dde.grad.jacobian(
        tf.convert_to_tensor(u_pred, dtype=tf.float32),
        tf.convert_to_tensor(x_test, dtype=tf.float32)
    ).numpy()

    # Compute stress (σ = E * strain)
    stress_pred = E * du_dx_pred

    # Plot and save results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(x_test, u_pred, label="Displacement u(x)", color='b')
    plt.xlabel("Length x (m)")
    plt.ylabel("Displacement (m)")
    plt.title("Displacement vs. Length")
    plt.grid()
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(x_test, du_dx_pred, label="Strain ε(x)", color='g')
    plt.xlabel("Length x (m)")
    plt.ylabel("Strain")
    plt.title("Strain vs. Length")
    plt.grid()
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(x_test, stress_pred, label="Stress σ(x)", color='r')
    plt.xlabel("Length x (m)")
    plt.ylabel("Stress (Pa)")
    plt.title("Stress vs. Length")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    image_path = "static/results.png"
    plt.savefig(image_path)
    plt.close()
    
    return image_path

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        E = float(request.form["E"])
        L = float(request.form["L"])
        P = float(request.form["P"])
        epochs = int(request.form["epochs"])
        
        image_path = run_pinn(E, L, P, epochs)
        return render_template("index.html", image_path=image_path)
    
    return render_template("index.html", image_path=None)

if __name__ == "__main__":
    app.run(debug=True)
