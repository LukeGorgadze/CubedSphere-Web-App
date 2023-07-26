import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return theta, phi

def spherical_to_cartesian(theta, phi, r):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def distance_estimator(p1, p2):
    return 2 * np.arcsin(np.sqrt(np.sin((p2[0] - p1[0]) / 2)**2 + np.cos(p1[0]) * np.cos(p2[0]) * np.sin((p2[1] - p1[1]) / 2)**2))

def compute_derivative(f, pointOrigin, direction, n, cubeOrigin):
    distance = float('inf')
    closestPoint = None
    prevPointX = None
    prevPointY = None
    for face in cubeOrigin:
        for i, row in enumerate(face):
            for j, point in enumerate(row):
                lat, longt = cartesian_to_spherical(point[0], point[1], point[2])
                dist = distance_estimator(pointOrigin, (lat, longt))
                if dist < distance:
                    distance = dist
                    closestPoint = point

    distance = float('inf')
    offsetX = np.array([1 / n, 0, 0])
    for face in cubeOrigin:
        for i, row in enumerate(face):
            for j, point in enumerate(row):
                dist = np.linalg.norm(point - closestPoint + offsetX)
                if dist < distance and (point != closestPoint).any():
                    distance = dist
                    prevPointX = point

    distance = float('inf')
    offsetY = np.array([0, 1 / n, 0])
    for face in cubeOrigin:
        for i, row in enumerate(face):
            for j, point in enumerate(row):
                dist = np.linalg.norm(point - closestPoint + offsetY)
                if dist < distance and (point != closestPoint).any() and (point != prevPointX).any():
                    distance = dist
                    prevPointY = point

    if direction == 'x':
        h = np.linalg.norm(closestPoint - prevPointX)
        return (closestPoint, prevPointX, prevPointY, (f(closestPoint[0], closestPoint[1], closestPoint[2]) - f(prevPointX[0], prevPointX[1], prevPointX[2])) / h)
    elif direction == 'y':
        h = np.linalg.norm(closestPoint - prevPointY)
        return (closestPoint, prevPointX, prevPointY, (f(closestPoint[0], closestPoint[1], closestPoint[2]) - f(prevPointY[0], prevPointY[1], prevPointY[2])) / h)
    else:
        return closestPoint, cubeOrigin

def CubedSphere(n, f, point, cubeOrigin):
    X = np.linspace(-1, 1, n)
    Y = np.linspace(-1, 1, n)
    cubeOrigin = []

    cubeOrigin.append([[[round(x, 2), round(y, 2), 1] / np.sqrt(x**2 + y**2 + 1) for y in Y] for x in X])
    cubeOrigin.append([[[round(x, 2), round(y, 2), -1] / np.sqrt(x**2 + y**2 + 1) for y in Y] for x in X])
    cubeOrigin.append([[[-1, round(x, 2), round(y, 2)] / np.sqrt(x**2 + y**2 + 1) for y in Y] for x in X])
    cubeOrigin.append([[[1, round(x, 2), round(y, 2)] / np.sqrt(x**2 + y**2 + 1) for y in Y] for x in X])
    cubeOrigin.append([[[round(x, 2), -1, round(y, 2)] / np.sqrt(x**2 + y**2 + 1) for y in Y] for x in X])
    cubeOrigin.append([[[round(x, 2), 1, round(y, 2)] / np.sqrt(x**2 + y**2 + 1) for y in Y] for x in X])

    closestPoint, prevPointX, prevPointY, value = compute_derivative(f, point, 'y', n, cubeOrigin)

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    x = closestPoint[0]
    y = closestPoint[1]
    z = closestPoint[2]

    ax.scatter(x, y, z, c='b', marker='o')
    ax.scatter(prevPointX[0], prevPointX[1], prevPointX[2], c='g', marker='o')
    ax.scatter(prevPointY[0], prevPointY[1], prevPointY[2], c='yellow', marker='o')

    for face in cubeOrigin:
        for row in face:
            X, Y, Z = zip(*row)
            ax.plot(X, Y, Z, color='red')
        for column in range(len(face)):
            X, Y, Z = zip(*[face[row][column] for row in range(len(face))])
            ax.plot(X, Y, Z, color='red')

    st.pyplot(fig)

# Streamlit App
def main():
    st.title('Cubed Sphere Mesh - Convergence of Finite Difference Scheme')
    st.subheader("Author : Luka Gorgadze")
    st.markdown('''To construct a cubed sphere mesh, we first start with a cube whose faces are divided into a 
regular grid of points. We then map each point on the cube to a point on the surface of a sphere 
inscribed inside the cube. This mapping preserves the regular grid structure and results in a mesh 
with six square faces that are each divided into a regular grid of points. This mesh is known as the cubed sphere mesh.

On a cubed sphere, we can compute the partial derivatives of f using the 
function values in nodal points only. These derivatives can be approximated using finite 
difference schemes that involve the function values at neighboring points on the mesh.

A stencil is a set of neighboring points on the mesh that are used to compute the derivative
of a function at a particular point. The convergence of a linear combination of 
function values depends on the order of the finite difference scheme being used. 
Higher-order schemes generally converge more quickly as the grid spacing vanishes.

To demonstrate convergence of a finite difference scheme on a cubed sphere mesh, 
we can compare the computed derivatives with the exact analytical solutions for simple 
functions. For example, we could compute the gradient of a function that varies linearly 
with x, y, and z and compare it to the exact analytical solution. We could also visualize 
the convergence by plotting the computed derivatives as a function of the grid spacing and 
comparing it to the expected rate of convergence''')
    # Interactive Inputs
    n = st.slider("Number of grid points per face:", 4, 50, 20)
    x = st.slider("X coordinate:", -100, 100, 50)
    y = st.slider("Y coordinate:", -100, 100, 6)
    z = st.slider("Z coordinate:", -100, 100, 100)

    # CubedSphere calculation and plotting
    def f(x, y, z):
        return x**2 + y**2 + z**2 + 60

    lat, longt = cartesian_to_spherical(x, y, z)

    X = np.linspace(-1, 1, n)
    Y = np.linspace(-1, 1, n)
    cubeOrigin = []

    cubeOrigin.append([[[round(x, 2), round(y, 2), 1] / np.sqrt(x**2 + y**2 + 1) for y in Y] for x in X])
    cubeOrigin.append([[[round(x, 2), round(y, 2), -1] / np.sqrt(x**2 + y**2 + 1) for y in Y] for x in X])
    cubeOrigin.append([[[-1, round(x, 2), round(y, 2)] / np.sqrt(x**2 + y**2 + 1) for y in Y] for x in X])
    cubeOrigin.append([[[1, round(x, 2), round(y, 2)] / np.sqrt(x**2 + y**2 + 1) for y in Y] for x in X])
    cubeOrigin.append([[[round(x, 2), -1, round(y, 2)] / np.sqrt(x**2 + y**2 + 1) for y in Y] for x in X])
    cubeOrigin.append([[[round(x, 2), 1, round(y, 2)] / np.sqrt(x**2 + y**2 + 1) for y in Y] for x in X])

    closestPoint, prevPointX, prevPointY, value = compute_derivative(f, (lat, longt), 'y', n, cubeOrigin)
    st.write("Partial derivative along y-axis on nodal point:", value)

    X = closestPoint[0]
    Y = closestPoint[1]
    Z = closestPoint[2]

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.scatter(X, Y, Z, c='b', marker='o')
    ax.scatter(prevPointX[0], prevPointX[1], prevPointX[2], c='g', marker='o')
    ax.scatter(prevPointY[0], prevPointY[1], prevPointY[2], c='yellow', marker='o')

    for face in cubeOrigin:
        for row in face:
            X, Y, Z = zip(*row)
            ax.plot(X, Y, Z, color='red')
        for column in range(len(face)):
            X, Y, Z = zip(*[face[row][column] for row in range(len(face))])
            ax.plot(X, Y, Z, color='red')

    st.pyplot(fig)

if __name__ == "__main__":
    main()
