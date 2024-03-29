import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset

if __name__ == "__main__":
    from polyreg import PolynomialRegression  # type: ignore
else:
    from .polyreg import PolynomialRegression

if __name__ == "__main__":
    """
        Main function to test polynomial regression
    """

    # load the data
    allData = load_dataset("polyreg")

    X = allData[:, [0]]
    y = allData[:, [1]]

    # regression with degree = d
    d = 8
    #reg_lambda = 10
    #model = PolynomialRegression(degree=d, reg_lambda=reg_lambda)
    #model.fit(X, y)

    # output predictions
    #xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
    #ypoints = model.predict(xpoints)

    # plot curve
    plt.figure(figsize=(15, 9), dpi=100)
    plt.subplot(2, 3, 1)

    d = 8
    reg_lambda = 0
    model = PolynomialRegression(degree=d, reg_lambda=reg_lambda)
    model.fit(X, y)

    # output predictions
    xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
    ypoints = model.predict(xpoints)
    
    plt.plot(X, y, "rx")
    plt.title(f"PolyRegression with d = {d} and lambda = {reg_lambda}")
    plt.plot(xpoints, ypoints, "b-")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.subplot(2, 3, 2)

    d = 8
    reg_lambda = 0.0001
    model = PolynomialRegression(degree=d, reg_lambda=reg_lambda)
    model.fit(X, y)

    # output predictions
    xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
    ypoints = model.predict(xpoints)
    
    plt.plot(X, y, "rx")
    plt.title(f"PolyRegression with d = {d} and lambda = {reg_lambda}")
    plt.plot(xpoints, ypoints, "b-")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.subplot(2, 3, 3)

    d = 8
    reg_lambda = 0.1
    model = PolynomialRegression(degree=d, reg_lambda=reg_lambda)
    model.fit(X, y)

    # output predictions
    xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
    ypoints = model.predict(xpoints)
    
    plt.plot(X, y, "rx")
    plt.title(f"PolyRegression with d = {d} and lambda = {reg_lambda}")
    plt.plot(xpoints, ypoints, "b-")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.subplot(2, 3, 4)

    d = 8
    reg_lambda = 1
    model = PolynomialRegression(degree=d, reg_lambda=reg_lambda)
    model.fit(X, y)

    # output predictions
    xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
    ypoints = model.predict(xpoints)
    
    plt.plot(X, y, "rx")
    plt.title(f"PolyRegression with d = {d} and lambda = {reg_lambda}")
    plt.plot(xpoints, ypoints, "b-")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.subplot(2, 3, 5)

    d = 8
    reg_lambda = 10
    model = PolynomialRegression(degree=d, reg_lambda=reg_lambda)
    model.fit(X, y)

    # output predictions
    xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
    ypoints = model.predict(xpoints)
    
    plt.plot(X, y, "rx")
    plt.title(f"PolyRegression with d = {d} and lambda = {reg_lambda}")
    plt.plot(xpoints, ypoints, "b-")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.subplot(2, 3, 6)

    d = 8
    reg_lambda = 100
    model = PolynomialRegression(degree=d, reg_lambda=reg_lambda)
    model.fit(X, y)

    # output predictions
    xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
    ypoints = model.predict(xpoints)
    
    plt.plot(X, y, "rx")
    plt.title(f"PolyRegression with d = {d} and lambda = {reg_lambda}")
    plt.plot(xpoints, ypoints, "b-")
    plt.xlabel("X")
    plt.ylabel("Y")







    plt.show()
