import matplotlib.pyplot as plt
import numpy as np


class BestFit:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.coef = 0
        self.intercept = 0

    @staticmethod
    def linear_function(x, c, i):
        return (x * c) + i

    def fit(self):
        global predicted, m2d_glist, m2e_glist, coef_list, intercept_list

        predicted = BestFit.linear_function(self.x, self.coef, self.intercept)

        m2e_list = list(map(lambda x, y: (x - y)**2, predicted, self.y))
        m2e = sum(m2e_list) / len(self.x)

        m2d_list = list(map(lambda x: (m2e - x)**2, m2e_list))
        m2d = sum(m2d_list)/len(predicted)
        prev_m2d = m2d

        m2d_glist = []
        coef_list = []
        running = True

        current = 0.01
        pos = 0.01
        neg = -0.01

        # Finds Line parallel with the line of best fit
        while running:
            self.coef = self.coef + current

            predicted = BestFit.linear_function(self.x, self.coef, self.intercept)
            m2e_list = list(map(lambda x, y: (x - y) ** 2, predicted, self.y))
            m2e = sum(m2e_list)/len(self.x)

            m2d_list = list(map(lambda x: (m2e - x) ** 2, m2e_list))
            m2d = sum(m2d_list)/len(predicted)

            recurring = m2d_glist.count(m2d)
            if recurring > 5: running = False

            m2d_glist.append(m2d)
            coef_list.append(self.coef)

            if prev_m2d < m2d:
                prev_m2d = m2d
                current = pos if current == neg else neg
            if prev_m2d > m2d: prev_m2d = m2d

        self.coef = round(self.coef, 2)

        predicted = BestFit.linear_function(self.x, self.coef, self.intercept)

        m2e_list = list(map(lambda x, y: (x - y) ** 2, predicted, self.y))
        m2e = sum(m2e_list) / len(self.x)
        prev_m2e = m2e

        running = True

        m2e_glist = []
        intercept_list = []

        current = 0.01
        pos = 0.01
        neg = -0.01

        # Using the new coefficient, find the best intercept
        while running:
            self.intercept = self.intercept + current

            predicted = BestFit.linear_function(self.x, self.coef, self.intercept)
            m2e_list = list(map(lambda x, y: (x - y) ** 2, predicted, self.y))
            m2e = sum(m2e_list) / len(self.x)

            recurring = m2e_glist.count(m2e)
            if recurring > 5: running = False

            intercept_list.append(self.intercept)
            m2e_glist.append(m2e)

            if prev_m2e < m2e:
                prev_m2e = m2e
                current = pos if current == neg else neg
            if prev_m2e > m2e: prev_m2e = m2e

        self.intercept = round(self.intercept, 2)

    def graph(self):
        # X and Y graph
        plt.subplot(2, 2, 1)
        plt.title("X and Y values")
        plt.scatter(self.x, self.y, marker="x", s=1, c="#00adff")
        plt.xlabel("X Values")
        plt.ylabel("Y Values")

        # Coef and M2D
        plt.subplot(2, 2, 2)
        plt.title("Coefficient and M2D")
        plt.plot(m2d_glist, coef_list, c="#00adff")
        plt.xlabel("Coefficient")
        plt.ylabel("Mean Squared Difference")

        # Intercept and M2E
        plt.subplot(2, 2, 3)
        plt.title("Intercept and M2E")
        plt.plot(m2e_glist, intercept_list, c="#00adff")
        plt.xlabel("Intercept")
        plt.ylabel("Mean Squared Error")

        # Prediction with X and Y Values
        plt.subplot(2, 2, 4)
        plt.title("Prediction")
        plt.scatter(self.x, self.y, marker="x", s=1, c="#00adff")
        plt.plot(np.array(range(len(predicted))), predicted, c="#00334bd4")
        plt.xlabel("X Values")
        plt.ylabel("Y Values")

        plt.tight_layout()

        plt.show()


Y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
     23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
     43, 44, 45, 46, 47, 48, 49, 50]

X = np.array(range(len(Y)))

model = BestFit(X, Y)
model.fit()

print(f"COEFFICIENT: {model.coef}")
print(f"INTERCEPT: {model.intercept}")

model.graph()
