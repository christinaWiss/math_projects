import constants
import main
import matplotlib.pyplot as plt

"""The CIR Model model: Here the differential equation is given by dr(t)=(b+beta*r(t))dt+sigma * sqrt(r(t)) dW^*(t).
    """

beta = -6
sigma = .148
b = 2*sigma+0.01
initial_r = 0.08


def short_rate_cir_model(b, beta, sigma, initial_r):
    if b < 0:
        print("b should not be negative but is", b)
        exit()
    else:
        Brownian_Motion = [i / (len(constants.gather_unit)) ** 0.5 for i in main.random_walk(prob=0.5, gather=constants.gather_unit)]

        short_rate = [initial_r]
        last = initial_r
        last_b = 0
        for i in Brownian_Motion:
            short_rate += [last + (b + beta * last)/constants.width + sigma * (last**0.5) * (i-last_b)]
            last = short_rate[-1]
            last_b = i

    short_rate.pop()
    return short_rate

r_path3 = short_rate_cir_model(b, beta, sigma, .1)
r_path2 = short_rate_cir_model(b, beta, sigma, .1)
r_path = short_rate_cir_model(b, beta, .0, .1)

plt.plot(constants.gather_unit, r_path3)
plt.plot(constants.gather_unit, r_path2)
plt.plot(constants.gather_unit, r_path)
plt.show()