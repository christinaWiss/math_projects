import constants
import main
import matplotlib.pyplot as plt
import math

"""The Ho-Lee-Model model: Here the differential equation is given by dr(t)=b(t)dt +σ dW∗(t)
    """


b = [math.sin(i*10) for i in constants.gather_unit]
sigma = 0.1
initial_r = 0.1


def short_rate_hl_model(b, sigma, initial_r):

    Brownian_Motion = [i / (len(constants.gather_unit)) ** 0.5 for i in main.random_walk(prob=0.5, gather=constants.gather_unit)]

    short_rate = [initial_r]
    last = initial_r
    last_b = 0
    count = 0
    for i in Brownian_Motion:
        short_rate += [short_rate[-1] + b[count]/constants.width + sigma * (i-last_b)]
        count += 1
        last_b = i

    short_rate.pop()
    return short_rate


r_path3 = short_rate_hl_model(b, sigma, .1)
print(r_path3)
r_path2 = short_rate_hl_model(b, sigma, .1)
r_path = short_rate_hl_model(b, sigma, .1)
r_path4 = short_rate_hl_model(b, 0, .1)

plt.plot(constants.gather_unit, r_path4)
plt.plot(constants.gather_unit, r_path3)
plt.plot(constants.gather_unit, r_path2)
plt.plot(constants.gather_unit, r_path)
plt.show()

