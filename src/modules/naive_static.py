
import numpy as np
from modules.util import intercept_reshaped_train_test
from modules.config import COST
def naive_static_pricing(X, y, preprocessor):
    X_train, X_test, y_train, y_test = intercept_reshaped_train_test(X, y, preprocessor)
    def demand_curve(axis, value):
        demand = []
        sample_size = len(value)
        for price in axis:
            demand.append(sum(1 if x >= price else 0 for x in value) / sample_size)
        return demand

    def APP_s(axis, value, c):
        demand = demand_curve(axis, value)
        app_s = []
        for i in range(len(axis)):
            app_s.append(max(axis[i] - c, 0) * demand[i])
        return app_s


    c = COST  #cost 0 since we do not have the information
    axis_2 = np.linspace(0, int(np.round(np.max(y_train), 0) + 1), int(2 * (np.round(np.max(y_train), 0) + 1)))
    app_s_train = APP_s(axis_2, y_train[0], c)
    index = np.argmax(app_s_train)
    app_s_test = APP_s(axis_2, y_test[0], c)
    app_s_star = app_s_test[index]

    optimised_static_app = app_s_star
    optimised_price = axis_2[index]
    return optimised_static_app, optimised_price, app_s_train