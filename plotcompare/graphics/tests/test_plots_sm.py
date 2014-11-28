
"""Trying out matplotplib testing


"""

import sys

import numpy as np
import matplotlib
from plotcompare.testing.decorators import image_comparison
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.graphics.regressionplots as rplots

@image_comparison(baseline_images=['test_added_variable1', 'test_added_variable2'], extensions=['png'])
def test_added_variable():

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    np.random.seed(3446)

    n = 100
    p = 3
    exog = np.random.normal(size=(n, p))
    lin_pred = 4 + exog[:, 0] + 0.2*exog[:, 1]**2
    expval = np.exp(lin_pred)
    endog = np.random.poisson(expval)

    model = sm.GLM(endog, exog, family=sm.families.Poisson())
    results = model.fit()

    focus_col = 0
    use_glm_weights = False#, True:
    resid_type = ["resid_deviance", "resid_response"]

    fig = rplots.plot_added_variable(results, focus_col,
                              use_glm_weights=use_glm_weights,
                              resid_type=resid_type[0], ax=ax)
    ti = "Added variable plot"
    #fig.savefig('test_added_variable-expected.pdf')
    fig2 = plt.figure()
    ax = fig2.add_subplot(1,1,1)
    fig2 = rplots.plot_added_variable(results, focus_col,
                              use_glm_weights=use_glm_weights,
                              resid_type=resid_type[1], ax=ax)
    return fig, fig2
