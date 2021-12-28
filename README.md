# Constrained Logistic Regression
Sample implementation of constructing a logistic regression with given ranges on each of the feature's coefficients (via [clogistic](https://github.com/guillermo-navas-palencia/clogistic) library).

# The Data
We will use the processed version of telco customer churn data from Kaggle. The data can be downloaded [here](https://www.kaggle.com/blastchar/telco-customer-churn).

# Steps
## Define the constraints
For example:
```python
# define constraints as dataframe
import numpy as np
constraint_df = pd.DataFrame(data=[
                                   ['gender',-np.inf,np.inf],
                                   ['SeniorCitizen',-np.inf,np.inf],
                                   ['Partner',-np.inf, 0],
                                   ['Dependents',-np.inf,0],
                                   ['tenure',-np.inf,0],
                                   ['PhoneService',-np.inf,0],
                                   ['PaperlessBilling',-np.inf,np.inf],
                                   ['MonthlyCharges',-np.inf,np.inf],
                                   ['intercept',-np.inf,np.inf]],
                             columns=['feature','lower_bound','upper_bound'])
constraint_df
```
```
|    | feature          |   lower_bound |   upper_bound |
|---:|:-----------------|--------------:|--------------:|
|  0 | gender           |          -inf |           inf |
|  1 | SeniorCitizen    |          -inf |           inf |
|  2 | Partner          |          -inf |             0 |
|  3 | Dependents       |          -inf |             0 |
|  4 | tenure           |          -inf |             0 |
|  5 | PhoneService     |          -inf |             0 |
|  6 | PaperlessBilling |          -inf |           inf |
|  7 | MonthlyCharges   |          -inf |           inf |
|  8 | intercept        |          -inf |           inf |
```

## Model training via clogistic
```python
# train using clogistic
from scipy.optimize import Bounds
from clogistic import LogisticRegression as clLogisticRegression

lower_bounds = constraint_df['lower_bound'].to_numpy()
upper_bounds = constraint_df['upper_bound'].to_numpy()
bounds = Bounds(lower_bounds, upper_bounds)

cl_logreg = clLogisticRegression(penalty='none')
cl_logreg.fit(X_train, y_train, bounds=bounds)
```

Retrieve the model coefficients
```python
# coefficients as dataframe
cl_coef = pd.DataFrame({
    'feature': df.drop(columns='Churn').columns.tolist() + ['intercept'],
    'coefficient': list(cl_logreg.coef_[0]) + [cl_logreg.intercept_[0]]
})

cl_coef
```

```
|    | feature          |   coefficient |
|---:|:-----------------|--------------:|
|  0 | gender           |   0.0184168   |
|  1 | SeniorCitizen    |   0.506692    |
|  2 | Partner          |   3.85603e-09 |
|  3 | Dependents       |  -0.35721     |
|  4 | tenure           |  -0.0557211   |
|  5 | PhoneService     |  -0.796233    |
|  6 | PaperlessBilling |   0.398824    |
|  7 | MonthlyCharges   |   0.033197    |
|  8 | intercept        |  -1.36086     |
```


