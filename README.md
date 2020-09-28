# lending_club_challenge

I do not have an analysis to report. I could not resolve the same error I would receive every time I tried to train a model. I will show an example below.

# Train the Logistic Regression model using the resampled data
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)

ValueError                                Traceback (most recent call last)
<ipython-input-13-ae986141e434> in <module>
      2 from sklearn.linear_model import LogisticRegression
      3 model = LogisticRegression(solver='lbfgs', random_state=1)
----> 4 model.fit(X_resampled, y_resampled)

~\miniconda3\envs\PythonData\lib\site-packages\sklearn\linear_model\_logistic.py in fit(self, X, y, sample_weight)
   1340             _dtype = [np.float64, np.float32]
   1341 
-> 1342         X, y = self._validate_data(X, y, accept_sparse='csr', dtype=_dtype,
   1343                                    order="C",
   1344                                    accept_large_sparse=solver != 'liblinear')

~\miniconda3\envs\PythonData\lib\site-packages\sklearn\base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    430                 y = check_array(y, **check_y_params)
    431             else:
--> 432                 X, y = check_X_y(X, y, **check_params)
    433             out = X, y
    434 

~\miniconda3\envs\PythonData\lib\site-packages\sklearn\utils\validation.py in inner_f(*args, **kwargs)
     71                           FutureWarning)
     72         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
---> 73         return f(**kwargs)
     74     return inner_f
     75 

~\miniconda3\envs\PythonData\lib\site-packages\sklearn\utils\validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
    794         raise ValueError("y cannot be None")
    795 
--> 796     X = check_array(X, accept_sparse=accept_sparse,
    797                     accept_large_sparse=accept_large_sparse,
    798                     dtype=dtype, order=order, copy=copy,

~\miniconda3\envs\PythonData\lib\site-packages\sklearn\utils\validation.py in inner_f(*args, **kwargs)
     71                           FutureWarning)
     72         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
---> 73         return f(**kwargs)
     74     return inner_f
     75 

~\miniconda3\envs\PythonData\lib\site-packages\sklearn\utils\validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)
    597                     array = array.astype(dtype, casting="unsafe", copy=False)
    598                 else:
--> 599                     array = np.asarray(array, order=order, dtype=dtype)
    600             except ComplexWarning:
    601                 raise ValueError("Complex data not supported\n"

~\miniconda3\envs\PythonData\lib\site-packages\numpy\core\_asarray.py in asarray(a, dtype, order)
     81 
     82     """
---> 83     return array(a, dtype, copy=False, order=order)
     84 
     85 

ValueError: could not convert string to float: 'N'
