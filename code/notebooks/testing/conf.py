# A configuration file for the randomized search for the best model.
rf_search_grid = {
  'n_estimators': [50,100,300,500],
  'max_depth': [10,30,40,70,80,100],
  'min_samples_leaf': [1, 2, 4],
  'min_samples_split': [2,3,4],
  'max_features': ['sqrt', 'auto'],
  'bootstrap': [False, True]
}

rand_search_cv_params = {
  'n_iter': 50,
  'cv': 4,
  'verbose': 2,
  'n_jobs': 4,
  'return_train_score': True,
  'verbose': True
}
