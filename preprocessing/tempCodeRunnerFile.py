numeric_features = df.select_dtypes(include=[np.number]).columns.drop('exam_score').tolist()
    # df = cap_outliers_iqr(df, numeric_features)