pf = PolynomialFeatures(degree=2, include_bias=False)
X_pf = pf.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_pf, y, test_size=0.3,
                                                    random_state=4321)
s = StandardScaler()
X_train_s = s.fit_transform(X_train)
bc_price = boxcox_test(y_train)
y_train_bc = bc_price[0]
lam2 = bc_price[1]
lr.fit(X_train_s, y_train_bc)
X_test_s = s.transform(X_test)
y_pred_bc = lr.predict(X_test_s)
inv_boxcox(y_train_bc, lam2)
y_pred_tran = inv_boxcox(y_pred_bc,lam2)
print(r2_score(y_pred_tran,y_pred_bc))