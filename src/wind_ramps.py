# def ramp_definition(X, Y, threshold):
#     Y_ = Y.copy()
#
#     Y_.insert(0, 'Norm. Wind Power(t)', X['Norm. Wind Power(t)'])
#     is_ramp = Y_.apply(lambda x: (x.diff() > threshold).any(), axis=1).astype(int)
#
#     return is_ramp
#
#
# def volatility_std(X):
#     return X.apply(lambda x: x / x.std(), axis=1)
