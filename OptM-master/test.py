import matlab.engine

eng = matlab.engine.start_matlab()
eng.cd(r'D:\MyResearch\Regions\Federated-Learning\Multi-modal-FL\code\Mine\ETF-PMR\OptM-master', nargout=0)

a_feature_file = "feature_a_matrix-new.mat"
a_ETF_file = "ETF_a_matrix-new.mat"
v_feature_file = "feature_v_matrix-new.mat"
v_ETF_file = "ETF_v_matrix-new.mat"
a = eng.optimalETF(a_feature_file, a_ETF_file)  #调用optimalETF函数，返回两个值，优化前后的F范数
print(a)