# function [diff_random, diff_opt] = optimalETF(feature_file, ETF_file)
#     clc
#     K = 6; % Number of classes, 10 for Cifar10
#     feature_size = 512;  % last layer feature size
#
#     %feature_file = "feature_v_matrix_warmup.mat";
#     %ETF_file = "ETF_v_matrix_warmup-new.mat";
#     G = load(feature_file).feature;
#
#     for i=1:K
#         m = norm(G(:,i));
#         G(:,i) = G(:,i) / m;
#     end
#
#
#     opts.record = 0;
#     opts.mxitr  = 1000;
#     opts.xtol = 1e-5;
#     opts.gtol = 1e-5;
#     opts.ftol = 1e-8;
#
#
#     % min ||G-UW|| <=> min -tr(XM), while M = W·G^T
#     % F(X) = -tr(XM)  s.t. X^T*X = I
#     % X: n*K, M:K*n
#
#     W = generate_ETF(K);
#     M = W * G.';
#     X0 = rand(feature_size,K);    X0 = orth(X0);
#     diff_random = norm(X0*W-G,"fro");
#     tic; [X, out]= OptStiefelGBB(X0, @target, opts, M); tsolve = toc;
#     out.fval = -2*out.fval;
#     W_p = X * W;
#     diff_opt = norm(W_p-G,"fro");
#
#
#     save(ETF_file, 'W_p');
#
#         function [F,G] = target(X,A)
#             F = -trace(X*A);
#             G = -A.';
#         end
#
#         function W = generate_ETF(K)
#             I = eye(K);
#             one = ones(K);
#             W = sqrt(K/(K-1)) * (I - (1/K)*one);
#         end
# end