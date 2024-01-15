function ETF_solver
clc

K = 200; % Number of classes, 10 for Cifar10
feature_size = 512*4;  % last layer feature size

dist_file = "tiny_sha10_100clients_dist.mat";
pETF_file = "tiny_sha10_100clients_ETF.mat";
dist_data = load(dist_file);
client_num = 20;

diff_random = 0;
diff_opt = 0;

U = rand(feature_size,K);   U = orth(U);  %n*K
G = U * generate_ETF(K);  % Global ETF
pETF.g = G;

opts.record = 0;
opts.mxitr  = 1000;
opts.xtol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;

for i = 0:client_num-1
    cur_client = ['client',int2str(i)];
    K_p = dist_data.(cur_client); %Local label distribution
    if length(K_p)==1
        pETF.(cur_client) = G(:,K_p+1);
    else
        % min ||G'-UW|| = min -tr(XM), while M = WG'^T
        % F(X) = -tr(XM)  s.t. X*X^T = I
        % X: n*K, M:K*n
        W = generate_ETF(length(K_p));  % Personal ETF
        G_p = G(:,K_p+1);
        M = W * G_p.';
        X0 = rand(feature_size,length(K_p));    X0 = orth(X0);
        diff_random = diff_random + norm(X0*W-G_p,"fro");
        tic; [X, out]= OptStiefelGBB(X0, @target, opts, M); tsolve = toc;
        out.fval = -2*out.fval;
        W_p = X * W;
        diff_opt = diff_opt + norm(W_p-G_p,"fro");
        pETF.(cur_client) = W_p;
    end
end

diff_random/client_num
diff_opt/client_num
save(pETF_file,"pETF");

    function [F,G] = target(X,A)
        F = -trace(X*A);
        G = -A.';
    end

    function W = generate_ETF(K)
        I = eye(K);
        one = ones(K);
        W = sqrt(K/(K-1)) * (I - (1/K)*one);  
    end
end