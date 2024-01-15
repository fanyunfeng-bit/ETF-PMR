dist_file = "cifar10_pat2_20clients_dist.mat";
i = 0;
dist_data = load(dist_file);
curclient = ['client',int2str(i)];
K_p = dist_data.(curclient)