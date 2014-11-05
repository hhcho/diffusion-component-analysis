addpath code

%network_prefix = 'data/networks/yeast_biogrid_physical'
network_prefix = 'data/networks/yeast_string_combined'
num_repeat = 50;
num_dim = 500;

fprintf('\n[Load network]\n');
[A, genes] = load_network(network_prefix);

fprintf('\n[Run diffusion]\n');
Q = run_diffusion(A, 'personalized-pagerank', struct('maxiter', 100, 'reset_prob', .5));

fprintf('\n[Learn vectors]\n');
x = learn_vectors(Q, num_dim);

fprintf('\n[Evaluate function prediction]\n');
D = squareform(pdist(x', 'cosine'));
acc = zeros(num_repeat, 3);
f1 = zeros(num_repeat, 3);
for mips_level = 1:3
  [acc(:,mips_level), f1(:,mips_level)] = function_pred_MV(mips_level, D, genes, num_repeat);
end
