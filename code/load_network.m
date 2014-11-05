% 
%
%
function [A g] = load_network(net_prefix, is_sparse)

[g1,g2,s] = textread([net_prefix, '_adjacency.txt'], '%d%d%f');
g = textread([net_prefix, '_genes.txt'], '%s');
n = length(g);
A = sparse(g1,g2,s,n,n);

is_sym = isequal(A,A');
fprintf('Symmetric? %d\n', is_sym);

if ~is_sym
  A = A + A';
end

if ~exist('is_sparse','var') || ~is_sparse
  A = full(A);
end

for i=1:n
  A(i,i) = 0;
end

fprintf('Size of network: %d\n', n);
fprintf('Max edge weight: %f\n', max(s));
fprintf('Min edge weight: %f\n', min(s));

end
