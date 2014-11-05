% mips_level: 1, 2, or 3. 
% D: n x n matrix, pairwise distances of n genes
% genes: n x 1 cell array, gene symbols in the same order as D
% nrepeat: scalar, number of repetitions of cross-validation
%
function [acc, f1] = function_pred_MV(mips_level, D, genes, nrepeat, nvote)
  if ~exist('nrepeat','var')
    nrepeat = 100;
  end
  if ~exist('nvote','var')
    nvote = 10;
  end
  nfold = 5;

  n = size(D, 1);

  assert(n == length(genes));

  mips_terms = textread(sprintf('data/annotations/yeast_mips_level%d_terms.txt',mips_level),'%s');
  mips_genes = textread(sprintf('data/annotations/yeast_mips_level%d_genes.txt',mips_level),'%s');

  genemap = containers.Map(mips_genes, 1:length(mips_genes));
  filt = genemap.isKey(genes);

  [g t] = textread(sprintf('data/annotations/yeast_mips_level%d_adjacency.txt',mips_level),'%d%d');
  S = sparse(t,g,true,length(mips_terms),length(mips_genes));
  S = S(:,cell2mat(values(genemap,genes(filt))));

  D = D(filt,filt);

  nf = sum(filt);
  fprintf('Number of annotated genes: %d\n', nf);
  ntest = floor(nf / nfold);

  acc = zeros(nrepeat, 1);
  f1 = zeros(nrepeat, 1);
  for p = 1:nrepeat
    fprintf('Permutation %d / %d ... ', p, nrepeat); tic
    test_ind = randperm(nf, ntest);
    rp = randperm(nf);
    test_ind = rp(1:ntest);
    train_ind = rp(ntest+1:end);

    class_score = zeros(ntest, size(S,1));
    for i = 1:ntest
      [v, o] = sort(D(test_ind(i), train_ind));
      o = o(~isinf(v));
      k = min(nvote, length(o));
      votes = sum(bsxfun(@rdivide, S(:,train_ind(o(1:k))), D(test_ind(i), train_ind(o(1:k)))), 2);
      class_score(i,:) = votes;
    end
    label = S(:,test_ind)';

    [acc(p), f1(p)] = evaluate_performance(class_score, label);
    fprintf('done. Accuracy: %f, F1: %f, ', acc(p), f1(p)); toc
  end

  fprintf('[Accuracy] Mean: %f, Stdev: %f\n', mean(acc), std(acc, 1));
  fprintf('[F1-score] Mean: %f, Stdev: %f\n', mean(f1), std(f1, 1));
end
