function [acc, f1] = function_pred_SVM(mips_level, x, genes)
  rng('shuffle')

  nfold = 5;
  cvrepeat = 10;

  rbf_kernel = @(X,Y,g) exp(-g .* pdist2(X,Y,'euclidean').^2);

  [d n] = size(x);

  assert(n == length(genes));

  % scale features
  maxval = max(x, [], 2);
  minval = min(x, [], 2);
  x = bsxfun(@times, bsxfun(@minus, x, minval), 1 ./ (maxval - minval));

  mips_terms = textread(sprintf('data/annotations/yeast_mips_level%d_terms.txt',mips_level),'%s');
  mips_genes = textread(sprintf('data/annotations/yeast_mips_level%d_genes.txt',mips_level),'%s');

  genemap = containers.Map(mips_genes, 1:length(mips_genes));
  filt = genemap.isKey(genes);

  [g t] = textread(sprintf('data/annotations/yeast_mips_level%d_adjacency.txt',mips_level),'%d%d');
  S = sparse(t,g,true,length(mips_terms),length(mips_genes));
  S = S(:,cell2mat(values(genemap,genes(filt))));
  
  x = x(:,filt);

  nf = sum(filt);
  fprintf('Number of annotated genes: %d\n', nf);
  ntest = floor(nf / nfold);

  fprintf('Number of annotations: %d\n', size(S, 1));

  rp = randperm(nf);
  fprintf('Permutation signature: ');
  fprintf('%d\t', rp(1:10));
  fprintf('\n');

  gvec = -3:-1;
  cvec = -1:1;

  fprintf('Pregenerating kernels:\n');
  rbfK = cell(length(gvec), 1);
  for i=1:length(gvec)
    fprintf('%d / %d ... ', i, length(gvec)); tic
    rbfK{i} = rbf_kernel(x', x', 2^gvec(i));
    fprintf('done. '); toc
  end

  fprintf('Total number of classes: %d\n', size(S,1));

  test_ind = rp(1:ntest);
  train_ind = rp(ntest+1:end);

  class_score = zeros(length(test_ind), size(S,1));
  parfor s = 1:size(S,1)
    tt = tic;

    Ytrain = full(double(S(s,train_ind)') * 2 - 1);
    Ytest = full(double(S(s,test_ind)') * 2 - 1);

    retmax = nan;
    trainauc = nan;
    testauc = nan;
    if sum(Ytrain == 1) > 0
      ret = zeros(length(gvec), length(cvec));
      retmax = -inf;
      gmax = 1;
      cmax = 1;
      
      for gi = 1:length(gvec)
        Ktrain = rbfK{gi}(train_ind,train_ind);

        for ci = 1:length(cvec)
          log2c = cvec(ci);

          ret(gi,ci) = 0;
          if cvrepeat > 0
            for it = 1:cvrepeat
              ret(gi,ci) = ret(gi,ci) + do_binary_cross_validation_kernel(Ytrain, Ktrain, ['-t 4 -q -c ', num2str(2^log2c)], 5);
            end
            ret(gi,ci) = ret(gi,ci) / cvrepeat;
          end

          if ret(gi,ci) > retmax
            retmax = ret(gi,ci);
            gmax = gi;
            cmax = ci;
          end
        end
      end

      Ktrain = rbfK{gmax}(train_ind,train_ind);
      Ktest = rbfK{gmax}(test_ind,train_ind);

      model = svmtrain(Ytrain, [(1:length(train_ind))', Ktrain], ['-b 1 -t 4 -q -c ', num2str(2^cvec(cmax))]);
      posind = find(model.Label > 0);
      if ~isempty(posind)
        [pred, acc, dec] = svmpredict(Ytrain, [(1:length(train_ind))', Ktrain], model, '-q -b 1');
        trainauc = validation_function(dec(:,posind), Ytrain);

        [pred, acc, dec] = svmpredict(Ytest, [(1:length(test_ind))', Ktest], model, '-q -b 1');
        testauc = validation_function(dec(:,posind), Ytest);

        class_score(:,s) = dec(:,posind);
      end
    end

    fprintf('[Class %d/%d] CV AUC: %f, Train AUC: %f, Test AUC: %f, ', s, size(S,1), retmax, trainauc, testauc); toc(tt)
  end

  label = S(:,test_ind)';

  [acc, f1] = evaluate_performance(class_score, label);
end
