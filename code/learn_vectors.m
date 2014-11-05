function [x, w, P, fval, heldout_error] = learn_vectors(Q, d, maxiter, heldout_inds)
    if ~exist('maxiter', 'var')
      maxiter = 1000;
    end
    if ~exist('heldout_inds', 'var')
      heldout_inds = [];
    end

    nheldout = sum(heldout_inds);

    Q_heldout = Q(heldout_inds);
    Q(heldout_inds) = 0;

    [nnode ncontext] = size(Q);
    nparam = (nnode + ncontext) * d;
        
    opts = struct('factr', 1e4, 'pgtol', 0, 'm', 5, 'printEvery', 50, 'maxIts', maxiter);
    if nheldout > 0
        opts.errFcn = @heldout_fn;
    end

    while true
      %% Initialize vectors
      fprintf('Initializing vectors ... '); tic
      wx = rand(d, nnode + ncontext) / 10 - .05;
      fprintf('done. '); toc

      opts.x0 = wx(:);
      [xopt, fval, info] = lbfgsb(@optim_fn, -inf(nparam,1), inf(nparam,1), opts);
      if info.iterations > 10
        break
      end
      fprintf('Premature termination; trying again.\n');
    end
    wx = reshape(xopt, d, nnode + ncontext);

    fprintf('Done.\n');
    
    %% Summarize output
    w = wx(:,1:ncontext);
    x = wx(:,ncontext+1:end);
    P = P_fn(w,x);
    fval = obj_fn(P);
    heldout_error = heldout_fn(wx);
    
    function [fval, grad] = optim_fn(wx)
        wx = reshape(wx, d, nnode + ncontext);

        P = P_fn(wx(:,1:ncontext), wx(:,ncontext+1:end));
        P(heldout_inds) = 0;

        fval = obj_fn(P);

        wgrad = wx(:,ncontext+1:end) * (P-Q);
        xgrad = wx(:,1:ncontext) * (P-Q)';
        grad = [wgrad, xgrad];

        grad = grad(:);
    end

    function P = P_fn(w, x)
        P = exp(x' * w);
        P = bsxfun(@rdivide, P, sum(P));
    end

    function res = obj_fn(P)
        v = zeros(ncontext,1);
        for j = 1:ncontext
            v(j) = kldiv(Q(:,j),P(:,j));
        end
        res = sum(v);
    end

    function res = heldout_fn(wx)
        wx = reshape(wx, d, nnode + ncontext);
        P = P_fn(wx(:,1:ncontext), wx(:,ncontext+1:end));
        res = mean((P(heldout_inds) - Q_heldout).^2);
    end
   
    function res = kldiv(p,q)
        filt = p > 0;
        res = sum(p(filt) .* log(p(filt) ./ q(filt)));
    end
end
