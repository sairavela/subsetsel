function demo_abess()
    rng(1);
    n = 200; p = 500; s_true = 10;

    % synthetic correlated design
    Sigma = 0.3.^(abs((1:p)'-(1:p)));
    Xraw = randn(n,p) * chol(Sigma);

    beta_true = zeros(p,1);
    supp = randperm(p, s_true);
    beta_true(supp) = [3; 2; 2; 1.5; 1.5; 1; 1; 0.8; 0.8; 0.5];

    y = Xraw*beta_true + 1.0*randn(n,1);

    opts = struct();
    opts.verbose = 1;
    opts.standardize = true;
    opts.kmax_rule = 'min(s,10)';  % or 's'
    opts.tau_rule  = 'paper';
    opts.smax = 30;

    out = abess(Xraw, y, opts);

    fprintf('\nSelected s = %d\n', out.s);
    fprintf('Selected active set size = %d\n', numel(out.A));
    fprintf('Overlap with truth: %d/%d\n', numel(intersect(out.A, supp)), s_true);
end
