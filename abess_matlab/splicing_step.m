function [beta_best, d_best, A_best, I_best, L_best, improved] = ...
    splicing_step(X, y, beta, d, A, I, kmax, tau_s, ridge)
%SPLICING_STEP  One call of Algorithm 2 "Splicing".
%
% Robust MATLAB implementation (column/row safe).
%
% Computes backward sacrifices xi_j for j in A:
%   xi_j = (Xj'Xj)/(2n) * beta_j^2   (Eq 3)
% Computes forward sacrifices zeta_j for j in I:
%   zeta_j = (Xj'Xj)/(2n) * (d_j / (Xj'Xj/n))^2  (Eq 4)
%         = n/(2*Xj'Xj) * d_j^2  (equivalent form)
%
% Tries k=1..kmax, swaps k least relevant from A with k most relevant from I,
% resolves LS on new active set, keeps best loss decrease.
% If L0-L < tau_s, returns original.

    % Force column vectors for set operations / concatenation safety
    A = A(:);
    I = I(:);

    [n, p] = size(X);

    r = y - X*beta;
    L0 = 0.5/n * (r' * r);
    L_best = L0;
    beta_best = beta; d_best = d; A_best = A; I_best = I;
    improved = false;

    xTx_diag = sum(X.^2, 1)';  % p-by-1

    % sacrifices
    xi   = zeros(p,1);
    zeta = zeros(p,1);

    % Eq (3): xi_j = (Xj'Xj)/(2n) * beta_j^2
    xi(A) = (xTx_diag(A) ./ (2*n)) .* (beta(A).^2);

    % Eq (4): zeta_j = n/(2*Xj'Xj) * d_j^2
    zeta(I) = (n ./ (2*max(xTx_diag(I), realmin))) .* (d(I).^2);

    % Sort once
    [~, orderA] = sort(xi(A), 'ascend');
    A_sorted = A(orderA);

    [~, orderI] = sort(zeta(I), 'descend');
    I_sorted = I(orderI);

    for k = 1:min([kmax, numel(A), numel(I)])
        Ak = A_sorted(1:k);      % k least relevant in A (column)
        Ik = I_sorted(1:k);      % k most relevant in I (column)

        % IMPORTANT: use vertical concatenation and keep as column
        A_tilde = sort([setdiff(A, Ak); Ik]);

        beta_tilde = zeros(p,1);
        betaA = solve_ls(X(:,A_tilde), y, ridge);
        beta_tilde(A_tilde) = betaA;

        r_tilde = y - X*beta_tilde;
        L_tilde = 0.5/n * (r_tilde' * r_tilde);

        if L_tilde < L_best
            L_best = L_tilde;
            beta_best = beta_tilde;

            d_best = (X' * r_tilde) / n;
            d_best(A_tilde) = 0;

            A_best = A_tilde;
            I_best = setdiff((1:p)', A_best);
            improved = true;
        end
    end

    % Thresholding (Alg 2 Step 4)
    if (L0 - L_best) < tau_s
        beta_best = beta;
        d_best    = d;
        A_best    = A;
        I_best    = I;
        L_best    = L0;
        improved  = false;
    end
end

function betaA = solve_ls(XA, y, ridge)
    G = XA' * XA;
    if ridge > 0
        G = G + ridge * eye(size(G));
    end
    betaA = G \ (XA' * y);
end