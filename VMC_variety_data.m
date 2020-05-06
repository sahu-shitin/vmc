% EE5111 Course Project
% EE16B058, EE16B150, EE16B157

% VMC for High Rank Matrix Completion
% Algorithm from: G. Ongie et. al, Algebraic Variety Models for High Rank
% Matrix Completion

% The code completes synthetic Union of Subspaces (UoS) data using VMC.
% This code runs the main algorithm several times to generate statistics 
% based on parameters given.

%% Parameters for iteration
% mbyn (1-D Array): Amount of sampling 
% k0 (1-D Array): Number of underlying subspaces used to generate the data
% Navg: Amount of averaging for the same set of sampling rate and number of
% subspaces

k0 = [1 3 5 7];
mbyn = [0.9 0.8 0.7 0.6 0.5 0.4 0.3];
Navg = 5;

%% Parameters of Matrix Completion
% d: Degree of lifted polynomial space
% p: Value of p in Schatten-p norm
% eta: Decay rate for gamma, gamm_n = gamm_0/eta^n
d = 2;
p = 0.5;
eta = 1.01;

recovered = zeros(length(mbyn), length(k0));
comperr = zeros(length(mbyn), length(k0));

tic
for mbyn_iter = 1:length(mbyn)
    for k_iter = 1:length(k0)
        for avgiter = 1:Navg
            
    % Parameters of UoS Data Generation
    % n: Dimension of generated data
    % r: (less than n) Dimension of underlying subspaces
    % k: (Value taken from array 'k0' above) Number of r-dimensional subspaces
    % p1: Number of points genreated from each subspace
    n = 10;
    r = 3;
    k = k0(k_iter);
    p1 = 40;

    % Generate Random UoS data
    X0 = [];
    for i = 1:k
        U = orth(randn(n,r));
        X0 = [X0, U*randn(r,p1)];
    end

    % X0 = mocapdata(:,[1:24 27:36 39:end]);
    scalefac = sqrt(max(sum(abs(X0).^2)));
    X0 = X0/scalefac;

    n = size(X0,1);
    s = size(X0,2);
    Omega = true(size(X0)); % (Logical Array) Index of arrays to sample

    m = ceil(mbyn(mbyn_iter)*n);
    nmiss =  n - m; % Number of missing entries per column
    for iter = 1:s
        missing = randperm(n);
        Omega(missing(1:nmiss), iter) = 0;
    end

    om  = find(Omega);
    om_ = find(~Omega);

    X = zeros(size(X0));
    X(om) = X0(om);     % Initialize with known samples, rest zeros
    I = eye(size(X,2)); % Identity matirx, size s

    kd = @(d,X,Y) (X.'*Y + 1).^d;    % Kernel function
    [~,e] = eig(kd(d,X,X));
    gamm = (0.1^d)*max(e,[],'all'); % Initial value of gamm
    gamm_min = 1e-16;    % Minimum step size for gradient descent

    iter = 0;
    tol = 3e-6;     % Tolerance value to determine convergence
    norms = [];
    temp = tol+1;   % True condition just to enter while loop

    % while iter < 8000     % For fixd number of iterations
    while temp > tol
        iter = iter + 1;

    % Inverse power of kernel matrix
        K = kd(d,X,X);  % Kernel Matrix
        [V,S] = eig(K);
        W = V*(real((S + gamm*I)^(p/2-1)))*V.';   % Weights

        % Projected Gradient descent steps
        % 1st Step
        tau = gamm^(1-p/2);
        Xprev = X;
        X = X - (tau)*X*(W.*kd(d-1,X,X));
        X(om) = X0(om);
        gamm = max(gamm/eta, gamm_min);

    %     % 2nd Step
    %     tau = gamm^(1-p/2);
    %     Xprev = X;
    %     X = X - (tau)*X*(W.*kd(d-1,X,X));
    %     X(om) = X0(om);
    %     gamm = max(gamm/eta, gamm_min);

        temp = norm(Xprev - X)/norm(Xprev);
        norms = [norms temp];       % Array stores norm of difference of previous and current estimate
                                    % Shows nature of convergence of algorithm

    % % Periodically display useful info if required                            
    %     if ~rem(iter,100)
    %         disp(iter)
    %         disp([temp norm(X-X0)/norm(X0)])
    %     end

    end     % end while

    norm_arr = zeros(1,s); % Store norms of column-wise difference
    for iter = 1:s
    norm_arr(iter) = norm(X(:,iter) - X0(:,iter))/norm(X0(:,iter));
    end

    % figure(f)
    % plot(norm_arr)
    % ylim([0 0.7]);
    % drawnow

    disp(["*******Trial"+num2str(avgiter)+ "********"]);
    rec = length(find(norm_arr < 1e-2))/p1/k;
    recovered(mbyn_iter, k_iter) = recovered(mbyn_iter, k_iter) + length(find(norm_arr < 1e-1))/p1/k/Navg;
    comperr(mbyn_iter) = comperr(mbyn_iter) + norm(X-X0)/norm(X0)/Navg;

        end
        disp("m/n = " + mbyn(mbyn_iter) + "k = " + k0(k_iter) + "done.");
    end
end
toc

% For UoS data
imshow(recovered)
colorbar

% % For mocapdata
% figure()
% plot(1-mbyn, recovered)
% title("Fraction of columns recovered d = " + d)
% xlabel("Missing rate")
% figure()
% plot(1-mbyn,comperr);
% title("Completion error d = " + d)
% xlabel("Missing rate")
