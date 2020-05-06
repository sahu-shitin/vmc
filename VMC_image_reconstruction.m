% EE5111 Course Project
% EE16B058, EE16B150, EE16B157

% VMC code customised for showing image completion

mbyn = 0.4;   % Amount of Sampling
k0 = 3; % Number of subspaces

% Parameters of Matrix completion
d = 1;      % Degree of lifted polynomial space
p = 0.5;    % p value for Schatten-p norm
eta = 1.01; % gamm_n = gamm_0/eta^n

% X0 = double([obj10__39 obj11__1; obj1__0 obj12__0])/512;
% X0 = double(yaleB10_P00A_005E_10)/512;
X0 = double(cat_2)/512;
scalefac = sqrt(max(sum(abs(X0).^2)));
X0 = X0/scalefac;

n = size(X0,1);
s = size(X0,2);
Omega = true(size(X0));

m = ceil(mbyn*n);
nmiss =  n - m;
% Random missing entries
for iter = 1:s
    missing = randperm(n);
    Omega(missing(1:nmiss), iter) = 0;
end

om  = find( Omega);
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
temp = tol+1;
figure()
imshow(X0*2*scalefac)
A = X0*2*scalefac;
figure()
imshow(X*2*scalefac);
A = [A X*2*scalefac];
%%
tic
f = figure();
% while iter < 800
while temp > tol
    iter = iter + 1;
%     disp(iter)
    
    % Inverse power of kernel matrix
    K = kd(d,X,X);  % Kernel Matrix
    [V,S] = eig(K);
    W = V*(real((S + gamm*I)^(p/2-1)))*V.';   % Weights
    
    % Projected Gradient descent step
    tau = gamm^(1-p/2);
    Xprev = X;
    X = X - (tau)*X*(W.*kd(d-1,X,X));
    X(om) = X0(om);
    gamm = max(gamm/eta, gamm_min);
    
%     tau = gamm^(1-p/2);
%     Xprev = X;
%     X = X - (tau)*X*(W.*kd(d-1,X,X));
%     X(om) = X0(om);
%     gamm = max(gamm/eta, gamm_min);
    
    temp = norm(Xprev - X)/norm(Xprev);
    norms = [norms temp];
    if ~rem(iter,50)
        disp(iter)
        figure(f)
        imshow(X*2*scalefac)
        drawnow
    end    
end
%%
toc
A = [A X*2*scalefac];
B = abs(X-X0);
A = [A B*2*scalefac];
disp(iter)
norm_arr = zeros(1,s);
for normiter = 1:s
norm_arr(normiter) = norm(X(:,normiter) - X0(:,normiter))/norm(X0(:,normiter));
end
figure()
plot(norms)
title({"Plot of norm(X-Xprev)/norm(Xprev) in each iteration", "Xprev is the previous estimate X is the current estimate"});
figure()
plot(norm_arr)
title("Plot of columnwise norm of difference between original and final estimate");
ylim([0 1])
