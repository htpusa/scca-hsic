function [U,V,finalObj] = scca_hsic(X,Y,varargin)

% The SCCA-HSIC implementation using the projected stochastic mini-batch
% gradient ascent.

% Input:
% X             n x dx data matrix
% Y             n x dy data matrix
%
% hyperparams structure with the following fields
% .M            number of components (default 1)
% .normtypeX 	norm for X view 1 = l1 (default) and 2 = l2
% .normtypeY 	norm for Y view 1 = l1 and 2 = l2 (default)
% .Cx           the value of the norm constraint on view X (default 1)
% .Cy           the value of the norm constraint on view Y (default 1)
% .Rep          number of repetitions from random initializations
%                   (default 5)
% .eps          convergence threshold (default 1e-7)
% .sigma1       the std of the rbf kernel, if empty = median heuristic
% .sigma2       the std of the rbf kernel, if empty = median heuristic
% .maxit        maximum iteration limit (default 500)
% .flag         print iteration results, 1: yes, 2: only the converged
%               result (default 1)

% Output:
% U             canonical coefficient vectors for X in the columns of U
% V             canonical coefficient vectors for Y in the columns of V

% InterMediate is a structure containing all intermediate results
% InterMediate(m,rep).u  contains all intermediate u for mth component
% InterMediate(m,rep).v  contains all intermediate v for mth component
% InterMEdiate(m,rep).obj contains intermediate objective values


%--------------------------------------------------------------------------
% Uurtio, V., Bhadra, S., Rousu, J.
% Sparse Non-Linear CCA through Hilbert-Schmidt Independence Criterion.
% IEEE International Conference on Data Mining (ICDM 2018)
%--------------------------------------------------------------------------
%% Set up parameters

if ~isempty(varargin)
    if size(varargin, 2) > 1
	    error('Check optional inputs.');
    end
    hyperparams = varargin{1,1};
else
    hyperparams = struct;
end

% default hyperparameter values
M = 1;
normtypeX = 1;
normtypeY = 2;
Cx = 1;
Cy = 1;
Rep = 5;
eps = 1e-7;
sigma1 = [];
sigma2 = [];
maxit = 500;
flag = 1;

params = fields(hyperparams);

for i =1:numel(params)
    switch params{i}
        case 'M'
            M = hyperparams.(params{i});
        case 'normtypeX'
            normtypeX = hyperparams.(params{i});
        case 'normtypeY'
            normtypeY = hyperparams.(params{i});
        case 'Cx'
            Cx = hyperparams.(params{i});
        case 'Cy'
            Cy = hyperparams.(params{i});
        case 'Rep'
            Rep = hyperparams.(params{i});
        case 'eps'
            eps = hyperparams.(params{i});
        case 'sigma1'
            sigma1 = hyperparams.(params{i});
        case 'sigma2'
            sigma2 = hyperparams.(params{i});
        case 'maxit'
            maxit = hyperparams.(params{i});
        case 'flag'
            flag = hyperparams.(params{i});
        otherwise
            warning('No hyperparameter named %s', params{i})
    end
end

% partition into training and validation sets
[~,indices] = partition(size(X,1), 3);
train = indices ~= 1;
test = indices == 1;
Xtrain = X(train,:); Xtest = X(test,:);
Ytrain = Y(train,:); Ytest = Y(test,:);

Xm = Xtrain;
Ym = Ytrain;
dx = size(Xm,2);
dy = size(Ym,2);

if size(Xm,1) ~= size(Ym,1)
    printf('sizes of data matrices are not same');
end

U = zeros(size(X,2),M);
V = zeros(size(Y,2),M);
finalObj = zeros(M);

for m=1:M

    candU = zeros(size(X,2),Rep);
    candV = zeros(size(Y,2),Rep);
    candObj = zeros(Rep,1);

    for rep=1:Rep
        %fprintf('Reps: #%d \n',rep);
        % intialization
        if normtypeX == 1
            umr = projL1(rand(dx,1),Cx);
        end
        if normtypeX == 2
            umr = projL2(rand(dx,1),Cx);
        end
        if normtypeY == 1
            vmr = projL1(rand(dy,1),Cy);
        end
        if normtypeY == 2
            vmr = projL2(rand(dy,1),Cy);
        end
        Xu = Xm * umr;
        Yv = Ym * vmr;
        
        % kernel for view x
        if isempty(sigma1)
            [Ku,au] = rbf_kernel(Xu);
        else
            [Ku,au] = rbf_kernel(Xu,sigma1);
        end
        % kernel fow view y
        if isempty(sigma2)
            [Kv,av] = rbf_kernel(Yv);
        else
            [Kv,av] = rbf_kernel(Yv,sigma2);
        end
        
        cKu = centre_kernel(Ku);
        cKv = centre_kernel(Kv);
        diff = 999999;
        ite = 0;
        
        while diff > eps && ite < maxit
            ite = ite + 1;
            obj_old = f(Ku,cKv);
            gradu = gradf_gauss_SGD(Ku,cKv,Xm,au,umr);
            
            %% line search for u
            gamma = norm(gradu,2); % initial step size
            chk = 1;
            while chk == 1
                if normtypeX == 1
                    umr_new  = projL1(umr + gradu * gamma, Cx);
                end
                if normtypeX == 2
                    umr_new  = projL2(umr + gradu * gamma, Cx);
                end
                
                if isempty(sigma1)
                    [Ku_new,au_new] = rbf_kernel(Xm * umr_new);
                else
                    [Ku_new,au_new] = rbf_kernel(Xm * umr_new,sigma1);
                end
                
                obj_new = f(Ku_new,cKv);
                
                if obj_new > obj_old + 1e-4*abs(obj_old)
                    chk = 0;
                    umr = umr_new;
                    Ku = Ku_new;
                    cKu = centre_kernel(Ku);
                    au = au_new;
                    obj = obj_new;
                else
                    gamma = gamma/2;
                    if gamma <1e-7
                        chk=0;
                        
                    end
                end
            end
            obj = obj_new;
            %% line search end
            
            obj_old = obj;
            gradv = gradf_gauss_SGD(Kv,cKu,Ym,av,vmr);
            %% line search for v
            gamma = norm(gradv,2); % initial step size
            chk = 1;
            while chk == 1
                if normtypeY == 1
                    vmr_new  = projL1(vmr + gradv * gamma,Cy);
                end
                if normtypeY == 2
                    vmr_new  = projL2(vmr + gradv * gamma,Cy);
                end
                
                if isempty(sigma2)
                    [Kv_new,av_new] = rbf_kernel(Ym * vmr_new);
                else
                    [Kv_new,av_new] = rbf_kernel(Ym * vmr_new, sigma2);
                end
                
                cKv_new = centre_kernel(Kv_new);
                obj_new = f(Ku,cKv_new);
                if obj_new > obj_old + 1e-4*abs(obj_old)
                    chk = 0;
                    vmr = vmr_new;
                    Kv = Kv_new;
                    cKv = cKv_new;
                    av = av_new;
                    obj = obj_new;
                else
                    gamma = gamma/2;
                    if gamma <1e-7
                        chk = 0;
                    end
                end
            end
            obj = obj_new;
            %% line search end
            %% check the value of test objective
            Kxtest = rbf_kernel(Xtest * umr);
            Kytest = centre_kernel(rbf_kernel(Ytest * vmr));
            test_obj = f(Kxtest,Kytest);
            
            %% compute the delta
            diff = abs(obj - obj_old) / abs(obj + obj_old);
            
            if flag == 1
                disp(['iter = ',num2str(ite),', objtr = ',num2str(obj),', diff = ', num2str(diff), ', test = ', num2str(test_obj)])
            end
        end
        candU(:,rep) = umr;
        candV(:,rep) = vmr;
        candObj(rep) = obj;
        
        if flag == 2
            disp(['Rep ', num2str(rep), ', Objective = ',num2str(obj,2)])
        end
    end
    
    [~,id] = max(candObj);
    U(:,m) = candU(:,id);
    V(:,m) = candV(:,id);
    finalObj(m) = max(candObj);
    
    % deflated data
    Xm = Xm - (U(:,m)*U(:,m)'*Xm')';
    Ym = Ym - (V(:,m)*V(:,m)'*Ym')';
    
end












