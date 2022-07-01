% Tutorial Script to Run SCCA-HSIC
clear

% generate data
n = 1000;
p = 20;
q = 20;
numx = 2;
type = 5;
[X,Y] = generate_data(n,p,q,numx,type);

% standardise
X = zscore(X); Y = zscore(Y);

% partition into training and test sets
[~,indices] = partition(size(X,1), 3);
train = indices ~= 1;
test = indices == 1;
Xtrain = X(train,:); Xtest = X(test,:);
Ytrain = Y(train,:); Ytest = Y(test,:);

% set the hyperparameters
hyperparams.M = 1;
hyperparams.normtypeX = 1;
hyperparams.normtypeY = 1;
hyperparams.Cx = 1;
hyperparams.Cy = 1;
hyperparams.Rep = 5;
hyperparams.eps = 1e-7;
hyperparams.sigma1 = [];
hyperparams.sigma2 = [];
hyperparams.maxit = 500;
hyperparams.flag = 2;

% train an scca-hsic model
[u,v] = scca_hsic(Xtrain,Ytrain,hyperparams);

% test the scca-hsic model 
Kxtest = rbf_kernel(Xtest * u);
Kytest = centre_kernel(rbf_kernel(Ytest * v));
hsic_test = f(Kxtest,Kytest);

% visualise
figure
    subplot(1,2,1)
    scatter(sum(X(:,1:2),2), sum(Y(:,1:2),2))
    title('true relation')
    subplot(1,2,2)
    scatter(Xtest*u,Ytest*v)
    title('inferred relation')




