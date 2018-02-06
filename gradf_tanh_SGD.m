function g = gradf_tanh_SGD(K1,cK2,X,a,u)


N = size(K1,1);
g = 0*u;
temp = 0*(u*u');
[~,id1]=sort(rand(1,N));
[~,id2]=sort(rand(1,N));
N=floor(N/10);
for i = id1(1:1:N)
    for j = id2(1:1:N)
        temp = temp + (1-K1(i,j)^2) * cK2(i,j) * (X(i,:)'*X(j,:) + X(j,:)'*X(i,:));
    end
end
g =temp*u*a;


end