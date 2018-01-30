using GLPKMathProgInterface
using JuMP
using Ipopt

function SMO(X,y,c)
    (d, n) = size(X');
    A = (X.*y)';
    M = A'*A;
    b = ones(n);
    eps = 0.0000001;
    f0 = eps;
    f1 = 0;
    a = rand(n);
    a[y.== 1] = 1/sum(y.== 1);
    a[y.== -1] = 1/sum(y.== -1);

    pt = []
    mx = 50000
    it = 0
    while (it <= mx)
        # choose random indices (i, j)
        i = rand(1:n);
        j = rand(1:n);
        if (i != j)
            di = gradFunc(M, b, a, i);
            dj = gradFunc(M, b, a, j);
            if (y[i] == y[j])
                vi = a[i] - (1/((M[i,i]+M[j,j]-2*M[i,j])))*(di-dj);
                vj = a[j] - (1/((M[i,i]+M[j,j]-2*M[i,j])))*(dj-di);
                aii = max(a[i]-(c-a[j]), max(0, min(c, min(a[i]+a[j], vi))));
                ajj = max(a[j]-(c-a[i]), max(0, min(c, min(a[j]+a[i], vj))));
            else
                vi = a[i] - (1/((M[i,i]+M[j,j]+2*M[i,j])))*(di+dj);
                vj = a[j] - (1/((M[i,i]+M[j,j]+2*M[i,j])))*(dj+di);
                aii = min(a[i]+(c-a[j]), min(c, max(0, max(a[i]-a[j], vi))));
                ajj = min(a[j]+(c-a[i]), min(c, max(0, max(a[j]-a[i], vj))));
            end
            a[i] = aii;
            a[j] = ajj;
            f0 = f1;
            it = it+1;
            f1 = objFunc(M, b, a);
            pt = [pt; f1];
        end
    end
    w = a'*(Xtrain.*ytrain);
    b = mean(ytrain[0 .< a .< c].-((Xtrain[0 .< a .< c, :]*w')));
    return a, w, b;
end

function objFunc(M, b, a)
    return 0.5*(a'*M*a)-b'*a;
end

function gradFunc(M, b, a, i)
    if (i == 0)
        return M'*a - b;
    else
        return M[i,:]'*a - b[i];
    end
end

function reg(Xtrain, ytrain)
    Xtrain2 = [Xtrain ones(size(Xtrain, 1))];
    (nTrain, nFeatures) = size(Xtrain2);
    m2 = Model(solver=IpoptSolver(print_level=0));
    @variable(m2, x[1:size(Xtrain2, 2)])
    @variable(m2, ep[1:size(Xtrain2, 1)] >= 0)
    @objective(m2, Min, 0.5*x'*x + c*sum(ep))
    for i = 1:size(Xtrain, 1)
        @constraint(m2, ytrain[i]*(Xtrain2[i,:]'*x) >= 1 - ep[i])
    end
    solve(m2)
    w2 = getvalue(x);
    return w2;
end

function unreg(Xtrain, ytrain)
    (nTrain, nFeatures) = size(Xtrain);
    m3 = Model(solver=IpoptSolver(print_level=0));
    @variable(m3, x[1:size(Xtrain, 2)])
    @variable(m3, ep[1:size(Xtrain, 1)] >= 0)
    @variable(m3, bb)
    @objective(m3, Min, 0.5*x'*x + c*sum(ep))
    for i = 1:size(Xtrain, 1)
        @constraint(m3, ytrain[i]*((Xtrain[i,:]'*x) + bb) >= 1 - ep[i])
    end
    solve(m3)
    w3 = getvalue(x);
    return w3, bb;
end
