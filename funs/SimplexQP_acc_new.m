function [x, obj] = SimplexQP_acc_new(A, b, x0)
NIter = 500;
NStop = 20;
lambda = 1;
epsilon = 1e-3;

n = size(A, 1);
if nargin < 3
    x = 1/n * ones(n, 1);
else
    x = x0;
end

x1 = x;
t = 1;
t1 = 0;
r = 0.5;

for iter = 1:NIter
    p = (t1 - 1) / t;
    s = x + p * (x - x1);
    x1 = x;
    g = 2 * A * s - b + 2 * lambda * s;
    ob1 = x' * A * x - x' * b + lambda * (x' * x);

    for it = 1:NStop
        z = s - r * g;
        z = EProjSimplex_new(z, 1);
        z = max(z, epsilon);
        ob = z' * A * z - z' * b + lambda * (z' * z); 

        if ob1 < ob
            r = 0.5 * r;
        else
            break;
        end
    end

    if it == NStop
        obj(iter) = ob;
        break;
    end

    x = z;
    t1 = t;
    t = (1 + sqrt(1 + 4 * t^2)) / 2;
    obj(iter) = ob;
end