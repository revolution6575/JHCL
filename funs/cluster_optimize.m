function [y1,y2,S] = cluster_optimize(B_fusion, cluster_num, optimize_iter_SF)


zr = 10e-11;
lambda = 0.1;



[n,m] = size(B_fusion);

B_fusion = sparse(B_fusion);
d_n0 = sum(B_fusion,2);
D_n0 = spdiags(1./sqrt(d_n0),0,n,n);
d_m0 = sum(B_fusion,1);
D_m0 = spdiags(1./sqrt(d_m0'),0,m,m);
A_all = D_n0*B_fusion*D_m0;

A_all = sparse(A_all);
B_fusion = sparse(B_fusion);

% SVD decomposition
SVD_0 = A_all'*A_all;
SVD_0 = full(SVD_0);

% Calculate eigenvalues and eigenvectors
[V, ev0, ev]=eig1(SVD_0,m);

% F
V = V(:,1:cluster_num);
U=(A_all*V)./(ones(n,1)*sqrt(ev0(1:cluster_num)'));
U = sqrt(2)/2*U;
V = sqrt(2)/2*V;



if sum(ev(1:cluster_num+1)) > (cluster_num+1)*(1-zr)
    error('The original graph has more than %d connected component', cluster_num);
end



%% Optimization
B_fusion = full(B_fusion);

index_r = cell(n,1);

for i=1:n
    index_r_0 = 1:m;
    index_r{i} = index_r_0;
end

index_c = cell(m,1);

for i=1:m
    index_c_0 = 1:n;
    index_c{i} = index_c_0;
end



D_n = 1; 
D_m = 10;

for iter_SF = 1:optimize_iter_SF

    U1 = D_n*U;
    V1 = D_m*V;
    
    dist = L2_distance_1(U1',V1');
    

    
    %% Optimize S (by row)
    S_r = zeros(n,m);

    for i=1:n
        
        index_r_0 = index_r{i};
        ai_r = B_fusion(i,index_r_0);
        di_r = dist(i,index_r_0);
        q_r = (ai_r-0.5*lambda*di_r);
        
        n_r = length(q_r);
        v0_r = q_r-mean(q_r) + 1/n_r;

        vmin_r = min(v0_r);
        if vmin_r < 0
            j = 1;
            lambda_m_r = 0;
            while 1
                v1_r = v0_r - lambda_m_r;
                posidx_r = find(v1_r>0);
                npos_r = length(posidx_r);
                g_r = -npos_r;
                f_r = sum(v1_r(posidx_r)) - 1;
                lambda_m_r = lambda_m_r - f_r/g_r;
                j = j+1;
                if abs(f_r) < 10^-6
                    break;
                end
            end
            vv_r = max(v1_r,0);
            S_r(i,index_r_0) = vv_r;
        else
            S_r(i,index_r_0) = v0_r;
        end
    end



    %% Optimize S (by column)
    S_c = zeros(m,n);
    
    for i=1:m
        index_c_0 = index_c{i};
        ai_c = B_fusion(index_c_0,i);
        di_c = dist(index_c_0,i);
        q_c = (ai_c-0.5*lambda*di_c);

        n_c = length(q_c);
        v0_c = q_c-mean(q_c) + 1/n_c;

        vmin_c = min(v0_c);
        if vmin_c < 0
            lambda_m_c = 0;
            while 1
                v1_c = v0_c - lambda_m_c;
                posidx_c = find(v1_c>0);
                npos_c = length(posidx_c);
                g_c = -npos_c;
                f_c = sum(v1_c(posidx_c)) - 1;
                lambda_m_c = lambda_m_c - f_c/g_c;
                if abs(f_c) < 10^-6
                    break;
                end
            end
            vv_c = max(v1_c,0);
            S_c(i,index_c_0) = vv_c;
        else
            S_c(i,index_c_0) = v0_c;
        end
    end



    %% Obtain S (take the average)
    S_r = sparse(S_r);
    S_c = sparse(S_c);
    S = (S_r+S_c')/2;



    %% Update the degree matrix, F
    D_n_old = D_n;
    D_m_old = D_m;
    d_n = sum(S,2);
    D_n = spdiags(1./sqrt(d_n),0,n,n);
    d_m = sum(S,1);
    D_m = spdiags(1./sqrt(d_m'),0,m,m);
    
    DSD = D_n*S*D_m;

    SVD_0 = DSD'*DSD; 
    SVD_0 = full(SVD_0); 
    
    U_old = U;
    V_old = V;
    [V, ev0, ev] = eig1(SVD_0,cluster_num);
    U = (DSD*V)./(ones(n,1)*sqrt(ev0'));
    U = sqrt(2)/2*U;
    V = sqrt(2)/2*V;
    


    %% Update lambda
    fn1 = sum(ev(1:cluster_num));
    fn2 = sum(ev(1:cluster_num+1));

    if fn1 < cluster_num-0.000000001
        lambda = 2*lambda;
    elseif fn2 > cluster_num+1-0.000000001
        lambda = lambda/1.5;
        U = U_old;
        V = V_old;
        D_n = D_n_old;
        D_m = D_m_old;
    else
        break;
    end
end


%% Z
Z=sparse(n+m,n+m);
Z(1:n,n+1:end)=S;
Z(n+1:end,1:n)=S';
y=conncomp(graph(Z));
y1=y(1:n)';
y2=y(n+1:end)';