function [y_pred, y, best_view, d_P, iter_P] = Cluster(dataname, X, y, cluster_num, views_num, order, beta, isGraph, select_best_view_style, anchor_num, k ,optimize_iter_P, optimize_iter_SF, delta, disrupt_index_all)

%% Find the most reliable view
% best_view_all = zeros(20,1);
% for i = 1:20
%     [best_view] = select_best_view(X, y, views_num, cluster_num, select_best_view_style);
%     best_view_all(i) = best_view;
% end

if strcmp(dataname,'ORL')
    best_view = 1;
elseif strcmp(dataname,'Caltech101-7')
    best_view = 6;
else
    [best_view] = select_best_view(X, y, views_num, cluster_num, select_best_view_style);
end



%% Normalization
for v = 1:views_num
    X{v} = zscore(X{v});
end



%% Randomly disrupted data
[X, y] = disrupt_data(X, y, views_num, best_view, disrupt_index_all);



%% Generate bipartite graphs
if isGraph == 1
    B = X;
else
    n = size(X{1},1);
    B = cell(views_num,1);

    [anchor] = gen_anchor_score(X,anchor_num,views_num);



    %% Initialization of graphs
    for v = 1:views_num
        D_v = L2_distance_1(X{v}', anchor{v}');
        [~, index] = sort(D_v, 2);
        B{v} = zeros(n,anchor_num);
        for j = 1:n
            index_11 = index(j,1:k+1);
            d_11 = D_v(j, index_11);
            B{v}(j,index_11) = (d_11(k+1)-d_11)/(k*d_11(k+1)-sum(d_11(1:k))+eps);
        end
    end

    %% Generate high-order bipartite graph
    B_high = cell(views_num,order);     
    
    for v = 1:views_num
        
        B_high{v,1} = B{v};
        [U,S,V] = svd(B{v}, 'econ');
        
        for d = 2:order
            Temp = U * S.^(2*d-1) * V';
            Temp = Temp ./ max(max(Temp));
            Temp(Temp < 1e-5) = 0;
            row_sums = sum(Temp, 2);
            row_sums(row_sums == 0) = 1;
            Temp_normalized = Temp ./ row_sums;
            B_high{v,d} = Temp_normalized;
        end
    end

end



%% Optimization of bipartite graphs
[n,m] = size(B{1});

P_all_all = cell(optimize_iter_P+1,1);
P_all_0 = zeros(n,n*views_num);
P_all_all{1} = P_all_0;

alpha_all = zeros(views_num*order,optimize_iter_P);
alpha(:,1) = ones(views_num*order,1)./(views_num*order);
alpha_all(:,1) = alpha;

B_high_aligned = B_high;

S = views_fusion_order(B_high_aligned, best_view, order ,alpha);

alpha_v = zeros(views_num,1);
for i = 1:views_num
    alpha_v(i) = alpha((i-1)*order+1)+alpha((i-1)*order+2)+alpha((i-1)*order+3);
end

B_fusion_v = cell(views_num,1);
for v = 1:views_num
    B_fusion_v{v} = views_fusion_order(B_high, v, order ,alpha);
end

I_n = eye(n);
I_m = eye(m);

P = cell(views_num,1);
Z = cell(views_num,1);
K = cell(views_num,1);
Y = cell(views_num,1);
for i = 1:views_num
    P{i} = zeros(n,n);
    Z{i} = zeros(n,m);
    K{i} = zeros(n,m);
    Y{i} = zeros(n,m);
end

mu = 10e-1;
max_mu = 10e8;
pho_mu = 1.5;

sX = [n, m, views_num];

loss_all_P = zeros(optimize_iter_P,1);

for iter_P = 1:optimize_iter_P

    %% Optimize P
    for v = 1:views_num
        if v == best_view
            P{v} = eye(n);
        else
            alphaPB = zeros(n,m);
            for vv = 1:views_num
                if vv ==v
                    alphaPB = alphaPB+alpha_v(vv)*P{vv}*B_fusion_v{vv};
                else
                    continue
                end
            end
            P{v} = (2*alpha_v(v)*(S-alphaPB)+mu*K{v}-Y{v})*B_fusion_v{v}'*(I_n/delta-B_fusion_v{v}/(I_m+B_fusion_v{v}'*B_fusion_v{v}/delta)*B_fusion_v{v}'/delta^2)/(2*alpha_v(v)^2+mu);

        end
    end

    P_all = zeros(n,n*views_num);
    for i = 1:views_num
        P_all(:,(i-1)*n+1:i*n) = P{i};
    end

    B_high_aligned = cell(views_num,order);
    for i = 1:views_num
        for j = 1:order
            B_high_aligned{i,j} = P{i}*B_high{i,j};
            B_high_aligned{i,j} = max(B_high_aligned{i,j},0);
            a = sum(B_high_aligned{i,j},2);
            B_high_aligned{i,j} = B_high_aligned{i,j}./a;
        end
    end

    [B_fusion] = views_fusion(B_high_aligned, views_num, order, alpha);

    B_fusion_v_aligned = cell(views_num,1);
    for v = 1:views_num
        B_fusion_v_aligned{v} = views_fusion_order(B_high_aligned, v, order ,alpha);
    end

    Z_tensor = cat(3, B_fusion_v_aligned{:,:});
    Y_tensor = cat(3,Y{:,:});

    z_row = Z_tensor(:);
    y_row = Y_tensor(:);
    [k_row, ~] = wshrinkObj_weight(z_row + 1/mu*y_row,beta/mu,sX,0,3); 
    K_tensor = reshape(k_row,sX);
    y_row = y_row+mu*(z_row-k_row);
    mu = min(mu*pho_mu, max_mu);

    for i = 1: views_num
        K{i}=K_tensor(:,:,i);
        Y_tensor = reshape(y_row,sX);
        Y{i}=Y_tensor(:,:,i);
    end 



    %% Optimize alpha

    B_row = zeros(n*m,views_num*order);
    for v = 1:views_num
        for j = 1:order
            B_row(:,order*(v-1)+j) = reshape(B_high_aligned{v,j},[n*m 1]); % B_row是一个n*m * n_view的矩阵
        end
    end
    
    A = B_row'*B_row;
    S_row = reshape(S,[n*m 1]);
    s = 2*B_row'*S_row;
    [alpha, ~] =SimplexQP_acc_new(A, s , alpha);
    alpha_all(:,iter_P+1) = alpha;

    alpha_v = zeros(views_num,1);
    for i = 1:views_num
        alpha_v(i) = alpha((i-1)*order+1)+alpha((i-1)*order+2)+alpha((i-1)*order+3);
    end

    B_fusion_v = cell(views_num,1);
    for v = 1:views_num
        B_fusion_v{v} = views_fusion_order(B_high, v, order ,alpha);
    end



    %% Optimize S,F
    [y_pred,~,S] = cluster_optimize(B_fusion, cluster_num, optimize_iter_SF);



    %% Calculate the loss function value
    alphaPB = zeros(n,m);
    for i = 1:views_num
        alphaPB = alphaPB+alpha_v(i)*P{i}*B_fusion_v{i};
    end
    loss_P = norm(alphaPB-S, 'fro')^2;
    loss_all_P(iter_P) = loss_P;



    %% Calculate the change in P and jump out of the loop
    P_all_all{iter_P+1} = P_all;
    d_P = zeros(iter_P,1);


    for i = 1:iter_P
        d_P(i) = norm(P_all_all{iter_P+1}-P_all_all{i},2);
    end


    if iter_P ~= 1
        if loss_all_P(iter_P)>1.01*loss_all_P(iter_P-1)
            y_pred = y_pred_0;
            break
        elseif loss_all_P(iter_P)>0.999*loss_all_P(iter_P-1)
            break
        end
    end
    y_pred_0 = y_pred;

    if min(d_P) < 0.5
        break;
    end
end